#!/usr/bin/env python3
"""Locates a workload's disruption windows from JobSet/Job Kubernetes events.

Container logs for a long-lived, many-host workload are enormous; scanning all
of them to find where it restarted is what makes a naive investigation slow.
The JobSet controller already records every disruption as a handful of small
`k8s_cluster` events (BackoffLimitExceeded, RestartJobSetFailurePolicyAction,
SliceDeleted, ...). Reading those first yields padded time windows that the
expensive container-log queries (node resolution, culprit extraction) can then
be confined to.

This does NOT replace the full-window crash sweep: a disruption the workload
recovered from without a JobSet restart leaves no event here.
"""

import argparse
import datetime
import json
import os
from typing import Any

import log_query

JOBSET_REASONS = (
    "BackoffLimitExceeded",
    "RestartJobSetFailurePolicyAction",
    "SliceDeleted",
    "JobSetSuspended",
    "JobSetUnsuspended",
    "SuspendedJobs",
    "ResumeJobs",
    "SliceCreated",
)
# Reasons that mark a disruption, as opposed to routine lifecycle churn.
DISRUPTION_REASONS = (
    "BackoffLimitExceeded",
    "RestartJobSetFailurePolicyAction",
)
TEARDOWN_REASONS = ("SliceDeleted",)


def build_filter(workload: str, namespace: str) -> str:
  reasons = " OR ".join(
      f'jsonPayload.reason="{reason}"' for reason in JOBSET_REASONS
  )
  ns = (
      f' AND jsonPayload.involvedObject.namespace="{namespace}"'
      if namespace
      else ""
  )
  return (
      'resource.type="k8s_cluster" AND ('
      '(jsonPayload.involvedObject.kind="JobSet" AND '
      f'jsonPayload.involvedObject.name="{workload}") OR '
      '(jsonPayload.involvedObject.kind="Job" AND '
      f'jsonPayload.involvedObject.name:"{workload}-")'
      f"){ns} AND ({reasons})"
  )


def normalize(entry: dict[str, Any]) -> dict[str, Any]:
  payload = entry.get("jsonPayload") or {}
  obj = payload.get("involvedObject") or {}
  return {
      "timestamp": entry.get("timestamp", ""),
      "kind": obj.get("kind", ""),
      "name": obj.get("name", ""),
      "reason": payload.get("reason", ""),
      "message": payload.get("message", ""),
  }


def merge_windows(
    points: list[tuple[str, str]], padding_minutes: int
) -> list[dict[str, Any]]:
  """Pads each (timestamp, reason) point and merges overlapping windows."""
  pad = datetime.timedelta(minutes=padding_minutes)
  spans = sorted(
      (log_query.parse_ts(ts) - pad, log_query.parse_ts(ts) + pad, reason)
      for ts, reason in points
  )
  merged = []
  for span_start, span_end, reason in spans:
    if merged and span_start <= merged[-1]["_end"]:
      merged[-1]["_end"] = max(merged[-1]["_end"], span_end)
      merged[-1]["reasons"].add(reason)
      merged[-1]["events"] += 1
    else:
      merged.append(
          {
              "_start": span_start,
              "_end": span_end,
              "reasons": {reason},
              "events": 1,
          }
      )
  return [
      {
          "start": log_query.fmt_ts(window["_start"]),
          "end": log_query.fmt_ts(window["_end"]),
          "reasons": sorted(window["reasons"]),
          "events": window["events"],
      }
      for window in merged
  ]


def analyze(
    events: list[dict[str, Any]], padding_minutes: int
) -> dict[str, Any]:
  """Extracts restarts, backoffs, teardown and padded disruption windows.

  Args:
    events: Normalized events (see `normalize`).
    padding_minutes: Minutes to pad each disruption point on both sides.

  Returns:
    A dict with restart, backoff and teardown timestamps, the merged
    disruption windows, and per-reason event counts.
  """
  restarts = [
      event["timestamp"]
      for event in events
      if event["reason"] == "RestartJobSetFailurePolicyAction"
  ]
  backoffs = [
      event["timestamp"]
      for event in events
      if event["reason"] == "BackoffLimitExceeded"
  ]
  # A SliceDeleted after the last restart/backoff is the workload being torn
  # down; earlier ones are part of a restart cycle and already covered.
  last_disruption = max(restarts + backoffs, default="")
  teardown = [
      event["timestamp"]
      for event in events
      if event["reason"] in TEARDOWN_REASONS
      and event["timestamp"] >= last_disruption
  ]
  points = [
      (event["timestamp"], event["reason"])
      for event in events
      if event["reason"] in DISRUPTION_REASONS
  ]
  points += [(ts, "SliceDeleted") for ts in teardown]
  counts = {}
  for event in events:
    counts[event["reason"]] = counts.get(event["reason"], 0) + 1
  return {
      "restarts": restarts,
      "backoffs": backoffs,
      "teardown": teardown,
      "windows": merge_windows(points, padding_minutes) if points else [],
      "counts": counts,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Locate disruption windows from JobSet/Job events."
  )
  parser.add_argument("--workload_name", required=True)
  parser.add_argument("--project_id", required=True)
  parser.add_argument("--start", required=True, help="UTC YYYY-MM-DDTHH:MM:SSZ")
  parser.add_argument("--end", required=True, help="UTC YYYY-MM-DDTHH:MM:SSZ")
  parser.add_argument("--namespace", default="default")
  parser.add_argument("--padding_minutes", type=int, default=15)
  parser.add_argument(
      "--output_file", default="scratch/disruption_windows.json"
  )
  parser.add_argument(
      "--limit", type=int, default=2000, help="Entry cap per one-day slice."
  )
  args = parser.parse_args()

  try:
    log_query.preflight(args.project_id)
    res = log_query.read_sliced(
        args.project_id,
        build_filter(args.workload_name, args.namespace),
        args.start,
        args.end,
        slice_minutes=24 * 60,
        limit_per_slice=args.limit,
        order="asc",
    )
  except log_query.LoggingQueryError as exc:
    log_query.fail(exc, context="locate-disruption-windows")
    return

  events = [normalize(entry) for entry in res["entries"]]
  result = {
      "status": "OK",
      "workload_name": args.workload_name,
      "project_id": args.project_id,
      "search_window": {"start": args.start, "end": args.end},
      "padding_minutes": args.padding_minutes,
      "total_events": len(events),
      "truncated": res["truncated"],
      "truncated_slices": [
          slice_stat for slice_stat in res["slices"] if slice_stat["truncated"]
      ],
      **analyze(events, args.padding_minutes),
      "events": events,
  }
  if res["truncated"]:
    result["warning"] = (
        "At least one one-day slice hit --limit; events (and windows) may be "
        "missing. Re-run with a higher --limit."
    )

  if args.output_file:
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

  print(
      json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2)
  )
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"  status: {result['status']}")
  print(f"  total_events: {result['total_events']}")
  print(f"  restarts: {len(result['restarts'])}")
  print(f"  teardown: {', '.join(result['teardown']) or 'none'}")
  print(f"  windows: {len(result['windows'])}")
  for window in result["windows"]:
    print(
        f"    {window['start']} -> {window['end']}"
        f" ({', '.join(window['reasons'])})"
    )
  print(f"  truncated: {result['truncated']}")
  if args.output_file:
    print(f"  output_file: {args.output_file}")
  print("---------------------------------------")


if __name__ == "__main__":
  main()

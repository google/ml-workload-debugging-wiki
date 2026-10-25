#!/usr/bin/env python3
"""Resolves the complete set of GKE nodes that hosted a workload's pods.

A single Cloud Logging query silently truncates at its entry limit, which
yields a partial node set. Downstream hardware analysis then renders a verdict
over only part of the fleet without any indication that it did so.

This script slices the window, unions the results, and then keeps halving the
slice width until the node set stops growing. Convergence -- not the absence of
entry-cap saturation -- is the completeness signal: a chatty workload saturates
on log volume long after every node has been seen, so treating saturation alone
as failure produces permanent false alarms.

If no container ever logged (pods rejected at admission / scheduling / mount /
image pull), it falls back to the pods' Kubernetes events: `Scheduled`
messages and kubelet `source.host` give the node set, and
`pre_start_failures` counts the reasons (`node_source` says which path ran).

It also emits two ready-made scoping clauses so downstream queries cannot
accidentally run unscoped: `node_filter_clause` for audit logs
(`protoPayload.resourceName`) and `node_name_filter_clause` for `k8s_node`
queries (`resource.labels.node_name`). They are not interchangeable -- each
field exists only on its own log shape.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
from collections.abc import Sequence
import concurrent.futures
import datetime
import json
import os
import random
import re
import subprocess
import sys
import time
from typing import Any

import log_query

NODE_LABEL = "compute.googleapis.com/resource_name"
_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Cloud Logging read quota (~60 entries.list/min/project) is shared by every
# script running in parallel; a 429 is transient. Same policy as
# extract_culprit_hosts.py: backoff + jitter on quota/unavailable only.
_RETRYABLE_RE = re.compile(
    r"429|RESOURCE_EXHAUSTED|Quota exceeded|503|UNAVAILABLE", re.IGNORECASE
)
_HTTP_STATUS_RE = re.compile(
    r"(?i)(?:http\w*|status|code)[^0-9\n]{0,20}\b([45]\d\d)\b"
)
_BACKOFF_SECONDS = (5, 10, 20)
_QUERY_TIMEOUT_SECONDS = 600
_ERROR_EXCERPT_CHARS = 200
_SAMPLE_MESSAGE_CHARS = 300
# Final (post-retry) failures of every query in this run, surfaced in the
# report so a failed query can never pass for an empty node set.
_query_errors: list[dict[str, object]] = []


def gcloud_read(command: Sequence[str], label: str) -> str | None:
  """Runs a gcloud read, retrying on quota/unavailable errors.

  Args:
    command: The gcloud argv.
    label: Query name recorded in the report if the read fails.

  Returns:
    The command's stdout, or None after recording the final error in
    `_query_errors`.
  """
  query_error = {}
  for attempt in range(len(_BACKOFF_SECONDS) + 1):
    try:
      completed = subprocess.run(
          command,
          capture_output=True,
          text=True,
          check=False,
          timeout=_QUERY_TIMEOUT_SECONDS,
      )
    except subprocess.TimeoutExpired:
      stderr = f"gcloud timed out after {_QUERY_TIMEOUT_SECONDS}s"
    except (subprocess.SubprocessError, OSError) as e:
      stderr = f"gcloud execution failed: {e}"
    else:
      if completed.returncode == 0:
        return completed.stdout
      stderr = completed.stderr or f"gcloud exited {completed.returncode}"
    status = _HTTP_STATUS_RE.search(stderr)
    query_error = {
        "query": label,
        "attempts": attempt + 1,
        "http_status": int(status.group(1)) if status else None,
        "error": " ".join(stderr.split())[:_ERROR_EXCERPT_CHARS],
    }
    if not _RETRYABLE_RE.search(stderr) or attempt == len(_BACKOFF_SECONDS):
      break
    delay = _BACKOFF_SECONDS[attempt]
    time.sleep(delay + random.uniform(0, delay / 2))
  _query_errors.append(query_error)
  print(
      f"Warning: query {label} FAILED: {query_error['error']}",
      file=sys.stderr,
  )
  return None


def parse_ts(value: str) -> datetime.datetime:
  return datetime.datetime.strptime(value.strip(), _TS_FMT).replace(
      tzinfo=datetime.timezone.utc
  )


def slice_window(
    start: datetime.datetime, end: datetime.datetime, minutes: int
) -> collections.abc.Iterator[tuple[datetime.datetime, datetime.datetime]]:
  """Splits [start, end] into <=minutes-long slices to dodge the entry cap."""
  step = datetime.timedelta(minutes=minutes)
  cursor = start
  while cursor < end:
    nxt = min(cursor + step, end)
    yield cursor, nxt
    cursor = nxt


def query_nodes(
    project_id: str,
    base_filter: str,
    start: datetime.datetime,
    end: datetime.datetime,
    limit: int,
) -> tuple[set[str], int]:
  """Returns the distinct node names seen in one time slice.

  `base_filter` is the workload selector plus scope clauses (no time bound).
  Access and exhausted-transient failures raise LoggingQueryError so main()
  aborts the run: no node set built on a refused query is worth reporting.

  Args:
    project_id: GCP project ID to read from.
    base_filter: Workload selector plus scope clauses.
    start: Slice start (inclusive).
    end: Slice end (inclusive).
    limit: Maximum number of log entries to read.

  Returns:
    A (nodes, line_count) tuple: the distinct node names and the number of
    non-empty output lines read, or (set(), -1) if the query failed for a
    reason other than access or an exhausted transient error.

  Raises:
    log_query.LoggingQueryError: the query was refused, or failed transiently
      after retries.
  """
  log_filter = (
      f"{base_filter} AND "
      f'timestamp>="{start.strftime(_TS_FMT)}" AND '
      f'timestamp<="{end.strftime(_TS_FMT)}"'
  )
  try:
    out = log_query.run_gcloud_read(
        project_id, log_filter, limit, fmt=f'value(labels."{NODE_LABEL}")'
    )
  except log_query.LoggingQueryError as exc:
    if exc.kind in ("access", "transient"):
      raise
    status = _HTTP_STATUS_RE.search(str(exc))
    _query_errors.append({
        "query": (
            f"container_logs:{start.strftime(_TS_FMT)}..{end.strftime(_TS_FMT)}"
        ),
        "attempts": 1,
        "http_status": int(status.group(1)) if status else None,
        "error": str(exc)[:_ERROR_EXCERPT_CHARS],
    })
    print(
        f"Warning: node query failed for slice {start}-{end}: {exc}",
        file=sys.stderr,
    )
    # Signal failure distinctly. Returning (set(), 0) would be indistinguishable
    # from a clean under-cap slice, letting one failed query produce a silently
    # truncated node set reported as COMPLETE -- precisely the failure this
    # module exists to prevent.
    return set(), -1
  lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
  return {ln for ln in lines if ln}, len(lines)


def group_of(node_name: str) -> str:
  """Strips the per-instance suffix to get the node-pool group prefix."""
  return re.sub(r"-[a-z0-9]+$", "", node_name)


def make_clauses(
    nodes: set[str], groups: dict[str, list[str]]
) -> tuple[str, str]:
  """Constructs audit-log and k8s_node scoping filter clauses."""
  _clause = " OR ".join(
      f'protoPayload.resourceName:"{group}"' for group in sorted(groups)
  )
  node_filter_clause = f"({_clause})" if _clause else ""
  _node_clause = " OR ".join(
      f'resource.labels.node_name="{node}"' for node in sorted(nodes)
  )
  node_name_filter_clause = f"({_node_clause})" if _node_clause else ""
  return node_filter_clause, node_name_filter_clause


# Pods that failed BEFORE any container started (admission, scheduling, mount,
# image pull, eviction) emit no k8s_container logs, so the slice queries above
# find no nodes. Their Kubernetes events still exist. Filter shape is the one
# pull_and_format_logs.py (pod-analyzer stages 1/2) already uses:
# logName:"/logs/events" AND resource.type="k8s_pod" + jsonPayload.reason.
PRE_START_REASONS = (
    "FailedScheduling",
    "UntoleratedTaint",
    "Inadmissible",
    "UnexpectedAdmissionError",
    "FailedMount",
    "Failed",
    "BackOff",
    "ErrImagePull",
    "ImagePullBackOff",
    "Evicted",
)
# Scheduler `Scheduled` event message, e.g. (verbatim in
# tests/test_cases/tc-my-training-job-multi-workload-mock/mock_data/
# run_6_pod_events.txt): "Successfully assigned default/<pod> to <node>".
SCHEDULED_RE = re.compile(
    r"Successfully assigned (?P<ns>[^/\s]+)/(?P<pod>\S+) to (?P<node>\S+)"
)


def query_pod_events(
    project_id: str,
    workload_name: str,
    start: datetime.datetime,
    end: datetime.datetime,
    limit: int,
) -> tuple[list[dict[str, str]], int]:
  """Returns (events, returned) for the workload's pod events; -1 on failure.

  Each event is {timestamp, pod, reason, message, source_host}. source_host is
  the k8s Event `source.host` (core/v1 EventSource.Host): set by the kubelet,
  i.e. the node, on kubelet-emitted events such as UnexpectedAdmissionError.

  Args:
    project_id: GCP project ID.
    workload_name: Pod-name prefix of the workload.
    start: Window start (UTC).
    end: Window end (UTC).
    limit: Entry cap for the read.

  Returns:
    (events, returned): returned is the raw entry count, or -1 on failure.
  """
  reasons = " OR ".join(
      f'jsonPayload.reason="{reason}"'
      for reason in ("Scheduled", *PRE_START_REASONS)
  )
  log_filter = (
      'logName:"/logs/events" AND resource.type="k8s_pod" AND '
      f'resource.labels.pod_name:"{workload_name}-" AND '
      f'timestamp>="{start.strftime(_TS_FMT)}" AND '
      f'timestamp<="{end.strftime(_TS_FMT)}" AND ({reasons})'
  )
  command = [
      "gcloud",
      "logging",
      "read",
      log_filter,
      "--project",
      project_id,
      "--limit",
      str(limit),
      "--order",
      "ASC",
      "--format",
      "json",
  ]
  stdout = gcloud_read(command, "pod_events")
  if stdout is None:
    return [], -1
  try:
    raw = json.loads(stdout or "[]")
  except json.JSONDecodeError as exc:
    _query_errors.append({
        "query": "pod_events",
        "attempts": 1,
        "http_status": None,
        "error": f"unparseable gcloud JSON: {exc}"[:_ERROR_EXCERPT_CHARS],
    })
    return [], -1
  events = []
  for entry in raw:
    payload = entry.get("jsonPayload") or {}
    labels = (entry.get("resource") or {}).get("labels") or {}
    events.append({
        "timestamp": entry.get("timestamp", ""),
        "pod": labels.get("pod_name", ""),
        "reason": payload.get("reason", ""),
        "message": payload.get("message", ""),
        "source_host": (payload.get("source") or {}).get("host", ""),
    })
  return events, len(raw)


def summarize_pod_events(
    events: Sequence[dict[str, str]], max_samples: int = 2
) -> tuple[set[str], dict[str, object]]:
  """Returns (nodes, pre_start_failures) from pod events.

  Nodes come from `Scheduled` messages and from kubelet `source.host`; both
  identify the node a pod was bound to even if no container ever started.

  Args:
    events: Dicts from query_pod_events().
    max_samples: Distinct sample messages kept per reason.

  Returns:
    (nodes, pre_start_failures).
  """
  nodes = set()
  counts = collections.Counter()
  samples = {}
  scheduled_pods = set()
  for event in events:
    reason = event["reason"]
    if reason == "Scheduled":
      match = SCHEDULED_RE.search(event["message"] or "")
      if match:
        nodes.add(match.group("node"))
        scheduled_pods.add(match.group("pod"))
      continue
    if event.get("source_host"):
      nodes.add(event["source_host"])
    counts[reason] += 1
    reason_samples = samples.setdefault(reason, [])
    message = (event["message"] or "")[:_SAMPLE_MESSAGE_CHARS]
    if len(reason_samples) < max_samples and message not in reason_samples:
      reason_samples.append(message)
  return nodes, {
      "pods_scheduled": len(scheduled_pods),
      "counts": dict(counts.most_common()),
      "samples": samples,
  }


def _extract_nodes_from_obj(obj: Any, nodes: set[str]) -> None:
  """Recursively extracts node names from pre-resolved mapping objects."""
  if isinstance(obj, dict):
    node_name = obj.get("node_name")
    if isinstance(node_name, str) and node_name.strip():
      nodes.add(node_name.strip())
    for key in ("node_to_host_map", "node_mapping", "nodes", "attempts"):
      val = obj.get(key)
      if isinstance(val, list):
        for item in val:
          if isinstance(item, str) and item.strip():
            nodes.add(item.strip())
          elif isinstance(item, dict):
            _extract_nodes_from_obj(item, nodes)
      elif isinstance(val, dict):
        for k, v in val.items():
          if isinstance(k, str) and k.startswith("gke-"):
            nodes.add(k.strip())
          elif isinstance(v, dict):
            _extract_nodes_from_obj(v, nodes)
  elif isinstance(obj, list):
    for item in obj:
      if isinstance(item, str) and item.strip():
        nodes.add(item.strip())
      elif isinstance(item, dict):
        _extract_nodes_from_obj(item, nodes)


def load_nodes_from_timestamps_json(json_path: str) -> set[str]:
  """Extracts distinct GKE node names from pre-resolved mapping JSON file."""
  path = json_path.lstrip("@").strip()
  if not path or not os.path.isfile(path):
    return set()
  nodes: set[str] = set()
  try:
    with open(path, encoding="utf-8") as f:
      raw_text = f.read().strip()
  except OSError:
    return set()
  if not raw_text:
    return set()
  try:
    parsed = json.loads(raw_text)
    _extract_nodes_from_obj(parsed, nodes)
    return nodes
  except json.JSONDecodeError:
    for line in raw_text.splitlines():
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      _extract_nodes_from_obj(row, nodes)
    return nodes


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Resolve the complete node set for a workload."
  )
  parser.add_argument("--workload_name", required=True)
  parser.add_argument("--project_id", required=True)
  parser.add_argument("--start", required=True, help="UTC YYYY-MM-DDTHH:MM:SSZ")
  parser.add_argument("--end", required=True, help="UTC YYYY-MM-DDTHH:MM:SSZ")
  parser.add_argument("--output_file", default="scratch/workload_nodes.json")
  parser.add_argument(
      "--expected_hosts",
      type=int,
      default=0,
      help=(
          "Expected TPU host count (slices x hosts_per_slice). 0 disables the"
          " check."
      ),
  )
  parser.add_argument("--slice_minutes", type=int, default=30)
  parser.add_argument(
      "--min_slice_minutes",
      type=int,
      default=5,
      help="Finest slice width to refine down to.",
  )
  parser.add_argument(
      "--max_refinements",
      type=int,
      default=4,
      help="How many times to halve the width of saturated intervals.",
  )
  parser.add_argument(
      "--max_workers",
      type=int,
      default=8,
      help="Concurrent Cloud Logging queries.",
  )
  parser.add_argument("--limit", type=int, default=5000)
  parser.add_argument(
      "--cluster_name", default="", help="Optional exact cluster scope."
  )
  parser.add_argument(
      "--namespace", default="", help="Optional exact namespace scope."
  )
  parser.add_argument(
      "--timestamps_json",
      default="",
      help=(
          "Optional pre-resolved timestamps and node-mapping JSON path."
          " When populated with GKE nodes, skips Cloud Logging slice queries."
      ),
  )
  args = parser.parse_args()

  _query_errors.clear()
  start, end = parse_ts(args.start), parse_ts(args.end)
  if end <= start:
    print("Error: --end must be after --start", file=sys.stderr)
    sys.exit(1)

  ts_candidate = (
      args.timestamps_json.strip()
      if hasattr(args, "timestamps_json") and args.timestamps_json
      else ""
  )
  if not ts_candidate:
    default_ts = f"scratch/workload_timestamps_{args.workload_name}.json"
    if os.path.isfile(default_ts):
      ts_candidate = default_ts
  ts_nodes = (
      load_nodes_from_timestamps_json(ts_candidate)
      if ts_candidate
      else set()
  )

  if ts_nodes:
    nodes = ts_nodes
    selector_mode = "timestamps_json"
    node_source = "task_mapping"
    converged = True
    passes = []
    total_failed = 0
    saturated_slices = 0
    total_queries = 0
    warnings = []
    pre_start_failures = {}
  else:
    try:
      log_query.preflight(args.project_id)
      selector, selector_mode = log_query.choose_workload_selector(
          args.project_id,
          args.workload_name,
          args.start,
          args.end,
          args.cluster_name,
          args.namespace,
      )
    except log_query.LoggingQueryError as exc:
      log_query.fail(exc, "resolve-workload-nodes pre-flight")
    base_filter = " AND ".join(
        ['resource.type="k8s_container"', selector]
        + log_query.scope_clauses(args.cluster_name, args.namespace)
    )

    # Adaptive subdivision, executed concurrently.
    #
    # Only intervals that actually saturated the entry cap are subdivided --
    # re-slicing the whole window on every pass repeats work already done and
    # grows the call count geometrically (uniform refinement over this window
    # issued 35 queries where adaptive subdivision needs a fraction of that).
    # A slice that came back under the cap was not truncated, so every node in
    # it has already been seen and it never needs revisiting.
    nodes = set()
    passes = []
    total_failed = 0
    pending = list(slice_window(start, end, args.slice_minutes))
    minutes = args.slice_minutes
    converged = False

    for _ in range(args.max_refinements + 1):
      if not pending:
        converged = True
        break

      results = []
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=args.max_workers
      ) as pool:
        futures = {
            pool.submit(
                query_nodes,
                args.project_id,
                base_filter,
                start_ts,
                end_ts,
                args.limit,
            ): (start_ts, end_ts)
            for start_ts, end_ts in pending
        }
        for fut in concurrent.futures.as_completed(futures):
          start_ts, end_ts = futures[fut]
          try:
            found, returned = fut.result()
          except log_query.LoggingQueryError as exc:
            if exc.kind in ("access", "transient"):
              # Abort: an expired token mid-run would otherwise yield a node set
              # from only the slices that ran before it expired. Cancel queued
              # slices first so the pool's exit does not wait for all of them.
              pool.shutdown(wait=False, cancel_futures=True)
              log_query.fail(
                  exc, f"resolve-workload-nodes slice {start_ts}-{end_ts}"
              )
            print(
                f"Warning: slice {start_ts}-{end_ts} failed: {exc}",
                file=sys.stderr,
            )
            found, returned = set(), -1
          except (
              subprocess.SubprocessError,
              OSError,
              json.JSONDecodeError,
              KeyError,
              ValueError,
          ) as exc:
            print(
                f"Warning: slice {start_ts}-{end_ts} raised {exc}",
                file=sys.stderr,
            )
            # -1, not 0: a raised exception is a failed query, not a clean
            # slice.
            found, returned = set(), -1
          results.append((start_ts, end_ts, found, returned))

      before = len(nodes)
      saturated_intervals = []
      failed_intervals = []
      for start_ts, end_ts, found, returned in results:
        nodes |= found
        if returned < 0:
          failed_intervals.append((start_ts, end_ts))
        elif returned >= args.limit:
          saturated_intervals.append((start_ts, end_ts))

      added = len(nodes) - before
      passes.append({
          "slice_minutes": minutes,
          "intervals_queried": len(pending),
          "nodes_total": len(nodes),
          "nodes_added": added,
          "saturated_intervals": len(saturated_intervals),
          "failed_intervals": len(failed_intervals),
      })
      total_failed += len(failed_intervals)

      # Convergence is "a finer pass found nothing new", NOT "no interval hit
      # the entry cap". For a chatty workload the cap is never cleared at any
      # usable slice width -- every node has long since been seen while the log
      # volume keeps saturating -- so terminating on saturation subdivides
      # forever and reports a permanent false INCOMPLETE.
      if not saturated_intervals:
        converged = True
        break
      if added == 0 and len(passes) > 1:
        converged = True
        break
      if minutes <= args.min_slice_minutes:
        break

      # Subdivide only the truncated intervals.
      minutes = max(args.min_slice_minutes, minutes // 2)
      pending = []
      for start_ts, end_ts in saturated_intervals:
        pending.extend(slice_window(start_ts, end_ts, minutes))

    saturated_slices = passes[-1]["saturated_intervals"] if passes else 0
    total_queries = sum(
        refinement_pass["intervals_queried"] for refinement_pass in passes
    )

    warnings = []
    node_source = "container_logs" if nodes else "none"
    pre_start_failures = {}
    if not nodes:
      # No container ever logged: fall back to pod events so the agent still
      # gets a scoped node set AND sees why the containers never started.
      events, returned = query_pod_events(
          args.project_id, args.workload_name, start, end, args.limit
      )
      total_queries += 1
      event_nodes, pre_start_failures = summarize_pod_events(events)
      pre_start_failures["events_examined"] = len(events)
      if returned < 0:
        warnings.append(
            "Pod event fallback query FAILED; containers logged nothing and the"
            " node set could not be derived from events either."
        )
      elif returned >= args.limit:
        warnings.append(
            f"Pod event fallback hit --limit ({args.limit}); node set and"
            " pre_start_failures may be truncated. Raise --limit."
        )
      if event_nodes:
        nodes = event_nodes
        node_source = "pod_events"

  groups = {}
  for node in sorted(nodes):
    groups.setdefault(group_of(node), []).append(node)

  if total_failed:
    warnings.append(
        f"{total_failed} slice quer(y/ies) FAILED to execute. The node set is "
        "not verifiable and must not be treated as complete; re-run once the "
        "underlying gcloud error is resolved."
    )
  if not nodes:
    warnings.append(
        "No nodes resolved from container logs or pod events. Check the"
        " workload name prefix, window and project before scoping by node."
    )
  if saturated_slices and not converged:
    warnings.append(
        f"Node set may be incomplete: {saturated_slices} interval(s) still hit"
        f" the entry cap at {passes[-1]['slice_minutes']}m width. Lower"
        " --min_slice_minutes or raise --limit."
    )
  if args.expected_hosts:
    # The head/CPU node is not a TPU host, so compare on TPU groups only.
    tpu_nodes = [node for node in nodes if "-tpu-" in node]
    if len(tpu_nodes) < args.expected_hosts:
      warnings.append(
          f"INCOMPLETE: resolved {len(tpu_nodes)} TPU nodes but expected "
          f"{args.expected_hosts}. Downstream hardware analysis MUST NOT be "
          "treated as fleet-wide until this is reconciled."
      )

  node_filter_clause, node_name_filter_clause = make_clauses(nodes, groups)

  report = {
      "workload_name": args.workload_name,
      "project_id": args.project_id,
      "window": {"start": args.start, "end": args.end},
      "selector_mode": selector_mode,
      "total_nodes": len(nodes),
      "node_source": node_source,
      "pre_start_failures": pre_start_failures,
      "node_groups": {
          group: len(members) for group, members in sorted(groups.items())
      },
      "converged": converged,
      "queries_issued": total_queries,
      "refinement_passes": passes,
      "node_filter_clause": node_filter_clause,
      "node_name_filter_clause": node_name_filter_clause,
      "nodes": sorted(nodes),
      "completeness_warnings": warnings,
      "query_errors": list(_query_errors),
      "status": "INCOMPLETE" if warnings else "COMPLETE",
  }

  with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

  print(json.dumps({k: v for k, v in report.items() if k != "nodes"}, indent=2))
  print(f"\nFull node list written to {args.output_file}", file=sys.stderr)
  for warning in warnings:
    print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
  main()

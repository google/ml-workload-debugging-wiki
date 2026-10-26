#!/usr/bin/env python3
"""Job Disruption Timeline Extractor CLI.

Queries Cloud Logging metadata and extracts chronological disruption JSON
timelines.
"""

import argparse
from concurrent import futures
import datetime
import json
import os
import subprocess
import sys
from typing import Any

import log_query
import pull_and_format_logs


def extract_timeline(
    workload: str,
    start: str,
    end: str,
    project: str,
    cluster: str,
    location: str,
    namespace: str,
    max_workers: int,
) -> list[dict[str, Any]]:
  """Extracts chronological disruption timeline events from logging queries.

  Args:
    workload: Target workload/JobSet name.
    start: Start UTC timestamp.
    end: End UTC timestamp.
    project: GCP Project ID.
    cluster: Cluster name where workload ran.
    location: GCP location/region/zone.
    namespace: K8s namespace where workload ran.
    max_workers: Maximum concurrent query worker threads.

  Returns:
    List of deduplicated timeline events.
  """
  execute_logging_query = pull_and_format_logs.execute_logging_query

  head_query = (
      'resource.type="k8s_container" '
      f'AND resource.labels.project_id="{project}" '
      f'AND resource.labels.location="{location}" '
      f'AND resource.labels.cluster_name="{cluster}" '
      f'AND resource.labels.namespace_name="{namespace}" '
      f'AND labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"="{workload}" '
      'AND resource.labels.pod_name:"pathways-head" '
      'AND textPayload:"Logging before InitGoogle() is written to" '
      f'AND timestamp >= "{start}" AND timestamp <= "{end}"'
  )
  worker_query = (
      'resource.type="k8s_container" '
      f'AND resource.labels.project_id="{project}" '
      f'AND resource.labels.location="{location}" '
      f'AND resource.labels.cluster_name="{cluster}" '
      f'AND resource.labels.namespace_name="{namespace}" '
      f'AND labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"="{workload}" '
      'AND resource.labels.pod_name:"-0-0-" '
      'AND textPayload:"Logging before InitGoogle() is written to" '
      f'AND timestamp >= "{start}" AND timestamp <= "{end}"'
  )

  with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    head_future = executor.submit(
        execute_logging_query,
        project_id=project,
        query_filter=head_query,
        limit=5000,
    )
    worker_future = executor.submit(
        execute_logging_query,
        project_id=project,
        query_filter=worker_query,
        limit=5000,
    )

    head_logs = head_future.result() or []
    worker_logs = worker_future.result() or []
    # The head pod is named <jobset>-pathways-head-0-0-<hash>, which
    # satisfies BOTH the "pathways-head" and "-0-0-" substring filters, so
    # every head entry appears in both result sets. Concatenating without
    # dedupe emits two timeline events per head initialization.
    logs = head_logs + worker_logs
    seen_keys = set()
    deduped = []
    for entry in logs:
      key = (
          entry.get("timestamp"),
          (entry.get("resource", {}).get("labels", {}) or {}).get("pod_name"),
          entry.get("textPayload"),
      )
      if key in seen_keys:
        continue
      seen_keys.add(key)
      deduped.append(entry)
    logs = deduped

  if not logs:
    return []

  # Sort logs by timestamp ascending
  logs.sort(key=lambda log_entry: log_entry.get("timestamp", ""))

  timeline = []
  attempt = 1

  current_job_name = None
  current_head_pod = ""
  current_worker_pod = ""

  for log in logs:
    ts = log.get("timestamp")
    labels = log.get("labels", {})
    res_labels = log.get("resource", {}).get("labels", {})
    pod_name = res_labels.get("pod_name", "")
    job_name = (
        labels.get("k8s-pod/jobset_sigs_k8s_io/jobset-name")
        or labels.get("k8s-pod/job-name")
        or ""
    )

    is_head = "pathways-head" in pod_name
    is_worker = "-0-0-" in pod_name

    if is_head:
      head_changed = pod_name != current_head_pod
      current_head_pod = pod_name
    else:
      head_changed = False

    if is_worker:
      worker_changed = pod_name != current_worker_pod
      current_worker_pod = pod_name
    else:
      worker_changed = False

    job_changed = current_job_name is not None and job_name != current_job_name

    if current_job_name is None:
      event_type = "Initial Launch"
      trigger_reason = "Initial Launch"
      current_job_name = job_name
    elif job_changed:
      event_type = "Manual"
      trigger_reason = "Job Name Change"
      current_job_name = job_name
    elif head_changed or worker_changed:
      event_type = "Automated"
      trigger_reason = "Pod Name Change"
    else:
      event_type = "In-Place"
      trigger_reason = "In-Place Restart"

    timeline.append({
        "attempt": attempt,
        "timestamp": ts,
        "type": event_type,
        "trigger_reason": trigger_reason,
        "job_name": job_name,
        "cpu_pathways_head_podname": current_head_pod,
        "tpu_worker_headname": current_worker_pod,
    })
    attempt += 1

  return deduplicate_timeline(timeline)


def _parse_ts(ts_str: str) -> datetime.datetime:
  """Parses an ISO-8601 timestamp, accepting a trailing "Z" for UTC."""
  return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def deduplicate_timeline(
    raw_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Deduplicates JobSet initial/manual launch cascades.

  If an event is 'Manual' or 'Initial Launch', any subsequent 'In-Place' or
  'Automated' events that share the EXACT SAME head pod name
  (cpu_pathways_head_podname) and occur within 120s of the launch are part of
  the initial container startup cascade and are merged.

  Args:
    raw_events: Chronological sequence of extracted timeline events.

  Returns:
    Consolidated list of timeline events with launch cascades collapsed.
  """
  if not raw_events:
    return []

  deduped = []
  i = 0
  while i < len(raw_events):
    current = raw_events[i]

    # Check if this event initiates a launch cascade
    if current.get("type") in ["Manual", "Initial Launch"]:
      launch_head_pod = current.get("cpu_pathways_head_podname")
      launch_job_name = current.get("job_name")

      j = i + 1
      last_event_in_cascade = current
      current_ts = _parse_ts(current["timestamp"])

      while j < len(raw_events):
        next_ev = raw_events[j]
        next_head_pod = next_ev.get("cpu_pathways_head_podname")
        next_job_name = next_ev.get("job_name")
        next_ts = _parse_ts(next_ev["timestamp"])

        # Check cascade conditions:
        # 1. Same job_name
        # 2. Same head pod name (head pod was NOT restarted)
        # 3. Within 120s of the initial launch start
        if (
            next_job_name == launch_job_name
            and next_head_pod == launch_head_pod
            and (next_ts - current_ts).total_seconds() <= 120
        ):

          last_event_in_cascade = next_ev
          j += 1
        else:
          break

      # If sub-events were merged:
      merged_event = dict(current)
      if j > i + 1:
        merged_event["tpu_worker_headname"] = last_event_in_cascade.get(
            "tpu_worker_headname"
        ) or current.get("tpu_worker_headname")
        merged_event["convergence_timestamp"] = last_event_in_cascade[
            "timestamp"
        ]
        merged_event["startup_sub_events_count"] = j - i

      deduped.append(merged_event)
      i = j
    else:
      deduped.append(current)
      i += 1

  # Re-index attempt numbers sequentially
  for index, ev in enumerate(deduped, start=1):
    ev["attempt"] = index

  return deduped


def main() -> None:
  """Extracts and outputs chronological ML workload disruption timeline."""
  parser = argparse.ArgumentParser(
      description="Extract ML Workload Job Disruption Timeline."
  )
  parser.add_argument(
      "--workload", required=True, help="Target workload name or prefix"
  )
  parser.add_argument(
      "--start",
      required=True,
      help="Start UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)",
  )
  parser.add_argument(
      "--end", required=True, help="End UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)"
  )
  parser.add_argument(
      "--project", default="[GCP_PROJECT_ID]", help="GCP Project ID"
  )
  parser.add_argument("--output", help="Optional output JSON filepath")
  parser.add_argument(
      "--workers",
      type=int,
      default=2,
      help="Max concurrent workers for querying logs",
  )

  args = parser.parse_args()

  # 1. Validate workload
  script_dir = os.path.dirname(os.path.abspath(__file__))
  val_script = os.path.join(script_dir, "validate_workload.py")

  val_cmd = [
      sys.executable,
      val_script,
      "--workload",
      args.workload,
      "--start",
      args.start,
      "--end",
      args.end,
      "--project",
      args.project,
  ]
  val_proc = subprocess.run(
      val_cmd, capture_output=True, text=True, check=False
  )
  # validate_workload exits 77/75 when Cloud Logging refused or failed a
  # query; the workload's state is then unknown. Pass that through unchanged
  # rather than collapsing it into a generic validation failure.
  if val_proc.returncode in (
      log_query.EXIT_ACCESS_DENIED,
      log_query.EXIT_QUERY_FAILED,
  ):
    sys.stdout.write(val_proc.stdout)
    sys.stderr.write(val_proc.stderr)
    sys.exit(val_proc.returncode)
  try:
    val_res = json.loads(val_proc.stdout) if val_proc.stdout.strip() else []
  except ValueError:
    val_res = []

  # run_command returns [] on every failure path (non-zero exit, empty
  # stdout, JSON decode error). Calling .get() on that raises AttributeError
  # and masks the intended "Validation failed" diagnostic below.
  if not isinstance(val_res, dict):
    sys.stderr.write(
        "Validation failed: validate_workload.py produced no parseable "
        f"result (got {type(val_res).__name__}). Command: {val_cmd}\n"
    )
    sys.exit(1)

  status = val_res.get("status")
  if status != "INITIALIZED":
    sys.stderr.write(f"Validation failed: {json.dumps(val_res, indent=2)}\n")
    print(json.dumps([val_res], indent=2))
    sys.exit(1)

  cluster = val_res.get("cluster_name")
  location = val_res.get("location")
  namespace = val_res.get("namespace_name", "default")

  # 2. Extract timeline
  try:
    timeline = extract_timeline(
        workload=args.workload,
        start=args.start,
        end=args.end,
        project=args.project,
        cluster=cluster,
        location=location,
        namespace=namespace,
        max_workers=args.workers,
    )
  except log_query.LoggingQueryError as exc:
    log_query.fail(exc, context="job disruption timeline queries")

  formatted_output = json.dumps(timeline, indent=2)
  print(formatted_output)

  if args.output:
    with open(args.output, "w") as f:
      f.write(formatted_output)
    sys.stderr.write(f"Wrote {len(timeline)} events to {args.output}\n")

if __name__ == "__main__":
  main()

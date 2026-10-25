#!/usr/bin/env python3
"""Workload Pre-flight Validation Tool.

Verifies if a workload initialized, failed startup, or was never
submitted.
"""

import argparse
import datetime
import json
import subprocess
import sys
from typing import Any

import log_query
from pull_and_format_logs import execute_logging_query

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _located(log: dict[str, Any], project: str) -> dict[str, Any]:
  res_labels = log.get("resource", {}).get("labels", {})
  return {
      "cluster_name": res_labels.get("cluster_name"),
      "location": res_labels.get("location"),
      "namespace_name": res_labels.get("namespace_name", "default"),
      "project_id": res_labels.get("project_id", project),
  }


def validate_workload(
    workload: str,
    start: str,
    end: str,
    project: str,
    cluster: str = "",
    namespace: str = "",
) -> dict[str, Any]:
  """Classifies a workload's launch state from Cloud Logging.

  Args:
    workload: Target workload name or prefix.
    start: Start UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).
    end: End UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).
    project: GCP project ID to query.
    cluster: Optional GKE cluster name to scope the queries to.
    namespace: Optional Kubernetes namespace to scope the queries to.

  Returns:
    A dict with a `status` of INITIALIZED, FAILED_STARTUP or
    WORKLOAD_NOT_FOUND plus supporting details.

  Raises:
    log_query.LoggingQueryError: a query was refused or failed. The workload's
      state is then unknown; it is never reported as WORKLOAD_NOT_FOUND.
  """
  if not project:
    return {
        "status": "FAILED_VALIDATION",
        "message": (
            "GCP project ID is required but was not provided, and no active "
            "project is configured in gcloud."
        ),
    }
  # The workload selector replaces a bare full-text `"<workload>"` term, which
  # searched every field of every container log in the window.
  selector, selector_mode = log_query.choose_workload_selector(
      project, workload, start, end, cluster=cluster, namespace=namespace
  )
  base_parts = [
      'resource.type="k8s_container"',
      selector,
  ] + log_query.scope_clauses(cluster, namespace)
  time_filter = log_query.time_clause(start, end)
  if time_filter:
    base_parts.append(time_filter)
  base = " AND ".join(base_parts)
  # Existence first: the earliest container log from the workload's own pods
  # is one indexed lookup and proves the workload launched, whatever framework
  # it runs. Searching the whole window for the initialization marker
  # first scanned every text payload of a non-Pathways workload end to end.
  container_logs = execute_logging_query(
      project_id=project, query_filter=base, limit=1, order="asc"
  )
  if container_logs:
    first = container_logs[0]
    first_ts = first.get("timestamp", "")
    # The marker, when present, is written at process start, so it is only
    # looked for shortly after the first container log.
    init_logs = []
    try:
      marker_end = log_query.fmt_ts(
          log_query.parse_ts(first_ts) + datetime.timedelta(hours=2)
      )
    except ValueError:
      marker_end = ""
    if marker_end:
      effective_end = (
          min(marker_end, end) if (marker_end and end) else (marker_end or end)
      )
      marker_parts = [
          'resource.type="k8s_container"',
          selector,
      ] + log_query.scope_clauses(cluster, namespace)
      marker_time_filter = log_query.time_clause(
          log_query.fmt_ts(log_query.parse_ts(first_ts)),
          effective_end,
      )
      if marker_time_filter:
        marker_parts.append(marker_time_filter)
      marker_base = " AND ".join(marker_parts)
      init_logs = execute_logging_query(
          project_id=project,
          query_filter=(
              f"{marker_base} AND "
              'textPayload:"Logging before InitGoogle() is written to"'
          ),
          limit=1,
      )
    if init_logs:
      return {
          "status": "INITIALIZED",
          **_located(init_logs[0], project),
          "first_init_timestamp": init_logs[0].get("timestamp"),
          "first_log_timestamp": first_ts,
          "evidence": "workload_init_marker",
          "selector_mode": selector_mode,
      }
    return {
        "status": "INITIALIZED",
        **_located(first, project),
        "first_init_timestamp": first_ts,
        "first_log_timestamp": first_ts,
        "evidence": "container_logs",
        "message": (
            "No Pathways init marker within 2h of the first container "
            "log (expected for non-Pathways workloads), but the workload's "
            "containers logged; first_init_timestamp is the earliest entry."
        ),
        "selector_mode": selector_mode,
    }

  audit_query = (
      f'logName:"projects/{project}/logs/'
      'cloudaudit.googleapis.com%2Factivity" '
      'AND protoPayload.methodName:"io.k8s.core.v1.pods.create" '
      f'AND (protoPayload.resourceName:"{workload}" OR '
      f'protoPayload.request.metadata.generateName:"{workload}" OR '
      f'protoPayload.request.metadata.name:"{workload}") '
      f"AND {log_query.time_clause(start, end)}"
  )
  audit_logs = execute_logging_query(
      project_id=project, query_filter=audit_query, limit=1
  )

  if audit_logs:
    return {
        "status": "FAILED_STARTUP",
        "message": (
            "Pod creation attempts found in Audit Logs, but the workload's "
            "containers never logged."
        ),
        "audit_attempt": audit_logs[0].get("timestamp"),
        "selector_mode": selector_mode,
    }

  return {
      "status": "WORKLOAD_NOT_FOUND",
      "message": (
          "No launch attempts or initialization logs found for workload "
          f"'{workload}' in specified timeframe."
      ),
      "selector_mode": selector_mode,
  }


def _active_gcloud_project() -> str:
  """Returns the active gcloud project, or "" if unset or unavailable."""
  try:
    p_res = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
  except (OSError, subprocess.SubprocessError):
    return ""
  value = p_res.stdout.strip()
  if p_res.returncode == 0 and value and value != "(unset)":
    return value
  return ""


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Validate workload initialization state."
  )
  parser.add_argument(
      "--workload", required=True, help="Target workload name or prefix"
  )
  parser.add_argument(
      "--start", default="", help="Start UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)"
  )
  parser.add_argument(
      "--end", default="", help="End UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)"
  )
  parser.add_argument(
      "--lookback-days",
      type=int,
      default=30,
      help="Lookback window in days when --start is omitted",
  )
  parser.add_argument("--project", default="", help="GCP Project ID")
  parser.add_argument("--cluster", default="", help="Optional GKE cluster")
  parser.add_argument("--namespace", default="", help="Optional namespace")

  args = parser.parse_args()

  now = datetime.datetime.now(datetime.timezone.utc)
  end = args.end or now.strftime(_TIMESTAMP_FORMAT)
  start = args.start or (
      now - datetime.timedelta(days=args.lookback_days)
  ).strftime(_TIMESTAMP_FORMAT)

  proj = args.project or _active_gcloud_project()
  if not proj:
    result = {
        "status": "FAILED_VALIDATION",
        "message": (
            "GCP project ID is required but was not provided via --project and"
            " no active project is set in gcloud config (`gcloud config set"
            " project <PROJECT_ID>`)."
        ),
    }
    print(json.dumps(result, indent=2))
    sys.exit(1)

  try:
    log_query.preflight(proj)
    result = validate_workload(
        args.workload,
        start,
        end,
        proj,
        cluster=args.cluster,
        namespace=args.namespace,
    )
  except log_query.LoggingQueryError as exc:
    log_query.fail(exc, context=f"validate_workload {args.workload}")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

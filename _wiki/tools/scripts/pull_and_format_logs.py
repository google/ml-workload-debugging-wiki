"""Pulls logs in 5-minute increments, formats and masks them."""

import argparse
from concurrent import futures
import datetime
import importlib
import json
import os
import re
import sys
import time
import types
from typing import Any

if __package__:
  build_log_graph = importlib.import_module(".build_log_graph", __package__)
  log_query = importlib.import_module(".log_query", __package__)
  mask_segment_tool = importlib.import_module(".mask_segment_tool", __package__)
else:
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  build_log_graph = importlib.import_module("build_log_graph")
  log_query = importlib.import_module("log_query")
  mask_segment_tool = importlib.import_module("mask_segment_tool")


class DummyLoggingError(Exception):
  """Fallback error when optional Google Cloud libraries are unavailable."""


class GoogleAPICallError(DummyLoggingError):
  """Fallback GoogleAPICallError."""


class _ResourceExhaustedError(GoogleAPICallError):
  """Fallback ResourceExhaustedError."""


class _ServiceUnavailableError(GoogleAPICallError):
  """Fallback ServiceUnavailableError."""


class _InternalServerError(GoogleAPICallError):
  """Fallback InternalServerError."""


class _DeadlineExceededError(GoogleAPICallError):
  """Fallback DeadlineExceededError."""


class _TooManyRequestsError(GoogleAPICallError):
  """Fallback TooManyRequestsError."""


class _PermissionDeniedError(GoogleAPICallError):
  """Fallback PermissionDeniedError."""


class _UnauthenticatedError(GoogleAPICallError):
  """Fallback UnauthenticatedError."""


class _ForbiddenError(GoogleAPICallError):
  """Fallback ForbiddenError."""


class _InvalidArgumentError(GoogleAPICallError):
  """Fallback InvalidArgumentError."""


class GoogleAuthError(DummyLoggingError):
  """Fallback GoogleAuthError."""


class DummyLoggingClient:
  """Fallback Logging Client stub."""

  def __init__(self, *unused_args: Any, **unused_kwargs: Any) -> None:
    pass

  def list_entries(self, *unused_args: Any, **unused_kwargs: Any) -> list[Any]:
    return []


_DummyGapiExcModule = types.SimpleNamespace(
    GoogleAPICallError=GoogleAPICallError,
    ResourceExhausted=_ResourceExhaustedError,
    ServiceUnavailable=_ServiceUnavailableError,
    InternalServerError=_InternalServerError,
    DeadlineExceeded=_DeadlineExceededError,
    TooManyRequests=_TooManyRequestsError,
    PermissionDenied=_PermissionDeniedError,
    Unauthenticated=_UnauthenticatedError,
    Forbidden=_ForbiddenError,
    InvalidArgument=_InvalidArgumentError,
)

_DummyGauthExcModule = types.SimpleNamespace(
    GoogleAuthError=GoogleAuthError,
)

_DummyLoggingModule = types.SimpleNamespace(
    Client=DummyLoggingClient,
)

_DummyCloudModule = types.SimpleNamespace(
    logging=_DummyLoggingModule,
)

_DummyGoogleModule = types.SimpleNamespace(
    cloud=_DummyCloudModule,
)

gauth_exc: Any = _DummyGauthExcModule
google: Any = _DummyGoogleModule
gapi_exc: Any = _DummyGapiExcModule
_HAS_GCLOUD_SDK: bool = False

try:
  google = importlib.import_module("google")
  cloud = importlib.import_module("google.cloud")
  importlib.import_module("google.cloud.logging")
  logging_v2 = importlib.import_module("google.cloud.logging_v2")
  gauth_exc = importlib.import_module("google.auth.exceptions")
  gapi_exc = importlib.import_module("google.api_core.exceptions")
  _HAS_GCLOUD_SDK = os.environ.get("USE_GCLOUD_CLI") != "1"
except ImportError:
  gauth_exc = _DummyGauthExcModule
  google = _DummyGoogleModule
  gapi_exc = _DummyGapiExcModule
  _HAS_GCLOUD_SDK = False


# Windows longer than this are read as concurrent slices (see read_sliced).
_SLICE_THRESHOLD = datetime.timedelta(hours=6)


def get_known_hostnames(workload_name: str) -> list[str]:
  """Returns known hostnames for the workload.

  hostnames are now dynamically extracted from pod logs.

  Args:
    workload_name: Name of the workload.

  Returns:
    List of known hostnames.
  """
  del workload_name  # Unused.
  return []


def _str_to_bool(val: Any) -> bool:
  if isinstance(val, bool):
    return val
  return str(val).strip().lower() in ("1", "true", "yes", "y")


def _has_matching_outer_parens(expr: str) -> bool:
  """Returns True if expr is enclosed by a single pair of outer parentheses."""
  if not (expr.startswith("(") and expr.endswith(")")):
    return False
  depth = 0
  in_quotes = False
  escape = False
  for idx, ch in enumerate(expr):
    if escape:
      escape = False
      continue
    if ch == "\\":
      escape = True
      continue
    if ch == '"':
      in_quotes = not in_quotes
      continue
    if in_quotes:
      continue
    if ch == "(":
      depth += 1
    elif ch == ")":
      depth -= 1
      if depth == 0 and idx < len(expr) - 1:
        return False
  return depth == 0


_MAX_FILTER_EXTRA_CHARS = 18000
_MAX_NODE_CLAUSE_COUNT = 400
_NODE_OR_BLOCK_RE = re.compile(
    r'\(?resource\.labels\.node_name="[^"]+"'
    r'(?:\s+OR\s+resource\.labels\.node_name="[^"]+")+\)?'
)
_NODE_NAME_ITEM_RE = re.compile(r'resource\.labels\.node_name="([^"]+)"')


def _compact_large_filter_extra(
    cleaned_extra: str,
    max_chars: int = _MAX_FILTER_EXTRA_CHARS,
    max_nodes: int = _MAX_NODE_CLAUSE_COUNT,
) -> str:
  """Compacts or truncates oversized node/OR filter clauses under max_chars."""
  node_matches = _NODE_NAME_ITEM_RE.findall(cleaned_extra)
  if len(cleaned_extra) <= max_chars and len(node_matches) <= max_nodes:
    return cleaned_extra

  def _replace_node_block(match: re.Match[str]) -> str:
    block = match.group(0)
    nodes = _NODE_NAME_ITEM_RE.findall(block)
    if len(nodes) <= max_nodes and len(cleaned_extra) <= max_chars:
      return block
    clause_set: set[str] = set()
    for node in nodes:
      pool_prefix = re.sub(r"-[a-z0-9]+$", "-", node)
      if pool_prefix != node:
        clause_set.add(f'resource.labels.node_name:"{pool_prefix}"')
      else:
        clause_set.add(f'resource.labels.node_name="{node}"')
    sorted_clauses = sorted(clause_set)
    kept: list[str] = []
    running_len = 2  # outer parens
    budget = max(256, max_chars - (len(cleaned_extra) - len(block)))
    for clause in sorted_clauses:
      sep_len = 4 if kept else 0
      if kept and running_len + sep_len + len(clause) > budget:
        break
      kept.append(clause)
      running_len += sep_len + len(clause)
    return f"({' OR '.join(kept)})"

  compacted = _NODE_OR_BLOCK_RE.sub(_replace_node_block, cleaned_extra)
  if len(compacted) <= max_chars:
    return compacted

  inner = (
      compacted[1:-1] if _has_matching_outer_parens(compacted) else compacted
  )
  or_parts = inner.split(" OR ")
  if len(or_parts) > 1:
    kept_parts: list[str] = []
    running_len = 2
    for part in or_parts:
      sep_len = 4 if kept_parts else 0
      if kept_parts and running_len + sep_len + len(part) > max_chars:
        break
      kept_parts.append(part)
      running_len += sep_len + len(part)
    return f"({' OR '.join(kept_parts)})"
  return compacted[:max_chars]


def build_logging_filter(
    project_id: str,
    resource_type: str = "k8s_container",
    workload_name: str = "",
    start_time_str: str = "",
    end_time_str: str = "",
    filter_extra: str = "",
    coordinator_only: bool = False,
    restart_attempt: int = -1,
    selector: str = "",
    cluster: str = "",
    namespace: str = "default",
) -> str:
  """Builds a Cloud Logging filter.

  `selector` is the workload-selection clause, normally from
  log_query.choose_workload_selector (the indexed JobSet label, or the
  substring fallback). Without one the substring form is used.

  Args:
    project_id: GCP project ID the logs belong to.
    resource_type: Cloud Logging monitored resource type.
    workload_name: Workload (JobSet) name; empty for no workload clause.
    start_time_str: Inclusive lower timestamp bound; empty for none.
    end_time_str: Inclusive upper timestamp bound; only applied together with
      `start_time_str`.
    filter_extra: Additional raw filter clause, parenthesized and ANDed in.
    coordinator_only: Restrict k8s_container logs to the coordinator pod.
    restart_attempt: JobSet restart attempt to match; negative for any.
    selector: Workload-selection clause (see above).
    cluster: GKE cluster name to scope k8s_container logs to.
    namespace: Kubernetes namespace to scope k8s_container logs to.

  Returns:
    The clauses joined with ` AND `.
  """
  filter_parts = [
      f'resource.type="{resource_type}"',
      f'resource.labels.project_id="{project_id}"',
  ]
  if resource_type == "k8s_container" and workload_name:
    filter_parts.extend(log_query.scope_clauses(cluster, namespace))
    filter_parts.append(selector or log_query.substring_selector(workload_name))
    if coordinator_only:
      filter_parts.append(
          '(resource.labels.pod_name:"-0-0-" OR'
          ' resource.labels.pod_name:"-pathways-head" OR'
          ' resource.labels.pod_name:"-pwhd" OR'
          ' resource.labels.pod_name:"-slice-0-job-0" OR'
          ' resource.labels.pod_name:"-job-0-")'
      )
  if restart_attempt >= 0:
    filter_parts.append(
        'labels."k8s-pod/jobset_sigs_k8s_io/restart-attempt":'
        f' "{restart_attempt}"'
    )
  if filter_extra:
    cleaned_extra = re.sub(r"\s+AND\s+\(\s*\)", "", filter_extra).strip()
    cleaned_extra = re.sub(
        r"^(?:AND\s+)+|(?:\s+AND)+$", "", cleaned_extra
    ).strip()
    if cleaned_extra and cleaned_extra != "()":
      cleaned_extra = _compact_large_filter_extra(cleaned_extra)
      if not _has_matching_outer_parens(cleaned_extra):
        cleaned_extra = f"({cleaned_extra})"
      filter_parts.append(cleaned_extra)
  if start_time_str and end_time_str:
    filter_parts.append(f'timestamp >= "{start_time_str}"')
    filter_parts.append(f'timestamp <= "{end_time_str}"')
  elif start_time_str:
    filter_parts.append(f'timestamp >= "{start_time_str}"')

  return " AND ".join(filter_parts)


def _normalize_cli_entries(raw_entries):
  """Adds the `payload` key the SDK path produces to raw gcloud JSON entries."""
  results = []
  for item in raw_entries:
    json_payload = item.get("jsonPayload", {})
    text_payload = item.get("textPayload")
    if json_payload:
      payload = json_payload
    elif text_payload:
      payload = {"message": text_payload}
    else:
      payload = {"message": json.dumps(item)}
    item["payload"] = payload
    results.append(item)
  return results


def _sdk_entry_to_dict(entry):
  """Converts a Cloud Logging SDK entry to the gcloud JSON entry shape.

  Args:
    entry: A log entry yielded by `google.cloud.logging.Client.list_entries`.

  Returns:
    A dict with timestamp, payload, jsonPayload, textPayload, labels and
    resource labels.
  """
  if isinstance(entry.payload, dict):
    json_payload = entry.payload
    text_payload = None
    payload = entry.payload
  else:
    json_payload = {}
    text_payload = str(entry.payload)
    payload = {"message": text_payload}
  ts_str = (
      entry.timestamp.isoformat()
      if isinstance(entry.timestamp, datetime.datetime)
      else str(entry.timestamp)
  )
  # SDK LogEntry objects always define `resource` (a Resource with `labels`,
  # or None) and `labels` (a dict or None).
  resource_labels = {}
  if entry.resource is not None and isinstance(entry.resource.labels, dict):
    resource_labels = entry.resource.labels
  entry_labels = entry.labels if isinstance(entry.labels, dict) else {}
  return {
      "timestamp": ts_str,
      "payload": payload,
      "jsonPayload": json_payload,
      "textPayload": text_payload,
      "labels": entry_labels,
      "resource": {"labels": resource_labels},
  }


def _execute_via_sdk(project_id, query_filter, limit, order):
  """Reads through the Python client. Returns None to fall back to gcloud.

  Falling back is reserved for credential problems: the client authenticates
  with Application Default Credentials while gcloud may carry different ones
  (an account, or an `auth/authorization_token_file` authentication
  token), so an SDK permission refusal says nothing about what gcloud can
  read. The gcloud result is authoritative and raises on failure. Every other
  SDK failure raises LoggingQueryError -- never an empty list.

  Args:
    project_id: GCP project ID to read from.
    query_filter: Cloud Logging filter.
    limit: Maximum number of entries to return.
    order: "asc" for oldest first; anything else for newest first.

  Returns:
    The matching entries as dicts, or None when gcloud should be used instead.

  Raises:
    log_query.LoggingQueryError: the SDK read failed for a reason other than
      credentials.
  """
  transient_exceptions = (
      gapi_exc.ResourceExhausted,
      gapi_exc.ServiceUnavailable,
      gapi_exc.InternalServerError,
      gapi_exc.DeadlineExceeded,
      gapi_exc.TooManyRequests,
  )
  credential_exceptions = (
      gapi_exc.PermissionDenied,
      gapi_exc.Unauthenticated,
      gapi_exc.Forbidden,
      gauth_exc.GoogleAuthError,
  )
  try:
    client = google.cloud.logging.Client(project=project_id)
  except (gauth_exc.GoogleAuthError, OSError, ValueError):
    # Client construction fails on missing/unusable ADC; gcloud may still work.
    return None
  order_by = "timestamp asc" if order.lower() == "asc" else "timestamp desc"
  max_retries, base_delay = 3, 2
  for attempt in range(max_retries):
    try:
      page_size = max(500, min(1000, limit)) if limit > 0 else 500
      entries = client.list_entries(
          filter_=query_filter,
          page_size=page_size,
          max_results=limit,
          order_by=order_by,
      )
      results = []
      for i, entry in enumerate(entries):
        if i >= limit:
          break
        results.append(_sdk_entry_to_dict(entry))
      return results
    except credential_exceptions:
      return None
    except transient_exceptions as exc:
      if attempt == max_retries - 1:
        raise log_query.LoggingQueryError("transient", str(exc)) from exc
      time.sleep(base_delay * (2**attempt))
    except gapi_exc.GoogleAPICallError as exc:
      raise log_query.LoggingQueryError("error", str(exc)) from exc
  return None


def execute_logging_query(
    project_id: str,
    query_filter: str,
    limit: int = 500,
    bucket: str = "",
    location: str = "",
    view: str = "",
    order: str = "asc",
) -> list[dict[str, Any]]:
  """Returns matching entries; raises log_query.LoggingQueryError on failure.

  An empty list means the query ran and matched nothing. A refused or failed
  query is never reported as empty -- that is how an expired token used to
  surface as "no events" / WORKLOAD_NOT_FOUND.

  Args:
    project_id: GCP project ID to read from.
    query_filter: Cloud Logging filter.
    limit: Maximum number of entries to return.
    bucket: Optional log bucket to read from instead of the default view.
    location: Location of `bucket`.
    view: Log view within `bucket`.
    order: "asc" for oldest first, "desc" for newest first.

  Returns:
    The matching log entries, each with a `payload` key.
  """
  if _HAS_GCLOUD_SDK and not (bucket or location or view):
    sdk_results = _execute_via_sdk(project_id, query_filter, limit, order)
    if sdk_results is not None:
      return sdk_results

  raw_entries = log_query.read_entries(
      project_id,
      query_filter,
      limit,
      order,
      extra_args=_bucket_args(bucket, location, view),
  )
  return _normalize_cli_entries(raw_entries)


def _bucket_args(bucket="", location="", view=""):
  extra = []
  if bucket:
    extra.append(f"--bucket={bucket}")
  if location:
    extra.append(f"--location={location}")
  if view:
    extra.append(f"--view={view}")
  return extra


def auto_slice_minutes(start: str, end: str, requested: int = 0) -> int:
  """Slice width for [start, end]; 0 means read the window in one query."""
  if requested > 0:
    return requested
  span = log_query.parse_ts(end) - log_query.parse_ts(start)
  if span <= _SLICE_THRESHOLD:
    return 0
  return 360 if span > datetime.timedelta(days=1) else 120


def analyze_workload(
    workload_name: str,
    project_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
  """Analyzes workload logs and extracts error events."""
  time_filter = ""
  if start_time and end_time:
    time_filter = (
        f' AND timestamp >= "{start_time}" AND timestamp <= "{end_time}"'
    )
  elif start_time:
    time_filter = f' AND timestamp >= "{start_time}"'

  now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
  report: dict[str, Any] = {
      "workload_name": workload_name,
      "project_id": project_id,
      "timestamp_analyzed": now_ts,
      "stage_1_scheduling_admitted": True,
      "stage_2_pods_running": True,
      # Seeded as INCONCLUSIVE, not as success. These two queries only ever
      # find *failure* events, so an empty result is ambiguous: either nothing
      # went wrong, or the query reached nothing (a window outside gcloud's
      # default freshness returns zero with exit 0). Asserting success from an
      # empty result turns an unreachable query into a clean bill of health.
      # `window_applied` records which of the two it was.
      "overall_status": "INCONCLUSIVE_NO_EVENTS_FOUND",
      "window_applied": bool(start_time or end_time),
      "events_examined": 0,
      "failure_stage": None,
      "primary_root_cause": None,
      "failure_counts": {},
      "image_pull_failures": {},
      "sample_errors": [],
  }

  stage1_reasons = (
      'jsonPayload.reason="FailedScheduling" OR '
      'jsonPayload.reason="UntoleratedTaint" OR '
      'jsonPayload.reason="QuotaReserved" OR '
      'jsonPayload.reason="Inadmissible" OR '
      'jsonPayload.reason="Evicted"'
  )
  stage1_query = (
      'logName:"/logs/events" AND resource.type="k8s_pod" '
      f'AND resource.labels.pod_name:"{workload_name}-" '
      f"AND ({stage1_reasons}){time_filter}"
  )
  stage2_reasons = (
      '(jsonPayload.reason="Failed" OR jsonPayload.reason="BackOff" OR '
      'jsonPayload.reason="FailedMount" OR '
      'jsonPayload.reason="ErrImagePull" OR '
      'jsonPayload.reason="ImagePullBackOff" OR jsonPayload.message:"pull" OR '
      'jsonPayload.message:"ImagePull" OR jsonPayload.message:"FailedMount") '
      'AND -jsonPayload.reason="Pulled" AND -jsonPayload.reason="Pulling"'
  )
  stage2_query = (
      'logName:"/logs/events" AND resource.type="k8s_pod" '
      f'AND resource.labels.pod_name:"{workload_name}-" '
      f"AND ({stage2_reasons}){time_filter}"
  )

  with futures.ThreadPoolExecutor(max_workers=2) as executor:
    stage1_future = executor.submit(
        execute_logging_query, project_id, stage1_query, limit
    )
    stage2_future = executor.submit(
        execute_logging_query, project_id, stage2_query, limit
    )

    stage1_entries = stage1_future.result()
    stage2_entries = stage2_future.result()

  stage1_counts = {}
  stage1_samples = []
  if stage1_entries:
    report["stage_1_scheduling_admitted"] = False
    for entry in stage1_entries:
      payload = entry.get("payload", {})
      reason = payload.get("reason", "UnknownSchedulingError")
      msg = payload.get("message", str(payload))
      stage1_counts[reason] = stage1_counts.get(reason, 0) + 1
      if len(stage1_samples) < 5 and msg not in stage1_samples:
        stage1_samples.append(f"[{reason}] {msg}")
  stage2_counts = {}
  stage2_samples = []
  image_failures = {}
  if stage2_entries:
    report["stage_2_pods_running"] = False
    for entry in stage2_entries:
      payload = entry.get("payload", {})
      reason = payload.get("reason", "UnknownStartupError")
      msg = payload.get("message", str(payload))
      if "pull" in msg.lower() or "image" in msg.lower():
        cat = f"ImagePull_{reason}"
        img_name = None
        img_match = re.search(r'image\s+"([^"]+)"', msg)
        if img_match:
          img_name = img_match.group(1)
        else:
          inv_obj = payload.get("involvedObject", {})
          field_path = inv_obj.get("fieldPath", "")
          cont_match = re.search(r"{(.*)}", field_path)
          img_name = (
              f"container:{cont_match.group(1)}"
              if cont_match
              else "unknown_image"
          )
        if img_name:
          if img_name not in image_failures:
            image_failures[img_name] = {"count": 0, "reasons": {}}
          image_failures[img_name]["count"] += 1
          image_failures[img_name]["reasons"][reason] = (
              image_failures[img_name]["reasons"].get(reason, 0) + 1
          )
      elif "mount" in msg.lower() or "forbidden" in msg.lower():
        cat = f"VolumeMount_{reason}"
      else:
        cat = reason

      stage2_counts[cat] = stage2_counts.get(cat, 0) + 1
      if len(stage2_samples) < 8 and msg not in stage2_samples:
        stage2_samples.append(f"[{cat}] {msg}")
    report["image_pull_failures"] = image_failures

  report["events_examined"] = len(stage1_entries) + len(stage2_entries)

  if report["stage_1_scheduling_admitted"] and report["stage_2_pods_running"]:
    # No failure events. That is only a pass if the query was scoped to a real
    # window; otherwise say so rather than implying the pods were healthy.
    report["overall_status"] = (
        "ALL_PODS_SCHEDULED_AND_RUNNING"
        if report["window_applied"]
        else "INCONCLUSIVE_NO_EVENTS_FOUND"
    )
    if not report["window_applied"]:
      report["primary_root_cause"] = (
          "No scheduling or startup failure events returned, but no time "
          "window was supplied, so gcloud applied its 1-day default "
          "freshness. For an "
          "incident older than that this is an empty query, not a healthy "
          "workload. Re-run with explicit start and end."
      )

  if (
      not report["stage_1_scheduling_admitted"]
      and not report["stage_2_pods_running"]
  ):
    report["overall_status"] = "POD_SCHEDULING_AND_STARTUP_FAILURE"
    report["failure_stage"] = (
        "Stage 1a & Stage 2a (Scheduling/Admission + Startup/Initialization)"
    )
    report["failure_counts"] = {**stage1_counts, **stage2_counts}
    report["sample_errors"] = stage1_samples + stage2_samples
    top_s1 = (
        max(stage1_counts, key=lambda k: stage1_counts[k])
        if stage1_counts
        else "Unknown"
    )
    top_s2 = (
        max(stage2_counts, key=lambda k: stage2_counts[k])
        if stage2_counts
        else "Unknown"
    )
    report["primary_root_cause"] = (
        "Workload encountered both Stage 1 scheduling/admission errors (Top:"
        f" {top_s1}) and Stage 2 container startup/initialization errors"
        f" (Top: {top_s2})."
    )
  elif not report["stage_1_scheduling_admitted"]:
    report["overall_status"] = "POD_SCHEDULING_OR_ADMISSION_FAILURE"
    report["failure_stage"] = "Stage 1a (Kueue Admission or Kube-Scheduler)"
    report["failure_counts"] = stage1_counts
    report["sample_errors"] = stage1_samples
    top_s1 = (
        max(stage1_counts, key=lambda k: stage1_counts[k])
        if stage1_counts
        else "Unknown"
    )
    report["primary_root_cause"] = (
        f"Workload pods failed to schedule or admit. Top reason: {top_s1}"
    )
  elif not report["stage_2_pods_running"]:
    report["overall_status"] = "POD_STARTUP_OR_INITIALIZATION_FAILURE"
    report["failure_stage"] = (
        "Stage 2a (Kubelet Sandbox, ImagePull, or Volume Mount)"
    )
    report["failure_counts"] = stage2_counts
    report["sample_errors"] = stage2_samples
    top_s2 = (
        max(stage2_counts, key=lambda k: stage2_counts[k])
        if stage2_counts
        else "Unknown"
    )
    report["primary_root_cause"] = (
        f"Pods scheduled but failed to reach Running state due to {top_s2}."
        " Total startup errors detected:"
        f" {sum(stage2_counts.values())}."
    )

  return report


def parse_logs(entries: Any) -> list[dict[str, Any]]:
  """Parses raw log entries into structured dictionaries."""
  parsed_entries = []
  if not isinstance(entries, list):
    return parsed_entries
  for item in entries:
    resource = item.get("resource", {}) if isinstance(item, dict) else {}
    res_labels = (
        resource.get("labels", {}) if isinstance(resource, dict) else {}
    )
    pod_name = res_labels.get("pod_name", "unknown-pod")
    container_name = res_labels.get("container_name", "unknown-container")

    labels = item.get("labels", {}) if isinstance(item, dict) else {}
    if not isinstance(labels, dict):
      labels = {}
    if not isinstance(res_labels, dict):
      res_labels = {}

    attempt = (
        labels.get("k8s-pod/jobset_sigs_k8s_io/restart-attempt")
        or labels.get("jobset.sigs.k8s.io/restart-attempt")
        or labels.get("jobset_sigs_k8s_io/restart-attempt")
        or labels.get("restart-attempt")
        or labels.get("restart_attempt")
        or res_labels.get("k8s-pod/jobset_sigs_k8s_io/restart-attempt")
        or res_labels.get("jobset.sigs.k8s.io/restart-attempt")
        or res_labels.get("jobset_sigs_k8s_io/restart-attempt")
        or res_labels.get("restart-attempt")
        or res_labels.get("restart_attempt")
        or "0"
    )

    payload = item.get("payload", {}) if isinstance(item, dict) else {}
    message = payload.get("message") if isinstance(payload, dict) else None
    if not message:
      message = item.get("textPayload") or (
          json.dumps(item) if isinstance(item, dict) else str(item)
      )

    parsed_entries.append({
        "pod_name": pod_name,
        "container_name": container_name,
        "attempt": str(attempt),
        "message": message,
    })
  return parsed_entries


def mask_message(
    message_str: str,
    known_hostnames: list[str],
    mask_counters: dict[str, int] | None = None,
) -> tuple[str, dict[str, Any]]:
  """Masks sensitive tokens in a log message."""
  spans = []
  for line in message_str.splitlines():
    tokens = mask_segment_tool.tokenize_line(line, known_hostnames)
    if tokens:
      spans.append(tokens)
  if not spans:
    return "", {}
  masked_str, tokens_map = mask_segment_tool.mask_segment(
      spans, mask_segment_tool.DEFAULT_MASKED_TYPES, mask_counters
  )
  return masked_str, tokens_map


def main():
  parser = argparse.ArgumentParser(
      description="Pull, format, mask and group workload logs."
  )
  parser.add_argument("--project_id", required=True)
  parser.add_argument("--workload_name", default="")
  parser.add_argument("--last_occurrence", default="")
  parser.add_argument(
      "--start",
      default="",
      help=(
          "Window start (RFC3339). With --end, queries the whole window instead"
          " of stepping back from --last_occurrence."
      ),
  )
  parser.add_argument("--end", default="", help="Window end (RFC3339).")
  parser.add_argument("--output_file", required=True)
  parser.add_argument("--mappings_file", default="")
  parser.add_argument("--deduplicate", default="False")
  parser.add_argument("--bucket", default="")
  parser.add_argument("--location", default="")
  parser.add_argument("--view", default="")
  parser.add_argument(
      "--severity_filter",
      default="",
      help="Filter logs by minimum severity (e.g. ERROR, FATAL)",
  )
  parser.add_argument(
      "--max_lines",
      type=int,
      default=5000,
      help="Maximum number of output lines",
  )
  parser.add_argument(
      "--unmask", default="False", help="Automatically unmask output if True"
  )
  parser.add_argument(
      "--filter_extra", default="", help="Additional log filter expressions"
  )
  parser.add_argument("--limit", type=int, default=5000, help="Log read limit")
  parser.add_argument(
      "--duration_minutes",
      type=int,
      default=5,
      help="Log lookback duration in minutes (default 5)",
  )
  parser.add_argument(
      "--coordinator_only",
      nargs="?",
      const=True,
      default=False,
      type=_str_to_bool,
      help="Only query coordinator logs",
  )
  parser.add_argument(
      "--full_window",
      nargs="?",
      const=True,
      default=False,
      type=_str_to_bool,
      help=(
          "Allow multi-day query windows without clamping to trailing"
          " duration_minutes."
      ),
  )
  parser.add_argument(
      "--resource_type",
      default="k8s_container",
      help="GCP resource type filter, e.g. k8s_container, gce_instance",
  )
  parser.add_argument(
      "--pod_analyzer_mode",
      action="store_true",
      help="Run 2-stage pod startup logic instead of standard parsing",
  )
  parser.add_argument(
      "--restart_attempt",
      type=int,
      default=-1,
      help=(
          "JobSet restart attempt index to filter by (default -1 for no filter)"
      ),
  )
  parser.add_argument(
      "--slice_minutes",
      type=int,
      default=0,
      help=(
          "Read an explicit --start/--end window as concurrent slices of this"
          " width. 0 (default) = auto: windows over 6h are sliced (120 min, or"
          " 360 min beyond 1 day); --limit then applies per slice."
      ),
  )
  parser.add_argument(
      "--cluster", default="", help="Optional GKE cluster name to scope to"
  )
  parser.add_argument(
      "--namespace", default="default", help="Kubernetes namespace"
  )

  args = parser.parse_args()
  try:
    _run(args)
  except log_query.LoggingQueryError as exc:
    log_query.fail(
        exc,
        context=(
            f"pull_and_format_logs project={args.project_id}"
            f" workload={args.workload_name}"
        ),
    )


def _query_window(args):
  """The window the run will read, or None if only gcloud's default applies."""
  if args.start:
    end = args.end or log_query.fmt_ts(
        datetime.datetime.now(datetime.timezone.utc)
    )
    if not args.full_window:
      try:
        start_dt = log_query.parse_ts(args.start)
        end_dt = log_query.parse_ts(end)
        if (end_dt - start_dt).total_seconds() > 86400:
          mins = args.duration_minutes or 5
          clamped_start = end_dt - datetime.timedelta(minutes=mins)
          sys.stderr.write(
              f"WARNING: Requested window ({args.start} to {end}) spans > 1"
              f" day; clamping to trailing {mins} minutes ("
              f"{log_query.fmt_ts(clamped_start)} to {end})."
              " --full_window=true is required for multi-day spans.\n"
          )
          return log_query.fmt_ts(clamped_start), end
      except (ValueError, TypeError):
        pass
    return args.start, end
  if args.last_occurrence:
    last = log_query.parse_ts(args.last_occurrence)
    mins = args.duration_minutes or 5
    start = last - datetime.timedelta(minutes=max(mins, 5))
    return log_query.fmt_ts(start), log_query.fmt_ts(last)
  return None


def _run_filter(
    args: argparse.Namespace,
    selector: str,
    start_time_str: str = "",
    end_time_str: str = "",
) -> str:
  """Builds the logging filter for `_run` from its parsed arguments.

  Args:
    args: Parsed command-line arguments.
    selector: Workload selector clause chosen by `_run`.
    start_time_str: Optional window start timestamp.
    end_time_str: Optional window end timestamp.

  Returns:
    The Cloud Logging filter expression.
  """
  return build_logging_filter(
      args.project_id,
      args.resource_type,
      args.workload_name,
      start_time_str,
      end_time_str,
      args.filter_extra,
      args.coordinator_only,
      restart_attempt=args.restart_attempt,
      selector=selector,
      cluster=args.cluster,
      namespace=args.namespace,
  )


def _fetch_logs_with_filter(
    args: argparse.Namespace,
    selector: str,
    truncation: dict[str, Any],
    start_time_str: str = "",
    end_time_str: str = "",
) -> list[dict[str, Any]]:
  """Runs one logging query for `_run`, flagging truncation at the limit.

  Args:
    args: Parsed command-line arguments.
    selector: Workload selector clause chosen by `_run`.
    truncation: `_run`'s truncation record; "truncated" is set to True when the
      query returns `args.limit` or more rows.
    start_time_str: Optional window start timestamp.
    end_time_str: Optional window end timestamp.

  Returns:
    The rows returned by `execute_logging_query`.
  """
  query_filter = _run_filter(args, selector, start_time_str, end_time_str)
  rows = execute_logging_query(
      args.project_id,
      query_filter,
      limit=args.limit,
      bucket=args.bucket,
      location=args.location,
      view=args.view,
  )
  if len(rows) >= args.limit:
    truncation["truncated"] = True
  return rows


def _run(args):
  """Runs the pull/format pipeline (or pod analyzer mode) for parsed flags.

  Args:
    args: The parsed command-line arguments.
  """
  log_query.preflight(args.project_id)

  if args.pod_analyzer_mode:
    report = analyze_workload(
        args.workload_name,
        args.project_id,
        start_time=args.start,
        end_time=args.end,
        limit=args.limit,
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    print(f"Successfully generated pod analyzer report to {args.output_file}")
    print("\n--- SUMMARY REPORT (summary_fields) ---")
    print(f"  overall_status: {report['overall_status']}")
    print(f"  failure_stage: {report.get('failure_stage')}")
    print(f"  events_examined: {report['events_examined']}")
    print(f"  window_applied: {report['window_applied']}")
    print(f"  output_file: {args.output_file}")
    print("---------------------------------------")
    return

  deduplicate_bool = args.deduplicate.lower() in ("true", "1", "yes")
  unmask_bool = args.unmask.lower() in ("true", "1", "yes")

  known_hostnames = (
      get_known_hostnames(args.workload_name) if args.workload_name else []
  )
  target_entries = []

  # Choose the workload selector once. The indexed JobSet label avoids a full
  # scan of every container log in the window; the probe falls back to the
  # substring match for workloads whose pods do not carry the label. A custom
  # bucket/view is not probed (the probe reads _Default) and keeps substring.
  selector, selector_mode = "", "substring"
  window = _query_window(args)
  if (
      args.resource_type == "k8s_container"
      and args.workload_name
      and not args.coordinator_only
      and window
      and not (args.bucket or args.location or args.view)
  ):
    selector, selector_mode = log_query.choose_workload_selector(
        args.project_id,
        args.workload_name,
        window[0],
        window[1],
        cluster=args.cluster,
        namespace=args.namespace,
    )
  elif args.coordinator_only:
    selector_mode = "coordinator_regex"

  truncation = {"truncated": False, "slices": []}

  # An explicit window wins: it is the only mode that can reach a historical
  # incident. Without one, gcloud applies its default freshness (1 day), so a
  # sweep over an older run returns zero hits and still reports success -- the
  # failure mode is silent, and it defeats the whole-window crash sweep that
  # Milestone 1 mandates.
  if args.start:
    win_start, win_end = window
    slice_minutes = auto_slice_minutes(win_start, win_end, args.slice_minutes)
    if slice_minutes:
      # Long windows: concurrent slices, each with its own --limit, so one
      # chatty interval cannot consume the whole budget and silently hide the
      # rest of the run. Truncation is tracked per slice.
      sliced = log_query.read_sliced(
          args.project_id,
          _run_filter(args, selector),
          win_start,
          win_end,
          slice_minutes,
          args.limit,
          max_workers=8,
          order="asc",
          extra_args=_bucket_args(args.bucket, args.location, args.view),
      )
      truncation["truncated"] = sliced["truncated"]
      truncation["slices"] = sliced["slices"]
      raw_output = _normalize_cli_entries(sliced["entries"])
    else:
      raw_output = _fetch_logs_with_filter(
          args,
          selector,
          truncation,
          start_time_str=win_start,
          end_time_str=win_end,
      )
    target_entries = parse_logs(raw_output)
  elif not args.last_occurrence:
    raw_output = _fetch_logs_with_filter(args, selector, truncation)
    target_entries = parse_logs(raw_output)
  else:
    last_occurrence_str = args.last_occurrence
    if last_occurrence_str.endswith("Z"):
      last_occurrence_str = last_occurrence_str[:-1] + "+00:00"
    last_occurrence_dt = datetime.datetime.fromisoformat(last_occurrence_str)

    match_pattern = re.compile(
        r"error|fatal|segfault|ret_check|exit_code", re.IGNORECASE
    )

    num_steps = max(1, args.duration_minutes // 5)
    found_matches = False
    for i in range(num_steps):
      end_dt = last_occurrence_dt - datetime.timedelta(minutes=5 * i)
      start_dt = last_occurrence_dt - datetime.timedelta(minutes=5 * (i + 1))

      start_time_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
      end_time_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

      raw_output = _fetch_logs_with_filter(
          args,
          selector,
          truncation,
          start_time_str=start_time_str,
          end_time_str=end_time_str,
      )
      parsed_entries = parse_logs(raw_output)

      if not parsed_entries:
        continue

      for entry in parsed_entries:
        if match_pattern.search(entry["message"]):
          found_matches = True
          break

      if found_matches:
        target_entries = parsed_entries
        break

    if not found_matches:
      start_dt = last_occurrence_dt - datetime.timedelta(
          minutes=args.duration_minutes
      )
      start_time_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
      end_time_str = last_occurrence_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

      raw_output = _fetch_logs_with_filter(
          args,
          selector,
          truncation,
          start_time_str=start_time_str,
          end_time_str=end_time_str,
      )
      target_entries = parse_logs(raw_output)

  total_found = len(target_entries)

  # Dynamic Hostname Extension
  for entry in target_entries:
    pod_name = entry["pod_name"]
    if pod_name not in known_hostnames:
      known_hostnames.append(pod_name)

  # Group by attempt, pod_name, and container
  grouped = {}
  error_pattern = re.compile(
      r"error|fatal|segfault|ret_check|exception|unavailable|refused",
      re.IGNORECASE,
  )
  error_logs_count = 0

  for entry in target_entries:
    message = entry["message"]
    if error_pattern.search(message):
      error_logs_count += 1

    if args.severity_filter:
      if args.severity_filter.upper() not in message.upper():
        continue

    attempt = entry["attempt"]
    if args.restart_attempt >= 0 and str(args.restart_attempt) != str(attempt):
      continue

    pod_name = entry["pod_name"]
    container = entry["container_name"]

    if attempt not in grouped:
      grouped[attempt] = {}
    if pod_name not in grouped[attempt]:
      grouped[attempt][pod_name] = {}
    if container not in grouped[attempt][pod_name]:
      grouped[attempt][pod_name][container] = []

    grouped[attempt][pod_name][container].extend(message.splitlines())

  # Run build_log_graph re-assembly per container/attempt, then mask and label
  global_id = 1
  output_lines = []
  seen_templates = set()
  all_mappings = {}
  global_mask_counters = {}

  for attempt in sorted(grouped.keys()):
    for pod_name in sorted(grouped[attempt].keys()):
      for container in sorted(grouped[attempt][pod_name].keys()):
        lines = grouped[attempt][pod_name][container]
        if not lines:
          continue

        assembled_msgs = build_log_graph.build_and_render_tree(lines)
        for msg in assembled_msgs:
          masked_pod, pod_mappings = mask_message(
              pod_name, known_hostnames, global_mask_counters
          )
          masked_msg, msg_mappings = mask_message(
              msg, known_hostnames, global_mask_counters
          )
          if not masked_msg:
            continue

          if deduplicate_bool:
            template = re.sub(r"_\d+\]", "]", masked_msg)
            if template in seen_templates:
              continue
            seen_templates.add(template)

          all_mappings.update(pod_mappings)
          all_mappings.update(msg_mappings)

          labeled_msg = f"[ID: {global_id}] [{masked_pod}] {masked_msg}"
          output_lines.append(labeled_msg)
          global_id += 1

          if len(output_lines) >= args.max_lines:
            break
        if len(output_lines) >= args.max_lines:
          break
      if len(output_lines) >= args.max_lines:
        break

  combined_messages_count = len(output_lines)
  final_output_text = "\n\n".join(output_lines) + "\n"

  if unmask_bool and all_mappings:
    sorted_tokens = sorted(all_mappings.keys(), key=len, reverse=True)
    for token in sorted_tokens:
      orig_value = all_mappings[token].get("value", "")
      if orig_value:
        final_output_text = final_output_text.replace(token, orig_value)

  # Ensure target directory exists
  os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
  with open(args.output_file, "w", encoding="utf-8") as f:
    f.write(final_output_text)

  if args.mappings_file:
    with open(args.mappings_file, "w", encoding="utf-8") as mf:
      json.dump(all_mappings, mf, indent=2)

  print(
      f"Successfully pulled and formatted {combined_messages_count} logs to"
      f" {args.output_file}"
  )
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  truncated_slices = [
      f"{slice_stat['start']}..{slice_stat['end']}"
      for slice_stat in truncation["slices"]
      if slice_stat["truncated"]
  ]
  print(f"  original_count: {total_found}")
  print(f"  deduplicated_count: {combined_messages_count}")
  print(f"  error_fatal_count: {error_logs_count}")
  print(f"  selector_mode: {selector_mode}")
  print(f"  slices: {len(truncation['slices']) or 1}")
  print(f"  truncated: {truncation['truncated']}")
  if truncated_slices:
    print(f"  truncated_slices: {', '.join(truncated_slices)}")
  if truncation["truncated"]:
    print(
        "  WARNING: at least one query hit --limit; results are TRUNCATED and"
        " prove nothing negative. Narrow the window or raise --limit."
    )
  print(f"  output_file: {args.output_file}")
  print("---------------------------------------")


if __name__ == "__main__":
  main()

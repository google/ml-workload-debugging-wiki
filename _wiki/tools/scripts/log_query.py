#!/usr/bin/env python3
"""Shared Cloud Logging access for the diagnostic scripts.

Three properties every diagnostic query needs, and that each script used to
re-implement (or skip):

1. A failed query is never an empty result. `gcloud logging read` exits non-zero
   on PERMISSION_DENIED, an ECP proxy fault or a backend 500; scripts that
   mapped that to `[]` reported "no events" / WORKLOAD_NOT_FOUND for a workload
   they were simply not allowed to see. Failures raise `LoggingQueryError`.
2. Transient faults are retried with backoff; permission faults are not.
3. Workloads are selected by the indexed JobSet pod label rather than a
   `pod_name:"<wl>-"` substring match, which forces a full scan of every
   container log in the window. The substring form remains as a fallback for
   workloads whose pods do not carry the label.

Long windows are split into slices queried concurrently (`read_sliced`), and
truncation is reported per slice so a capped slice cannot hide behind an
uncapped total.
"""

import collections.abc
import concurrent.futures
import datetime
import functools
import json
import re
import subprocess
import sys
import time
from typing import Any

JOBSET_LABEL = "k8s-pod/jobset_sigs_k8s_io/jobset-name"
_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# sysexits.h: a script that cannot read the logs exits with one of these, so
# callers can tell "no access" from "looked and found nothing".
EXIT_ACCESS_DENIED = 77  # EX_NOPERM
EXIT_QUERY_FAILED = 75  # EX_TEMPFAIL

_PERMISSION_MARKERS = (
    "PERMISSION_DENIED",
    "does not have permission",
    "Permission denied for all log views",
    "AUTH_REQUIRED",
    "is closed (See http",
    "UNAUTHENTICATED",
    "Reauthentication required",
    "Unable to read file",  # authorization_token_file removed or expired
)
_TRANSIENT_MARKERS = (
    "ECP Proxy",
    "AuthProxyError",
    "Internal error encountered",
    "'status': 500",
    "'status': 502",
    "'status': 503",
    'status": 500',
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "RESOURCE_EXHAUSTED",
    "Quota exceeded",
    "code=429",
    "Too Many Requests",
    "Connection reset",
    "timed out",
)


class LoggingQueryError(Exception):
  """A Cloud Logging query failed; its result is unknown, not empty.

  `kind` is "access" (permission/auth -- retrying will not help), "transient"
  (retries exhausted) or "error" (anything else).
  """

  def __init__(self, kind: str, message: str):
    super().__init__(message)
    self.kind = kind

  @property
  def exit_code(self) -> int:
    return EXIT_ACCESS_DENIED if self.kind == "access" else EXIT_QUERY_FAILED


def classify_error(stderr: str) -> str:
  text = stderr or ""
  if any(marker in text for marker in _PERMISSION_MARKERS):
    return "access"
  if any(marker in text for marker in _TRANSIENT_MARKERS):
    return "transient"
  return "error"


def _summarize(stderr: str, limit: int = 300) -> str:
  """First meaningful stderr line, without gcloud's x-google header dump."""
  for line in (stderr or "").splitlines():
    line = line.strip()
    if line and "x-google" not in line:
      return line[:limit]
  return (stderr or "").strip()[:limit]


def run_gcloud_read(
    project_id: str,
    log_filter: str,
    limit: int,
    order: str = "desc",
    fmt: str = "json",
    extra_args: list[str] | None = None,
    retries: int = 3,
    base_delay: float = 2.0,
    timeout: float | None = None,
) -> str:
  """Runs `gcloud logging read` and returns stdout, or raises LoggingQueryError.

  Exit 0 with empty stdout is a genuine empty result. Any non-zero exit is a
  failure, even if some stdout was produced.

  Args:
    project_id: GCP project ID to read from.
    log_filter: Cloud Logging filter.
    limit: Maximum number of entries (`--limit`).
    order: Sort order (`--order`), "asc" or "desc".
    fmt: gcloud output format (`--format`).
    extra_args: Additional gcloud arguments appended verbatim.
    retries: Total attempts; only transient failures are retried.
    base_delay: Initial backoff in seconds, doubled per retry.
    timeout: Per-attempt timeout in seconds; None for no timeout.

  Returns:
    gcloud's stdout.

  Raises:
    LoggingQueryError: gcloud could not be run, was refused, or failed (after
      retries, for transient failures).
  """
  cmd = [
      "gcloud",
      "logging",
      "read",
      log_filter,
      "--project",
      project_id,
      "--limit",
      str(limit),
      "--order",
      order.lower(),
      "--format",
      fmt,
  ] + list(extra_args or [])
  last_err = ""
  for attempt in range(retries):
    try:
      proc = subprocess.run(
          cmd, capture_output=True, text=True, check=False, timeout=timeout
      )
    except subprocess.TimeoutExpired:
      last_err, kind = (
          f"gcloud logging read timed out after {timeout}s",
          "transient",
      )
    except OSError as exc:
      raise LoggingQueryError("error", f"cannot run gcloud: {exc}") from exc
    else:
      if proc.returncode == 0:
        return proc.stdout
      last_err = proc.stderr
      kind = classify_error(proc.stderr)
    if kind != "transient" or attempt == retries - 1:
      raise LoggingQueryError(kind, _summarize(last_err))
    time.sleep(base_delay * (2**attempt))
  raise LoggingQueryError("transient", _summarize(last_err))


def read_entries(
    project_id: str, log_filter: str, limit: int, order: str = "desc", **kwargs
) -> list[dict[str, Any]]:
  """Reads log entries as parsed JSON via `run_gcloud_read`.

  Args:
    project_id: GCP project ID to read from.
    log_filter: Cloud Logging filter.
    limit: Maximum number of entries.
    order: Sort order, "asc" or "desc".
    **kwargs: Passed through to `run_gcloud_read`.

  Returns:
    The entries; an empty list when the query matched nothing.
  """
  out = run_gcloud_read(project_id, log_filter, limit, order, "json", **kwargs)
  return json.loads(out) if out.strip() else []


def preflight(project_id: str) -> None:
  """Fails fast if the project's logs cannot be read at all.

  A one-minute window keeps this cheap; it is the permission check, not the
  data, that matters. Without it a long run discovers an expired token only
  after it has already written partial results.

  Args:
    project_id: GCP project ID whose logs must be readable.

  Raises:
    LoggingQueryError: the probe query was refused or failed.
  """
  # An explicit recent window, not --freshness: gcloud ignores --freshness
  # whenever the filter carries its own timestamp clause, which turned a
  # one-minute probe into a scan of the project's whole retention.
  since = fmt_ts(
      datetime.datetime.now(datetime.timezone.utc)
      - datetime.timedelta(minutes=5)
  )
  run_gcloud_read(
      project_id,
      f'timestamp>="{since}"',
      1,
      retries=2,
      fmt="value(timestamp)",
      timeout=120,
  )


def substring_selector(workload: str) -> str:
  return f'resource.labels.pod_name:"{workload}-"'


def label_selector(workload: str) -> str:
  return f'labels."{JOBSET_LABEL}"="{workload}"'


def scope_clauses(cluster: str = "", namespace: str = "") -> list[str]:
  """Exact resource-label clauses; these are indexed and narrow every scan."""
  clauses = []
  if cluster:
    clauses.append(f'resource.labels.cluster_name="{cluster}"')
  if namespace:
    clauses.append(f'resource.labels.namespace_name="{namespace}"')
  return clauses


def time_clause(start: str, end: str, end_inclusive: bool = True) -> str:
  """Returns a Cloud Logging timestamp filter for [start, end] or [start, end).

  Args:
    start: Window start (RFC3339), always inclusive.
    end: Window end (RFC3339).
    end_inclusive: Whether `end` itself is included. Adjacent slices use an
      exclusive end so an entry exactly on a shared boundary is read once.

  Returns:
    The filter clause.
  """
  parts = []
  if start:
    parts.append(f'timestamp>="{start}"')
  if end:
    op = "<=" if end_inclusive else "<"
    parts.append(f'timestamp{op}"{end}"')
  return " AND ".join(parts)


def choose_workload_selector(
    project_id: str,
    workload: str,
    start: str,
    end: str,
    cluster: str = "",
    namespace: str = "",
) -> tuple[str, str]:
  """Returns (selector_clause, mode) where mode is "label" or "substring".

  Probes the indexed JobSet label first (limit 1, cheap when it matches). Only
  if the label finds nothing does it fall back to the substring scan -- a
  workload not launched as a JobSet has no such label, and silently querying a
  label nobody carries would return the very false negative this module exists
  to prevent.

  Args:
    project_id: GCP project identifier.
    workload: Name of workload to probe.
    start: Start timestamp in UTC ISO format.
    end: End timestamp in UTC ISO format.
    cluster: Optional GKE cluster name filter.
    namespace: Optional Kubernetes namespace filter.

  Returns:
    Tuple of (filter_clause, selector_mode).
  """
  parts = [
      'resource.type="k8s_container"',
      label_selector(workload),
  ] + scope_clauses(cluster, namespace)
  time_filter = time_clause(start, end)
  if time_filter:
    parts.append(time_filter)
  probe = " AND ".join(parts)
  if run_gcloud_read(project_id, probe, 1, fmt="value(timestamp)").strip():
    return label_selector(workload), "label"
  return substring_selector(workload), "substring"


_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:?\d{2})?"
)


def parse_ts(value: str) -> datetime.datetime:
  """Parses an RFC3339 timestamp into an aware UTC datetime.

  Keeps fractional seconds (Cloud Logging emits up to nanoseconds; precision
  beyond microseconds is truncated) and honours non-UTC offsets. A timestamp
  without an offset is taken as UTC.

  Args:
    value: The timestamp string.

  Returns:
    The timestamp converted to UTC.

  Raises:
    ValueError: `value` does not start with an RFC3339 timestamp.
  """
  match = _TS_RE.match(value.strip())
  if not match:
    raise ValueError(f"unparseable timestamp: {value!r}")
  base, frac, offset = match.groups()
  frac = (frac or "")[:7]  # The dot plus at most 6 digits.
  if not offset or offset == "Z":
    offset = "+00:00"
  elif ":" not in offset:
    offset = offset[:3] + ":" + offset[3:]
  dt = datetime.datetime.fromisoformat(base + frac + offset)
  return dt.astimezone(datetime.timezone.utc)


def fmt_ts(dt: datetime.datetime) -> str:
  return dt.astimezone(datetime.timezone.utc).strftime(_TS_FMT)


def split_window(
    start: str, end: str, slice_minutes: int
) -> collections.abc.Iterator[tuple[str, str]]:
  """Splits [start, end] time range into smaller minute intervals.

  Args:
    start: Beginning timestamp ISO string.
    end: Ending timestamp ISO string.
    slice_minutes: Duration of each slice in minutes.

  Yields:
    Tuples of (slice_start, slice_end) formatted timestamp strings.
  """
  start_ts, end_ts = parse_ts(start), parse_ts(end)
  if slice_minutes <= 0 or end_ts <= start_ts:
    yield start, end
    return
  step = datetime.timedelta(minutes=slice_minutes)
  cursor = start_ts
  while cursor < end_ts:
    next_cursor = min(cursor + step, end_ts)
    # The outer bounds are the caller's strings, so sub-second precision on
    # `start` / `end` is never truncated away.
    yield (
        start if cursor == start_ts else fmt_ts(cursor),
        end if next_cursor == end_ts else fmt_ts(next_cursor),
    )
    cursor = next_cursor


def _read_one_slice(
    indexed_window: tuple[int, tuple[str, str]],
    *,
    project_id: str,
    base_filter: str,
    limit_per_slice: int,
    order: str,
    last: int,
    read_kwargs: dict[str, Any],
) -> tuple[tuple[str, str], list[dict[str, Any]]]:
  """Reads one (index, window) slice for `read_sliced`.

  Args:
    indexed_window: (index, (slice_start, slice_end)) from `enumerate`.
    project_id: GCP project identifier.
    base_filter: Logging query filter expression.
    limit_per_slice: Maximum log entries per slice.
    order: Direction of results ("asc" or "desc").
    last: Index of the final slice, the only one that includes its end.
    read_kwargs: Additional keyword arguments passed to read_entries.

  Returns:
    The slice window and the entries read for it.
  """
  index, window = indexed_window
  # Only the final slice includes its end: slices share boundaries.
  clause = time_clause(window[0], window[1], end_inclusive=index == last)
  slice_filter = f"({base_filter}) AND {clause}"
  rows = read_entries(
      project_id, slice_filter, limit_per_slice, order, **read_kwargs
  )
  return window, rows


def read_sliced(
    project_id: str,
    base_filter: str,
    start: str,
    end: str,
    slice_minutes: int,
    limit_per_slice: int,
    max_workers: int = 8,
    order: str = "asc",
    **kwargs,
) -> dict[str, Any]:
  """Reads [start, end] as concurrent slices and merges them in time order.

  Args:
    project_id: GCP project identifier.
    base_filter: Logging query filter expression.
    start: Start timestamp string.
    end: End timestamp string.
    slice_minutes: Minute granularity per query slice.
    limit_per_slice: Maximum log entries per slice.
    max_workers: Concurrency thread pool limit.
    order: Direction of results ("asc" or "desc").
    **kwargs: Additional keyword arguments passed to read_entries.

  Returns:
    Dictionary containing combined log entries and slicing status summary.
  """
  windows = list(split_window(start, end, slice_minutes))
  read_one_slice = functools.partial(
      _read_one_slice,
      project_id=project_id,
      base_filter=base_filter,
      limit_per_slice=limit_per_slice,
      order=order,
      last=len(windows) - 1,
      read_kwargs=kwargs,
  )
  slices, entries = [], []
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max(1, min(max_workers, len(windows)))
  ) as pool:
    try:
      for (slice_start, slice_end), rows in pool.map(
          read_one_slice, enumerate(windows)
      ):
        slices.append({
            "start": slice_start,
            "end": slice_end,
            "count": len(rows),
            "truncated": len(rows) >= limit_per_slice,
        })
        entries.extend(rows)
    except LoggingQueryError:
      # Don't let the pool's exit wait for every queued slice (and its retry
      # back-off) before the caller can report the failure.
      pool.shutdown(wait=False, cancel_futures=True)
      raise
  entries.sort(
      key=lambda entry: entry.get("timestamp", ""),
      reverse=order.lower() == "desc",
  )
  slices.sort(key=lambda slice_info: slice_info["start"])
  return {
      "entries": entries,
      "slices": slices,
      "truncated": any(slice_info["truncated"] for slice_info in slices),
  }


def fail(exc: LoggingQueryError, context: str = "") -> None:
  """Prints a machine-readable failure record and exits with its code."""
  status = "ACCESS_DENIED" if exc.kind == "access" else "QUERY_FAILED"
  record = {"status": status, "error_kind": exc.kind, "error": str(exc)}
  if context:
    record["context"] = context
  if exc.kind == "access":
    record["hint"] = (
        "Cloud Logging access was refused. Results would be "
        "unknown, not empty -- refresh credentials or the access "
        "justification and re-run."
    )
  print(json.dumps(record, indent=2))
  print(
      f"\n--- SUMMARY REPORT (summary_fields) ---\nstatus: {status}",
      file=sys.stderr,
  )
  sys.exit(exc.exit_code)

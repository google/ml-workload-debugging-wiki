"""Queries workload timestamps via Cloud Logging."""

import argparse
from collections.abc import Sequence
import datetime
import functools
import importlib
import json
import os
import re
import subprocess
import sys

if __package__:
  log_query = importlib.import_module(".log_query", __package__)
else:
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  log_query = importlib.import_module("log_query")

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
_DEFAULT_WINDOW_DAYS = 30
_TWO_DIGIT_TZ_RE = re.compile(r"([+-]\d{2})$")
_SUBSECOND_RE = re.compile(r"\.\d+")
_YYYYMMDD_TOKEN_RE = re.compile(r"(?:^|-)(\d{4})(\d{2})(\d{2})(?=-|$)")
_MMDD_TOKEN_RE = re.compile(r"(?:^|-)(\d{2})(\d{2})(?=-|$)")
_NON_DATE_4DIGIT_TOKENS = frozenset({
    "0064",
    "0128",
    "0256",
    "0512",
    "1024",
    "2048",
    "4096",
    "8192",
})


def _infer_date_window_from_name(
    jobset_name: str, now: datetime.datetime
) -> tuple[str, str] | None:
  """Infers narrow [date - 1d, date + 7d] window if name encodes a date.

  Args:
    jobset_name: Workload or JobSet name that may contain a hyphen-delimited
      YYYYMMDD or MMDD date token.
    now: Current UTC datetime used to bound the inferred window and resolve the
      year for MMDD tokens.

  Returns:
    A `(start_timestamp, end_timestamp)` tuple of UTC RFC3339 strings covering
    `[date - 1d, min(now, date + 7d)]`, or `None` if `jobset_name` does not
    contain a valid date token or if the inferred start date is not before
    `now`.
  """
  m8 = _YYYYMMDD_TOKEN_RE.search(jobset_name or "")
  if m8:
    year, month, day = int(m8.group(1)), int(m8.group(2)), int(m8.group(3))
    try:
      dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
      start_dt = dt - datetime.timedelta(days=1)
      end_dt = min(now, dt + datetime.timedelta(days=7))
      if start_dt < end_dt:
        return start_dt.strftime(_TS_FMT), end_dt.strftime(_TS_FMT)
    except ValueError:
      pass
  for m4 in reversed(list(_MMDD_TOKEN_RE.finditer(jobset_name or ""))):
    token = m4.group(1) + m4.group(2)
    if token in _NON_DATE_4DIGIT_TOKENS:
      continue
    month, day = int(m4.group(1)), int(m4.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
      continue
    for year in (now.year, now.year - 1):
      try:
        dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
      except ValueError:
        continue
      start_dt = dt - datetime.timedelta(days=1)
      if start_dt <= now:
        end_dt = min(now, dt + datetime.timedelta(days=7))
        if start_dt < end_dt:
          return start_dt.strftime(_TS_FMT), end_dt.strftime(_TS_FMT)
  return None


def format_timestamp(ts_str: str) -> str:
  """Formats timestamp string to standard UTC RFC3339 (YYYY-MM-DDTHH:MM:SSZ)."""
  ts_str = ts_str.strip().strip('"').strip("'")
  if not ts_str or ts_str.upper() in ("NULL", "NONE"):
    return ""

  # Standardize separators
  ts_str = ts_str.replace(" ", "T")

  # Standardize trailing Z to +00:00
  if ts_str.endswith("Z"):
    ts_str = ts_str[:-1] + "+00:00"

  # If it ends with -XX or +XX (2-digit timezone), pad it to -XX:00
  if _TWO_DIGIT_TZ_RE.search(ts_str):
    ts_str = ts_str + ":00"

  # Remove subseconds (anything after dot up to the timezone offset or end)
  ts_str = _SUBSECOND_RE.sub("", ts_str)

  try:
    parsed_dt = datetime.datetime.fromisoformat(ts_str)
    if parsed_dt.tzinfo is not None:
      parsed_dt = parsed_dt.astimezone(datetime.timezone.utc)
    else:
      parsed_dt = parsed_dt.replace(tzinfo=datetime.timezone.utc)
    return parsed_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  except (ValueError, TypeError, OverflowError):
    for time_format in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
      try:
        parsed_dt = datetime.datetime.strptime(ts_str, time_format).replace(
            tzinfo=datetime.timezone.utc
        )
        return parsed_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
      except ValueError:
        # Try the next candidate timestamp format.
        pass
    print(f"Failed to parse timestamp '{ts_str}'", file=sys.stderr)
    return ts_str


def parse_utc_timestamp(ts_str: str) -> datetime.datetime | None:
  """Parses RFC3339/ISO UTC timestamp string into a datetime object."""
  if not ts_str:
    return None
  normalized_ts = ts_str.strip()
  if normalized_ts.endswith("Z"):
    normalized_ts = normalized_ts[:-1] + "+00:00"
  try:
    parsed_dt = datetime.datetime.fromisoformat(normalized_ts)
    if parsed_dt.tzinfo is not None:
      return parsed_dt.astimezone(datetime.timezone.utc)
    return parsed_dt.replace(tzinfo=datetime.timezone.utc)
  except (ValueError, TypeError):
    for time_format in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
      try:
        return datetime.datetime.strptime(normalized_ts, time_format).replace(
            tzinfo=datetime.timezone.utc
        )
      except ValueError:
        # Try the next candidate timestamp format.
        pass
    return None


def main() -> None:
  """Main CLI entrypoint for querying workload timestamps."""
  parser = argparse.ArgumentParser(
      description="Query workload timestamps via Cloud Logging."
  )
  parser.add_argument(
      "--jobset_name",
      "--workload_name",
      "--workload",
      dest="jobset_name",
      required=True,
      help="JobSet workload name.",
  )
  parser.add_argument(
      "--project_id", default="", help="GCP project ID (optional)."
  )
  parser.add_argument(
      "--project_number", default="", help="GCP project number (optional)."
  )
  parser.add_argument(
      "--namespace",
      default="default",
      help="Kubernetes namespace (default 'default').",
  )
  parser.add_argument(
      "--restart_attempt",
      type=int,
      default=-1,
      help=(
          "JobSet restart attempt index (default -1 to query across all"
          " attempts)."
      ),
  )
  parser.add_argument(
      "--latest_attempts",
      "--latest-attempts",
      type=int,
      default=0,
      help=(
          "If > 0, return only the latest N restart attempts discovered across"
          " the workload in reverse chronological order."
      ),
  )
  parser.add_argument(
      "--window_minutes",
      type=int,
      default=5,
      help=(
          "Trailing triage window duration in minutes prior to"
          " last_occurrence (default 5). Set <= 0 for full run window."
      ),
  )
  parser.add_argument(
      "--start", default="", help="Optional window start timestamp (RFC3339)."
  )
  parser.add_argument(
      "--end", default="", help="Optional window end timestamp (RFC3339)."
  )
  parser.add_argument(
      "--bucket", default="", help="Cloud Logging bucket (optional)."
  )
  parser.add_argument(
      "--location", default="", help="Cloud Logging location (optional)."
  )
  parser.add_argument(
      "--view", default="", help="Cloud Logging view (optional)."
  )
  parser.add_argument(
      "--output_file", default="", help="Optional output JSON file path."
  )
  parser.add_argument(
      "--lookback_days",
      "--lookback-days",
      type=int,
      default=0,
      help=(
          "Number of days to look back for workload logs (default 0, which uses"
          f" the standard {_DEFAULT_WINDOW_DAYS}-day window or date inference)."
      ),
  )

  args = parser.parse_args()

  jobset_name = args.jobset_name.strip() if args.jobset_name else ""
  if not jobset_name or jobset_name in ("None", "null"):
    print(
        "Error: --jobset_name must be a non-empty workload name.",
        file=sys.stderr,
    )
    sys.exit(1)
  args.jobset_name = jobset_name

  project_id = (
      args.project_id.strip()
      if args.project_id and args.project_id.strip() not in ("", "None")
      else ""
  )
  if (
      not project_id
      and args.project_number
      and args.project_number.strip() not in ("", "0", "None")
  ):
    project_id = args.project_number.strip()
  # The project_id resource-label clause is only added when the caller named a
  # project; a project resolved from gcloud config is used for --project only.
  explicit_project = bool(project_id)
  if not project_id:
    project_id = _active_gcloud_project()
  if not project_id:
    print(
        "Error: no GCP project given (--project_id/--project_number) and no"
        " active gcloud project is configured.",
        file=sys.stderr,
    )
    sys.exit(1)

  try:
    _lookup(args, project_id, explicit_project)
  except log_query.LoggingQueryError as exc:
    log_query.fail(exc, context="lookup-workload-timestamps")


def _active_gcloud_project() -> str:
  """Returns the active gcloud project, or "" if unset or unavailable."""
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
  except (OSError, subprocess.SubprocessError):
    return ""
  value = result.stdout.strip()
  if result.returncode == 0 and value and value != "(unset)":
    return value
  return ""


def _run_logging_query(
    project_id: str,
    extra_args: Sequence[str],
    filter_arg: str,
    order: str = "DESC",
    limit: int = 1,
    format_str: str = "value(timestamp)",
) -> str:
  """Returns stdout; a failed query raises LoggingQueryError.

  Args:
    project_id: GCP project identifier.
    extra_args: Extra gcloud arguments (bucket/location/view).
    filter_arg: Logging query filter expression.
    order: Result order ("DESC" or "ASC").
    limit: Maximum number of entries to read.
    format_str: gcloud --format expression.

  Returns:
    The query's stdout.
  """
  return log_query.run_gcloud_read(
      project_id,
      filter_arg,
      limit,
      order=order,
      fmt=format_str,
      extra_args=list(extra_args),
  )


def _first_line(out: str) -> str:
  """Returns the first stripped line of `out`, or "" when it is blank."""
  return out.strip().splitlines()[0].strip() if out.strip() else ""


def _lookup(
    args: argparse.Namespace, project_id: str, explicit_project: bool
) -> None:
  """Runs the timestamp discovery; LoggingQueryError propagates to main()."""
  now = datetime.datetime.now(datetime.timezone.utc)
  default_window_applied = False
  inferred_window_applied = False
  effective_lookback_days = (
      args.lookback_days
      if args.lookback_days and args.lookback_days > 0
      else _DEFAULT_WINDOW_DAYS
  )
  if args.start and args.end:
    window_start, window_end = args.start, args.end
  elif args.start:
    window_start, window_end = args.start, now.strftime(_TS_FMT)
  elif args.end:
    end_dt = parse_utc_timestamp(args.end) or now
    window_start = (
        end_dt - datetime.timedelta(days=effective_lookback_days)
    ).strftime(_TS_FMT)
    window_end = args.end
  else:
    # A bounded default window, never an unbounded scan: an unbounded query
    # over a long-lived, many-host workload is what timed out and returned
    # backend 500s. If lookback_days is explicitly set > 0, use it. Otherwise,
    # if the workload name encodes a date (e.g. -0827- or -20260827-), scope
    # the window around that date first.
    if args.lookback_days and args.lookback_days > 0:
      default_window_applied = True
      window_start = (
          now - datetime.timedelta(days=effective_lookback_days)
      ).strftime(_TS_FMT)
      window_end = now.strftime(_TS_FMT)
    else:
      inferred = _infer_date_window_from_name(args.jobset_name, now)
      if inferred is not None:
        inferred_window_applied = True
        window_start, window_end = inferred
      else:
        default_window_applied = True
        window_start = (
            now - datetime.timedelta(days=_DEFAULT_WINDOW_DAYS)
        ).strftime(_TS_FMT)
        window_end = now.strftime(_TS_FMT)
  time_clauses = [
      f'timestamp >= "{window_start}"',
      f'timestamp <= "{window_end}"',
  ]

  extra_args = []
  if args.bucket:
    extra_args.extend(["--bucket", args.bucket])
  if args.location:
    extra_args.extend(["--location", args.location])
  if args.view:
    extra_args.extend(["--view", args.view])

  log_query.preflight(project_id)

  # The selector probe reads the _Default bucket; when a custom bucket or view
  # is requested it cannot speak for that bucket, so keep the substring form.
  if extra_args:
    selector = log_query.substring_selector(args.jobset_name)
    selector_mode = "substring"
  else:
    selector, selector_mode = log_query.choose_workload_selector(
        project_id,
        args.jobset_name,
        window_start,
        window_end,
        cluster="",
        namespace=args.namespace,
    )

  base_clauses = [
      'resource.type="k8s_container"',
      f'resource.labels.namespace_name="{args.namespace}"',
      selector,
  ]
  if explicit_project:
    base_clauses.append(f'resource.labels.project_id="{project_id}"')
  if args.restart_attempt >= 0:
    base_clauses.append(
        'labels."k8s-pod/jobset_sigs_k8s_io/restart-attempt":'
        f' "{args.restart_attempt}"'
    )

  head_filter = (
      '(resource.labels.pod_name:"-0-0-" OR'
      ' resource.labels.pod_name:"-pathways-head" OR'
      ' resource.labels.pod_name:"-pwhd")'
  )
  fast_clauses = base_clauses + [head_filter]
  broad_clauses = list(base_clauses)

  run_logging_query = functools.partial(
      _run_logging_query, project_id, extra_args
  )

  # Fast path targets coordinator/head pods. Only a genuinely empty result
  # (exit 0, no rows) falls through to the broad query; a failure raises.
  used_fast_path = False
  last_occ_line = _first_line(
      run_logging_query(" AND ".join(fast_clauses + time_clauses))
  )
  if last_occ_line:
    used_fast_path = True
    active_clauses = fast_clauses
  else:
    last_occ_line = _first_line(
        run_logging_query(" AND ".join(broad_clauses + time_clauses))
    )
    active_clauses = broad_clauses

  # If the date-inferred window returned no logs (e.g. a job config named
  # -0827- that was re-run >7 days later), fall back to the standard
  # default window before failing.
  if not last_occ_line and inferred_window_applied:
    inferred_window_applied = False
    default_window_applied = True
    window_start = (
        now - datetime.timedelta(days=effective_lookback_days)
    ).strftime(_TS_FMT)
    window_end = now.strftime(_TS_FMT)
    time_clauses = [
        f'timestamp >= "{window_start}"',
        f'timestamp <= "{window_end}"',
    ]
    last_occ_line = _first_line(
        run_logging_query(" AND ".join(fast_clauses + time_clauses))
    )
    if last_occ_line:
      used_fast_path = True
      active_clauses = fast_clauses
    else:
      last_occ_line = _first_line(
          run_logging_query(" AND ".join(broad_clauses + time_clauses))
      )
      active_clauses = broad_clauses

  if not last_occ_line:
    scope = (
        f"default {effective_lookback_days}-day window"
        if default_window_applied
        else f"window {window_start} .. {window_end}"
    )
    print(
        f"Error: No logs found for workload {args.jobset_name} in Cloud"
        f" Logging ({scope}).",
        file=sys.stderr,
    )
    sys.exit(1)

  overall_last = format_timestamp(last_occ_line)

  # Earliest log line timestamp (workload_start) via --order=ASC --limit=1.
  workload_start_raw = _first_line(
      run_logging_query(
          " AND ".join(active_clauses + time_clauses), order="ASC"
      )
  )
  if not workload_start_raw and used_fast_path:
    workload_start_raw = _first_line(
        run_logging_query(
            " AND ".join(broad_clauses + time_clauses), order="ASC"
        )
    )
  if not workload_start_raw:
    workload_start_raw = last_occ_line

  workload_start = format_timestamp(workload_start_raw)

  if args.start:
    overall_first = format_timestamp(args.start)
  elif args.window_minutes > 0 and workload_start and overall_last:
    start_dt = parse_utc_timestamp(workload_start)
    last_dt = parse_utc_timestamp(overall_last)
    if start_dt and last_dt:
      first_dt = max(
          start_dt, last_dt - datetime.timedelta(minutes=args.window_minutes)
      )
      overall_first = first_dt.strftime(_TS_FMT)
    else:
      overall_first = workload_start
  else:
    overall_first = workload_start

  if args.restart_attempt >= 0:
    attempts = [{
        "restart_attempt": args.restart_attempt,
        "first_occurrence": overall_first,
        "last_occurrence": overall_last,
        "attempt_start": workload_start,
    }]
  else:
    # Discover restart attempts across all workload logs.
    attempt_filter = " AND ".join(
        [clause for clause in active_clauses if "restart-attempt" not in clause]
        + ['labels."k8s-pod/jobset_sigs_k8s_io/restart-attempt":*']
        + time_clauses
    )
    attempt_out = run_logging_query(
        attempt_filter,
        order="DESC",
        limit=100,
        format_str=(
            'value(labels."k8s-pod/jobset_sigs_k8s_io/restart-attempt",'
            "timestamp)"
        ),
    )

    discovered_attempts = set()
    for line in attempt_out.strip().splitlines():
      for token in line.strip().split():
        try:
          discovered_attempts.add(int(token))
          break
        except ValueError:
          # Ignore non-integer tokens (such as timestamps) on the line.
          pass

    if args.latest_attempts > 0:
      sorted_attempts = sorted(discovered_attempts, reverse=True)[
          : args.latest_attempts
      ]
    else:
      sorted_attempts = sorted(discovered_attempts)

    if len(discovered_attempts) <= 1:
      attempts = [{
          "restart_attempt": sorted_attempts[0] if sorted_attempts else 0,
          "first_occurrence": overall_first,
          "last_occurrence": overall_last,
          "attempt_start": workload_start,
      }]
    else:
      min_attempt = min(discovered_attempts)
      attempts = []
      for attempt in sorted_attempts:
        per_attempt_filter = " AND ".join(
            [
                clause
                for clause in active_clauses
                if "restart-attempt" not in clause
            ]
            + [
                'labels."k8s-pod/jobset_sigs_k8s_io/restart-attempt":'
                f' "{attempt}"'
            ]
            + time_clauses
        )
        last_line = _first_line(
            run_logging_query(per_attempt_filter, order="DESC")
        )
        attempt_last = (
            format_timestamp(last_line) if last_line else overall_last
        )
        first_line = _first_line(
            run_logging_query(per_attempt_filter, order="ASC")
        )
        if first_line:
          attempt_start = format_timestamp(first_line)
        else:
          attempt_start = (
              workload_start if attempt == min_attempt else attempt_last
          )

        attempt_first = attempt_start
        if args.window_minutes > 0 and attempt_start and attempt_last:
          attempt_start_dt = parse_utc_timestamp(attempt_start)
          attempt_last_dt = parse_utc_timestamp(attempt_last)
          if attempt_start_dt and attempt_last_dt:
            attempt_first = max(
                attempt_start_dt,
                attempt_last_dt
                - datetime.timedelta(minutes=args.window_minutes),
            ).strftime(_TS_FMT)

        attempts.append({
            "restart_attempt": attempt,
            "first_occurrence": attempt_first,
            "last_occurrence": attempt_last,
            "attempt_start": attempt_start,
        })

  output_data = {}
  if args.restart_attempt >= 0:
    output_data["restart_attempt"] = args.restart_attempt
  if args.latest_attempts > 0:
    output_data["latest_attempts"] = args.latest_attempts
  output_data["first_occurrence"] = overall_first
  output_data["last_occurrence"] = overall_last
  output_data["workload_start"] = workload_start
  output_data["window_minutes"] = args.window_minutes
  output_data["attempts"] = attempts
  output_data["search_window"] = {"start": window_start, "end": window_end}
  output_data["default_window_applied"] = default_window_applied
  if args.lookback_days and args.lookback_days > 0:
    output_data["lookback_days"] = args.lookback_days
  output_data["selector_mode"] = selector_mode
  output_data["used_fast_path"] = used_fast_path

  if args.output_file:
    file_payload = dict(output_data)
    file_payload["output_file"] = args.output_file
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as output_file:
      json.dump(file_payload, output_file, indent=2)

  print(json.dumps(output_data))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"  first_occurrence: {overall_first}")
  print(f"  last_occurrence: {overall_last}")
  print(f"  workload_start: {workload_start}")
  print(f"  attempts: {len(attempts)}")
  if args.latest_attempts > 0:
    print(f"  latest_attempts: {args.latest_attempts}")
  print(f"  selector_mode: {selector_mode}")
  print(f"  used_fast_path: {used_fast_path}")
  if default_window_applied:
    print(
        f"  default_window_applied: last {effective_lookback_days} days"
        f" ({window_start} .. {window_end}); pass --start/--end for older runs"
    )
  if len(attempts) > 1 or (len(attempts) == 1 and args.latest_attempts > 0):
    for attempt_entry in attempts:
      print(
          f"    attempt {attempt_entry['restart_attempt']}:"
          f" {attempt_entry['first_occurrence']} ->"
          f" {attempt_entry['last_occurrence']} (attempt_start:"
          f" {attempt_entry['attempt_start']})"
      )
  if args.output_file:
    print(f"  output_file: {args.output_file}")
  print("---------------------------------------")
  sys.exit(0)


if __name__ == "__main__":
  main()

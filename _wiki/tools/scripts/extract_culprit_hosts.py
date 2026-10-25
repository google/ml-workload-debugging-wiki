#!/usr/bin/env python3
"""Extracts culprit hosts that the runtime names explicitly in its own logs.

The Pathways resource manager and worker pipes print the failing host directly,
e.g. `lost connection to peer at http://machine/<node>/events` or
`... distributor <id> <node> exiting`. Attribution should always be quoted from
these lines rather than inferred from DATA_LOSS device coordinates: the device
coords identify where the *symptom* surfaced, which is frequently a different
slice from the host that actually died.

McJax / MaxText / Megascale multi-slice JobSets never print those lines, so the
script also reads (see `signature_family`):
  * megascale: the Megascale hang digest `Potential culprit workers: ...`.
  * barrier: JAX coordination-service `Barrier timed out` (which lists the
    tasks that did NOT arrive) and heartbeat timeouts (which name the dead
    task). The timeout itself is a victim; the non-arriving task is the lead.
  * earliest_error_candidates: pods ranked by their EARLIEST crash line, so
    the culprit is never "worker 0" or "the loudest pod" by default.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable, Mapping, Sequence
import concurrent.futures
import dataclasses
import datetime
import json
import random
import re
import subprocess
import sys
import time
from typing import Any

import log_query

# `lost connection to peer at http://machine/gke-tpu-node-pool-xxxx/events#...`
PEER_RE = re.compile(
    r"lost connection to peer at http://machine/([A-Za-z0-9._-]+)"
)
# `... first level distributor 4292922668782146471 gke-tpu-node-pool-xxxx
# exiting`
EXITING_RE = re.compile(r"distributor\s+\d+\s+([A-Za-z0-9._-]+)\s+exiting")

# A new log entry begins with the RFC3339 timestamp column; continuation
# lines of a multi-line payload do not.
_ENTRY_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

SEARCH_TERMS = [
    "lost connection to peer",
    "exiting [reported by worker pipe closing]",
]

# --- Non-Pathways signatures (McJax / MaxText / Megascale multi-slice). ---
# Megascale hang digest. Source: FormatDigest() / FormatWorker() in
# megascale_error_aggregator.cc:
#   "    Potential cause: $0"  (human text, e.g. "Unrecoverable error")
#   "    Potential culprit workers: $0", workers = "<worker_id>:<hostname>
#   [@<loc>:<core>]" joined by ", ", optionally + " and N more devices.".
# Verbatim example: error_reporter_test.cc
# ("Potential culprit workers: slice0-task0:worker-cell, ...").
# Wiki: symptoms/megascale-hang.md.
MEGASCALE_CULPRIT_RE = re.compile(r"Potential culprit workers:[ \t]*([^\n]*)")
MEGASCALE_CAUSE_RE = re.compile(r"Potential cause:[ \t]*([^\n]+)")
MEGASCALE_WORKER_RE = re.compile(
    r"^(?P<worker>[^\s:,]+):(?P<host>[^\s:@,]+)(?:@\S*)?$"
)
MEGASCALE_MORE_RE = re.compile(r"\s+and\s+\d+\s+more devices\.?\s*$")

# JAX coordination-service barrier timeout. Source: CheckBarrierTimeout() in
# third_party/tensorflow/compiler/xla/tsl/distributed_runtime/coordination/
# coordination_service.cc (and the pjrt/distributed/coordination copy):
#   "Barrier timed out. Id: %s. ... # of tasks that reached the barrier: %d/%d.
#   \nThe first task at the barrier: %s. Some timed out task names:\n%s"
# Task names are "/job:<job>/replica:0/task:<id>" (tsl GetTaskName) or a bare
# int (pjrt copy); at most kPendingStragglerLogLimit-1 = 2 names are listed.
# Wiki: playbooks/rca-invariants.md (#4), srcs/cases/incident-rca-cards-review-
# 2026-10.md excerpt 4 ("... 1023/1024. The first task at the barrier: 466.
# Some timed out task names: 106").
BARRIER_TIMEOUT_RE = re.compile(r"Barrier timed out\. Id: (\S+)")
BARRIER_REACHED_RE = re.compile(
    r"# of tasks that reached the barrier: (\d+)/(\d+)"
)
BARRIER_FIRST_TASK_RE = re.compile(
    r"The first task at the barrier: (?:/job:[\w-]+/replica:\d+/task:)?(\d+)"
)
BARRIER_TIMED_OUT_NAMES = "Some timed out task names:"
# Heartbeat timeouts name the task that stopped heartbeating. Source:
# CheckHeartbeatTimeout() in the same coordination_service.cc files:
#   "Task <task> heartbeat timeout. This indicates that the remote task has
#   failed, got preempted, or crashed unexpectedly. ..."
#   "The following tasks are unhealthy (stopped sending heartbeats):\n<tasks>
#   \nThe tasks have crashed. ..."
HEARTBEAT_TASK_RE = re.compile(r"Task (\S+) heartbeat timeout")
UNHEALTHY_RE = re.compile(
    r"The following tasks are unhealthy \(stopped sending heartbeats\):"
    r"(.*?)(?:The tasks have crashed|$)",
    re.DOTALL,
)
TASK_TOKEN_RE = re.compile(r"^(?:/job:[\w-]+/replica:\d+/task:)?(\d+)$")

MEGASCALE_TERMS = ("Potential culprit workers", "Potential cause")
BARRIER_TERMS = (
    "Barrier timed out",
    "heartbeat timeout",
    "stopped sending heartbeats",
)
# Crash signatures used ONLY for earliest-error ranking. Source: signature
# table in concepts/application-log-analysis.md (CRASH_PYTHON_FATAL,
# CRASH_TRACEBACK, CRASH_SEGFAULT, CRASH_FATAL_HARDWARE, HALT_DEVICE,
# CRASH_CORE_HALT). Barrier / heartbeat lines are deliberately excluded:
# timeouts are victims (playbooks/rca-invariants.md #4).
EARLY_ERROR_TERMS = (
    "Fatal Python error",
    "Traceback (most recent call",
    "Segmentation fault",
    "!!!! FATAL ERROR !!!!",
    "halted prematurely",
    "RuntimeUnexpectedCoreHalt",
    "*** SIGILL",
)
_VICTIM_MARKERS = (
    "Barrier timed out",
    "heartbeat timeout",
    "stopped sending heartbeats",
    "Shutdown barrier has failed",
)
# JobSet pod name: <jobset>-<replicatedJob>-<jobIndex>-<podIndex>-<hash>.
_JOBSET_POD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<job_index>\d+)-(?P<pod_index>\d+)-[a-z0-9]{5}$"
)
_INDEX_0_LEADER_POD_RE = re.compile(r"-worker-job0-\d+-0-[a-z0-9]+$")
_SIGILL_SI_USER_RE = re.compile(r"\*\*\*\s*SIGILL\b.*?\bsi_code=0\b")
_PATHWAYS_POD_MARKER = "pathways"

# Cloud Logging read quota (~60 entries.list/min/project) is shared by every
# script an agent runs in parallel; a 429 there is transient, not "no logs".
_RETRYABLE_RE = re.compile(
    r"429|RESOURCE_EXHAUSTED|Quota exceeded|503|UNAVAILABLE", re.IGNORECASE
)
_HTTP_STATUS_RE = re.compile(
    r"(?i)(?:http\w*|status|code)[^0-9\n]{0,20}\b([45]\d\d)\b"
)
_BACKOFF_SECONDS = (5, 10, 20)
_QUERY_TIMEOUT_SECONDS = 600
_ERROR_EXCERPT_CHARS = 200
# When the status keyword sits past the excerpt, keep this much context.
_STATUS_CONTEXT_BEFORE = 20
_STATUS_CONTEXT_AFTER = 55
_EXCERPT_HEAD_CHARS = _ERROR_EXCERPT_CHARS - 80
_PAYLOAD_EXCERPT_CHARS = 200
# Report keys written to --output_file but left out of the printed summary.
_ENTRY_COLUMNS = 4  # timestamp, pod, node, payload.
_FULL_REPORT_ONLY_KEYS = ("events", "incidents", "pod_to_node")

# A parsed log entry: timestamp, pod, node, payload.
LogEntry = dict[str, str]
QueryError = dict[str, str | int | None]
# (items, error or None, truncated at --limit).
QueryResult = tuple[list[str], QueryError | None, bool]


def error_excerpt(stderr: str) -> str:
  """Shortens gcloud stderr for the report.

  Args:
    stderr: Raw gcloud stderr.

  Returns:
    Whitespace-collapsed text of at most about _ERROR_EXCERPT_CHARS chars,
    keeping the HTTP status / quota keyword even if it appears late.
  """
  text = " ".join(stderr.split())
  match = _RETRYABLE_RE.search(text) or _HTTP_STATUS_RE.search(text)
  if (
      len(text) <= _ERROR_EXCERPT_CHARS
      or not match
      or match.start() < _EXCERPT_HEAD_CHARS
  ):
    return text[:_ERROR_EXCERPT_CHARS]
  snippet_start = max(0, match.start() - _STATUS_CONTEXT_BEFORE)
  snippet = text[snippet_start : match.start() + _STATUS_CONTEXT_AFTER]
  return f"{text[:_EXCERPT_HEAD_CHARS]} ... {snippet}"


def gcloud_read(
    command: Sequence[str], label: str
) -> tuple[str | None, QueryError | None]:
  """Runs a gcloud read, retrying only transient errors.

  Retries with exponential backoff + jitter only on quota / unavailable
  errors; anything else (bad filter, auth) fails fast.

  Args:
    command: gcloud argv.
    label: Query label recorded in the error.

  Returns:
    (stdout, None) on success, or (None, error) with keys query, attempts,
    http_status and error after the final failed attempt.
  """
  error = None
  for attempt in range(len(_BACKOFF_SECONDS) + 1):
    try:
      process = subprocess.run(
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
      if process.returncode == 0:
        return process.stdout, None
      stderr = process.stderr or f"gcloud exited {process.returncode}"
    status = _HTTP_STATUS_RE.search(stderr)
    error = {
        "query": label,
        "attempts": attempt + 1,
        "http_status": int(status.group(1)) if status else None,
        "error": error_excerpt(stderr),
    }
    if not _RETRYABLE_RE.search(stderr) or attempt == len(_BACKOFF_SECONDS):
      break
    delay = _BACKOFF_SECONDS[attempt]
    time.sleep(delay + random.uniform(0, delay / 2))
  print(f"Warning: query {label} FAILED: {error['error']}", file=sys.stderr)
  return None, error


def run_query(
    project_id: str,
    base_filter: str,
    start: str,
    end: str,
    term: str,
    limit: int,
    end_inclusive: bool = True,
) -> tuple[list[str], int, bool]:
  """Returns (lines, entry_count, truncated) for one term over one window.

  `base_filter` is the workload selector plus scope clauses (no time bound).
  Failures raise LoggingQueryError: mapping a refused query to [] once turned
  an expired token into a published "0 culprit incidents" result.

  Args:
    project_id: GCP project ID to read from.
    base_filter: Workload selector plus scope clauses.
    start: Window start (RFC3339, inclusive).
    end: Window end (RFC3339).
    term: Substring to match in textPayload or jsonPayload.message.
    limit: Maximum number of log entries to read.
    end_inclusive: Whether `end` itself is included; adjacent slices pass
      False so an entry on a shared boundary is counted once.

  Returns:
    A (lines, entry_count, truncated) tuple: the non-empty output lines, the
    number of log entries they contain, and whether `limit` was reached.

  Raises:
    log_query.LoggingQueryError: the query was refused or failed.
  """
  # Pathways splits its output across textPayload and jsonPayload.message
  # depending on the emitting component; filtering only textPayload silently
  # drops resource-manager lines, which are exactly the ones that name hosts.
  log_filter = (
      f"{base_filter} AND "
      f"{log_query.time_clause(start, end, end_inclusive)} AND "
      f'(textPayload:"{term}" OR jsonPayload.message:"{term}")'
  )
  out = log_query.run_gcloud_read(
      project_id,
      log_filter,
      limit,
      order="asc",
      fmt=(
          "value(timestamp,resource.labels.pod_name,textPayload,"
          "jsonPayload.message)"
      ),
  )
  lines = [ln for ln in out.splitlines() if ln.strip()]
  # --limit bounds log ENTRIES, but multi-line payloads (stack traces) emit
  # several output lines each, so len(lines) overstates the entry count and
  # would raise spurious truncation warnings. Count entries instead: only a
  # line carrying the leading RFC3339 timestamp field starts a new entry.
  entries = sum(1 for ln in lines if _ENTRY_START_RE.match(ln))
  return lines, entries, entries >= limit


def auto_slice_minutes(start: str, end: str) -> int:
  """360-minute slices for windows over a day; otherwise one query."""
  span = log_query.parse_ts(end) - log_query.parse_ts(start)
  return 360 if span.total_seconds() > 86400 else 0


def query_term(
    project_id: str,
    base_filter: str,
    start: str,
    end: str,
    term: str,
    limit: int,
    slice_minutes: int,
    max_workers: int,
) -> tuple[list[str], dict[str, Any]]:
  """Queries one term over concurrent slices; returns (lines, stats).

  Slices are returned in window order and each is ascending, so concatenation
  preserves time order without re-sorting (which would separate multi-line
  payloads from their header line). The entry cap applies per slice, so one
  saturated slice is reported even when the total looks modest.

  Args:
    project_id: GCP project ID to read from.
    base_filter: Workload selector plus scope clauses.
    start: Window start (RFC3339, inclusive).
    end: Window end (RFC3339, inclusive).
    term: Substring to match in textPayload or jsonPayload.message.
    limit: Maximum number of log entries to read per slice.
    slice_minutes: Slice length in minutes; 0 for a single query.
    max_workers: Maximum number of concurrent slice queries.

  Returns:
    A (lines, stats) tuple: the output lines of all slices in time order, and
    a dict with slice count, entry total and truncation details.

  Raises:
    log_query.LoggingQueryError: a slice query was refused or failed.
  """
  windows = list(log_query.split_window(start, end, slice_minutes))
  last = len(windows) - 1
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max(1, min(max_workers, len(windows)))
  ) as pool:
    try:
      results = list(
          pool.map(
              lambda iw: run_query(
                  project_id,
                  base_filter,
                  iw[1][0],
                  iw[1][1],
                  term,
                  limit,
                  end_inclusive=iw[0] == last,
              ),
              enumerate(windows),
          )
      )
    except log_query.LoggingQueryError:
      # Cancel queued slices so the caller can fail fast instead of waiting
      # for every remaining query (and its retry back-off) to finish.
      pool.shutdown(wait=False, cancel_futures=True)
      raise
  lines, truncated_slices, slice_errors, total = [], [], [], 0
  for (start_ts, end_ts), (rows, second, truncated) in zip(windows, results):
    lines.extend(rows)
    if isinstance(second, int):
      entries = second
    else:
      entries = sum(1 for ln in rows if _ENTRY_START_RE.match(ln))
      if isinstance(second, dict):
        slice_errors.append(second)
    total += entries
    if truncated:
      truncated_slices.append(
          {"start": start_ts, "end": end_ts, "entries": entries}
      )
  if truncated_slices:
    print(
        f"WARNING: term '{term}' hit --limit ({limit}) in "
        f"{len(truncated_slices)} of {len(windows)} slice(s). Results are "
        "TRUNCATED and attribution may be incomplete; re-run with a higher "
        "--limit or smaller --slice_minutes.",
        file=sys.stderr,
    )
  stats: dict[str, Any] = {
      "slices": len(windows),
      "entries": total,
      "truncated": bool(truncated_slices),
      "truncated_slices": truncated_slices,
  }
  if slice_errors:
    stats["errors"] = slice_errors
  return lines, stats


def parse_line(line: str) -> tuple[str, str, str] | None:
  """Returns (timestamp, reporting_pod, culprit_node) or None."""
  parts = line.split("\t")
  if len(parts) < 3:
    return None
  timestamp, pod, payload = parts[0], parts[1], "\t".join(parts[2:])
  match = PEER_RE.search(payload) or EXITING_RE.search(payload)
  if not match:
    return None
  return timestamp, pod, match.group(1)


def _to_epoch(ts: str) -> float | None:
  """Best-effort RFC3339 -> epoch seconds; returns None if unparseable."""
  match = re.search(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?", ts)
  if not match:
    return None
  base = datetime.datetime.strptime(
      match.group(1), "%Y-%m-%dT%H:%M:%S"
  ).replace(tzinfo=datetime.timezone.utc)
  frac = float("0." + match.group(2)) if match.group(2) else 0.0
  return base.timestamp() + frac


def cluster_incidents(
    events: Sequence[dict[str, Any]], window_seconds: int
) -> list[dict[str, Any]]:
  """Collapses same-node events within window_seconds into one incident.

  A single host death is reported independently by every surviving peer within
  roughly the same second, so raw event counts dramatically overstate how many
  times a node actually failed.

  Args:
    events: Parsed events with timestamp, reporting_pod and culprit_node keys.
    window_seconds: Maximum seconds after an incident's first event for a
      later event on the same node to join it.

  Returns:
    The incidents, sorted by first_seen.
  """
  by_node = {}
  for event in events:
    by_node.setdefault(event["culprit_node"], []).append(event)

  incidents = []
  for node, evs in by_node.items():
    evs.sort(key=lambda event: event["timestamp"])
    current = None
    for event in evs:
      epoch = _to_epoch(event["timestamp"])
      # Anchor the window to the incident's FIRST event, not the most recent
      # one. Anchoring to the most recent event chains a continuous drip of
      # retry lines into a single incident spanning hours, which hides genuine
      # repeat failures.
      if (
          current is not None
          and epoch is not None
          and current["_anchor"] is not None
          and (epoch - current["_anchor"]) <= window_seconds
      ):
        current["report_count"] += 1
        current["reporting_pods"].append(event["reporting_pod"])
        current["last_seen"] = event["timestamp"]
        continue
      current = {
          "culprit_node": node,
          "first_seen": event["timestamp"],
          "last_seen": event["timestamp"],
          "report_count": 1,
          "reporting_pods": [event["reporting_pod"]],
          "_anchor": epoch,
      }
      incidents.append(current)

  for incident in incidents:
    incident.pop("_anchor", None)
    incident["reporting_pods"] = sorted(set(incident["reporting_pods"]))
  incidents.sort(key=lambda incident: incident["first_seen"])
  return incidents


def classify_incidents(
    incidents: Sequence[dict[str, Any]], min_reporting_pods: int
) -> list[dict[str, Any]]:
  """Tags each incident primary/cascade by reporting breadth; returns primaries.

  A host that genuinely dies is observed by most of the fleet at once. The
  teardown that follows makes every node briefly report every other node,
  producing narrow incidents reported by only one or two pods. Counting those
  toward host isolation manufactures false repeat offenders -- on a 128-chip
  multi-slice run it turned 3 real failures into 29 and nominated 7 healthy
  nodes for host replacement.

  Args:
    incidents: Incidents from `cluster_incidents`; tagged in place with
      `distinct_reporters` and `classification`.
    min_reporting_pods: Distinct reporting pods needed to count as primary.

  Returns:
    The incidents classified as primary.
  """
  for incident in incidents:
    incident["distinct_reporters"] = len(incident["reporting_pods"])
    incident["classification"] = (
        "primary"
        if incident["distinct_reporters"] >= min_reporting_pods
        else "cascade"
    )
  return [
      incident
      for incident in incidents
      if incident["classification"] == "primary"
  ]


def group_entries(lines: Sequence[str]) -> list[str]:
  """Re-joins continuation lines of multi-line payloads onto their entry.

  Barrier errors put the timed-out task names on the lines AFTER the header,
  so parsing line by line would lose exactly the field that names the culprit.

  Args:
    lines: Non-empty gcloud output lines.

  Returns:
    Entries; each starts with an RFC3339 timestamp line.
  """
  entries = []
  for line in lines:
    if _ENTRY_START_RE.match(line) or not entries:
      entries.append(line)
    else:
      entries[-1] += "\n" + line
  return entries


def run_query_terms(
    project_id: str,
    workload_name: str,
    start: str,
    end: str,
    terms: Sequence[str],
    limit: int,
) -> QueryResult:
  """Runs one OR-query over `terms` for the workload's container logs.

  Output columns: timestamp, pod, node label, payload. The node label is the
  same `compute.googleapis.com/resource_name` that resolve_workload_nodes.py
  reads, so pod -> node mapping needs no extra query.

  Args:
    project_id: GCP project ID.
    workload_name: Pod-name prefix of the workload.
    start: Window start (RFC3339).
    end: Window end (RFC3339).
    terms: Substrings OR-ed over textPayload and jsonPayload.message.
    limit: Entry cap for the read.

  Returns:
    (entries, error or None, truncated).
  """
  clause = " OR ".join(
      f'textPayload:"{term}" OR jsonPayload.message:"{term}"' for term in terms
  )
  log_filter = (
      'resource.type="k8s_container" AND '
      f'resource.labels.pod_name:"{workload_name}-" AND '
      f'timestamp>="{start}" AND timestamp<="{end}" AND ({clause})'
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
      (
          "value(timestamp,resource.labels.pod_name,"
          'labels."compute.googleapis.com/resource_name",'
          "textPayload,jsonPayload.message)"
      ),
  ]
  stdout, error = gcloud_read(command, query_label(terms))
  if error:
    return [], error, False
  entries = group_entries(
      [line for line in stdout.splitlines() if line.strip()]
  )
  if len(entries) >= limit:
    # ASC order: truncation drops the LATEST entries, never the earliest.
    print(
        f"WARNING: terms {terms} returned {len(entries)} entries, hitting "
        f"--limit ({limit}); later entries are TRUNCATED.",
        file=sys.stderr,
    )
  return entries, None, len(entries) >= limit


def query_label(terms: Sequence[str]) -> str:
  """Returns the label of a term group in query_errors / truncated_queries.

  Args:
    terms: The term group; its first term names the group.

  Returns:
    "terms:<first term>".
  """
  first_term, *_ = terms
  return f"terms:{first_term}"


def parse_entry(entry: str) -> LogEntry | None:
  """Splits one grouped entry into its columns.

  Args:
    entry: A tab-separated (timestamp, pod, node, payload) entry.

  Returns:
    The columns keyed by name, or None if the entry has too few columns.
  """
  parts = entry.split("\t", _ENTRY_COLUMNS - 1)
  if len(parts) < _ENTRY_COLUMNS:
    return None
  timestamp, pod, node, payload = parts
  return {
      "timestamp": timestamp.strip(),
      "pod": pod.strip(),
      "node": node.strip(),
      "payload": payload.strip("\t"),
  }


def _task_tokens(text: str) -> list[int]:
  """Task ids from the leading whitespace-separated task names in `text`."""
  tasks = []
  for token in text.split():
    match = TASK_TOKEN_RE.match(token.rstrip(".,;"))
    if not match:
      break
    tasks.append(int(match.group(1)))
  return tasks


def parse_megascale(payload: str) -> dict[str, object] | None:
  """Parses a Megascale digest line.

  Args:
    payload: Log payload.

  Returns:
    {"cause": str | None, "workers": [(worker_id, host)]}, or None when the
    payload has no `Potential culprit workers:` line.
  """
  match = MEGASCALE_CULPRIT_RE.search(payload)
  if not match:
    return None
  text = MEGASCALE_MORE_RE.sub("", match.group(1).strip())
  workers = []
  for token in text.split(","):
    worker_match = MEGASCALE_WORKER_RE.match(token.strip())
    if worker_match:
      workers.append((worker_match.group("worker"), worker_match.group("host")))
  cause = MEGASCALE_CAUSE_RE.search(payload)
  return {
      "cause": cause.group(1).strip() if cause else None,
      "workers": workers,
  }


def parse_non_arriving(payload: str) -> list[dict[str, object]]:
  """Finds tasks that did not reach a barrier or stopped heartbeating.

  Args:
    payload: Log payload.

  Returns:
    One record per barrier timeout / heartbeat signature with keys source,
    barrier_id, reached, total, first_task_at_barrier and tasks.
  """
  records = []
  barrier = BARRIER_TIMEOUT_RE.search(payload)
  if barrier:
    reached = BARRIER_REACHED_RE.search(payload)
    first = BARRIER_FIRST_TASK_RE.search(payload)
    _, names_marker, names_text = payload.partition(BARRIER_TIMED_OUT_NAMES)
    records.append({
        "source": "barrier_timeout",
        "barrier_id": barrier.group(1).rstrip("."),
        "reached": int(reached.group(1)) if reached else None,
        "total": int(reached.group(2)) if reached else None,
        "first_task_at_barrier": int(first.group(1)) if first else None,
        "tasks": _task_tokens(names_text) if names_marker else [],
    })
  heartbeat_tasks = []
  for heartbeat in HEARTBEAT_TASK_RE.finditer(payload):
    heartbeat_tasks.extend(_task_tokens(heartbeat.group(1)))
  unhealthy = UNHEALTHY_RE.search(payload)
  if unhealthy:
    for token in unhealthy.group(1).split():
      heartbeat_tasks.extend(_task_tokens(token))
  if heartbeat_tasks:
    records.append({
        "source": "heartbeat_timeout",
        "barrier_id": None,
        "reached": None,
        "total": None,
        "first_task_at_barrier": None,
        "tasks": sorted(set(heartbeat_tasks)),
    })
  return records


def task_to_pods(
    pods: Iterable[str], hosts_per_slice: int = 0
) -> dict[int, list[str]]:
  """Maps JAX process id -> pod names for JobSet-named pods.

  JAX: process_id = process_id_in_slice + slice_id * processes_per_slice
  (third_party/py/jax/_src/clusters/cloud_tpu_cluster.py get_process_id).
  Assumes JobSet job-index = slice id and pod index = worker id in slice; this
  is a heuristic, so callers label the result as a candidate.

  Args:
    pods: Pod names seen in the logs.
    hosts_per_slice: Pods per JobSet job; 0 infers max(pod index) + 1.

  Returns:
    {task_id: sorted pod names}, or {} when pod names do not share one JobSet
    prefix.
  """
  parsed = []
  for pod in pods:
    if _PATHWAYS_POD_MARKER in pod:
      continue
    match = _JOBSET_POD_RE.match(pod)
    if match:
      parsed.append((
          match.group("prefix"),
          int(match.group("job_index")),
          int(match.group("pod_index")),
          pod,
      ))
  if not parsed or len({prefix for prefix, *_ in parsed}) != 1:
    return {}
  per_slice = hosts_per_slice or (
      max(pod_index for _, _, pod_index, _ in parsed) + 1
  )
  task_pods = {}
  for _, job_index, pod_index, pod in parsed:
    task_pods.setdefault(job_index * per_slice + pod_index, set()).add(pod)
  return {task: sorted(names) for task, names in task_pods.items()}


def _sort_key(timestamp: str) -> tuple[int, float, str]:
  """Orders parseable timestamps chronologically, unparseable ones last."""
  epoch = _to_epoch(timestamp)
  if epoch is None:
    return (1, 0.0, timestamp)
  return (0, epoch, timestamp)


def _excerpt(payload: str) -> str:
  """First payload line, capped for the report."""
  first_line, *_ = payload.splitlines() or [""]
  return first_line[:_PAYLOAD_EXCERPT_CHARS]


def _pod_to_node(entries: Sequence[LogEntry]) -> dict[str, str]:
  """First node label seen per pod."""
  mapping = {}
  for entry in entries:
    if entry["pod"] and entry["node"]:
      mapping.setdefault(entry["pod"], entry["node"])
  return mapping


def _resolve_host_node(host: str, pod_to_node: Mapping[str, str]) -> str | None:
  """Node for a digest hostname that is a pod name or JobSet pod hostname."""
  if host in pod_to_node:
    return pod_to_node[host]
  for pod, node in sorted(pod_to_node.items()):
    # JobSet pod hostname = pod name minus the random suffix.
    if pod.startswith(host + "-"):
      return node
  return None


def _megascale_culprits(
    entries: Sequence[LogEntry], pod_to_node: Mapping[str, str]
) -> list[dict[str, object]]:
  """Aggregates digest culprits per host, pairing each with its cause.

  FormatDigest logs `Potential cause` and `Potential culprit workers` as
  separate LOG lines, so a digest without an inline cause takes the latest
  cause logged by the same pod at or before it.

  Args:
    entries: Parsed entries matching MEGASCALE_TERMS.
    pod_to_node: Pod -> node mapping from the same queries.

  Returns:
    One record per culprit host, sorted by first_seen.
  """
  causes_by_pod = {}
  for entry in sorted(entries, key=lambda item: _sort_key(item["timestamp"])):
    cause = MEGASCALE_CAUSE_RE.search(entry["payload"])
    if cause:
      causes_by_pod.setdefault(entry["pod"], []).append(
          (entry["timestamp"], cause.group(1).strip())
      )
  culprits = {}
  for entry in entries:
    parsed = parse_megascale(entry["payload"])
    if not parsed:
      continue
    cause = parsed["cause"]
    if not cause:
      prior = [
          logged_cause
          for logged_at, logged_cause in causes_by_pod.get(entry["pod"], [])
          if _sort_key(logged_at) <= _sort_key(entry["timestamp"])
      ]
      cause = prior[-1] if prior else None
    for worker, host in parsed["workers"]:
      culprit = culprits.setdefault(
          host,
          {
              "host": host,
              "node": _resolve_host_node(host, pod_to_node),
              "worker_ids": set(),
              "potential_causes": set(),
              "first_seen": entry["timestamp"],
              "digest_count": 0,
          },
      )
      culprit["worker_ids"].add(worker)
      if cause:
        culprit["potential_causes"].add(cause)
      culprit["digest_count"] += 1
      if _sort_key(entry["timestamp"]) < _sort_key(culprit["first_seen"]):
        culprit["first_seen"] = entry["timestamp"]
  records = []
  for culprit in culprits.values():
    culprit["worker_ids"] = sorted(culprit["worker_ids"])
    culprit["potential_causes"] = sorted(culprit["potential_causes"])
    records.append(culprit)
  records.sort(key=lambda record: _sort_key(record["first_seen"]))
  return records


def _non_arriving_tasks(
    entries: Sequence[LogEntry],
    pod_to_node: Mapping[str, str],
    task_pods: Mapping[int, list[str]],
) -> list[dict[str, object]]:
  """Aggregates the tasks named by barrier / heartbeat timeouts.

  The named tasks are the leads, not the pods that reported the timeout.

  Args:
    entries: Parsed entries matching BARRIER_TERMS.
    pod_to_node: Pod -> node mapping from the same queries.
    task_pods: JAX task id -> candidate pods (task_to_pods()).

  Returns:
    One record per (source, barrier_id, task), sorted by first_seen.
  """
  by_key = {}
  for entry in entries:
    for record in parse_non_arriving(entry["payload"]):
      missing = None
      if record["total"] is not None and record["reached"] is not None:
        missing = record["total"] - record["reached"]
      for task in record["tasks"] or [None]:
        aggregate = by_key.setdefault(
            (record["source"], record["barrier_id"], task),
            {
                "task": task,
                "source": record["source"],
                "barrier_id": record["barrier_id"],
                "reached": record["reached"],
                "total": record["total"],
                "missing_count": missing,
                # Only kPendingStragglerLogLimit-1 names are printed.
                "names_truncated": bool(
                    missing is not None and missing > len(record["tasks"])
                ),
                "first_task_at_barrier": record["first_task_at_barrier"],
                "first_seen": entry["timestamp"],
                "reporting_pods": set(),
            },
        )
        aggregate["reporting_pods"].add(entry["pod"])
        if _sort_key(entry["timestamp"]) < _sort_key(aggregate["first_seen"]):
          aggregate["first_seen"] = entry["timestamp"]
  results = []
  for aggregate in sorted(
      by_key.values(), key=lambda item: _sort_key(item["first_seen"])
  ):
    pods = task_pods.get(aggregate["task"], [])
    aggregate["reporting_pods"] = len(aggregate["reporting_pods"])
    aggregate["candidate_pods"] = pods
    aggregate["candidate_nodes"] = sorted(
        {pod_to_node[pod] for pod in pods if pod in pod_to_node}
    )
    results.append(aggregate)
  return results


def _earliest_error_candidates(
    entries: Sequence[LogEntry], pod_to_node: Mapping[str, str], top_n: int
) -> list[dict[str, object]]:
  """Ranks pods by their earliest crash line; timeout lines never rank.

  Args:
    entries: Parsed entries matching EARLY_ERROR_TERMS.
    pod_to_node: Pod -> node mapping from the same queries.
    top_n: Number of candidates to return.

  Returns:
    Up to top_n records (rank, pod, node, first_error, excerpt,
    error_entries), earliest first.
  """
  first_by_pod = {}
  for entry in entries:
    if any(marker in entry["payload"] for marker in _VICTIM_MARKERS):
      continue
    candidate = first_by_pod.setdefault(
        entry["pod"],
        {
            "pod": entry["pod"],
            "node": pod_to_node.get(entry["pod"]),
            "first_error": entry["timestamp"],
            "excerpt": _excerpt(entry["payload"]),
            "error_entries": 0,
        },
    )
    candidate["error_entries"] += 1
    if _sort_key(entry["timestamp"]) < _sort_key(candidate["first_error"]):
      candidate["first_error"] = entry["timestamp"]
      candidate["excerpt"] = _excerpt(entry["payload"])
  ranked = sorted(
      first_by_pod.values(), key=lambda item: _sort_key(item["first_error"])
  )
  return [
      dict(candidate, rank=rank)
      for rank, candidate in enumerate(ranked[:top_n], start=1)
  ]


def analyze_non_pathways(
    megascale_entries: Sequence[LogEntry],
    barrier_entries: Sequence[LogEntry],
    error_entries: Sequence[LogEntry],
    hosts_per_slice: int = 0,
    top_n: int = 5,
) -> dict[str, object]:
  """Builds the non-Pathways report keys from parsed entries (no I/O).

  Args:
    megascale_entries: Entries matching MEGASCALE_TERMS.
    barrier_entries: Entries matching BARRIER_TERMS.
    error_entries: Entries matching EARLY_ERROR_TERMS.
    hosts_per_slice: Passed to task_to_pods(); 0 infers it.
    top_n: Number of earliest-error candidates.

  Returns:
    Dict with megascale_culprits, non_arriving_tasks,
    earliest_error_candidates and pod_to_node.
  """
  all_entries = [*megascale_entries, *barrier_entries, *error_entries]
  pod_to_node = _pod_to_node(all_entries)
  all_pods = {entry["pod"] for entry in all_entries}
  return {
      "megascale_culprits": _megascale_culprits(megascale_entries, pod_to_node),
      "non_arriving_tasks": _non_arriving_tasks(
          barrier_entries,
          pod_to_node,
          task_to_pods(all_pods, hosts_per_slice),
      ),
      "earliest_error_candidates": _earliest_error_candidates(
          error_entries, pod_to_node, top_n
      ),
      "pod_to_node": pod_to_node,
  }


def query_status(ok: int, failed: int, truncated: bool) -> str:
  """Returns the run's overall query status.

  Args:
    ok: Number of queries that succeeded.
    failed: Number of queries that failed after retries.
    truncated: Whether any query hit --limit.

  Returns:
    "NO_DATA" if every query failed, "INCOMPLETE" if any failed or was
    truncated, else "COMPLETE".
  """
  if failed and not ok:
    return "NO_DATA"
  if failed or truncated:
    return "INCOMPLETE"
  return "COMPLETE"


def signature_family(
    pathways_events: Sequence[object], extra: Mapping[str, object]
) -> str:
  """Returns the primary signature family that named a culprit.

  Args:
    pathways_events: Pathways peer/distributor events.
    extra: Output of analyze_non_pathways().

  Returns:
    "pathways", "megascale", "barrier" or "none", in that priority.
  """
  if pathways_events:
    return "pathways"
  if extra["megascale_culprits"]:
    return "megascale"
  if extra["non_arriving_tasks"]:
    return "barrier"
  return "none"


@dataclasses.dataclass
class QueryOutcomes:
  """Every query outcome, so a failed read can never pass as "no culprit".

  Attributes:
    ok: Queries that succeeded.
    failed: Queries that failed after retries.
    errors: The failed queries' errors.
    truncated: Labels of queries that hit --limit.
  """

  ok: int = 0
  failed: int = 0
  errors: list[QueryError] = dataclasses.field(default_factory=list)
  truncated: list[str] = dataclasses.field(default_factory=list)

  def record(self, label: str, result: QueryResult) -> list[str]:
    """Records one query result.

    Args:
      label: The query's label.
      result: (items, error, truncated) from run_query / run_query_terms.

    Returns:
      The query's items (empty if it failed).
    """
    items, error, truncated = result
    if error:
      self.failed += 1
      self.errors.append(error)
    else:
      self.ok += 1
    if truncated:
      self.truncated.append(label)
    return items

  @property
  def status(self) -> str:
    """The overall status; see query_status()."""
    return query_status(self.ok, self.failed, bool(self.truncated))


def fetch_entries(
    outcomes: QueryOutcomes, args: argparse.Namespace, terms: Sequence[str]
) -> list[LogEntry]:
  """Runs one term-group query and records its outcome.

  Args:
    outcomes: Accumulator the query outcome is recorded in.
    args: Parsed command-line flags.
    terms: The term group to query.

  Returns:
    The parsed entries.
  """
  raw_entries = outcomes.record(
      query_label(terms),
      run_query_terms(
          args.project_id,
          args.workload_name,
          args.start,
          args.end,
          terms,
          args.limit,
      ),
  )
  parsed_entries = []
  for raw in raw_entries:
    parsed = parse_entry(raw)
    if parsed:
      parsed_entries.append(parsed)
  return parsed_entries


def filter_role_confound_offenders(
    repeat_offenders: Sequence[str],
    pod_to_node: Mapping[str, str],
    error_entries: Sequence[LogEntry] = (),
) -> tuple[list[str], list[str]]:
  """Separates repeat offenders confounded by index-0 role or SIGILL(si_code=0).

  In elastic resiliency / snapshot workloads, fault injection or
  slice-leader bugs repeatedly kill `-worker-job0-<slice>-0-<hash>` with
  user-space `SIGILL (si_code=0, SI_USER)`. Because pod index 0 is often placed
  on the same node across restarts or killed multiple times in place, a healthy
  host can accumulate >=2 primary incidents solely due to its logical index-0
  role. When all known pods/crashes on a repeat-offender node are index-0 slice
  leaders or `SIGILL (si_code=0)` kills, hold it back from `isolate_recommended`
  and surface it in `role_confound_candidates` instead.

  Args:
    repeat_offenders: Nodes with >=2 primary incidents.
    pod_to_node: Pod -> node mapping observed in logs.
    error_entries: Parsed crash entries from EARLY_ERROR_TERMS.

  Returns:
    (isolate_recommended, role_confound_candidates).
  """
  pods_by_node: dict[str, set[str]] = {}
  for pod, node in pod_to_node.items():
    if pod and node:
      pods_by_node.setdefault(node, set()).add(pod)

  entries_by_node: dict[str, list[LogEntry]] = {}
  for entry in error_entries:
    node = entry.get("node") or pod_to_node.get(entry.get("pod", ""), "")
    if node:
      entries_by_node.setdefault(node, []).append(entry)
    if entry.get("pod") and node:
      pods_by_node.setdefault(node, set()).add(entry["pod"])

  isolate_recommended: list[str] = []
  role_confound_candidates: list[str] = []

  for node in repeat_offenders:
    node_pods = pods_by_node.get(node, set())
    node_entries = entries_by_node.get(node, [])

    only_index_0_pods = bool(node_pods) and all(
        bool(_INDEX_0_LEADER_POD_RE.search(pod)) for pod in node_pods
    )
    only_sigill_user_or_index_0_entries = bool(node_entries) and all(
        bool(_SIGILL_SI_USER_RE.search(entry.get("payload", "")))
        or bool(_INDEX_0_LEADER_POD_RE.search(entry.get("pod", "")))
        for entry in node_entries
    )

    if only_index_0_pods or only_sigill_user_or_index_0_entries:
      role_confound_candidates.append(node)
    else:
      isolate_recommended.append(node)

  return isolate_recommended, role_confound_candidates


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Extract culprit hosts from Pathways, Megascale and JAX barrier logs."
      )
  )
  parser.add_argument("--workload_name", required=True)
  parser.add_argument("--project_id", required=True)
  parser.add_argument("--start", required=True)
  parser.add_argument("--end", required=True)
  parser.add_argument("--output_file", default="scratch/culprit_hosts.json")
  parser.add_argument("--limit", type=int, default=5000)
  parser.add_argument(
      "--cluster_seconds",
      type=int,
      default=120,
      help="Same-node events within this window count as ONE incident.",
  )
  parser.add_argument(
      "--min_reporting_pods",
      type=int,
      default=3,
      help=(
          "Distinct reporting pods required to classify an incident as primary."
          " When a host genuinely dies most of the fleet reports it; a teardown"
          " cascade produces incidents reported by only 1-2 pods."
      ),
  )
  parser.add_argument(
      "--cluster_name", default="", help="Optional exact cluster scope."
  )
  parser.add_argument(
      "--namespace", default="", help="Optional exact namespace scope."
  )
  parser.add_argument(
      "--slice_minutes",
      type=int,
      default=0,
      help="Slice width for concurrent queries; 0 = auto (360 if > 1 day).",
  )
  parser.add_argument("--max_workers", type=int, default=8)
  parser.add_argument(
      "--hosts_per_slice",
      type=int,
      default=0,
      help=(
          "Pods per JobSet job (TPU hosts per slice) for task -> pod mapping;"
          " 0 infers it from the highest pod index seen."
      ),
  )
  parser.add_argument("--top_candidates", type=int, default=5)
  args = parser.parse_args()

  # Every query outcome is recorded: a failed or truncated query must never
  # read as "no culprit found" (playbooks/rca-invariants.md, NO_DATA rule).
  outcomes = QueryOutcomes()
  slice_minutes = args.slice_minutes or auto_slice_minutes(args.start, args.end)
  term_stats = {}
  term_lines = {}
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
    base_filter = " AND ".join(
        ['resource.type="k8s_container"', selector]
        + log_query.scope_clauses(args.cluster_name, args.namespace)
    )
    for term in SEARCH_TERMS:
      term_lines[term], term_stats[term] = query_term(
          args.project_id,
          base_filter,
          args.start,
          args.end,
          term,
          args.limit,
          slice_minutes,
          args.max_workers,
      )
      slice_errors = term_stats[term].pop("errors", None)
      if slice_errors:
        for err in slice_errors:
          outcomes.record(
              f"pathways:{term}",
              ([], err, term_stats[term]["truncated"]),
          )
      else:
        outcomes.record(
            f"pathways:{term}",
            (term_lines[term], None, term_stats[term]["truncated"]),
        )
  except log_query.LoggingQueryError as exc:
    log_query.fail(exc, "extract-culprit-hosts")

  events = []
  seen = set()
  for term in SEARCH_TERMS:
    for line in term_lines[term]:
      parsed = parse_line(line)
      if not parsed:
        continue
      timestamp, pod, node = parsed
      key = (timestamp, pod, node)
      if key in seen:
        continue
      seen.add(key)
      events.append(
          {"timestamp": timestamp, "reporting_pod": pod, "culprit_node": node}
      )

  events.sort(key=lambda event: event["timestamp"])
  incidents = cluster_incidents(events, args.cluster_seconds)

  # Host isolation is decided on distinct INCIDENTS, never on raw event count.
  # When a host dies, every surviving peer reports losing it within the same
  # second, so raw counts inflate one failure into a dozen and manufacture false
  # repeat offenders.
  primary = classify_incidents(incidents, args.min_reporting_pods)

  offences = collections.Counter(
      incident["culprit_node"] for incident in primary
  )
  repeat_offenders = sorted(
      node for node, count in offences.items() if count > 1
  )

  # Non-Pathways families. Always run: a Pathways job can also hit a barrier,
  # and the earliest-error ranking is the fallback when nothing names a host.
  megascale_entries = fetch_entries(outcomes, args, MEGASCALE_TERMS)
  barrier_entries = fetch_entries(outcomes, args, BARRIER_TERMS)
  error_entries = fetch_entries(outcomes, args, EARLY_ERROR_TERMS)
  extra = analyze_non_pathways(
      megascale_entries,
      barrier_entries,
      error_entries,
      hosts_per_slice=args.hosts_per_slice,
      top_n=args.top_candidates,
  )
  isolate_recommended, role_confound_candidates = (
      filter_role_confound_offenders(
          repeat_offenders,
          extra["pod_to_node"],
          error_entries,
      )
  )

  status = outcomes.status
  family = signature_family(events, extra)
  if family == "none" and status != "COMPLETE":
    family = "unknown"

  report = {
      "workload_name": args.workload_name,
      "selector_mode": selector_mode,
      "slice_minutes": slice_minutes,
      "truncated": (
          any(st["truncated"] for st in term_stats.values())
          or bool(outcomes.truncated)
      ),
      "term_stats": term_stats,
      "status": status,
      "signature_family": family,
      "query_errors": outcomes.errors,
      "truncated_queries": outcomes.truncated,
      "total_events": len(events),
      "total_incidents": len(incidents),
      "primary_incidents": len(primary),
      "cascade_incidents": len(incidents) - len(primary),
      "distinct_culprit_nodes": len(offences),
      "primary_incidents_by_node": dict(offences.most_common()),
      "primary_timeline": [
          {
              "first_seen": incident["first_seen"],
              "culprit_node": incident["culprit_node"],
              "distinct_reporters": incident["distinct_reporters"],
          }
          for incident in primary
      ],
      "repeat_offenders": repeat_offenders,
      "role_confound_candidates": role_confound_candidates,
      "isolate_recommended": isolate_recommended,
      "megascale_culprits": extra["megascale_culprits"],
      "non_arriving_tasks": extra["non_arriving_tasks"],
      "earliest_error_candidates": extra["earliest_error_candidates"],
      "pod_to_node": extra["pod_to_node"],
      "incidents": incidents,
      "events": events,
  }
  if role_confound_candidates:
    report["role_confound_warning"] = (
        f"Repeat incidents on {role_confound_candidates} occurred exclusively"
        " on index-0 slice leader pods (-worker-job0-*-0-) and/or user-space"
        " SIGILL (si_code=0, SI_USER) kills. Verify role-vs-host confound"
        " before isolating/replacing hosts."
    )

  with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

  print(
      json.dumps(
          {
              key: value
              for key, value in report.items()
              if key not in _FULL_REPORT_ONLY_KEYS
          },
          indent=2,
      )
  )
  print(f"\nFull event list written to {args.output_file}", file=sys.stderr)
  if isolate_recommended:
    print(
        "RECOMMEND HOST REPLACEMENT (>=2 distinct failures of the same"
        f" instance): {isolate_recommended}",
        file=sys.stderr,
    )
  if role_confound_candidates:
    print(
        "WARNING: Role-vs-host confound detected on repeat-offender node(s) "
        f"{role_confound_candidates} (index-0 slice leader / user-space SIGILL "
        "si_code=0); excluded from isolate_recommended.",
        file=sys.stderr,
    )
  if status != "COMPLETE":
    print(
        f"STATUS {status}: {outcomes.failed} quer(y/ies) failed,"
        f" {len(outcomes.truncated)} truncated. An empty result here is NOT"
        " evidence that no host is to blame; see query_errors.",
        file=sys.stderr,
    )
  if family == "barrier":
    print(
        "Barrier/heartbeat timeouts are victims: debug the non_arriving_tasks"
        " (and earliest_error_candidates) BEFORE the timeout, not the pods"
        " that reported it.",
        file=sys.stderr,
    )


if __name__ == "__main__":
  main()

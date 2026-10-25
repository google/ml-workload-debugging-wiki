#!/usr/bin/env python3
"""Queries ML Diagnostics Workload Monitoring events and maps analyzer findings.

Discovers matching `machineLearningRun` resources (`<job-name>-<timestamp>`) on
GKE (`>= 1.36.0-gke.4447000`) via `gcloud alpha mldiagnostics` (with REST API
fallback to `https://hypercomputecluster.googleapis.com/v1alpha`), retrieves
`monitoredEvents` and `analyzerReports`, extracts affected instances, and
deterministically maps `DETECTED` analyzer reports across all 12 Infrastructure,
Orchestrator, and Framework analyzers (and 11 Megascale XLA hang causes) to wiki
symptoms, workflows, and actions.
"""

import argparse
from collections.abc import Sequence
import json
import os
import re
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

EXIT_ACCESS_DENIED = 77

_ACCESS_DENIED_MARKERS = (
    "PERMISSION_DENIED",
    "ACCESS_DENIED",
    "403 FORBIDDEN",
    "HTTP 403",
    "INVALID_GRANT",
    "REAUTH",
    "HTTPS://CLOUD.GOOGLE.COM/IAM",
    "ACCESS APPROVAL",
    "CALLER DOES NOT HAVE PERMISSION",
    "DOES NOT HAVE PERMISSION",
    "INSUFFICIENT AUTHENTICATION SCOPES",
)

_MXLA_CAUSE_PATTERNS = (
    "BAD_TPU_CHIP",
    "BAD_SC_CHIP",
    "NETWORKING_ISSUE",
    "DIFFERENT_MODULE",
    "FINGERPRINT_MISMATCH",
    "DATA_INPUT_STALL",
    "ICI_ERROR",
    "PROGRAM_NOT_QUEUED",
    "LAUNCHES_ORDER_INCONSISTENT",
    "UNRECOVERABLE_ERROR",
    "UNKNOWN",
)


class AccessDeniedError(RuntimeError):
  """Raised when GCP IAM or authentication denies access to ML Diagnostics API."""


class CommandExecutionError(RuntimeError):
  """Raised when both CLI and REST API calls fail with a non-permission error."""


def is_access_denied_error(text: str) -> bool:
  """Returns True if `text` contains IAM or authentication access-denial markers."""
  if not text:
    return False
  upper_text = text.upper()
  return any(marker in upper_text for marker in _ACCESS_DENIED_MARKERS)


def _append_unique(target: list[str], items: Sequence[str]) -> None:
  """Appends items to `target` preserving order without duplicates."""
  for item in items:
    if item and item not in target:
      target.append(item)


ANALYZER_WIKI_MAP: dict[str, dict[str, list[str]]] = {
    "slice interruption": {
        "symptoms": [
            "tpu-provisioner-slice-recreation",
            "tpu7x-cube-unhealthy",
        ],
        "workflows": [
            "tpu7x-cube-health-triage",
            "workload-disruption-orchestration",
        ],
        "actions": [],
    },
    "nodepool interruption": {
        "symptoms": [
            "tpu-provisioner-slice-recreation",
            "false-preemption-sigterm",
            "data-loss-pipe-closed",
        ],
        "workflows": ["workload-disruption-orchestration"],
        "actions": [],
    },
    "node pool interruption": {
        "symptoms": [
            "tpu-provisioner-slice-recreation",
            "false-preemption-sigterm",
            "data-loss-pipe-closed",
        ],
        "workflows": ["workload-disruption-orchestration"],
        "actions": [],
    },
    "ici link": {
        "symptoms": ["slow-step-time", "grpc-error-start-session-failed"],
        "workflows": ["ici-straggler-triage"],
        "actions": [],
    },
    "thermal": {
        "symptoms": ["slow-step-time"],
        "workflows": ["repair-in-place", "node-isolation-and-rr-api"],
        "actions": [],
    },
    "throttling": {
        "symptoms": ["slow-step-time"],
        "workflows": ["repair-in-place", "node-isolation-and-rr-api"],
        "actions": [],
    },
    "hbm bandwidth": {
        "symptoms": ["slow-step-time"],
        "workflows": ["repair-in-place", "node-isolation-and-rr-api"],
        "actions": [],
    },
    "host memory": {
        "symptoms": ["tpu-coordinator-oom", "slow-step-time"],
        "workflows": ["workload-disruption-orchestration"],
        "actions": [],
    },
    "hbm capacity": {
        "symptoms": ["slow-step-time"],
        "workflows": ["libtpu-timing-outlier-analysis"],
        "actions": [],
    },
    "cpu utilization": {
        "symptoms": ["slow-step-time"],
        "workflows": ["libtpu-timing-outlier-analysis"],
        "actions": [],
    },
    "tor nic": {
        "symptoms": ["slow-step-time"],
        "workflows": ["libtpu-timing-outlier-analysis"],
        "actions": [],
    },
    "tor uplink congestion": {
        "symptoms": ["megascale-hang", "slow-step-time"],
        "workflows": ["debugging-megascale-hangs"],
        "actions": ["query-node-health-events"],
    },
    "pygrain data": {
        "symptoms": ["slow-step-time", "step-timebomb-watchdog"],
        "workflows": ["libtpu-timing-outlier-analysis"],
        "actions": ["analyze-libtpu-timing", "query-workload-logs"],
    },
    "async checkpointing": {
        "symptoms": ["incomplete-checkpoint-restore", "step-timebomb-watchdog"],
        "workflows": ["workload-disruption-orchestration"],
        "actions": ["query-workload-logs"],
    },
}


def _normalize_analyzer_key(text: str) -> str:
  """Normalizes analyzer names by lowercasing and stripping separators."""
  return re.sub(r"[\s_-]+", "", (text or "").lower())


def extract_affected_gce_instance_ids(
    details: str,
    report: dict[str, Any] | None = None,
) -> list[str]:
  """Extracts numeric GCE Instance IDs from an analyzer report."""
  extracted: list[str] = []
  if isinstance(report, dict):
    for key in ("instanceIds", "instance_ids", "gceInstanceIds"):
      val = report.get(key)
      if isinstance(val, list):
        for item in val:
          cleaned = str(item).strip()
          if cleaned.isdigit():
            _append_unique(extracted, [cleaned])
  if details:
    matches = re.findall(
        r"\bInstance\s+Id\s*:?\s*(\d+)\b", details, flags=re.IGNORECASE
    )
    _append_unique(extracted, matches)
  return extracted


def extract_affected_instances(
    details: str,
    report: dict[str, Any] | None = None,
) -> list[str]:
  """Extracts affected node/instance/slice/pod/nodepool identifiers.

  Args:
    details: The human-readable `details` string from an `analyzerReport`.
    report: Optional raw `analyzerReport` dict containing structured fields.

  Returns:
    Ordered, deduplicated list of extracted instance, node, pod, nodepool,
    slice, or sub-block identifiers.
  """
  extracted: list[str] = []
  if isinstance(report, dict):
    for key in (
        "affectedInstances",
        "affected_instances",
        "instances",
        "nodes",
        "affectedNodes",
        "affected_nodes",
    ):
      val = report.get(key)
      if isinstance(val, list):
        for item in val:
          if isinstance(item, str) and item.strip():
            _append_unique(extracted, [item.strip()])
      elif isinstance(val, str) and val.strip():
        _append_unique(extracted, [val.strip()])

  if not details:
    return extracted

  # 1. Standard GKE TPU node names (e.g. gke-tpu-..., gk3-...)
  gke_nodes = re.findall(r"\b(?:gke|gk3)-[a-zA-Z0-9-]+", details)
  _append_unique(extracted, gke_nodes)

  # 2. Pod lists in "[Pods: pod-a, pod-b]" format (before stripping pod suffix)
  pod_lists = re.findall(
      r"\[\s*Pods?\s*:\s*([^\]]+)\]", details, flags=re.IGNORECASE
  )
  for raw_pods in pod_lists:
    for token in re.split(r"[,\s]+", raw_pods):
      cleaned = token.strip().strip("'\"`.;:()")
      if cleaned:
        _append_unique(extracted, [cleaned])

  # 3. Interrupted Nodepool name ("Nodepool <name> is interrupted")
  nodepool_matches = re.findall(
      r"\bNodepool\s+([a-zA-Z0-9-]+)\s+is\s+interrupted\b",
      details,
      flags=re.IGNORECASE,
  )
  _append_unique(extracted, nodepool_matches)

  # 4. Slice / JobSet replicated job or pod IDs (e.g. <jobset>-job-0-1, slice-0)
  slice_matches = re.findall(
      r"\b(?:[a-zA-Z0-9-]+-job-\d+(?:-\d+)?(?:-[a-zA-Z0-9]+)?|slice-\d+|sub-block-[a-zA-Z0-9-]+)\b",
      details,
  )
  _append_unique(extracted, slice_matches)

  # 5. Bracketed/colon-delimited lists after "instances", "nodes", "sub-block"
  bracket_lists = re.findall(
      r"(?:instances?|nodes?|sub-?blocks?)\s*(?::|are|is|in)?\s*\[([^\]]+)\]",
      details,
      flags=re.IGNORECASE,
  )
  for raw_list in bracket_lists:
    for token in re.split(r"[,\s]+", raw_list):
      cleaned = token.strip().strip("'\"`.;:()")
      if cleaned:
        _append_unique(extracted, [cleaned])

  colon_lists = re.findall(
      r"(?:on\s+instances?|affected\s+instances?|affected\s+nodes?|instances?\s+experienced[^:]*|instances?\s+with[^:]*):\s*([a-zA-Z0-9_.,\s-]+?)(?:\s+in\s+|\.|$|\n)",
      details,
      flags=re.IGNORECASE,
  )
  stopwords = (
      "and",
      "the",
      "with",
      "for",
      "from",
      "slice",
      "slices",
      "node",
      "nodes",
      "instance",
      "instances",
      "pods",
      "pod",
  )
  for raw_list in colon_lists:
    for token in re.split(r"[,\s]+", raw_list):
      cleaned = token.strip().strip("'\"`.;:()")
      if cleaned and len(cleaned) >= 3 and cleaned.lower() not in stopwords:
        _append_unique(extracted, [cleaned])

  return extracted


def map_analyzer_report_to_wiki(
    analyzer_name: str,
    details: str,
    event_type: str = "",
) -> dict[str, list[str]]:
  """Maps a single DETECTED analyzer report to wiki symptoms, workflows, actions.

  Args:
    analyzer_name: Name of the analyzer (e.g. 'Megascale XLA Hang Analyzer' or
      'NodepoolInterruptionAnalyzer').
    details: Details string from the analyzer report.
    event_type: Optional parent MonitoredEvent `type` (e.g. 'HANG_DETECTED').

  Returns:
    Dictionary with keys `matched_symptoms`, `recommended_workflows`,
    `recommended_actions`, and `mxla_hang_causes`.
  """
  norm_analyzer = (analyzer_name or "").strip().lower()
  compact_analyzer = _normalize_analyzer_key(analyzer_name)
  upper_details = (details or "").upper()
  upper_event = (event_type or "").upper()

  symptoms: list[str] = []
  workflows: list[str] = []
  actions: list[str] = []
  mxla_causes: list[str] = []

  # 1. Framework Layer: Megascale XLA Hang Analyzer
  if "megascale" in compact_analyzer or (
      "hang" in compact_analyzer and "xla" in compact_analyzer
  ):
    _append_unique(symptoms, ["megascale-hang"])
    for cause in _MXLA_CAUSE_PATTERNS:
      if cause in upper_details:
        _append_unique(mxla_causes, [cause])

    if "BAD_TPU_CHIP" in mxla_causes or "UNRECOVERABLE_ERROR" in mxla_causes:
      _append_unique(
          symptoms,
          [
              "megascale-hang",
              "mxla-hardware-fault",
              "worker-hardware-fatal-error",
          ],
      )
      _append_unique(
          workflows, ["mxla-hardware-triage", "node-isolation-and-rr-api"]
      )

    if "BAD_SC_CHIP" in mxla_causes:
      _append_unique(symptoms, ["wait-timeout-on-sflag", "megascale-hang"])
      _append_unique(workflows, ["tpu-state-analysis"])
      _append_unique(actions, ["diagnose-sflag-hang"])

    if "ICI_ERROR" in mxla_causes:
      _append_unique(
          symptoms, ["megascale-hang", "grpc-error-start-session-failed"]
      )
      _append_unique(workflows, ["ici-straggler-triage"])

    if "DATA_INPUT_STALL" in mxla_causes:
      _append_unique(
          symptoms,
          [
              "megascale-hang",
              "gcs-429-503-data-loading",
              "stopiteration-disk-full",
          ],
      )
      _append_unique(actions, ["ai-check-data-input-stalls"])

    if any(
        c in mxla_causes
        for c in (
            "DIFFERENT_MODULE",
            "FINGERPRINT_MISMATCH",
            "LAUNCHES_ORDER_INCONSISTENT",
        )
    ):
      _append_unique(symptoms, ["megascale-hang"])
      _append_unique(
          actions, ["check-program-consistency", "inspect-hlo-dumps"]
      )

    if "PROGRAM_NOT_QUEUED" in mxla_causes:
      _append_unique(symptoms, ["step-timebomb-watchdog", "megascale-hang"])
      _append_unique(actions, ["ai-check-blocking-user-logic"])

    if (
        "NETWORKING_ISSUE" in mxla_causes
        or "UNKNOWN" in mxla_causes
        or not mxla_causes
    ):
      _append_unique(symptoms, ["megascale-hang", "deadline-exceeded"])
      _append_unique(workflows, ["debugging-megascale-hangs"])

  else:
    matched_entry = False
    for key, mapping in ANALYZER_WIKI_MAP.items():
      compact_key = _normalize_analyzer_key(key)
      if key in norm_analyzer or (
          compact_key and compact_key in compact_analyzer
      ):
        _append_unique(symptoms, mapping.get("symptoms", []))
        _append_unique(workflows, mapping.get("workflows", []))
        _append_unique(actions, mapping.get("actions", []))
        matched_entry = True
        break

    if not matched_entry:
      if "HANG" in upper_event:
        _append_unique(symptoms, ["megascale-hang"])
        _append_unique(workflows, ["debugging-megascale-hangs"])
      elif "PERFORMANCE_DEGRADATION" in upper_event:
        _append_unique(symptoms, ["slow-step-time"])
        _append_unique(workflows, ["libtpu-timing-outlier-analysis"])

  return {
      "matched_symptoms": symptoms,
      "recommended_workflows": workflows,
      "recommended_actions": actions,
      "mxla_hang_causes": mxla_causes,
  }


def parse_and_match_analyzer_reports(
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
  """Parses MonitoredEvent objects and maps all DETECTED analyzer reports.

  Args:
    events: Sequence of MonitoredEvent dicts (each containing `analyzerReports`
      or `analyzer_reports`).

  Returns:
    Aggregated dictionary containing `detected_analyzers`,
    `not_detected_analyzers`, `matched_symptoms`, `recommended_workflows`,
    `recommended_actions`, `mxla_hang_causes`, `affected_instances`,
    `affected_gce_instance_ids`, and `detected_reports`.
  """
  detected_analyzers: list[str] = []
  not_detected_analyzers: list[str] = []
  matched_symptoms: list[str] = []
  recommended_workflows: list[str] = []
  recommended_actions: list[str] = []
  mxla_hang_causes: list[str] = []
  affected_instances: list[str] = []
  affected_gce_instance_ids: list[str] = []
  detected_reports: list[dict[str, Any]] = []

  for event in events:
    if not isinstance(event, dict):
      continue
    event_name = str(event.get("name", ""))
    event_type = str(event.get("type", event.get("eventType", "")))
    reports = event.get("analyzerReports")
    if not isinstance(reports, list):
      reports = event.get("analyzer_reports", [])
    if not isinstance(reports, list):
      continue

    for report in reports:
      if not isinstance(report, dict):
        continue
      state = str(
          report.get("detectionState", report.get("detection_state", ""))
      ).strip()
      analyzer = str(report.get("analyzer", "")).strip()
      upper_state = state.upper()
      if upper_state == "NOT_DETECTED":
        if analyzer:
          _append_unique(not_detected_analyzers, [analyzer])
        continue
      if upper_state != "DETECTED":
        continue

      details = str(report.get("details", "")).strip()
      rec_actions = report.get(
          "recommendedActions", report.get("recommended_actions", [])
      )
      instances = extract_affected_instances(details, report)
      gce_ids = extract_affected_gce_instance_ids(details, report)
      wiki_mapping = map_analyzer_report_to_wiki(analyzer, details, event_type)

      _append_unique(detected_analyzers, [analyzer])
      _append_unique(matched_symptoms, wiki_mapping["matched_symptoms"])
      _append_unique(
          recommended_workflows, wiki_mapping["recommended_workflows"]
      )
      _append_unique(recommended_actions, wiki_mapping["recommended_actions"])
      _append_unique(mxla_hang_causes, wiki_mapping["mxla_hang_causes"])
      _append_unique(affected_instances, instances)
      _append_unique(affected_gce_instance_ids, gce_ids)

      detected_reports.append({
          "event_name": event_name,
          "event_type": event_type,
          "analyzer": analyzer,
          "detection_state": state,
          "details": details,
          "recommended_actions_text": rec_actions,
          "affected_instances": instances,
          "affected_gce_instance_ids": gce_ids,
          "matched_symptoms": wiki_mapping["matched_symptoms"],
          "recommended_workflows": wiki_mapping["recommended_workflows"],
          "recommended_actions": wiki_mapping["recommended_actions"],
          "mxla_hang_causes": wiki_mapping["mxla_hang_causes"],
      })

  return {
      "detected_analyzers": detected_analyzers,
      "not_detected_analyzers": not_detected_analyzers,
      "matched_symptoms": matched_symptoms,
      "recommended_workflows": recommended_workflows,
      "recommended_actions": recommended_actions,
      "mxla_hang_causes": mxla_hang_causes,
      "affected_instances": affected_instances,
      "affected_gce_instance_ids": affected_gce_instance_ids,
      "detected_reports": detected_reports,
  }


def _run_gcloud_json(args: Sequence[str], timeout: int = 45) -> Any:
  """Executes a `gcloud` command and returns parsed JSON."""
  cmd = ["gcloud"] + list(args) + ["--quiet", "--format=json"]
  proc = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
  )
  if proc.returncode != 0:
    combined = f"{proc.stdout}\n{proc.stderr}"
    if is_access_denied_error(combined):
      raise AccessDeniedError(proc.stderr.strip() or proc.stdout.strip())
    raise CommandExecutionError(
        proc.stderr.strip() or f"gcloud exited with code {proc.returncode}"
    )
  raw = proc.stdout.strip()
  if not raw:
    return []
  return json.loads(raw)


def _get_access_token() -> str:
  """Obtains bearer token via auth/authorization_token_file or gcloud auth."""
  cfg_proc = subprocess.run(
      [
          "gcloud",
          "config",
          "get-value",
          "auth/authorization_token_file",
          "--quiet",
      ],
      capture_output=True,
      text=True,
      timeout=10,
      check=False,
  )
  if cfg_proc.returncode == 0:
    token_file = cfg_proc.stdout.strip()
    if token_file and token_file != "(unset)" and os.path.exists(token_file):
      try:
        with open(token_file, "r", encoding="utf-8") as f:
          token_from_file = f.read().strip()
        if token_from_file:
          return token_from_file
      except OSError:
        pass

  token_proc = subprocess.run(
      ["gcloud", "auth", "print-access-token", "--quiet"],
      capture_output=True,
      text=True,
      timeout=15,
      check=False,
  )
  if token_proc.returncode != 0 or not token_proc.stdout.strip():
    stderr_msg = token_proc.stderr.strip()
    if is_access_denied_error(stderr_msg):
      raise AccessDeniedError(stderr_msg)
    raise CommandExecutionError(
        f"Failed to obtain gcloud access token: {stderr_msg}"
    )
  return token_proc.stdout.strip()


def _fetch_rest_json(url: str, timeout: int = 30) -> Any:
  """Fetches JSON from `hypercomputecluster.googleapis.com` using bearer token."""
  token = _get_access_token()
  req = urllib.request.Request(
      url,
      headers={
          "Authorization": f"Bearer {token}",
          "Accept": "application/json",
      },
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      body = resp.read().decode("utf-8", errors="replace").strip()
      return json.loads(body) if body else {}
  except urllib.error.HTTPError as e:
    err_body = ""
    try:
      err_body = e.read().decode("utf-8", errors="replace")
    except OSError:
      pass
    msg = f"HTTP {e.code}: {err_body or e.reason}"
    if e.code in (401, 403) or is_access_denied_error(msg):
      raise AccessDeniedError(msg) from e
    raise CommandExecutionError(msg) from e
  except urllib.error.URLError as e:
    raise CommandExecutionError(f"REST request failed: {e}") from e


def _extract_items_list(payload: Any, list_keys: Sequence[str]) -> list[Any]:
  """Extracts a list of resource objects from either CLI list or REST dict."""
  if isinstance(payload, list):
    return payload
  if isinstance(payload, dict):
    for key in list_keys:
      val = payload.get(key)
      if isinstance(val, list):
        return val
  return []


def _matches_workload_id(run_obj: dict[str, Any], workload_id: str) -> bool:
  """Returns True if `run_obj` matches `workload_id`."""
  if not workload_id:
    return True
  wl_lower = workload_id.lower()
  name = str(run_obj.get("name", "")).lower()
  display_name = str(
      run_obj.get("displayName", run_obj.get("display_name", ""))
  ).lower()
  run_set = str(run_obj.get("runSet", run_obj.get("run_set", ""))).lower()
  gke_id = ""
  wl_details = run_obj.get(
      "workloadDetails", run_obj.get("workload_details", {})
  )
  if isinstance(wl_details, dict):
    gke_block = wl_details.get("gke", {})
    if isinstance(gke_block, dict):
      gke_id = str(gke_block.get("id", "")).lower()
  return any(
      wl_lower in candidate
      for candidate in (name, display_name, run_set, gke_id)
      if candidate
  )


def discover_ml_runs(
    project_id: str,
    location: str,
    workload_id: str = "",
) -> list[str]:
  """Discovers matching `machineLearningRun` IDs via gcloud CLI or REST API."""
  try:
    raw = _run_gcloud_json([
        "alpha",
        "mldiagnostics",
        "machine-learning-run",
        "list",
        f"--location={location}",
        f"--project={project_id}",
    ])
  except (CommandExecutionError, subprocess.SubprocessError, OSError):
    q_proj = urllib.parse.quote(project_id, safe="")
    q_loc = urllib.parse.quote(location, safe="")
    url = (
        "https://hypercomputecluster.googleapis.com/v1alpha/"
        f"projects/{q_proj}/locations/{q_loc}/machineLearningRuns"
    )
    raw = _fetch_rest_json(url)

  items = _extract_items_list(
      raw, ("machineLearningRuns", "machine_learning_runs", "items")
  )
  matched_ids: list[str] = []
  for item in items:
    if not isinstance(item, dict):
      continue
    if not _matches_workload_id(item, workload_id):
      continue
    run_id = str(item.get("name", "")).split("/")[-1].strip()
    if run_id:
      _append_unique(matched_ids, [run_id])
  return matched_ids


def fetch_monitored_events_for_run(
    project_id: str,
    location: str,
    ml_run_id: str,
    event_id: str = "",
) -> list[dict[str, Any]]:
  """Fetches `monitoredEvents` and full `analyzerReports` for an ML run."""
  q_proj = urllib.parse.quote(project_id, safe="")
  q_loc = urllib.parse.quote(location, safe="")
  q_run = urllib.parse.quote(ml_run_id, safe="")
  base_rest_url = (
      "https://hypercomputecluster.googleapis.com/v1alpha/"
      f"projects/{q_proj}/locations/{q_loc}/machineLearningRuns/{q_run}/monitoredEvents"
  )

  normalized_event_id = event_id.strip().split("/")[-1] if event_id else ""
  if normalized_event_id:
    event_ids = [normalized_event_id]
    listed_events_by_id: dict[str, dict[str, Any]] = {}
  else:
    try:
      raw_list = _run_gcloud_json([
          "alpha",
          "mldiagnostics",
          "monitored-events",
          "list",
          f"--machine-learning-run={ml_run_id}",
          f"--location={location}",
          f"--project={project_id}",
      ])
    except (CommandExecutionError, subprocess.SubprocessError, OSError):
      raw_list = _fetch_rest_json(base_rest_url)

    items = _extract_items_list(
        raw_list, ("monitoredEvents", "monitored_events", "items")
    )
    event_ids = []
    listed_events_by_id = {}
    for ev in items:
      if not isinstance(ev, dict):
        continue
      ev_id = str(ev.get("name", "")).split("/")[-1].strip()
      if ev_id:
        _append_unique(event_ids, [ev_id])
        listed_events_by_id[ev_id] = ev

  detailed_events: list[dict[str, Any]] = []
  for ev_id in event_ids:
    existing = listed_events_by_id.get(ev_id)
    if isinstance(existing, dict) and (
        "analyzerReports" in existing or "analyzer_reports" in existing
    ):
      detailed_events.append(existing)
      continue

    try:
      ev_detail = _run_gcloud_json([
          "alpha",
          "mldiagnostics",
          "monitored-events",
          "describe",
          ev_id,
          f"--machine-learning-run={ml_run_id}",
          f"--location={location}",
          f"--project={project_id}",
      ])
    except (CommandExecutionError, subprocess.SubprocessError, OSError):
      q_ev = urllib.parse.quote(ev_id, safe="")
      ev_detail = _fetch_rest_json(f"{base_rest_url}/{q_ev}")

    if isinstance(ev_detail, dict) and ev_detail:
      detailed_events.append(ev_detail)
    elif isinstance(existing, dict):
      detailed_events.append(existing)

  return detailed_events


def extract_profiler_sessions(
    raw_sessions: Sequence[dict[str, Any]],
    ml_run_id: str = "",
) -> list[dict[str, Any]]:
  """Extracts structured profiler session metadata (AutoCapture, targets, GCS paths)."""
  extracted: list[dict[str, Any]] = []
  for sess in raw_sessions:
    if not isinstance(sess, dict):
      continue
    name = str(sess.get("name", "")).strip()
    auto_captured = bool(
        sess.get("autoCaptured", sess.get("auto_captured", False))
    )
    kind = str(sess.get("kind", sess.get("sessionKind", ""))).strip()
    state = str(sess.get("state", sess.get("status", ""))).strip()

    targets: list[str] = []
    for t_key in ("profilerTargets", "profiler_targets", "targets"):
      t_val = sess.get(t_key)
      if isinstance(t_val, list):
        for item in t_val:
          if isinstance(item, str) and item.strip():
            _append_unique(targets, [item.strip()])
          elif isinstance(item, dict):
            t_name = str(
                item.get("name", item.get("target", item.get("id", "")))
            ).strip()
            if t_name:
              _append_unique(targets, [t_name])
      elif isinstance(t_val, str) and t_val.strip():
        _append_unique(targets, [t_val.strip()])

    gcs_paths: list[str] = []
    for g_key in (
        "gcsPath",
        "gcs_path",
        "storagePath",
        "storage_path",
        "artifactUri",
        "artifact_uri",
    ):
      g_val = sess.get(g_key)
      if isinstance(g_val, str) and g_val.strip():
        _append_unique(gcs_paths, [g_val.strip()])
    raw_dump = json.dumps(sess)
    gcs_matches = re.findall(r"gs://[^\s\"'`,;\]]+", raw_dump)
    _append_unique(gcs_paths, gcs_matches)

    extracted.append({
        "name": name,
        "ml_run_id": ml_run_id,
        "auto_captured": auto_captured,
        "kind": kind,
        "state": state,
        "profiler_targets": targets,
        "gcs_paths": gcs_paths,
    })
  return extracted


def fetch_profiler_sessions_for_run(
    project_id: str,
    location: str,
    ml_run_id: str,
) -> list[dict[str, Any]]:
  """Best-effort query for profilerSessions on an ML run (never raises on error)."""
  raw_list: Any = []
  try:
    raw_list = _run_gcloud_json([
        "alpha",
        "mldiagnostics",
        "profiler-session",
        "list",
        f"--machine-learning-run={ml_run_id}",
        f"--location={location}",
        f"--project={project_id}",
    ])
  except (
      AccessDeniedError,
      CommandExecutionError,
      subprocess.SubprocessError,
      OSError,
      ValueError,
  ):
    try:
      q_proj = urllib.parse.quote(project_id, safe="")
      q_loc = urllib.parse.quote(location, safe="")
      q_run = urllib.parse.quote(ml_run_id, safe="")
      url = (
          "https://hypercomputecluster.googleapis.com/v1alpha/"
          f"projects/{q_proj}/locations/{q_loc}/machineLearningRuns/{q_run}/profilerSessions"
      )
      raw_list = _fetch_rest_json(url)
    except (
        AccessDeniedError,
        CommandExecutionError,
        subprocess.SubprocessError,
        OSError,
        ValueError,
    ):
      return []

  items = _extract_items_list(
      raw_list, ("profilerSessions", "profiler_sessions", "items")
  )
  return extract_profiler_sessions(items, ml_run_id=ml_run_id)


def _write_output_and_print_summary(
    output_file: str,
    full_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> None:
  """Writes `full_payload` to `output_file` and prints `summary_payload`."""
  out_dir = os.path.dirname(output_file)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(full_payload, f, indent=2)
  print(json.dumps(summary_payload))


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Query ML Diagnostics MonitoredEvents and map analyzer findings to"
          " wiki symptoms and workflows."
      )
  )
  parser.add_argument("--project_id", "--project-id", required=True)
  parser.add_argument("--location", required=True)
  parser.add_argument("--ml_run_id", "--ml-run-id", default="")
  parser.add_argument("--workload_id", "--workload-id", default="")
  parser.add_argument("--event_id", "--event-id", default="")
  parser.add_argument(
      "--output_file",
      "--output-file",
      default="scratch/mldiagnostics_monitored_events.json",
  )
  args = parser.parse_args(argv)

  try:
    if args.ml_run_id.strip():
      runs = [args.ml_run_id.strip().split("/")[-1]]
    else:
      runs = discover_ml_runs(
          project_id=args.project_id,
          location=args.location,
          workload_id=args.workload_id,
      )

    all_events: list[dict[str, Any]] = []
    all_profiler_sessions: list[dict[str, Any]] = []
    results_by_run: list[dict[str, Any]] = []
    for run_id in runs:
      run_events = fetch_monitored_events_for_run(
          project_id=args.project_id,
          location=args.location,
          ml_run_id=run_id,
          event_id=args.event_id.strip(),
      )
      all_events.extend(run_events)
      has_perf_degradation = any(
          "PERFORMANCE_DEGRADATION"
          in str(ev.get("type", ev.get("eventType", ""))).upper()
          for ev in run_events
          if isinstance(ev, dict)
      )
      run_profiler_sessions: list[dict[str, Any]] = []
      if has_perf_degradation:
        run_profiler_sessions = fetch_profiler_sessions_for_run(
            project_id=args.project_id,
            location=args.location,
            ml_run_id=run_id,
        )
        all_profiler_sessions.extend(run_profiler_sessions)
      results_by_run.append({
          "ml_run_id": run_id,
          "events": run_events,
          "profiler_sessions": run_profiler_sessions,
      })

    parsed = parse_and_match_analyzer_reports(all_events)
    if not runs:
      exec_status = "NO_RUNS_FOUND"
    elif not all_events:
      exec_status = "NO_EVENTS_FOUND"
    else:
      exec_status = "SUCCESS"

    summary = {
        "execution_status": exec_status,
        "status": exec_status,
        "ml_runs_queried": len(runs),
        "events_count": len(all_events),
        "detected_analyzers": parsed["detected_analyzers"],
        "not_detected_analyzers": parsed["not_detected_analyzers"],
        "matched_symptoms": parsed["matched_symptoms"],
        "recommended_workflows": parsed["recommended_workflows"],
        "recommended_actions": parsed["recommended_actions"],
        "mxla_hang_causes": parsed["mxla_hang_causes"],
        "affected_instances": parsed["affected_instances"],
        "affected_gce_instance_ids": parsed["affected_gce_instance_ids"],
        "profiler_sessions": all_profiler_sessions,
        "output_file": args.output_file,
    }
    full_payload = {
        **summary,
        "project_id": args.project_id,
        "location": args.location,
        "workload_id": args.workload_id,
        "detected_reports": parsed["detected_reports"],
        "results": results_by_run,
    }
    _write_output_and_print_summary(args.output_file, full_payload, summary)
    return 0

  except AccessDeniedError as e:
    err_msg = str(e)
    print(
        "[ML Diagnostics] ACCESS_DENIED querying project"
        f" '{args.project_id}': {err_msg}",
        file=sys.stderr,
    )
    summary = {
        "execution_status": "ACCESS_DENIED",
        "status": "ACCESS_DENIED",
        "error": err_msg,
        "events_count": 0,
        "detected_analyzers": [],
        "not_detected_analyzers": [],
        "matched_symptoms": [],
        "recommended_workflows": [],
        "recommended_actions": [],
        "affected_instances": [],
        "affected_gce_instance_ids": [],
        "profiler_sessions": [],
        "output_file": args.output_file,
    }
    _write_output_and_print_summary(args.output_file, summary, summary)
    return EXIT_ACCESS_DENIED

  except (
      CommandExecutionError,
      subprocess.SubprocessError,
      OSError,
      ValueError,
  ) as e:
    err_msg = str(e)
    print(
        f"[ML Diagnostics] ERROR querying project '{args.project_id}':"
        f" {err_msg}",
        file=sys.stderr,
    )
    summary = {
        "execution_status": "COMMAND_ERROR",
        "status": "COMMAND_ERROR",
        "error": err_msg,
        "events_count": 0,
        "detected_analyzers": [],
        "not_detected_analyzers": [],
        "matched_symptoms": [],
        "recommended_workflows": [],
        "recommended_actions": [],
        "affected_instances": [],
        "affected_gce_instance_ids": [],
        "profiler_sessions": [],
        "output_file": args.output_file,
    }
    _write_output_and_print_summary(args.output_file, summary, summary)
    return 1


if __name__ == "__main__":
  sys.exit(main())

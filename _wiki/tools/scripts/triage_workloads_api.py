#!/usr/bin/env python3
"""Triage related TPU workloads using Python Native google-cloud-logging API."""

import argparse
import csv
import importlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set


class WorkloadTriageData:
  """Holds categorized logs for a single workload."""

  def __init__(self) -> None:
    self.exit_logs: List[Dict[str, Any]] = []
    self.evict_logs: List[Dict[str, Any]] = []
    self.quota_logs: List[Dict[str, Any]] = []
    self.oom_logs: List[Dict[str, Any]] = []
    self.mxla_logs: List[Dict[str, Any]] = []
    self.all_wl_logs: List[Dict[str, Any]] = []


def match_token(row: Dict[str, str], token: str) -> bool:
  """Matches a single query filter token against a CSV row."""
  if ":" in token:
    key, val = token.split(":", 1)
    val_clean = val.replace('"', "").strip()
    if key == "logName" and val_clean == "events":
      msg = row.get("message", "")
      return "Evicted" in msg or "Slice" in msg or "finalizing" in msg
    elif "pod_name" in key:
      return val_clean in row.get("slice_name", "") or val_clean in row.get(
          "message", ""
      )

  if "=" in token:
    key, val = token.split("=", 1)
    val_clean = val.replace('"', "").strip()
    if "reason" in key:
      return val_clean in row.get("message", "") or val_clean in row.get(
          "state", ""
      )

  token_clean = token.replace('"', "").strip()
  return (
      token_clean.lower() in row.get("message", "").lower()
      or token_clean.lower() in row.get("state", "").lower()
  )


def match_csv_row(row: Dict[str, str], filter_str: str) -> bool:
  """Evaluates GCloud filter string against CSV row columns."""
  filter_clean = filter_str.strip()
  if filter_clean.startswith("(") and filter_clean.endswith(")"):
    filter_clean = filter_clean[1:-1].strip()

  tokens = filter_clean.split(" AND ")
  for token in tokens:
    token = token.strip()
    if not token:
      continue
    if " OR " in token:
      or_content = token
      if or_content.startswith("(") and or_content.endswith(")"):
        or_content = or_content[1:-1]
      or_tokens = [ot.strip() for ot in or_content.split(" OR ")]
      if not any(match_token(row, ot) for ot in or_tokens):
        return False
    else:
      if not match_token(row, token):
        return False
  return True


def run_gcloud_api(
    filter_str: str,
    project: str,
    limit: int = 100,
    order: str = "ASC",
    workload_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
  """Queries Cloud Logging API or falls back to local GKE CSV telemetry."""
  # 1. Try API call using google-cloud-logging
  try:
    logging_pkg = importlib.import_module("google.cloud.logging")

    client = logging_pkg.Client(project=project)
    order_by = "timestamp asc" if order == "ASC" else "timestamp desc"

    entries = client.list_entries(
        filter_=filter_str, page_size=limit, order_by=order_by
    )
    results: List[Dict[str, Any]] = []

    for entry in entries:
      payload = entry.payload
      if isinstance(payload, dict):
        payload_str = json.dumps(payload)
      else:
        payload_str = str(payload)

      results.append({
          "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
          "textPayload": payload_str,
          "resource": {
              "labels": entry.resource.labels if entry.resource else {}
          },
          "labels": entry.labels or {},
      })
      if len(results) >= limit:
        break
    return results
  except (
      OSError,
      ValueError,
      RuntimeError,
      KeyError,
      TypeError,
      AttributeError,
      ImportError,
  ):
    pass

  # 2. Fallback to local CSV
  csv_name = f"slice_events_{workload_prefix}.csv"
  csv_path = os.environ.get("SLICE_EVENTS_CSV_PATH")
  if not csv_path or not os.path.exists(csv_path):
    home = os.path.expanduser("~")
    csv_path = os.path.join(home, ".gemini", "logs", csv_name)

  if not os.path.exists(csv_path):
    return []

  results = []
  try:
    with open(csv_path, "r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        if match_csv_row(row, filter_str):
          msg = row.get("message", "")
          ts_match = re.search(
              r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s*", msg
          )
          ts_val = ts_match.group(1) if ts_match else row.get("timestamp") or ""

          log_entry = {
              "timestamp": ts_val,
              "textPayload": msg,
              "resource": {"labels": {"pod_name": row.get("slice_name", "")}},
              "labels": {
                  "k8s-pod/jobset_sigs_k8s_io/jobset-name": row.get(
                      "jobset", ""
                  )
              },
          }
          results.append(log_entry)
          if len(results) >= limit:
            break
  except (OSError, csv.Error, ValueError, KeyError) as e:
    print(f"Error reading CSV: {e}", file=sys.stderr)
    return []

  if order == "DESC":
    results.reverse()
  return results


def extract_workload_name(
    pod_name: str, prefix: str, labels: Dict[str, str]
) -> Optional[str]:
  """Resolves the GKE workload/JobSet name from pod labels or name prefix."""
  for k in [
      "k8s-pod/jobset_sigs_k8s_io/jobset-name",
      "k8s-pod/jobset.sigs.k8s.io/jobset-name",
      "k8s-pod/xpk_google_com/workload",
  ]:
    if k in labels:
      return labels[k]

  if not pod_name.startswith(prefix):
    return None

  remainder = pod_name[len(prefix) :]
  match = re.search(r"^-(\d+)", remainder)
  if match:
    return prefix + "-" + match.group(1)
  else:
    return prefix


def clean_message_for_dedup(msg: str) -> str:
  """Strips standard timestamp and severity prefixes to compare body content."""
  msg = re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*", "", msg)
  msg = re.sub(r"^LEVEL\([^)]+\)\s*", "", msg)
  return msg.strip()


def classify_logs(
    all_logs: List[Dict[str, Any]], workload_prefix: str
) -> Dict[str, WorkloadTriageData]:
  """Categorizes all log lines into triaging buckets grouped by workload."""
  workloads: Dict[str, WorkloadTriageData] = {}

  exit_code_re = re.compile(r"EXIT_CODE=(\d+)")
  quota_re = re.compile(r"quota|429|bandwidth|egress", re.IGNORECASE)
  oom_re = re.compile(r"oom|hbm|out of memory|tpuoom", re.IGNORECASE)
  mxla_re = re.compile(
      "DIFFERENT_MODULE|FINGERPRINT_MISMATCH|BAD_TPU_CHIP|"
      "DATA_INPUT_STALL|UNRECOVERABLE_ERROR|NETWORKING_ISSUE"
  )

  for log in all_logs:
    pod_name = log.get("resource", {}).get("labels", {}).get("pod_name", "")
    labels = log.get("labels", {})
    wl_name = extract_workload_name(pod_name, workload_prefix, labels)

    if not wl_name:
      continue

    if wl_name not in workloads:
      workloads[wl_name] = WorkloadTriageData()

    payload = log.get("textPayload", "")
    workloads[wl_name].all_wl_logs.append(log)

    if exit_code_re.search(payload):
      workloads[wl_name].exit_logs.append(log)
    if "Evicted" in payload or "evicting" in payload or "finalizing" in payload:
      workloads[wl_name].evict_logs.append(log)
    if quota_re.search(payload):
      workloads[wl_name].quota_logs.append(log)
    if oom_re.search(payload):
      workloads[wl_name].oom_logs.append(log)
    if mxla_re.search(payload):
      workloads[wl_name].mxla_logs.append(log)

  return workloads


def write_logs_to_file(
    all_logs: List[Dict[str, Any]],
    workloads: Dict[str, WorkloadTriageData],
    workload_prefix: str,
    output_file: str,
    deduplicate: bool,
) -> None:
  """Writes the pulled log entries to a local output file."""
  try:
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    written_count = 0
    seen_contents: Set[str] = set()

    with open(output_file, "w", encoding="utf-8") as f:
      wl_names_set = set(workloads.keys())
      for log in all_logs:
        pod_name = log.get("resource", {}).get("labels", {}).get("pod_name", "")
        wl_name = extract_workload_name(
            pod_name, workload_prefix, log.get("labels", {})
        )
        if wl_name not in wl_names_set:
          continue

        msg = log.get("textPayload", "")
        if not msg:
          continue

        if deduplicate:
          cleaned = clean_message_for_dedup(msg)
          if cleaned in seen_contents:
            continue
          seen_contents.add(cleaned)

        f.write(msg + "\n")
        written_count += 1

    print(
        f"Successfully wrote {written_count} log messages to {output_file} "
        f"(Deduplicated={deduplicate})"
    )
  except (OSError, ValueError, TypeError) as we:
    print(f"Error writing to output file: {we}", file=sys.stderr)
    sys.exit(1)


def print_triage_summary(workloads: Dict[str, WorkloadTriageData]) -> None:
  """Prints console-side formatted triage summary findings."""
  exit_code_re = re.compile(r"EXIT_CODE=(\d+)")
  traceback_re = re.compile(r"Traceback|ValueError|KeyError|AttributeError")

  for wl in sorted(workloads.keys()):
    print(f"=== Triaging Workload: {wl} ===")
    wl_data = workloads[wl]

    exit_counts: Dict[str, int] = {}
    earliest_exit: Optional[Dict[str, str]] = None

    for log in wl_data.exit_logs:
      payload = log.get("textPayload", "")
      pod = log.get("resource", {}).get("labels", {}).get("pod_name", "")
      ts = log.get("timestamp", "")
      m = exit_code_re.search(payload)
      if m:
        code = m.group(1)
        exit_counts[code] = exit_counts.get(code, 0) + 1
        if not earliest_exit:
          earliest_exit = {"pod": pod, "code": code, "timestamp": ts}

    print(f"Exit code distribution: {exit_counts}")
    if earliest_exit:
      print(
          f"Earliest Exit Code: {earliest_exit['code']} on pod"
          f" {earliest_exit['pod']} at {earliest_exit['timestamp']}"
      )
    else:
      print("No exit codes logged (possible silent eviction).")

    if wl_data.evict_logs:
      first_evict = wl_data.evict_logs[0]
      msg = first_evict.get("textPayload", "")
      pod = (
          first_evict.get("resource", {}).get("labels", {}).get("pod_name", "")
      )
      ts = first_evict.get("timestamp", "")
      print(f"Eviction detected: {msg[:150].strip()}... on pod {pod} at {ts}")
    else:
      print("No eviction events found.")

    if wl_data.quota_logs:
      first_quota = wl_data.quota_logs[0]
      msg = first_quota.get("textPayload", "")
      pod = (
          first_quota.get("resource", {}).get("labels", {}).get("pod_name", "")
      )
      ts = first_quota.get("timestamp", "")
      print(
          f"Quota/API Limit error detected: {msg[:200].strip()}... on pod"
          f" {pod} at {ts}"
      )
    else:
      print("No quota keywords found in logs.")

    if wl_data.oom_logs:
      first_oom = wl_data.oom_logs[0]
      msg = first_oom.get("textPayload", "")
      pod = first_oom.get("resource", {}).get("labels", {}).get("pod_name", "")
      ts = first_oom.get("timestamp", "")
      print(f"TPU OOM detected: {msg[:200].strip()}... on pod {pod} at {ts}")
    else:
      print("No OOM keywords found in logs.")

    if wl_data.mxla_logs:
      first_mxla = wl_data.mxla_logs[0]
      msg = first_mxla.get("textPayload", "")
      pod = (
          first_mxla.get("resource", {}).get("labels", {}).get("pod_name", "")
      )
      ts = first_mxla.get("timestamp", "")
      print(
          f"MXLA Hang Error Digest detected: {msg[:200].strip()}... on pod"
          f" {pod} at {ts}"
      )
    else:
      print("No MXLA Hang Error Digests found in logs.")

    if earliest_exit and earliest_exit["code"] in ["1", "134"]:
      target_pod = earliest_exit["pod"]
      tb_found = False
      for log in wl_data.all_wl_logs:
        pod = log.get("resource", {}).get("labels", {}).get("pod_name", "")
        if pod != target_pod:
          continue
        payload = log.get("textPayload", "")
        if traceback_re.search(payload):
          print("Traceback/Error signature found in earliest exit pod:")
          lines = payload.strip().split("\n")
          print("  " + "\n  ".join(lines[:5]))
          tb_found = True
          break
      if not tb_found:
        print("No tracebacks found in earliest exit pod logs.")
    print()


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Triage related TPU workloads using Python Native API"
  )
  parser.add_argument(
      "workload_prefix", help="Prefix of the workloads (e.g. my-training-job)"
  )
  parser.add_argument(
      "--project",
      required=False,
      default="[GCP_PROJECT_ID]",
      help="GCP Project ID",
  )
  parser.add_argument(
      "--output_file",
      required=False,
      help="Local file path to write log messages",
  )
  parser.add_argument(
      "--deduplicate",
      action="store_true",
      help="Deduplicate log messages by content",
  )
  args = parser.parse_args()

  wl_prefix = args.workload_prefix

  print(
      f"Retrieving all telemetry logs for workload prefix '{wl_prefix}' via"
      " google-cloud-logging API..."
  )
  filter_all = f'resource.labels.pod_name:"{wl_prefix}-"'
  all_logs = run_gcloud_api(
      filter_all, args.project, limit=10000, workload_prefix=wl_prefix
  )
  all_logs.sort(key=lambda x: x.get("timestamp", ""))

  workloads = classify_logs(all_logs, wl_prefix)

  if args.output_file:
    write_logs_to_file(
        all_logs,
        workloads,
        wl_prefix,
        args.output_file,
        args.deduplicate,
    )

  print_triage_summary(workloads)


if __name__ == "__main__":
  main()

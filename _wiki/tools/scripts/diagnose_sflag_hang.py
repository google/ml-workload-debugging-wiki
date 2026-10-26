#!/usr/bin/env python3
"""Diagnose MegaScale / MXLA sflag and collective hangs in TPU workloads.

Queries Cloud Logging in chronological ascending order (--order=ASC) with an
explicit --freshness window, mirroring the multi-tier hang attribution
heuristics of Megascale error aggregation (GetBottlenecks / GetErrorDigest):
  1. TensorCore 'Wait timeout on sflag' (including HLO op extraction).
  2. SparseCore 'Sflag wait timeout on op' and sequencer halts ('tag:pc').
  3. Program Counter (pc == 0 -> DATA_INPUT_STALL vs pc != 0 -> BAD_TPU_CHIP).
  4. Cross-slice DCN collective action stalls (NETWORK_SEND, NETWORK_RECEIVE,
     D2H_START, H2D_START, WAIT_BARRIER / device_to_host, host_to_device,
     network_send, network_receive, compute).
  5. Silent / Missing Pod Detection across <slice>-<worker> topology indices.
"""

import argparse
import collections
from concurrent import futures
import json
import os
import re
import subprocess
from typing import Any

_SFLAG_TC_RE = re.compile(
    r"Wait timeout on sflag.*?['\"]+([^'\"]+)['\"]+"
    r"(?:.*?\bhlo:\s*([a-zA-Z0-9_.-]+))?"
)
_SFLAG_SC_RE = re.compile(r"Sflag wait timeout on op(?::\s*|\s*\{)([^\s,;}]+)")
_SC_HALT_TAG_PC_RE = re.compile(
    r"(?:halted unexpectedly|halted"
    r" prematurely).*?\btag:pc\s*[=:]?\s*([^\s,;]+)"
    r"|\btag:pc\s*[=:]\s*([^\s,;]+)",
    re.IGNORECASE,
)
_PC_VALUE_RE = re.compile(
    r"(?<!tag:)\bpc\s*[=:]\s*(0x[0-9a-fA-F]+|\d+)\b",
    re.IGNORECASE,
)
_DCN_ACTION_RE = re.compile(
    r"\b(D2H_START|NETWORK_SEND|NETWORK_RECEIVE|H2D_START|WAIT_BARRIER)\b"
)
_POD_SLICE_WORKER_RE = re.compile(
    r"^(?P<stem>.+?)-(?P<slice>\d+)-(?P<worker>\d+)"
    r"(?:-(?P<suffix>[a-z0-9]{4,10}))?$"
)
_POD_SINGLE_WORKER_RE = re.compile(
    r"^(?P<stem>.+?)-(?P<worker>\d+)(?:-(?P<suffix>[a-z0-9]{4,10}))?$"
)

_OUTLIER_RATIO_THRESHOLD = 0.20


def run_gcloud(
    filter_str: str,
    project: str,
    limit: int = 5000,
    freshness: str = "180d",
) -> list[dict[str, Any]]:
  """Queries Cloud Logging in chronological ascending order."""
  command = [
      "gcloud",
      "logging",
      "read",
      filter_str,
      f"--project={project}",
      "--order=ASC",
      f"--freshness={freshness}",
      "--format=json",
      f"--limit={limit}",
  ]
  completed_proc = subprocess.run(
      command, capture_output=True, text=True, check=False
  )
  if completed_proc.returncode != 0:
    return []
  try:
    parsed_logs = json.loads(completed_proc.stdout)
    return parsed_logs if isinstance(parsed_logs, list) else []
  except json.JSONDecodeError:
    return []


def run_node_node_telemetry(node_name: str) -> str | None:
  """Resolves a GKE node name to its physical machine via host node_telemetry."""
  if not node_name or node_name == "unknown":
    return None
  command = ["echo", node_name]
  try:
    completed_proc = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
  except FileNotFoundError:
    return None
  if completed_proc.returncode == 0:
    return completed_proc.stdout.strip()
  return None


def _fetch_node_node_telemetry_for_pod(
    pod_name: str, node_name: str
) -> tuple[str, str, str | None]:
  """Helper for concurrent physical host node_telemetry resolution."""
  if not node_name or node_name == "unknown":
    return pod_name, "unknown", None
  return pod_name, node_name, run_node_node_telemetry(node_name)


def _extract_log_message(log_entry: dict[str, Any]) -> str:
  """Extracts text message from a Cloud Logging entry."""
  json_payload = log_entry.get("jsonPayload")
  if isinstance(json_payload, dict):
    msg = json_payload.get("message")
    if isinstance(msg, str) and msg:
      return msg
  text_payload = log_entry.get("textPayload")
  if isinstance(text_payload, str):
    return text_payload
  return ""


def _detect_silent_pods(
    reporting_pods: set[str], expected_pods: int = 0
) -> list[str]:
  """Identifies missing <slice>-<worker> pods absent from sflag reporters."""
  stem_to_slices: dict[str, dict[int, set[int]]] = collections.defaultdict(
      lambda: collections.defaultdict(set)
  )
  single_stem_to_workers: dict[str, set[int]] = collections.defaultdict(set)

  for pod_name in reporting_pods:
    slice_match = _POD_SLICE_WORKER_RE.match(pod_name)
    if slice_match:
      stem = slice_match.group("stem")
      slice_idx = int(slice_match.group("slice"))
      worker_idx = int(slice_match.group("worker"))
      stem_to_slices[stem][slice_idx].add(worker_idx)
      continue

    single_match = _POD_SINGLE_WORKER_RE.match(pod_name)
    if single_match:
      stem = single_match.group("stem")
      worker_idx = int(single_match.group("worker"))
      single_stem_to_workers[stem].add(worker_idx)

  silent_pod_patterns: list[str] = []

  if stem_to_slices:
    apply_expected_to_all = len(stem_to_slices) == 1
    for stem, slices_map in sorted(stem_to_slices.items()):
      inferred_workers_per_slice = max(
          (max(workers) + 1 for workers in slices_map.values() if workers),
          default=0,
      )
      use_expected = expected_pods > 0 and (
          apply_expected_to_all or inferred_workers_per_slice > 1
      )
      target_workers_per_slice = (
          max(inferred_workers_per_slice, expected_pods)
          if use_expected
          else inferred_workers_per_slice
      )
      if target_workers_per_slice > 1:
        max_slice_idx = max(slices_map.keys(), default=-1)
        for slice_idx in range(max_slice_idx + 1):
          observed_workers = slices_map.get(slice_idx, set())
          for worker_idx in range(target_workers_per_slice):
            if worker_idx not in observed_workers:
              silent_pod_patterns.append(f"{stem}-{slice_idx}-{worker_idx}-*")
    return silent_pod_patterns

  if single_stem_to_workers:
    for stem, observed_workers in sorted(single_stem_to_workers.items()):
      inferred_count = max(observed_workers) + 1 if observed_workers else 0
      target_count = (
          max(inferred_count, expected_pods)
          if expected_pods > 0
          else inferred_count
      )
      if target_count > 1:
        for worker_idx in range(target_count):
          if worker_idx not in observed_workers:
            silent_pod_patterns.append(f"{stem}-{worker_idx}-*")

  return silent_pod_patterns


def _classify_primary_cause(
    pc_zero_pods: set[str],
    pc_nonzero_pods: set[str],
    dcn_action_counts: dict[str, int],
    sc_halt_groups: dict[str, list[str]],
    outlier_pods: list[str],
    silent_pods: list[str],
) -> str:
  """Infers hang category following Megascale hang attribution priority."""
  if pc_zero_pods and (
      not pc_nonzero_pods or len(pc_zero_pods) <= len(pc_nonzero_pods)
  ):
    return "DATA_INPUT_STALL"
  if dcn_action_counts:
    net_edges = dcn_action_counts.get(
        "NETWORK_SEND", 0
    ) + dcn_action_counts.get("NETWORK_RECEIVE", 0)
    device_edges = (
        dcn_action_counts.get("D2H_START", 0)
        + dcn_action_counts.get("H2D_START", 0)
        + dcn_action_counts.get("WAIT_BARRIER", 0)
    )
    if net_edges > device_edges:
      return "NETWORKING_ISSUE"
    if device_edges > 0:
      return "BAD_TPU_CHIP"
  if outlier_pods or len(sc_halt_groups) > 1:
    return "BAD_TPU_CHIP"
  if silent_pods:
    return "SILENT_WORKER_STALL"
  if pc_nonzero_pods and pc_zero_pods:
    return "BAD_TPU_CHIP"
  return "UNKNOWN_UNIFORM_SFLAG_WAIT"


def analyze_sflag_logs(
    logs: list[dict[str, Any]],
    expected_pods: int = 0,
    max_concurrent_workers: int = 10,
) -> dict[str, Any]:
  """Groups sflag timeouts, SparseCore halts, PCs, DCN, and silent pods."""
  sflag_groups: dict[str, list[str]] = collections.defaultdict(list)
  sc_halt_groups: dict[str, list[str]] = collections.defaultdict(list)
  dcn_action_counts: dict[str, int] = collections.Counter()
  pc_zero_pods: set[str] = set()
  pc_nonzero_pods: set[str] = set()
  pod_to_node: dict[str, str] = {}
  reporting_pods: set[str] = set()

  for log_entry in logs:
    message = _extract_log_message(log_entry)
    pod_name = (
        log_entry.get("resource", {}).get("labels", {}).get("pod_name", "")
    )
    node_name = log_entry.get("labels", {}).get(
        "compute.googleapis.com/resource_name", ""
    )

    if pod_name and node_name:
      pod_to_node[pod_name] = node_name

    match_tc = _SFLAG_TC_RE.search(message)
    match_sc = _SFLAG_SC_RE.search(message)
    match_sc_halt = _SC_HALT_TAG_PC_RE.search(message)

    if match_sc_halt:
      tag_pc = match_sc_halt.group(1) or match_sc_halt.group(2) or "unknown"
      sc_halt_groups[f"tag:pc={tag_pc}"].append(pod_name)

    if match_tc:
      if pod_name:
        reporting_pods.add(pod_name)
      sflag_name = match_tc.group(1)
      hlo_op = match_tc.group(2)
      group_key = f"{sflag_name} (hlo: {hlo_op})" if hlo_op else sflag_name
      sflag_groups[group_key].append(pod_name)
    elif match_sc:
      if pod_name:
        reporting_pods.add(pod_name)
      sc_op = match_sc.group(1)
      if match_sc_halt:
        tag_pc = match_sc_halt.group(1) or match_sc_halt.group(2) or "unknown"
        sflag_groups[f"SparseCore op: {sc_op} (tag:pc={tag_pc})"].append(
            pod_name
        )
      else:
        sflag_groups[f"SparseCore op: {sc_op}"].append(pod_name)
    elif match_sc_halt:
      if pod_name:
        reporting_pods.add(pod_name)
      tag_pc = match_sc_halt.group(1) or match_sc_halt.group(2) or "unknown"
      sflag_groups[f"SparseCore halt tag:pc={tag_pc}"].append(pod_name)
    else:
      sflag_groups["unknown"].append(pod_name)

    for pc_match in _PC_VALUE_RE.finditer(message):
      raw_pc = pc_match.group(1)
      pc_int = (
          int(raw_pc, 16) if raw_pc.lower().startswith("0x") else int(raw_pc)
      )
      if pc_int == 0 and pod_name:
        pc_zero_pods.add(pod_name)
      elif pc_int != 0 and pod_name:
        pc_nonzero_pods.add(pod_name)

    for dcn_match in _DCN_ACTION_RE.finditer(message):
      dcn_action_counts[dcn_match.group(1)] += 1

  outlier_groups: list[dict[str, Any]] = []
  outlier_pods_set: set[str] = set()

  non_unknown_groups = [
      (group_name, pods, sorted(set(pod for pod in pods if pod)))
      for group_name, pods in sflag_groups.items()
      if group_name != "unknown"
  ]
  sorted_groups = sorted(
      non_unknown_groups, key=lambda item: (len(item[2]), len(item[1]))
  )

  if len(sorted_groups) > 1:
    _, majority_pods, unique_majority_pods = sorted_groups[-1]
    unique_majority_count = len(unique_majority_pods) or len(majority_pods)
    outlier_threshold = max(
        1.0, unique_majority_count * _OUTLIER_RATIO_THRESHOLD
    )
    for group_name, pods, unique_group_pods in sorted_groups[:-1]:
      unique_count = len(unique_group_pods) or len(pods)
      is_minority = unique_count < unique_majority_count and (
          unique_count < unique_majority_count * _OUTLIER_RATIO_THRESHOLD
          or (unique_majority_count > 1 and unique_count <= outlier_threshold)
      )
      if is_minority:
        outlier_pods_set.update(unique_group_pods)
        outlier_groups.append({
            "sflag_group": group_name,
            "occurrences": len(pods),
            "unique_pods": unique_group_pods,
        })

  sorted_outlier_pods = sorted(outlier_pods_set)
  host_mappings: dict[str, dict[str, str | None]] = {}
  if sorted_outlier_pods:
    worker_pool_size = max(
        1, min(max_concurrent_workers, len(sorted_outlier_pods))
    )
    with futures.ThreadPoolExecutor(max_workers=worker_pool_size) as executor:
      submitted_futures = [
          executor.submit(
              _fetch_node_node_telemetry_for_pod,
              pod_name,
              pod_to_node.get(pod_name, "unknown"),
          )
          for pod_name in sorted_outlier_pods
      ]
      for future in submitted_futures:
        pod_name, node_name, node_telemetry_output = future.result()
        host_mappings[pod_name] = {
            "gke_node": node_name,
            "physical_machine": node_telemetry_output,
        }

  silent_pods = _detect_silent_pods(reporting_pods, expected_pods=expected_pods)
  primary_cause = _classify_primary_cause(
      pc_zero_pods=pc_zero_pods,
      pc_nonzero_pods=pc_nonzero_pods,
      dcn_action_counts=dcn_action_counts,
      sc_halt_groups=sc_halt_groups,
      outlier_pods=sorted_outlier_pods,
      silent_pods=silent_pods,
  )

  distribution_summary = [
      {
          "sflag_group": group_name,
          "count": len(pods),
          "unique_pods_count": len(set(pod for pod in pods if pod)),
      }
      for group_name, pods in sorted(
          sflag_groups.items(), key=lambda item: len(item[1]), reverse=True
      )
  ]

  return {
      "execution_status": "SUCCESS" if logs else "NO_LOGS_FOUND",
      "total_logs_analyzed": len(logs),
      "unique_reporting_pods": len(reporting_pods),
      "silent_pods_count": len(silent_pods),
      "silent_pod_candidates": silent_pods,
      "outlier_pods_count": len(sorted_outlier_pods),
      "outlier_pods": sorted_outlier_pods,
      "outlier_groups": outlier_groups,
      "host_mappings": host_mappings,
      "pc_zero_pods": sorted(pc_zero_pods),
      "pc_nonzero_pods": sorted(pc_nonzero_pods),
      "dcn_action_counts": dict(dcn_action_counts),
      "sflag_distribution": distribution_summary,
      "primary_hang_cause": primary_cause if logs else "NO_SFLAG_LOGS",
  }


def _print_summary_fields_block(summary_block: dict[str, Any]) -> None:
  """Prints JSON summary and glue-runner SUMMARY REPORT (summary_fields)."""
  print("\n--- Summary JSON ---")
  print(json.dumps(summary_block, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  for key, value in summary_block.items():
    print(f"  {key}: {value}")
  print("---------------------------------------")


def _print_human_report(report: dict[str, Any]) -> None:
  """Prints a human-readable diagnostic report and JSON summary block."""
  print("\n--- sflag & SparseCore Timeout Distribution ---")
  for entry in report["sflag_distribution"]:
    print(
        f"  sflag: {entry['sflag_group']:<48} "
        f"Count: {entry['count']:<6} "
        f"(Unique Pods: {entry['unique_pods_count']})"
    )

  if report["pc_zero_pods"] or report["pc_nonzero_pods"]:
    print("\n--- Program Counter (PC) Outlier Signals ---")
    if report["pc_zero_pods"]:
      print(
          "  [DATA_INPUT_STALL indicator] Pods with pc == 0:"
          f" {report['pc_zero_pods']}"
      )
    if report["pc_nonzero_pods"]:
      pc_label = (
          "[BAD_TPU_CHIP indicator] "
          if report["pc_zero_pods"] or report["outlier_pods"]
          else ""
      )
      print(f"  {pc_label}Pods with pc != 0: {report['pc_nonzero_pods']}")

  if report["dcn_action_counts"]:
    print("\n--- Cross-Slice DCN Collective Action Signals ---")
    for action_name, count in sorted(report["dcn_action_counts"].items()):
      print(f"  {action_name}: {count}")

  print("\n--- Silent / Missing Pod Topology Check ---")
  if report["silent_pod_candidates"]:
    print(
        "  [!] Silent Culprit Candidates (missing from sflag timeout logs):"
        f" {report['silent_pod_candidates']}"
    )
  else:
    print("  No missing <slice>-<worker> pods detected among reporting slices.")

  print("\n--- Minority Outlier Diagnosis (< 20% of majority) ---")
  if not report["outlier_groups"]:
    print(
        "  No minority sflag outlier groups detected. If all reporting pods are"
        " uniform, inspect silent pod candidates above or run"
        " analyze_libtpu_timing.py to check per-launch_id enqueue skew."
    )
  else:
    for group in report["outlier_groups"]:
      print(
          f"  Detected outlier sflag: '{group['sflag_group']}'"
          f" ({group['occurrences']} occurrences)"
      )
      print(f"  Culprit Pods: {group['unique_pods']}")
      for pod_name in group["unique_pods"]:
        mapping = report["host_mappings"].get(pod_name, {})
        node_name = mapping.get("gke_node", "unknown")
        phys_machine = mapping.get("physical_machine")
        print(f"    Pod: {pod_name}")
        print(f"      GKE Node: {node_name}")
        if node_name != "unknown":
          if phys_machine:
            print(f"      Host Telemetry Output:\n{phys_machine}")
          else:
            print("      Host node_telemetry mapping unavailable.")

  summary_block = {
      "execution_status": report["execution_status"],
      "total_logs_analyzed": report["total_logs_analyzed"],
      "unique_reporting_pods": report["unique_reporting_pods"],
      "silent_pods_count": report["silent_pods_count"],
      "outlier_pods_count": report["outlier_pods_count"],
      "primary_hang_cause": report["primary_hang_cause"],
      "output_file": report.get("output_file"),
  }
  _print_summary_fields_block(summary_block)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Diagnose sflag and MegaScale collective hangs in TPU workloads."
      )
  )
  parser.add_argument(
      "workload_prefix", help="Workload name or prefix (e.g. my-tpu-workload)"
  )
  parser.add_argument(
      "--project",
      "--project_id",
      "--project-id",
      dest="project",
      required=True,
      help="GCP Project ID",
  )
  parser.add_argument(
      "--freshness",
      default="180d",
      help="Cloud Logging freshness window (default: 180d)",
  )
  parser.add_argument(
      "--limit",
      type=int,
      default=5000,
      help="Maximum log entries to fetch (default: 5000)",
  )
  parser.add_argument(
      "--expected_pods",
      "--expected-pods",
      dest="expected_pods",
      type=int,
      default=0,
      help=(
          "Expected worker pods per slice (0 = auto-infer from"
          " <slice>-<worker>)"
      ),
  )
  parser.add_argument(
      "--max_concurrent_workers",
      "--max-concurrent-workers",
      dest="max_concurrent_workers",
      type=int,
      default=10,
      help="Max concurrent workers for host resolution lookups (default: 10)",
  )
  parser.add_argument(
      "--output_file",
      "--output-file",
      dest="output_file",
      default=None,
      help="Optional path to write full JSON diagnostic report",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Print full diagnostic report as JSON",
  )
  args = parser.parse_args()

  if not args.json:
    print(
        f"Querying sflag timeout logs for {args.workload_prefix} in project"
        f" {args.project} (freshness={args.freshness}, order=ASC)..."
    )

  filter_str = (
      f'resource.labels.pod_name:"{args.workload_prefix}-" AND '
      '("Wait timeout on sflag" OR "Sflag wait timeout on op" OR '
      '"halted unexpectedly at tag:pc" OR "halted prematurely")'
  )
  logs = run_gcloud(
      filter_str=filter_str,
      project=args.project,
      limit=args.limit,
      freshness=args.freshness,
  )

  report = analyze_sflag_logs(
      logs=logs,
      expected_pods=args.expected_pods,
      max_concurrent_workers=args.max_concurrent_workers,
  )
  report["output_file"] = args.output_file

  if args.output_file:
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as out_f:
      json.dump(report, out_f, indent=2)

  if args.json:
    print(json.dumps(report, indent=2))
    return

  if not logs:
    print("No sflag timeout logs found.")
    summary_block = {
        "execution_status": report["execution_status"],
        "total_logs_analyzed": 0,
        "unique_reporting_pods": 0,
        "silent_pods_count": 0,
        "outlier_pods_count": 0,
        "primary_hang_cause": report["primary_hang_cause"],
        "output_file": args.output_file,
    }
    _print_summary_fields_block(summary_block)
    return

  print(
      f"Found {len(logs)} log entries. Analyzing sflags, ops, and pod"
      " topology..."
  )
  _print_human_report(report)


if __name__ == "__main__":
  main()

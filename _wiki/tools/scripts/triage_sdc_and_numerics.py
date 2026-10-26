#!/usr/bin/env python3
"""Deterministic XLA Silent Data Corruption (SDC) and Numerics Error Triage Tool.

Parses workload logs for XLA Silent Data Corruption (`XLA_SDC_CHECKER`,
`SDC Checker`, `checksum mismatch`), LLO runtime bounds check violations
(`BoundsCheck <ID>`), and `NaN` / `Inf` numerical divergence to classify
hardware defects vs compiler/kernel bugs vs application non-determinism.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

_POD_BRACKET_RE = re.compile(
    r"^\s*(?:\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*)?\[(?P<pod>[a-zA-Z0-9_.-]+)\]:"
)
_SDC_CHIP_HOST_RE = re.compile(
    r"\[XLA_SDC_CHECKER\]\s*chip\s*\[(?P<chip>[^\]]+)\]\s*host\s*\[(?P<host>[^\]]+)\]"
    r"(?:\s*::>\s*(?P<msg>[^\n]+))?",
    re.IGNORECASE,
)
_PHYSICAL_LOC_RE = re.compile(
    r"Physical location:\s*(?P<loc>[^.\n]+)", re.IGNORECASE
)
_CHECKSUM_MISMATCH_OP_RE = re.compile(
    r"checksum does not match\s*[:>]+\s*(?P<op>[a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)
_BOUNDS_CHECK_RE = re.compile(
    r"BoundsCheck\s+(?P<bc_id>\d+)(?:\s*\([^)]*\))?\s*\[(?P<deref>[^\]]+)\]"
    r"(?:\s*for\s*(?P<op>[^\n]+))?",
    re.IGNORECASE,
)
_SEQUENCER_TAG_PC_RE = re.compile(
    r"(?:TensorCoreSequencer|SparseCoreSequencer):\d+:0x[0-9a-fA-F]+"
)
_GKE_NODE_RE = re.compile(r"\bgke-[a-zA-Z0-9-]+-[0-9a-f]{4,}-[a-z0-9]{4}\b")
_HLO_OP_RE = re.compile(
    r"\b(?:fusion|all-gather|all-reduce|reduce-scatter|custom-call|convolution|"
    r"dot|dma\.[a-z0-9_]+)\.[0-9a-zA-Z_.-]+\b|\bdma\.[a-z0-9_]+\b"
)


def _add_unique(target: list[str], item: str) -> None:
  cleaned = item.strip()
  if cleaned and cleaned not in target:
    target.append(cleaned)


def triage_sdc_and_numerics_logs(
    log_text: str,
    workload_name: str = "",
    project_id: str = "",
) -> dict[str, Any]:
  """Analyzes log text for XLA SDC checksum mismatches, bounds checks, and NaNs."""
  lines = log_text.splitlines()
  culprit_hosts_or_pods: list[str] = []
  failing_operations: list[str] = []

  for line in lines:
    pm = _POD_BRACKET_RE.search(line)
    if pm:
      _add_unique(culprit_hosts_or_pods, pm.group("pod"))
    for nm in _GKE_NODE_RE.finditer(line):
      _add_unique(culprit_hosts_or_pods, nm.group(0))

  sdc_matches = list(_SDC_CHIP_HOST_RE.finditer(log_text))
  for m in sdc_matches:
    _add_unique(culprit_hosts_or_pods, m.group("chip"))
    _add_unique(culprit_hosts_or_pods, m.group("host"))

  for m in _PHYSICAL_LOC_RE.finditer(log_text):
    _add_unique(culprit_hosts_or_pods, m.group("loc"))

  for m in _CHECKSUM_MISMATCH_OP_RE.finditer(log_text):
    _add_unique(failing_operations, m.group("op"))

  bounds_matches = list(_BOUNDS_CHECK_RE.finditer(log_text))
  for m in bounds_matches:
    bc_label = f"BoundsCheck {m.group('bc_id')} [{m.group('deref')}]"
    _add_unique(failing_operations, bc_label)
    if m.group("op"):
      _add_unique(failing_operations, m.group("op"))

  for m in _SEQUENCER_TAG_PC_RE.finditer(log_text):
    _add_unique(failing_operations, m.group(0))

  for m in _HLO_OP_RE.finditer(log_text):
    _add_unique(failing_operations, m.group(0))

  lower_text = log_text.lower()
  has_sdc_checker = (
      "xla_sdc_checker" in lower_text
      or "sdc event" in lower_text
      or "output data checksum does not match" in lower_text
      or "checksum mismatch" in lower_text
  )
  has_input_mismatch = (
      "input data checksum does not match" in lower_text
      or bool(re.search(r"input_identical\s*[:=]\s*false", lower_text))
  )
  has_bounds_check = bool(bounds_matches) or "boundscheck" in lower_text
  has_nan_inf = (
      bool(re.search(r"\b(?:nan|inf)\b", lower_text))
      or "numerical divergence" in lower_text
      or "check_numerics" in lower_text
  )

  # Classify failure mode
  recommended_actions: list[str] = []
  if has_input_mismatch:
    failure_classification = "application_nondeterminism"
    root_cause = (
        "Input buffer checksum mismatch across repeated HLO executions"
        " (--xla_tpu_sdc_check_input_identical=true), indicating application"
        " input non-determinism or in-place input buffer aliasing/mutation."
    )
    recommended_actions.append("inspect-hlo-dumps")
  elif has_sdc_checker:
    failure_classification = "hardware_sdc"
    ops_str = (
        ", ".join(failing_operations[:4]) if failing_operations else "HLO op"
    )
    hosts_str = (
        ", ".join(culprit_hosts_or_pods[:4])
        if culprit_hosts_or_pods
        else "unknown host"
    )
    root_cause = (
        "Silent Data Corruption (SDC) on-device output checksum mismatch"
        f" detected by XLA SDC Checker ({ops_str}) localized to"
        f" {hosts_str}."
    )
    recommended_actions.extend([
        "isolate-node-via-rr-label",
        "inspect-hlo-dumps",
    ])
  elif has_bounds_check:
    failure_classification = "compiler_or_kernel_numerics"
    ops_str = (
        ", ".join(failing_operations[:4])
        if failing_operations
        else "LLO memory instruction"
    )
    root_cause = (
        "On-device execution halt caused by an LLO runtime bounds check"
        f" violation ({ops_str}) exceeding allocated buffer bounds."
    )
    recommended_actions.append("inspect-hlo-dumps")
  elif has_nan_inf:
    failure_classification = "compiler_or_kernel_numerics"
    ops_str = (
        ", ".join(failing_operations[:4])
        if failing_operations
        else "model forward/backward pass"
    )
    root_cause = (
        "Numerical divergence (NaN/Inf or precision drift) detected in"
        f" {ops_str}. Evaluate compiler fusion/lowering passes with hlo-diff"
        " and host-assisted verification."
    )
    recommended_actions.append("inspect-hlo-dumps")
  else:
    failure_classification = "inconclusive"
    root_cause = (
        "No XLA_SDC_CHECKER checksum mismatch, LLO BoundsCheck violation, or"
        " NaN/Inf divergence signature detected in the provided logs."
    )
    recommended_actions.append("query-workload-logs")

  return {
      "execution_status": (
          "success"
          if failure_classification != "inconclusive"
          else "no_signature_found"
      ),
      "workload_name": workload_name,
      "project_id": project_id,
      "failure_classification": failure_classification,
      "culprit_hosts_or_pods": culprit_hosts_or_pods,
      "failing_operations": failing_operations,
      "root_cause": root_cause,
      "recommended_actions": recommended_actions,
  }


def _fetch_logs_via_helper(
    workload_name: str,
    project_id: str,
    start_time: str,
    end_time: str,
) -> str:
  """Invokes pull_and_format_logs.py when inline/file logs are not supplied."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  helper = os.path.join(script_dir, "pull_and_format_logs.py")
  if not os.path.exists(helper) or not workload_name or not project_id:
    return ""
  out_tmp = os.path.join(
      os.path.dirname(os.path.dirname(script_dir)),
      "scratch",
      "sdc_numerics_queried_logs.txt",
  )
  os.makedirs(os.path.dirname(out_tmp), exist_ok=True)
  cmd = [
      sys.executable,
      helper,
      f"--workload_name={workload_name}",
      f"--project_id={project_id}",
      (
          '--filter_extra=("XLA_SDC_CHECKER" OR "SDC Checker" OR "checksum'
          ' mismatch" OR "BoundsCheck" OR "NaN" OR "Inf")'
      ),
      f"--output_file={out_tmp}",
  ]
  if start_time:
    cmd.append(f"--start_time={start_time}")
  if end_time:
    cmd.append(f"--end_time={end_time}")
  subprocess.run(cmd, check=False, capture_output=True, text=True)
  if os.path.exists(out_tmp):
    with open(out_tmp, "r", encoding="utf-8", errors="replace") as f:
      return f.read()
  return ""


def main() -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Triage XLA Silent Data Corruption (SDC), BoundsCheck, and Numerics"
          " logs."
      )
  )
  parser.add_argument(
      "--workload_name", default="", help="Target TPU workload name."
  )
  parser.add_argument("--project_id", default="", help="GCP Project ID.")
  parser.add_argument(
      "--logs_file",
      default="",
      help="Path to file containing queried workload logs.",
  )
  parser.add_argument(
      "--logs",
      default="",
      help="Raw or formatted log text passed inline.",
  )
  parser.add_argument(
      "--start_time", default="", help="Optional ISO-8601 start timestamp."
  )
  parser.add_argument(
      "--end_time", default="", help="Optional ISO-8601 end timestamp."
  )
  parser.add_argument(
      "--output_file",
      default="scratch/sdc_numerics_triage.json",
      help="Output JSON file path.",
  )
  args = parser.parse_args()

  log_text = ""
  if args.logs_file and os.path.exists(args.logs_file):
    with open(args.logs_file, "r", encoding="utf-8", errors="replace") as f:
      log_text = f.read()
  elif args.logs:
    log_text = args.logs
  else:
    default_logs = "scratch/queried_logs.txt"
    if os.path.exists(default_logs):
      with open(default_logs, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    else:
      log_text = _fetch_logs_via_helper(
          args.workload_name, args.project_id, args.start_time, args.end_time
      )

  result = triage_sdc_and_numerics_logs(
      log_text=log_text,
      workload_name=args.workload_name,
      project_id=args.project_id,
  )

  output_file = args.output_file or "scratch/sdc_numerics_triage.json"
  if output_file:
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)
    result["output_file"] = output_file

  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())

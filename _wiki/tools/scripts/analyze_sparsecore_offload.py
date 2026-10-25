#!/usr/bin/env python3
"""Deterministic SparseCore Offload, Memory, and Sync Failure Analyzer.

Parses workload logs for SparseCore (`[XLA:SPARSECORE]`, `SparseCoreSequencer`,
`SC0`, `SC1`, `sc_tpu.dma_wait`) failures, including SparseCore Mismatch
Detector divergences, SparseCore sequencer halts / `sflag` timeouts, and
SparseCore `smem`/`vmem`/HBM stack exhaustion.
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
_SC_CORE_RE = re.compile(
    r"\b(?:\d+x\d+x\d+_SC\d+|SC\d+|SparseCoreSequencer:\d+)\b"
)
_MISMATCH_OP_RE = re.compile(
    r"Found Mismatch in offloaded op\s+(?P<op>[a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)
_SC_SEQ_PC_RE = re.compile(
    r"SparseCoreSequencer:\d+:0x[0-9a-fA-F]+"
    r"(?:\s*\(from\s+SparseCoreSequencer:\d+:0x[0-9a-fA-F]+\))?"
)
_GKE_NODE_RE = re.compile(r"\bgke-[a-zA-Z0-9-]+-[0-9a-f]{4,}-[a-z0-9]{4}\b")


def _add_unique(target: list[str], item: str) -> None:
  cleaned = item.strip()
  if cleaned and cleaned not in target:
    target.append(cleaned)


def analyze_sparsecore_offload_logs(
    log_text: str,
    workload_name: str = "",
    project_id: str = "",
) -> dict[str, Any]:
  """Analyzes log text for SparseCore offload mismatch, halt, or OOM events."""
  lines = log_text.splitlines()
  culprit_sc_cores: list[str] = []
  culprit_nodes_or_pods: list[str] = []

  for line in lines:
    pm = _POD_BRACKET_RE.search(line)
    if pm:
      _add_unique(culprit_nodes_or_pods, pm.group("pod"))
    for nm in _GKE_NODE_RE.finditer(line):
      _add_unique(culprit_nodes_or_pods, nm.group(0))
    for cm in _SC_CORE_RE.finditer(line):
      _add_unique(culprit_sc_cores, cm.group(0))

  lower_text = log_text.lower()
  has_mismatch = (
      "found mismatch in offloaded op" in lower_text
      or "sparsecore mismatch detector" in lower_text
      or "sparse_core_mismatch_detector" in lower_text
      or (
          "[xla:sparsecore]" in lower_text
          and ("mismatch" in lower_text or "diverges" in lower_text)
      )
  )
  has_sc_oom = (
      "sparsecore" in lower_text
      or "sparse_core" in lower_text
      or "xla_sc_" in lower_text
  ) and (
      "out of memory" in lower_text
      or "resource_exhausted" in lower_text
      or "smem" in lower_text
      and "overflow" in lower_text
      or "vmem" in lower_text
      and "overflow" in lower_text
  )
  has_sc_halt_or_sflag = (
      "sparsecoresequencer" in lower_text
      or "sc_tpu.dma_wait" in lower_text
      or ("sparsecore" in lower_text and "sflag" in lower_text)
      or ("sparsecore" in lower_text and "boundscheck" in lower_text)
  )

  offending_op_or_pc = "unknown"
  mm_op = _MISMATCH_OP_RE.search(log_text)
  if mm_op:
    offending_op_or_pc = mm_op.group("op")
  else:
    seq_pc = _SC_SEQ_PC_RE.search(log_text)
    if seq_pc:
      offending_op_or_pc = seq_pc.group(0)
    elif "sc_tpu.dma_wait" in lower_text:
      offending_op_or_pc = "sc_tpu.dma_wait"

  if not culprit_sc_cores and (
      has_mismatch or has_sc_oom or has_sc_halt_or_sflag
  ):
    culprit_sc_cores.append("SC0")

  recommended_mitigations: list[str] = []
  if has_mismatch:
    sc_failure_mode = "offload_mismatch"
    root_cause = (
        "Numerical/functional divergence between SparseCore offloaded execution"
        f" and TensorCore reference baseline on op '{offending_op_or_pc}'"
        " detected by SparseCore Mismatch Detector (XLA compiler offload bug)."
    )
    recommended_mitigations.extend([
        (
            "Disable SparseCore offloading for the affected op (or set"
            " --xla_sc_enable_instruction_fusion=false) to execute on"
            " TensorCore as an immediate workaround."
        ),
        (
            "Narrow mismatch detection via"
            " --xla_tpu_sparse_core_mismatch_detector_on_specified_ops and"
            " capture HLO dumps (--xla_dump_to) to file an XLA compiler bug."
        ),
    ])
  elif has_sc_halt_or_sflag:
    sc_failure_mode = "sequencer_halt_or_sflag_timeout"
    root_cause = (
        "SparseCore sequencer halt or TensorCore-to-SparseCore sflag DMA"
        f" synchronization timeout at '{offending_op_or_pc}'."
    )
    recommended_mitigations.extend([
        (
            "Enable SparseCore log recorder (--xla_tpu_enable_sc_log_recorder="
            "true) and LLO bounds checking (--xla_jf_bounds_check=true)."
        ),
        (
            "If isolated to a single physical chip/node across retries, cordon"
            " and isolate the node via isolate-node-via-rr-label."
        ),
    ])
  elif has_sc_oom:
    sc_failure_mode = "sc_memory_oom"
    root_cause = (
        "SparseCore scratchpad (SMEM/VMEM) or SparseCore HBM stack memory"
        " exhaustion during embedding or collective offload compilation."
    )
    recommended_mitigations.extend([
        (
            "Tune --xla_sc_num_serialized_tables_to_optimize_hbm to serialize"
            " embedding table stack usage and lower"
            " maximum_parallel_iterations."
        ),
        (
            "Verify 32B (8-float) embedding dimension alignment and mod-shard"
            " large vocabulary tables across chips."
        ),
    ])
  else:
    sc_failure_mode = "unknown"
    root_cause = (
        "No SparseCore mismatch detector alert, SparseCoreSequencer halt, or"
        " SparseCore memory exhaustion signature found in the provided logs."
    )
    recommended_mitigations.append(
        "Re-query logs with SparseCore, SparseCoreSequencer, or"
        " [XLA:SPARSECORE] filters."
    )

  return {
      "execution_status": (
          "success" if sc_failure_mode != "unknown" else "no_sc_failure_found"
      ),
      "workload_name": workload_name,
      "project_id": project_id,
      "sc_failure_mode": sc_failure_mode,
      "culprit_sc_cores": culprit_sc_cores,
      "culprit_nodes_or_pods": culprit_nodes_or_pods,
      "offending_op_or_pc": offending_op_or_pc,
      "root_cause": root_cause,
      "recommended_mitigations": recommended_mitigations,
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
      "sparsecore_queried_logs.txt",
  )
  os.makedirs(os.path.dirname(out_tmp), exist_ok=True)
  cmd = [
      sys.executable,
      helper,
      f"--workload_name={workload_name}",
      f"--project_id={project_id}",
      (
          '--filter_extra=("SparseCore" OR "sparse_core" OR'
          ' "SparseCoreSequencer" OR "sc_tpu" OR "SC0" OR "SC1")'
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
          "Analyze SparseCore offload mismatch, sequencer halt, and OOM logs."
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
      default="scratch/sparsecore_offload_analysis.json",
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

  result = analyze_sparsecore_offload_logs(
      log_text=log_text,
      workload_name=args.workload_name,
      project_id=args.project_id,
  )

  output_file = args.output_file or "scratch/sparsecore_offload_analysis.json"
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

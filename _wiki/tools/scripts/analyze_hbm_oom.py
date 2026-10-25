#!/usr/bin/env python3
"""Deterministic HBM Out of Memory (Error E1000) and Host RAM OOM Analyzer.

Parses XLA TPU HBM OOM allocation summaries (`Out of memory in memory space hbm`
Error E1000, `Allocation (size=...) would exceed memory`), runtime HBM buffer
exhaustion, and Kubernetes container host RAM OOMs (`OOMKilled`, `exit code
137`) to classify the OOM category, extract peak memory demand and top offending
HLO operations, and generate targeted XLA/JAX mitigations.
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
_PEAK_MEM_RE = re.compile(
    r"(?:Peak memory usage|peak_memory_usage|Peak HBM usage)\s*[:=]\s*"
    r"(?P<val>[0-9.]+\s*(?:[GMKT]i?B|bytes|B))",
    re.IGNORECASE,
)
_TRIED_ALLOC_RE = re.compile(
    r"(?:tried to allocate|Allocation\s*\(size\s*=\s*)"
    r"(?P<alloc>[0-9.]+\s*(?:[GMKT]i?B|bytes|B)?|\d+)\)?\s*"
    r"(?:would exceed memory\s*\(size\s*=\s*(?P<limit>\d+)\)|"
    r"[^\n]*?(?:limit|capacity)\s*[:=]?\s*(?P<limit2>[0-9.]+\s*[GMKT]i?B))?",
    re.IGNORECASE,
)
_CATEGORY_BYTES_RE = {
    "temporaries": re.compile(
        r"(?:Temporaries|Temporary buffers|Activations)\s*[:=]\s*"
        r"(?P<val>[0-9.]+\s*(?:[GMKT]i?B|bytes|B))",
        re.IGNORECASE,
    ),
    "parameters": re.compile(
        r"(?:Arguments|Parameters|Weights|Input/Output arguments)\s*[:=]\s*"
        r"(?P<val>[0-9.]+\s*(?:[GMKT]i?B|bytes|B))",
        re.IGNORECASE,
    ),
    "outputs": re.compile(
        r"(?:Outputs|Output buffers)\s*[:=]\s*"
        r"(?P<val>[0-9.]+\s*(?:[GMKT]i?B|bytes|B))",
        re.IGNORECASE,
    ),
    "aliases": re.compile(
        r"(?:Aliases|Aliased buffers)\s*[:=]\s*"
        r"(?P<val>[0-9.]+\s*(?:[GMKT]i?B|bytes|B))",
        re.IGNORECASE,
    ),
    "sparsecore_hbm": re.compile(
        r"(?:SparseCore HBM|sparse_core[^\n:]*hbm|sc_hbm)\s*[:=]\s*"
        r"(?P<val>[0-9.]+\s*(?:[GMKT]i?B|bytes|B))",
        re.IGNORECASE,
    ),
}
_ALLOCATION_LINE_RE = re.compile(
    r"(#allocation\d+\s*\[[^\]]+\]|"
    r"(?:fusion|all-gather|all-reduce|reduce-scatter|custom-call|convolution|"
    r"dot|parameter|tuple|bitcast|reshape|transpose|broadcast|slice|"
    r"dynamic-slice|sort|while|conditional)\.[0-9a-zA-Z_.-]+"
    r"(?:\s*\[[^\]]+\])?|"
    r"tag\s*=\s*'[^']+')"
)
_SHAPE_RE = re.compile(
    r"\b(?:bf16|f32|f16|s32|u32|s8|u8|f8e4m3b11fnuz|f8e4m3fn|f8e5m2)"
    r"\[[0-9,]+\](?:\{[0-9,:T() ]+\})?"
)


def _parse_size_to_bytes(size_str: str) -> float:
  """Converts a human-readable or integer byte string into float bytes."""
  cleaned = size_str.strip()
  m = re.fullmatch(
      r"(?P<num>[0-9.]+)\s*(?P<unit>[GMKT]i?B|bytes|B)?",
      cleaned,
      re.IGNORECASE,
  )
  if not m:
    return 0.0
  num = float(m.group("num"))
  unit = (m.group("unit") or "B").upper()
  multipliers = {
      "B": 1.0,
      "BYTES": 1.0,
      "KB": 1e3,
      "KIB": 1024.0,
      "MB": 1e6,
      "MIB": 1024.0**2,
      "GB": 1e9,
      "GIB": 1024.0**3,
      "TB": 1e12,
      "TIB": 1024.0**4,
  }
  return num * multipliers.get(unit, 1.0)


def _format_bytes_gib(num_bytes: float) -> str:
  """Formats raw byte count as GiB string."""
  gib = num_bytes / (1024.0**3)
  return f"{gib:.2f}GiB ({int(num_bytes)} bytes)"


def analyze_hbm_oom_logs(
    log_text: str,
    workload_name: str = "",
    project_id: str = "",
) -> dict[str, Any]:
  """Analyzes log text for TPU HBM OOM (E1000) and Host RAM OOM signatures."""
  lines = log_text.splitlines()
  pods: list[str] = []
  for line in lines:
    pm = _POD_BRACKET_RE.search(line)
    if pm:
      pod = pm.group("pod")
      if pod not in pods:
        pods.append(pod)

  lower_text = log_text.lower()
  has_hbm_space = (
      "memory space hbm" in lower_text
      or "space=hbm" in lower_text
      or "space: hbm" in lower_text
      or "hbm oom" in lower_text
      or "error code 1000" in lower_text
      or "e1000" in lower_text
  )
  has_resource_exhausted = (
      "resource_exhausted" in lower_text
      or "out of memory" in lower_text
      or "would exceed memory" in lower_text
  )
  has_host_oom = (
      "oomkilled" in lower_text
      or "exit code 137" in lower_text
      or "memory cgroup out of memory" in lower_text
      or "host ram oom" in lower_text
  )
  has_compile_marker = (
      "xlacompile" in lower_text
      or "compiletimehbmoom" in lower_text
      or "jit compiling" in lower_text
      or "allocation summary" in lower_text
      or "hbm memory usageest" in lower_text
      or "compile time hbm oom" in lower_text
  )
  has_runtime_marker = (
      "xlaexecute" in lower_text
      or "runtime allocation" in lower_text
      or "runtime hbm oom" in lower_text
      or "fragmentation" in lower_text
  )
  has_sparsecore_oom = (
      "sparsecore" in lower_text
      or "sparse_core" in lower_text
      or "xla_sc_" in lower_text
  ) and (has_hbm_space or has_resource_exhausted)

  # 1. Classify OOM category
  if has_hbm_space or (has_resource_exhausted and not has_host_oom):
    if has_runtime_marker and not has_compile_marker:
      oom_category = "runtime_hbm_oom"
    else:
      oom_category = "compile_time_hbm_oom"
  elif has_host_oom:
    oom_category = "host_ram_oom"
  else:
    oom_category = "unknown"

  # 2. Extract peak HBM bytes & category breakdowns
  peak_hbm_bytes: str | None = None
  device_limit_bytes: str | None = None
  pm_match = _PEAK_MEM_RE.search(log_text)
  if pm_match:
    peak_hbm_bytes = pm_match.group("val").strip()

  ta_match = _TRIED_ALLOC_RE.search(log_text)
  if ta_match:
    raw_alloc = ta_match.group("alloc").strip()
    if raw_alloc.isdigit():
      peak_hbm_bytes = _format_bytes_gib(float(raw_alloc))
    elif not peak_hbm_bytes:
      peak_hbm_bytes = raw_alloc
    raw_limit = ta_match.group("limit") or ta_match.group("limit2")
    if raw_limit:
      raw_limit = raw_limit.strip()
      if raw_limit.isdigit():
        device_limit_bytes = _format_bytes_gib(float(raw_limit))
      else:
        device_limit_bytes = raw_limit

  allocation_breakdown: dict[str, str] = {}
  category_numeric: dict[str, float] = {}
  for cat, pat in _CATEGORY_BYTES_RE.items():
    m = pat.search(log_text)
    if m:
      val_str = m.group("val").strip()
      allocation_breakdown[cat] = val_str
      category_numeric[cat] = _parse_size_to_bytes(val_str)

  # 3. Determine dominant consumer
  if oom_category == "host_ram_oom":
    dominant_consumer = "host_ram"
  elif has_sparsecore_oom or (
      "sparsecore_hbm" in category_numeric
      and category_numeric["sparsecore_hbm"] > 0
  ):
    dominant_consumer = "sparsecore_hbm"
  elif category_numeric:
    dominant_consumer = max(category_numeric, key=lambda k: category_numeric[k])
  elif "output of " in lower_text or "temporaries" in lower_text:
    dominant_consumer = "temporaries"
  elif "parameter" in lower_text or "argument" in lower_text:
    dominant_consumer = "parameters"
  elif oom_category in ("compile_time_hbm_oom", "runtime_hbm_oom"):
    dominant_consumer = "temporaries"
  else:
    dominant_consumer = "unknown"

  # 4. Extract top offending ops and shapes
  top_offending_ops: list[str] = []
  for m in _ALLOCATION_LINE_RE.finditer(log_text):
    op_str = m.group(0).strip()
    if op_str not in top_offending_ops:
      top_offending_ops.append(op_str)
  for m in _SHAPE_RE.finditer(log_text):
    shape_str = m.group(0).strip()
    if shape_str not in top_offending_ops:
      top_offending_ops.append(shape_str)
  top_offending_ops = top_offending_ops[:10]

  # 5. Build root cause and recommended mitigations
  mitigations: list[str] = []
  if oom_category in ("compile_time_hbm_oom", "runtime_hbm_oom"):
    limit_clause = (
        f" against device HBM capacity {device_limit_bytes}"
        if device_limit_bytes
        else ""
    )
    peak_clause = (
        f" (requested/peak allocation: {peak_hbm_bytes}{limit_clause})"
        if peak_hbm_bytes
        else ""
    )
    ops_clause = (
        f" Top offending allocations/ops: {', '.join(top_offending_ops[:4])}."
        if top_offending_ops
        else ""
    )
    root_cause = (
        f"TPU High Bandwidth Memory (HBM) exhaustion ({oom_category}, Error"
        f" E1000){peak_clause} dominated by {dominant_consumer}.{ops_clause}"
    )
    if dominant_consumer == "sparsecore_hbm":
      mitigations.extend([
          (
              "Tune --xla_sc_num_serialized_tables_to_optimize_hbm to serialize"
              " SparseCore embedding table HBM stack usage."
          ),
          (
              "Lower maximum_parallel_iterations to reduce SparseCore HBM heap"
              " prefetching and verify 32B (8-float) embedding table alignment."
          ),
      ])
    elif dominant_consumer == "parameters":
      mitigations.extend([
          (
              "Enable buffer donation (donate_argnums in jax.jit) so input"
              " weight and optimizer state buffers are aliased in-place."
          ),
          (
              "Apply FSDP / Megatron-style parameter and optimizer state"
              " sharding across ICI/DCN and align tensor dimensions to 128x8."
          ),
      ])
    else:
      mitigations.extend([
          (
              "Reduce per-device microbatch size, apply tensor/SPMD sharding,"
              " or enable activation checkpointing (jax.checkpoint /"
              " jax.remat)."
          ),
          (
              "Enable compiler rematerialization and packing flags:"
              "  and"
              " --xla_tpu_rematerialization_algo=PEAK_PRIORITY."
          ),
      ])
    mitigations.append(
        "Capture HLO dumps (--xla_dump_to) and inspect peak live buffer"
        " lifespans in the XProf Memory Viewer."
    )
  elif oom_category == "host_ram_oom":
    root_cause = (
        "Kubernetes container terminated by Linux kernel OOM killer"
        " (OOMKilled / exit code 137) due to host CPU RAM exhaustion rather"
        " than TPU on-chip HBM exhaustion."
    )
    mitigations.extend([
        (
            "Reduce host-side data loader prefetch workers, in-memory dataset"
            " caching, or concurrent Orbax checkpoint staging buffers."
        ),
        (
            "Increase the Kubernetes pod memory limit or offload host-side"
            " tensor initialization directly to device shards."
        ),
    ])
  else:
    root_cause = (
        "No definitive TPU HBM OOM (Error E1000) or Host RAM OOM signature"
        " found in the provided log window."
    )
    mitigations.append(
        "Re-query workload logs around the pod termination timestamp with"
        " RESOURCE_EXHAUSTED, Allocation Summary, or OOMKilled filters."
    )

  return {
      "execution_status": (
          "success" if oom_category != "unknown" else "no_oom_found"
      ),
      "workload_name": workload_name,
      "project_id": project_id,
      "oom_category": oom_category,
      "peak_hbm_bytes": peak_hbm_bytes,
      "device_limit_bytes": device_limit_bytes,
      "dominant_consumer": dominant_consumer,
      "allocation_breakdown": allocation_breakdown,
      "top_offending_ops": top_offending_ops,
      "culprit_pods": pods,
      "root_cause": root_cause,
      "recommended_mitigations": mitigations,
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
      "hbm_oom_queried_logs.txt",
  )
  os.makedirs(os.path.dirname(out_tmp), exist_ok=True)
  cmd = [
      sys.executable,
      helper,
      f"--workload_name={workload_name}",
      f"--project_id={project_id}",
      (
          '--filter_extra=("Out of memory in memory space hbm" OR "Allocation'
          ' Summary" OR "OOMKilled" OR "RESOURCE_EXHAUSTED" OR "exit code 137")'
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
      description="Analyze XLA TPU HBM OOM (Error E1000) and Host RAM OOM logs."
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
      default="scratch/hbm_oom_analysis.json",
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

  result = analyze_hbm_oom_logs(
      log_text=log_text,
      workload_name=args.workload_name,
      project_id=args.project_id,
  )

  output_file = args.output_file or "scratch/hbm_oom_analysis.json"
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

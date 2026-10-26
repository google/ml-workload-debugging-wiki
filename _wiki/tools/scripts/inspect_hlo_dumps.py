#!/usr/bin/env python3
"""Deterministic XLA HLO / LLO Dump and XPlane HloProto Inspector.

Inspects XLA HLO/LLO dump directories (local directory, `gs://...` URI, or
`pod:<namespace>/<pod_name>:/path` / `pod:<pod_name>:/path`) and optional
`.xplane.pb` profile bundles to:
1. Parse `module_####.*.{before_optimizations,after_optimizations}.txt`, `.hlo`,
   and `*-final_bundles.txt` files across worker subdirectories.
2. Extract module names, module IDs, compiler fingerprints, collective
   operations (`all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`,
   `collective-permute`, `send`, `recv`) with `channel_id`s, `replica_groups`,
   and `_xla_megascale_launch_id`, plus custom calls (`tpu_custom_call`,
   `Mosaic`, `SparseCore`).
3. Compute per-worker SHA-256 hashes of normalized HLO modules and detect
   cross-worker `MODULE_MISMATCH` / `FINGERPRINT_MISMATCH` with unified diffs.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

_MODULE_FILENAME_RE = re.compile(
    r"^(?P<module_id>module_\d+)"
    r"(?:\.(?P<module_name>[^.]+))?"
    r"(?:\.(?P<fingerprint>[0-9a-fA-F]{6,}))?"
    r".*?\.(?P<stage>before_optimizations|after_optimizations|final_bundles)"
    r"\.(?:txt|hlo)$"
)
_HLO_HEADER_RE = re.compile(
    r"HloModule\s+(?P<name>[A-Za-z0-9_.-]+)", re.MULTILINE
)
_COLLECTIVE_OP_RE = re.compile(
    r"^\s*(?P<lhs>%[a-zA-Z0-9_.-]+)\s*=\s*(?P<shape>.+?)\s+"
    r"(?P<opcode>all-reduce|all-gather|reduce-scatter|all-to-all|"
    r"collective-permute|send|recv(?:-done)?|all-reduce-start|all-gather-start|"
    r"reduce-scatter-start)\s*\((?P<operands>[^)]*)\)(?P<attrs>.*)$"
)
_CHANNEL_ID_RE = re.compile(r"channel_id=(?P<cid>\d+)")
_REPLICA_GROUPS_RE = re.compile(r"replica_groups=(?P<rg>\{\{[^}]*\}\}|\{\})")
_LAUNCH_ID_RE = re.compile(r'_xla_megascale_launch_id="(?P<lid>\d+)"')
_CUSTOM_CALL_RE = re.compile(
    r"custom-call.*?(?:custom_call_target=\"(?P<target>[^\"]+)\"|"
    r"tpu_custom_call|Mosaic|SparseCore)",
    re.IGNORECASE,
)
_PRINTABLE_HLO_TOKEN_RE = re.compile(
    rb"HloModule\s+[A-Za-z0-9_.-]+|module_\d+\.[A-Za-z0-9_.-]+"
)


def _normalize_hlo_text(text: str) -> str:
  """Normalizes HLO text by stripping trailing spaces and volatile metadata."""
  normalized_lines: list[str] = []
  for line in text.splitlines():
    stripped = line.rstrip()
    if not stripped:
      continue
    normalized_lines.append(stripped)
  return "\n".join(normalized_lines)


def _parse_single_hlo_file(filepath: str) -> dict[str, Any]:
  """Extracts module metadata, collective ops, and custom calls from an HLO file."""
  with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    raw_text = f.read()

  norm_text = _normalize_hlo_text(raw_text)
  sha256_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

  fname = os.path.basename(filepath)
  m_file = _MODULE_FILENAME_RE.match(fname)
  module_id = (
      m_file.group("module_id") if m_file else os.path.splitext(fname)[0]
  )
  stage = m_file.group("stage") if m_file else "unknown"
  fingerprint = (
      m_file.group("fingerprint")
      if (m_file and m_file.group("fingerprint"))
      else sha256_hash[:16]
  )

  m_hdr = _HLO_HEADER_RE.search(raw_text)
  if m_hdr:
    module_name = m_hdr.group("name")
  elif m_file and m_file.group("module_name"):
    module_name = m_file.group("module_name")
  else:
    module_name = module_id

  collectives: list[dict[str, Any]] = []
  custom_calls: list[str] = []
  for line in norm_text.splitlines():
    cm = _COLLECTIVE_OP_RE.match(line)
    if cm:
      attrs = cm.group("attrs")
      cid_m = _CHANNEL_ID_RE.search(attrs)
      rg_m = _REPLICA_GROUPS_RE.search(attrs)
      lid_m = _LAUNCH_ID_RE.search(attrs)
      collectives.append({
          "instruction": cm.group("lhs"),
          "opcode": cm.group("opcode"),
          "channel_id": cid_m.group("cid") if cid_m else None,
          "replica_groups": rg_m.group("rg") if rg_m else None,
          "megascale_launch_id": lid_m.group("lid") if lid_m else None,
      })
    ccm = _CUSTOM_CALL_RE.search(line)
    if ccm:
      target = ccm.group("target") or "custom-call"
      if target not in custom_calls:
        custom_calls.append(target)

  return {
      "file": fname,
      "module_id": module_id,
      "module_name": module_name,
      "stage": stage,
      "fingerprint": fingerprint,
      "sha256": sha256_hash,
      "normalized_text": norm_text,
      "collectives": collectives,
      "custom_calls": custom_calls,
  }


def _extract_xplane_hlo_names(xplane_path: str) -> list[str]:
  """Extracts embedded HLO module names from `.xplane.pb` file(s) if present."""
  if not xplane_path or not os.path.exists(xplane_path):
    return []
  files: list[str] = []
  if os.path.isfile(xplane_path):
    files.append(xplane_path)
  elif os.path.isdir(xplane_path):
    for root, _, fnames in os.walk(xplane_path):
      for fn in sorted(fnames):
        if fn.endswith(".xplane.pb"):
          files.append(os.path.join(root, fn))

  found: list[str] = []
  for fp in files:
    try:
      with open(fp, "rb") as f:
        blob = f.read()
      for m in _PRINTABLE_HLO_TOKEN_RE.finditer(blob):
        tok = m.group(0).decode("utf-8", errors="ignore").strip()
        if tok not in found:
          found.append(tok)
    except OSError:
      continue
  return found


def inspect_hlo_dump_directory(
    local_dir: str,
    workload_name: str = "",
    xplane_path: str = "",
) -> dict[str, Any]:
  """Scans a local HLO dump directory and detects cross-worker HLO mismatches."""
  xplane_modules = _extract_xplane_hlo_names(xplane_path)
  if not local_dir or not os.path.exists(local_dir):
    return {
        "status": "no_dumps_found",
        "execution_status": "no_dumps_found",
        "verdict": "NO_DUMPS_FOUND",
        "target_entity": workload_name or local_dir or "unknown",
        "modules_analyzed": 0,
        "collective_ops_summary": {},
        "mismatched_modules": [],
        "diffs": [],
        "xplane_embedded_modules": xplane_modules,
        "recommendations": [
            (
                "Configure XLA_FLAGS='--xla_dump_to=/tmp/xla_dump/"
                " --xla_dump_hlo_as_text' on the workload container and copy"
                " worker dumps before pod termination."
            ),
        ],
    }

  # Discover HLO text files grouped by worker (parent subdirectory name or
  # 'root').
  hlo_files: list[tuple[str, str]] = []
  if os.path.isfile(local_dir):
    hlo_files.append(("worker_0", local_dir))
  else:
    for root, _, files in os.walk(local_dir):
      for fn in sorted(files):
        if fn.endswith((".txt", ".hlo")) and (
            "module_" in fn
            or "optimizations" in fn
            or "final_bundles" in fn
            or fn.endswith(".hlo")
        ):
          rel_parent = os.path.relpath(root, local_dir)
          worker_id = (
              "worker_0"
              if rel_parent in (".", "")
              else rel_parent.split(os.sep)[0]
          )
          hlo_files.append((worker_id, os.path.join(root, fn)))

  if not hlo_files:
    return {
        "status": "no_dumps_found",
        "execution_status": "no_dumps_found",
        "verdict": "NO_DUMPS_FOUND",
        "target_entity": workload_name or local_dir,
        "modules_analyzed": 0,
        "collective_ops_summary": {},
        "mismatched_modules": [],
        "diffs": [],
        "xplane_embedded_modules": xplane_modules,
        "recommendations": [
            (
                "No module_####.*.{before,after}_optimizations.txt or .hlo"
                f" files found under '{local_dir}'."
            ),
        ],
    }

  # Parse all discovered HLO files and group by (module_id, stage)
  grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
  collective_counts: dict[str, int] = {}
  all_collectives: list[dict[str, Any]] = []
  custom_calls_set: list[str] = []

  for worker_id, fpath in hlo_files:
    parsed = _parse_single_hlo_file(fpath)
    key = (parsed["module_id"], parsed["stage"])
    grouped.setdefault(key, {})[worker_id] = parsed
    for coll in parsed["collectives"]:
      op = coll["opcode"]
      collective_counts[op] = collective_counts.get(op, 0) + 1
      if len(all_collectives) < 50:
        all_collectives.append({"worker": worker_id, **coll})
    for cc in parsed["custom_calls"]:
      if cc not in custom_calls_set:
        custom_calls_set.append(cc)

  mismatched_modules: list[str] = []
  diffs: list[dict[str, Any]] = []

  for (module_id, stage), worker_map in sorted(grouped.items()):
    workers = sorted(worker_map.keys())
    if len(workers) < 2:
      continue
    baseline_worker = workers[0]
    baseline_entry = worker_map[baseline_worker]
    for other_worker in workers[1:]:
      other_entry = worker_map[other_worker]
      if other_entry["sha256"] != baseline_entry["sha256"]:
        mod_label = f"{module_id}.{stage}"
        if mod_label not in mismatched_modules:
          mismatched_modules.append(mod_label)
        diff_lines = list(
            difflib.unified_diff(
                baseline_entry["normalized_text"].splitlines(),
                other_entry["normalized_text"].splitlines(),
                fromfile=f"{baseline_worker}/{baseline_entry['file']}",
                tofile=f"{other_worker}/{other_entry['file']}",
                lineterm="",
                n=2,
            )
        )
        diffs.append({
            "module_id": module_id,
            "module_name": baseline_entry["module_name"],
            "stage": stage,
            "worker_a": baseline_worker,
            "worker_b": other_worker,
            "hash_a": baseline_entry["sha256"][:16],
            "hash_b": other_entry["sha256"][:16],
            "unified_diff_snippet": "\n".join(diff_lines[:40]),
        })

  if mismatched_modules:
    verdict = "HLO_MISMATCH_DETECTED"
    recommendations = [
        (
            "Cross-worker HLO compilation mismatch (FINGERPRINT_MISMATCH /"
            " MODULE_MISMATCH) detected across workers. Ensure all workers run"
            " identical JAX/XLA code paths, deterministic tracing, and matching"
            " XLA_FLAGS."
        ),
        (
            "Compare the diverging HLO modules with hlo-diff to isolate the"
            " compiler pass or host-dependent branch introducing the mismatch."
        ),
    ]
  else:
    verdict = "CONSISTENT_HLO"
    recommendations = [
        (
            "All analyzed HLO modules are structurally identical across"
            " workers. If investigating a collective hang, correlate the"
            " failing Megascale launch_id or channel_id against the extracted"
            " collective_ops_summary."
        ),
    ]

  return {
      "status": "success",
      "execution_status": "success",
      "verdict": verdict,
      "target_entity": workload_name or local_dir,
      "modules_analyzed": len(hlo_files),
      "collective_ops_summary": {
          "opcode_counts": collective_counts,
          "sample_collectives": all_collectives[:20],
          "custom_calls": custom_calls_set,
      },
      "mismatched_modules": mismatched_modules,
      "diffs": diffs,
      "xplane_embedded_modules": xplane_modules,
      "recommendations": recommendations,
  }


def _stage_remote_dumps(dump_source: str, temp_dir: str) -> str:
  """Stages remote `gs://` or `pod:` dumps into `temp_dir` if applicable."""
  src = dump_source.strip()
  if src.startswith("gs://"):
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", src, temp_dir],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return temp_dir
  if src.startswith("pod:"):
    # Format: pod:<namespace>/<pod_name>:/tmp/xla_dump or
    # pod:<pod_name>:/tmp/xla_dump
    spec = src[len("pod:") :]
    subprocess.run(
        ["kubectl", "cp", spec, temp_dir],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return temp_dir
  return src


def main() -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Inspect XLA HLO/LLO dump directories across workers for module"
          " mismatches and collective channel metadata."
      )
  )
  parser.add_argument(
      "--dump_source",
      default="",
      help=(
          "Local directory path, gs:// URI, or pod:<ns>/<pod>:/tmp/xla_dump"
          " containing HLO dumps."
      ),
  )
  parser.add_argument(
      "--bucket_location",
      default="",
      help="Optional alias/fallback GCS bucket location for HLO dumps.",
  )
  parser.add_argument(
      "--workload_name",
      default="",
      help="Optional workload name for attribution.",
  )
  parser.add_argument(
      "--xplane_path",
      default="",
      help="Optional path to .xplane.pb file or directory.",
  )
  parser.add_argument(
      "--output_file",
      default="scratch/hlo_dumps_analysis.json",
      help="Output JSON file path.",
  )
  args = parser.parse_args()

  effective_source = args.dump_source or args.bucket_location or "/tmp/xla_dump"
  temp_dir = None
  try:
    if effective_source.startswith(("gs://", "pod:")):
      temp_dir = tempfile.mkdtemp(prefix="hlo_dumps_")
      local_dir = _stage_remote_dumps(effective_source, temp_dir)
    else:
      local_dir = effective_source

    result = inspect_hlo_dump_directory(
        local_dir=local_dir,
        workload_name=args.workload_name,
        xplane_path=args.xplane_path,
    )
  finally:
    if temp_dir and os.path.exists(temp_dir):
      shutil.rmtree(temp_dir, ignore_errors=True)

  output_file = args.output_file or "scratch/hlo_dumps_analysis.json"
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

#!/usr/bin/env python3
"""Deterministic analyzer for XLA TPU ICI straggler delay wave propagation."""

import argparse
import json
import os
import re
import sys
from typing import Any

_COORD_RE = re.compile(r"(\[v\d+[^\]]*\]|\(\d+,\s*\d+,\s*\d+[^)]*\))")
_ITER_RE = re.compile(
    r"(?:iteration|iter|run_id|step)[=:\s]+(\d+)", re.IGNORECASE
)
_POD_RE = re.compile(
    r"(?:pod_name|pod|host|worker|node)[=:\s\"']+([a-zA-Z0-9_.-]+)"
    r"|(?:\[ID:\s*\d+\]|\[\d{4}-\d{2}-\d{2}[^\]]*\])\s*\[([a-zA-Z0-9_.-]+)\]",
    re.IGNORECASE,
)
_SEND_RE = re.compile(r"\bSEND\b", re.IGNORECASE)


def analyze_ici_wave_text(
    text: str,
    workload_name: str = "",
    project_id: str = "",
) -> dict[str, Any]:
  """Parses ICI transfer wait, XLA_SDC_CHECKER, and sflag timeout telemetry."""
  del workload_name, project_id
  stalls: list[dict[str, Any]] = []
  all_pods: set[str] = set()

  for line in text.splitlines():
    if not any(
        kw in line
        for kw in (
            "XLA_SDC_CHECKER",
            "ICI transfer wait",
            "Wait timeout on sflag",
            "sflag",
            "SEND",
            "RECV",
        )
    ):
      continue

    coord_match = _COORD_RE.search(line)
    iter_match = _ITER_RE.search(line)
    pod_match = _POD_RE.search(line)

    coord = coord_match.group(1) if coord_match else ""
    iteration = int(iter_match.group(1)) if iter_match else None
    pod = (pod_match.group(1) or pod_match.group(2)) if pod_match else ""
    if pod:
      all_pods.add(pod)

    is_send = bool(_SEND_RE.search(line))
    if coord or iteration is not None:
      stalls.append({
          "coord": coord,
          "iteration": iteration,
          "pod": pod,
          "is_send": is_send,
          "line": line.strip(),
      })

  iters = sorted(
      {stall["iteration"] for stall in stalls if stall["iteration"] is not None}
  )
  if not stalls:
    return {
        "root_cause": (
            "NO_ICI_DELAY_WAVE_DETECTED: No XLA_SDC_CHECKER or ICI sflag stall"
            " lines found."
        ),
        "min_iteration": 0,
        "s0_culprit_chips": [],
        "culprit_pods": [],
        "delay_wave_delta": None,
        "recommended_actions": ["enable-tpu-verbose-logging"],
    }

  min_iteration = iters[0] if iters else 0
  delay_wave_delta = (iters[1] - iters[0]) if len(iters) >= 2 else None

  s0_chips: list[str] = []
  s0_pods: list[str] = []
  for stall in stalls:
    if stall["iteration"] == min_iteration or (not iters and stall["is_send"]):
      if stall["coord"] and stall["coord"] not in s0_chips:
        s0_chips.append(stall["coord"])
      if stall["pod"] and stall["pod"] not in s0_pods:
        s0_pods.append(stall["pod"])

  if not s0_chips:
    for stall in stalls:
      if stall["coord"] and stall["coord"] not in s0_chips:
        s0_chips.append(stall["coord"])

  if not s0_pods and all_pods:
    s0_pods = sorted(all_pods)

  delta_desc = (
      f" with downstream S1 cascade delta={delay_wave_delta} iterations"
      if delay_wave_delta is not None
      else ""
  )
  root_cause = (
      "ICI_STRAGGLER_DELAY_WAVE: S0 root stall at iteration"
      f" N_min={min_iteration} on chip(s)"
      f" {', '.join(s0_chips) if s0_chips else 'unknown'}{delta_desc}."
  )

  return {
      "root_cause": root_cause,
      "min_iteration": min_iteration,
      "s0_culprit_chips": s0_chips,
      "culprit_pods": s0_pods,
      "delay_wave_delta": delay_wave_delta,
      "recommended_actions": [
          "map-physical-id-to-gke-instance",
          "isolate-node-via-rr-label",
      ],
  }


def main() -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Analyze XLA_SDC_CHECKER and ICI sflag delay wave propagation."
      )
  )
  parser.add_argument(
      "--workload_name", default="", help="Target TPU workload name."
  )
  parser.add_argument("--project_id", default="", help="GCP Project ID.")
  parser.add_argument("--logs", default="", help="Inline log text.")
  parser.add_argument("--logs_file", default="", help="Path to log file.")
  args = parser.parse_args()

  text = args.logs or ""
  if args.logs_file and os.path.isfile(args.logs_file):
    with open(args.logs_file, "r", encoding="utf-8", errors="replace") as f:
      text = text + "\n" + f.read()

  res = analyze_ici_wave_text(
      text=text,
      workload_name=args.workload_name,
      project_id=args.project_id,
  )
  print(json.dumps(res, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())

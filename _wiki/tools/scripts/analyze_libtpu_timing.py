#!/usr/bin/env python3
"""libtpu Timing and Outlier Analysis Tool.

Parses libtpu runtime logs (enqueueing program to last_finished_run_id updated)
across TPU workers, computes per-step device execution time and host dispatch
gaps,
and detects outlier workers/stragglers in distributed training jobs.
"""

import argparse
import collections
import datetime
import json
import math
import re
import sys
from typing import Any

_TS_PARSE_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)

ENQUEUE_RE = re.compile(
    r"(?:INFO\s+)?"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"(?:[\s\t,]+(?:\[(?:resource\.labels\.)?(?:containerName|pod_name|worker_id):\s*)?(?P<worker>[a-zA-Z0-9_.-]+)\]?)?"
    r"(?:[\s\t,]+(?P<instance_id>[a-zA-Z0-9_.-]+))?"
    r"[\s\t,]+(?P<tc>[0-9a-zA-Z_x]+_TC[0-9]+):\s*Enqueueing"
    r" program\s+(?P<prog_id>\d+):\s*"
    r"(?P<fingerprint>[0-9a-fA-F]+)\s+to continuation"
    r" queue\s*\(launch_id=(?P<launch_id>\d+)\)"
)

FINISH_RE = re.compile(
    r"(?:INFO\s+)?"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"(?:[\s\t,]+(?:\[(?:resource\.labels\.)?(?:containerName|pod_name|worker_id):\s*)?(?P<worker>[a-zA-Z0-9_.-]+)\]?)?"
    r"(?:[\s\t,]+(?P<instance_id>[a-zA-Z0-9_.-]+))?"
    r"[\s\t,]+(?P<tc>[0-9a-zA-Z_x]+_TC[0-9]+):\s*last_finished_run_id updated"
    r" to\s+(?P<prog_id>\d+)"
    r"(?:,\s*remaining programs=(?P<remaining>\d+))?"
)

APP_LOG_RE = re.compile(
    r"(?:INFO\s+)?"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"(?:[\s\t,]+(?:\[(?:resource\.labels\.)?(?:containerName|pod_name|worker_id):\s*)?(?P<worker>[a-zA-Z0-9_.-]+)\]?)?"
    r"(?:[\s\t,]+(?P<instance_id>[a-zA-Z0-9_.-]+))?"
    r"[\s\t,]+(?P<app_msg>(?:trainer|train|jax_trainer|process\s+\d+|app).*?step\s+(?P<step>\d+).*)"
)


def parse_timestamp(ts_str: str) -> float:
  """Parses ISO 8601 timestamp string into epoch seconds with float precision.

  Supports arbitrary subsecond precision (e.g. microseconds, nanoseconds) and
  timezone offsets (e.g. 'Z', '+00:00', '-07:00', '-0700').

  Args:
    ts_str: ISO 8601 formatted timestamp string.

  Returns:
    Timestamp as epoch seconds with float precision.
  """
  clean_ts = ts_str.strip()
  m = _TS_PARSE_RE.match(clean_ts)
  if not m:
    dt = datetime.datetime.fromisoformat(clean_ts)
    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()

  base = m.group("base").replace(" ", "T")
  frac_str = m.group("frac")
  frac_val = float("0." + frac_str) if frac_str else 0.0

  tz_str = m.group("tz")
  if not tz_str or tz_str == "Z":
    tz_norm = "+00:00"
  elif ":" not in tz_str:
    tz_norm = tz_str[:3] + ":" + tz_str[3:]
  else:
    tz_norm = tz_str

  dt = datetime.datetime.fromisoformat(base + tz_norm)
  return dt.timestamp() + frac_val


def compute_median(sorted_vals: list[float]) -> float:
  """Computes median from a sorted list of numbers."""
  if not sorted_vals:
    return 0.0
  n = len(sorted_vals)
  if n % 2 == 1:
    return float(sorted_vals[n // 2])
  return float(sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def compute_stats(values: list[float]) -> dict[str, float]:
  """Computes robust statistics (min, max, mean, median, stddev, IQR, MAD)."""
  if not values:
    return {
        "count": 0.0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "median": 0.0,
        "stddev": 0.0,
        "iqr": 0.0,
        "mad": 0.0,
    }
  sorted_vals = sorted(values)
  n = len(sorted_vals)
  min_val = sorted_vals[0]
  max_val = sorted_vals[-1]
  mean_val = sum(sorted_vals) / n
  median_val = compute_median(sorted_vals)

  variance = sum((x - mean_val) ** 2 for x in sorted_vals) / max(1, n - 1)
  stddev_val = math.sqrt(variance)

  q1 = sorted_vals[n // 4]
  q3 = sorted_vals[(3 * n) // 4]
  iqr_val = q3 - q1

  # Median Absolute Deviation (MAD)
  abs_devs = sorted([abs(x - median_val) for x in sorted_vals])
  mad_val = compute_median(abs_devs)

  return {
      "count": float(n),
      "min": min_val,
      "max": max_val,
      "mean": mean_val,
      "median": median_val,
      "stddev": stddev_val,
      "iqr": iqr_val,
      "mad": mad_val,
  }


def is_outlier_value(
    val: float,
    stats: dict[str, float],
    min_abs_diff: float = 0.15,
    ratio_threshold: float = 2.0,
) -> bool:
  """Determines if a value is an outlier using MAD, IQR, or ratio-to-median."""
  if stats["count"] < 2:
    return False

  median = stats["median"]
  diff = val - median
  if diff <= 0:
    return False

  if diff < min_abs_diff:
    return False

  # Ratio check
  if median > 0.001 and (val / median) >= ratio_threshold:
    return True

  # MAD / modified z-score check
  if stats["mad"] > 0.001:
    mod_z = 0.6745 * diff / stats["mad"]
    if mod_z >= 3.0:
      return True

  # Standard IQR check
  if stats["iqr"] > 0.001 and diff >= 1.5 * stats["iqr"]:
    return True

  return False


def analyze_logs(log_lines: list[str]) -> dict[str, Any]:
  """Parses log lines and groups executions per worker, TC, and step."""
  raw_events = []
  app_events = []

  for line_idx, line in enumerate(log_lines):
    line = line.strip()
    if not line:
      continue

    # Support JSON lines from GCS sinks or Cloud Logging JSON exports
    worker_hint = None
    ts_hint = None
    if line.startswith("{") and line.endswith("}"):
      try:
        j_entry = json.loads(line)
        if "timestamp" in j_entry:
          ts_hint = j_entry["timestamp"]
        if "resource" in j_entry and isinstance(j_entry["resource"], dict):
          labels = j_entry["resource"].get("labels", {})
          worker_hint = (
              labels.get("container_name")
              or labels.get("pod_name")
              or labels.get("instance_id")
          )
        if "textPayload" in j_entry:
          line = j_entry["textPayload"]
        elif "jsonPayload" in j_entry and isinstance(
            j_entry["jsonPayload"], dict
        ):
          line = j_entry["jsonPayload"].get(
              "message", str(j_entry["jsonPayload"])
          )
      except (ValueError, TypeError, KeyError):
        pass

    m_enq = ENQUEUE_RE.search(line)
    if m_enq:
      raw_events.append({
          "type": "enqueue",
          "ts": parse_timestamp(ts_hint or m_enq.group("timestamp")),
          "ts_raw": ts_hint or m_enq.group("timestamp"),
          "worker": worker_hint or m_enq.group("worker") or "worker-default",
          "instance_id": (
              m_enq.group("instance_id")
              if "instance_id" in m_enq.groupdict()
              else None
          ),
          "tc": m_enq.group("tc"),
          "prog_id": int(m_enq.group("prog_id")),
          "fingerprint": m_enq.group("fingerprint"),
          "launch_id": int(m_enq.group("launch_id")),
          "line_idx": line_idx,
      })
      continue

    m_fin = FINISH_RE.search(line)
    if m_fin:
      raw_events.append({
          "type": "finish",
          "ts": parse_timestamp(ts_hint or m_fin.group("timestamp")),
          "ts_raw": ts_hint or m_fin.group("timestamp"),
          "worker": worker_hint or m_fin.group("worker") or "worker-default",
          "instance_id": (
              m_fin.group("instance_id")
              if "instance_id" in m_fin.groupdict()
              else None
          ),
          "tc": m_fin.group("tc"),
          "prog_id": int(m_fin.group("prog_id")),
          "remaining": int(m_fin.group("remaining") or 0),
          "line_idx": line_idx,
      })
      continue

    m_app = APP_LOG_RE.search(line)
    if m_app:
      app_events.append({
          "ts": parse_timestamp(m_app.group("timestamp")),
          "worker": m_app.group("worker") or "worker-default",
          "step": int(m_app.group("step")),
          "msg": m_app.group("app_msg"),
      })

  worker_tc_runs = collections.defaultdict(
      lambda: collections.defaultdict(dict)
  )

  for ev in raw_events:
    w = ev["worker"]
    tc = ev["tc"]
    p = ev["prog_id"]
    if p not in worker_tc_runs[w][tc]:
      worker_tc_runs[w][tc][p] = {}
    worker_tc_runs[w][tc][p][ev["type"]] = ev

  worker_cycles = collections.defaultdict(list)

  for w, tcs in worker_tc_runs.items():
    tc_ordered_runs = {}
    for tc, p_map in tcs.items():
      sorted_progs = sorted(
          [
              (p, d)
              for p, d in p_map.items()
              if "enqueue" in d and "finish" in d
          ],
          key=lambda item: item[1]["enqueue"]["ts"],
      )
      tc_ordered_runs[tc] = sorted_progs

    if not tc_ordered_runs:
      continue

    num_cycles = min(len(runs) for runs in tc_ordered_runs.values())
    for cycle_idx in range(num_cycles):
      enq_times = [
          tc_ordered_runs[tc][cycle_idx][1]["enqueue"]["ts"]
          for tc in tc_ordered_runs
      ]
      fin_times = [
          tc_ordered_runs[tc][cycle_idx][1]["finish"]["ts"]
          for tc in tc_ordered_runs
      ]
      prog_ids = [tc_ordered_runs[tc][cycle_idx][0] for tc in tc_ordered_runs]
      fps = {
          tc_ordered_runs[tc][cycle_idx][1]["enqueue"]["fingerprint"]
          for tc in tc_ordered_runs
      }
      launch_ids = {
          tc_ordered_runs[tc][cycle_idx][1]["enqueue"]["launch_id"]
          for tc in tc_ordered_runs
      }

      t_enq_min = min(enq_times)
      t_enq_max = max(enq_times)
      t_fin_min = min(fin_times)
      t_fin_max = max(fin_times)

      worker_cycles[w].append({
          "cycle_index": cycle_idx,
          "prog_ids": prog_ids,
          "t_enqueue_min": t_enq_min,
          "t_enqueue_max": t_enq_max,
          "t_finish_min": t_fin_min,
          "t_finish_max": t_fin_max,
          "exec_duration": t_fin_max - t_enq_min,
          "tc_enqueue_skew_ms": (t_enq_max - t_enq_min) * 1000.0,
          "tc_finish_skew_ms": (t_fin_max - t_fin_min) * 1000.0,
          "fingerprints": list(fps),
          "launch_ids": list(launch_ids),
          "num_tcs": len(tc_ordered_runs),
      })

  for w, cycles in worker_cycles.items():
    for i in range(len(cycles)):
      if i > 0:
        prev_finish = cycles[i - 1]["t_finish_max"]
        curr_enqueue = cycles[i]["t_enqueue_min"]
        cycles[i]["dispatch_gap"] = max(0.0, curr_enqueue - prev_finish)
      else:
        cycles[i]["dispatch_gap"] = None

  all_cycle_indices = set()
  for cycles in worker_cycles.values():
    for c in cycles:
      all_cycle_indices.add(c["cycle_index"])

  cycle_reports = []
  for c_idx in sorted(all_cycle_indices):
    workers_in_cycle = {}
    for w, cycles in worker_cycles.items():
      if c_idx < len(cycles):
        workers_in_cycle[w] = cycles[c_idx]

    if not workers_in_cycle:
      continue

    exec_durations = [c["exec_duration"] for c in workers_in_cycle.values()]
    dispatch_gaps = [
        c["dispatch_gap"]
        for c in workers_in_cycle.values()
        if c["dispatch_gap"] is not None
    ]
    tc_skews = [c["tc_enqueue_skew_ms"] for c in workers_in_cycle.values()]

    exec_stats = compute_stats(exec_durations)
    dispatch_stats = compute_stats(dispatch_gaps)
    skew_stats = compute_stats(tc_skews)

    all_fps = set()
    for c in workers_in_cycle.values():
      all_fps.update(c["fingerprints"])

    outliers = []
    for w, c in workers_in_cycle.items():
      reasons = []

      # Check host dispatch gap outlier (host/straggler before enqueue)
      if c["dispatch_gap"] is not None and is_outlier_value(
          c["dispatch_gap"],
          dispatch_stats,
          min_abs_diff=0.2,
          ratio_threshold=2.5,
      ):
        reasons.append(
            "Elevated host dispatch gap (Host Straggler):"
            f" {c['dispatch_gap']*1000.0:.1f}ms vs median"
            f" {dispatch_stats['median']*1000.0:.1f}ms"
        )

      # Check device execution outlier (slow accelerator execution)
      if is_outlier_value(
          c["exec_duration"], exec_stats, min_abs_diff=0.3, ratio_threshold=1.5
      ):
        reasons.append(
            "Elevated device execution duration (Device Straggler):"
            f" {c['exec_duration']:.3f}s vs median"
            f" {exec_stats['median']:.3f}s"
        )

      # Check TC skew outlier within host
      if c["tc_enqueue_skew_ms"] > 100.0:
        reasons.append(
            f"High intra-host TC enqueue skew: {c['tc_enqueue_skew_ms']:.1f}ms"
        )

      # Check fingerprint mismatch
      if len(all_fps) > 1 and len(c["fingerprints"]) == 1:
        reasons.append(
            f"Fingerprint divergence: {c['fingerprints'][0]} differs from peers"
        )

      if reasons:
        outliers.append({
            "worker": w,
            "reasons": reasons,
            "exec_duration": c["exec_duration"],
            "dispatch_gap": c["dispatch_gap"],
            "tc_skew_ms": c["tc_enqueue_skew_ms"],
            "fingerprints": c["fingerprints"],
        })

    cycle_reports.append({
        "cycle_index": c_idx,
        "num_workers": len(workers_in_cycle),
        "fingerprints": list(all_fps),
        "exec_duration_stats": exec_stats,
        "dispatch_gap_stats": dispatch_stats,
        "tc_enqueue_skew_stats": skew_stats,
        "workers": workers_in_cycle,
        "outliers": outliers,
    })

  # Compute Launch Enqueue Timing Reports (per launch_id)
  enqueue_events = [e for e in raw_events if e["type"] == "enqueue"]
  launches = collections.defaultdict(list)
  for e in enqueue_events:
    launches[e["launch_id"]].append(e)

  launch_reports = []
  sorted_launch_ids = sorted(launches.keys())
  prev_launch_min_ts = None

  for l_idx, l_id in enumerate(sorted_launch_ids):
    evs = launches[l_id]
    timestamps = [e["ts"] for e in evs]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    sorted_ts = sorted(timestamps)
    median_ts = sorted_ts[len(sorted_ts) // 2]
    p90_ts = sorted_ts[int(len(sorted_ts) * 0.90)]
    p99_ts = sorted_ts[int(len(sorted_ts) * 0.99)]

    fps = list(set(e["fingerprint"] for e in evs))
    tcs = set(e["tc"] for e in evs)

    pod_events = collections.defaultdict(list)
    for e in evs:
      pod_events[e["worker"]].append(e)

    pod_summaries = []
    for w, p_evs in pod_events.items():
      p_min = min(e["ts"] for e in p_evs)
      p_max = max(e["ts"] for e in p_evs)
      inst_ids = list(
          set(e["instance_id"] for e in p_evs if e.get("instance_id"))
      )
      pod_summaries.append({
          "worker": w,
          "instance_id": inst_ids[0] if inst_ids else None,
          "first_enqueue_ts": p_min,
          "last_enqueue_ts": p_max,
          "delay_from_cluster_start_ms": (p_min - min_ts) * 1000.0,
          "intra_pod_tc_skew_ms": (p_max - p_min) * 1000.0,
          "num_tcs": len(set(e["tc"] for e in p_evs)),
      })

    pod_summaries.sort(
        key=lambda x: x["delay_from_cluster_start_ms"], reverse=True
    )

    outliers = []
    for p in pod_summaries:
      if p["delay_from_cluster_start_ms"] > 500.0 or (
          p["delay_from_cluster_start_ms"] > 100.0
          and p["delay_from_cluster_start_ms"]
          > (median_ts - min_ts) * 1000.0 + 100.0
      ):
        outliers.append(p)

    step_duration_from_prev = None
    if prev_launch_min_ts is not None:
      step_duration_from_prev = min_ts - prev_launch_min_ts
    prev_launch_min_ts = min_ts

    launch_reports.append({
        "launch_id": l_id,
        "step_index": l_idx + 1,
        "num_enqueues": len(evs),
        "num_unique_tcs": len(tcs),
        "num_workers": len(pod_events),
        "fingerprints": fps,
        "t_enqueue_min": min_ts,
        "t_enqueue_max": max_ts,
        "total_enqueue_skew_ms": (max_ts - min_ts) * 1000.0,
        "median_offset_ms": (median_ts - min_ts) * 1000.0,
        "p90_offset_ms": (p90_ts - min_ts) * 1000.0,
        "p99_offset_ms": (p99_ts - min_ts) * 1000.0,
        "step_duration_sec": step_duration_from_prev,
        "top_outliers": outliers[:10],
        "all_pods_count": len(pod_events),
    })

  total_outlier_workers = set()
  for c in cycle_reports:
    for o in c.get("outliers", []):
      total_outlier_workers.add(o["worker"])
  for l in launch_reports:
    for o in l.get("top_outliers", []):
      total_outlier_workers.add(o["worker"])

  return {
      "total_raw_events": len(raw_events),
      "total_app_events": len(app_events),
      "total_workers": max(
          len(worker_cycles), len(set(e["worker"] for e in raw_events))
      ),
      "total_cycles": len(cycle_reports),
      "outlier_workers_count": len(total_outlier_workers),
      "cycle_reports": cycle_reports,
      "launch_reports": launch_reports,
  }


def print_report(results: dict[str, Any], as_json: bool = False) -> None:
  """Outputs structured analysis report."""
  if as_json:
    print(json.dumps(results, indent=2))
    return

  print("=" * 80)
  print(" LIBTPU PROGRAM ENQUEUE & EXECUTION TIMING ANALYSIS REPORT")
  print("=" * 80)
  print(f"Total Log Events Parsed: {results['total_raw_events']}")
  print(f"Total Workers Tracked:   {results['total_workers']}")
  print(f"Total Launch Batches:    {len(results.get('launch_reports', []))}")
  print(f"Total Execution Cycles:  {len(results['cycle_reports'])}")
  print()

  if results.get("launch_reports"):
    print("=" * 80)
    print(" LAUNCH ENQUEUE ARRIVAL & SKEW ANALYSIS (Across All Workers)")
    print("=" * 80)
    for lr in results["launch_reports"]:
      step_str = f"Step #{lr['step_index']}"
      if lr["step_duration_sec"] is not None:
        step_str += f" (Inter-Step Elapsed: {lr['step_duration_sec']:.3f}s)"
      print(f"\n--- Launch ID: {lr['launch_id']} [{step_str}] ---")
      print(
          f"Workers / Cores Tracked: {lr['num_workers']} pods /"
          f" {lr['num_unique_tcs']} TensorCores"
      )
      print(f"HLO Fingerprint:         {', '.join(lr['fingerprints'])}")
      print(
          "Enqueue Arrival Spread:  Max Skew:"
          f" {lr['total_enqueue_skew_ms']:.2f}ms | Median Offset:"
          f" {lr['median_offset_ms']:.2f}ms | P90: {lr['p90_offset_ms']:.2f}ms"
          f" | P99: {lr['p99_offset_ms']:.2f}ms"
      )

      if lr["top_outliers"]:
        print("  [!] LATE-ENQUEUE STRAGGLER PODS DETECTED:")
        for rank, out in enumerate(lr["top_outliers"], 1):
          delay_sec = out["delay_from_cluster_start_ms"] / 1000.0
          inst_str = (
              f" [GKE Instance/Node: {out['instance_id']}]"
              if out.get("instance_id")
              else ""
          )
          print(
              f"    #{rank} Worker: {out['worker']}{inst_str}\n        Delay vs"
              f" Cluster Start: +{out['delay_from_cluster_start_ms']:.2f}ms"
              f" (+{delay_sec:.3f}s)\n        Intra-Pod TC Skew:     "
              f" {out['intra_pod_tc_skew_ms']:.2f}ms"
          )
      else:
        print(
            "  [✓] Synchronous Enqueue: All workers enqueued within expected"
            " threshold."
        )

  if results.get("cycle_reports"):
    print("\n" + "=" * 80)
    print(" PAIRED DEVICE EXECUTION & DISPATCH GAP CYCLES")
    print("=" * 80)
    for report in results["cycle_reports"]:
      c_idx = report["cycle_index"]
      e_stats = report["exec_duration_stats"]
      d_stats = report["dispatch_gap_stats"]
      s_stats = report["tc_enqueue_skew_stats"]

      print(
          f"\n--- Execution Cycle #{c_idx + 1} (Workers:"
          f" {report['num_workers']}) ---"
      )
      print(f"HLO Fingerprint(s): {', '.join(report['fingerprints'])}")
      print(
          "Device Execution Duration (T_exec):     Median:"
          f" {e_stats['median']:.3f}s | Min: {e_stats['min']:.3f}s | Max:"
          f" {e_stats['max']:.3f}s | StdDev: {e_stats['stddev']:.3f}s"
      )
      if d_stats["count"] > 0:
        print(
            "Host Dispatch Gap (T_dispatch_gap):     Median:"
            f" {d_stats['median']*1000:.1f}ms | Min:"
            f" {d_stats['min']*1000:.1f}ms | Max: {d_stats['max']*1000:.1f}ms |"
            f" StdDev: {d_stats['stddev']*1000:.1f}ms"
        )
      print(
          "Intra-Host TC Enqueue Skew:             Median:"
          f" {s_stats['median']:.2f}ms | Max: {s_stats['max']:.2f}ms"
      )

      if report["outliers"]:
        print("  [!] OUTLIERS DETECTED:")
        for out in report["outliers"]:
          print(f"    - Worker: {out['worker']}")
          for r in out["reasons"]:
            print(f"        * {r}")
      else:
        print(
            "  [✓] No timing outliers or fingerprint mismatches detected across"
            " workers."
        )

  print("\n" + "=" * 80)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Analyze libtpu program enqueue and finish logs across TPU workers."
      )
  )
  parser.add_argument(
      "logfile",
      nargs="?",
      help="Path to file containing libtpu logs (reads stdin if omitted)",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Output analysis report as JSON",
  )
  args = parser.parse_args()

  if args.logfile:
    with open(args.logfile, "r", encoding="utf-8") as f:
      lines = f.readlines()
  else:
    if sys.stdin.isatty():
      print(
          "Error: No logfile specified and no input piped via stdin.",
          file=sys.stderr,
      )
      sys.exit(1)
    lines = sys.stdin.readlines()

  results = analyze_logs(lines)
  print_report(results, as_json=args.json)


if __name__ == "__main__":
  main()

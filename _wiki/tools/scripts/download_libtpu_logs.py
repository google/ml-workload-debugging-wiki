#!/usr/bin/env python3
"""Batch Downloader for libtpu Program Lifecycle Logs.

Queries Google Cloud Logging in targeted slice or worker batches to retrieve
libtpu program enqueue and finish events across multi-slice TPU clusters without
hitting unbounded query timeouts or API throttles.
"""

import argparse
import os
import subprocess
import time


def parse_args() -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Download libtpu program lifecycle logs across TPU cluster slices."
      )
  )
  parser.add_argument(
      "--project",
      required=True,
      help="GCP Project ID hosting the TPU workload",
  )
  parser.add_argument(
      "--cluster",
      default="",
      help="GKE cluster name (optional filter)",
  )
  parser.add_argument(
      "--namespace",
      default="default",
      help="Kubernetes namespace (default: default)",
  )
  parser.add_argument(
      "--container",
      default="",
      help="Container name filter (e.g. trainer or server container)",
  )
  parser.add_argument(
      "--pod-prefix",
      default="job-",
      help=(
          "Pod name prefix pattern matching slice/job worker naming convention"
      ),
  )
  parser.add_argument(
      "--start-time",
      required=True,
      help="Start timestamp (ISO 8601, e.g. 2026-08-12T03:22:00-07:00)",
  )
  parser.add_argument(
      "--end-time",
      required=True,
      help="End timestamp (ISO 8601, e.g. 2026-08-12T03:22:30-07:00)",
  )
  parser.add_argument(
      "--num-slices",
      type=int,
      default=4,
      help="Total number of TPU slices in the workload (default: 4)",
  )
  parser.add_argument(
      "--events",
      choices=["enqueue", "finish", "all"],
      default="all",
      help=(
          "Log event types to capture: 'enqueue', 'finish', or 'all' (default:"
          " all)"
      ),
  )
  parser.add_argument(
      "--output",
      default="workload_libtpu_logs.txt",
      help=(
          "Output file path to save downloaded logs (default:"
          " workload_libtpu_logs.txt)"
      ),
  )
  return parser.parse_args()


def build_log_filter(args: argparse.Namespace, slice_id: int) -> str:
  """Builds a Cloud Logging query filter for a specific slice."""
  clauses = [
      'resource.type="k8s_container"',
      f'resource.labels.namespace_name="{args.namespace}"',
  ]
  if args.cluster:
    clauses.append(f'resource.labels.cluster_name="{args.cluster}"')
  if args.container:
    clauses.append(f'resource.labels.container_name="{args.container}"')

  # Target specific slice pods
  clauses.append(f'resource.labels.pod_name:"{args.pod_prefix}{slice_id}-"')

  # Target libtpu driver source
  clauses.append('sourceLocation.file="real_program_continuator.cc"')

  # Event filtering
  if args.events == "enqueue":
    clauses.append('jsonPayload.message:"Enqueueing program"')
  elif args.events == "finish":
    clauses.append('jsonPayload.message:"last_finished_run_id updated"')
  else:
    clauses.append(
        '(jsonPayload.message:"Enqueueing program" OR'
        ' jsonPayload.message:"last_finished_run_id updated")'
    )

  clauses.append(f'timestamp >= "{args.start_time}"')
  clauses.append(f'timestamp <= "{args.end_time}"')

  return "\n".join(clauses)


def main() -> None:
  args = parse_args()

  # Initialize / clear output file
  out_dir = os.path.dirname(args.output)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  open(args.output, "w", encoding="utf-8").close()

  print("=" * 75)
  print(" LIBTPU BATCH LOG DOWNLOADER")
  print("=" * 75)
  print(f"Project:     {args.project}")
  print(f"Time Window: {args.start_time} -> {args.end_time}")
  print(f"Slices:      {args.num_slices} slice(s)")
  print(f"Events:      {args.events}")
  print(f"Output File: {args.output}")
  print("=" * 75)
  print()

  total_logs = 0
  start_all = time.time()

  for slice_id in range(args.num_slices):
    query = build_log_filter(args, slice_id)
    print(
        f"[{slice_id + 1}/{args.num_slices}] Querying Slice {slice_id}...",
        end="",
        flush=True,
    )

    t0 = time.time()
    cmd = [
        "gcloud",
        "logging",
        "read",
        query,
        f"--project={args.project}",
        (
            "--format=value(timestamp, resource.labels.pod_name,"
            " resource.labels.instance_id, jsonPayload.message)"
        ),
        "--order=asc",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - t0

    if res.returncode != 0:
      print(f" -> [Error] {res.stderr.strip()}")
      continue

    lines = [l for l in res.stdout.split("\n") if l.strip()]
    with open(args.output, "a", encoding="utf-8") as f:
      for line in lines:
        f.write(line + "\n")

    total_logs += len(lines)
    print(
        f" -> {len(lines):,} logs retrieved in {elapsed:.2f}s | Cumulative:"
        f" {total_logs:,}"
    )

  total_duration = time.time() - start_all
  print()
  print("=" * 75)
  print(
      f"[Done] Successfully downloaded {total_logs:,} logs from"
      f" {args.num_slices} slices in {total_duration:.1f}s."
  )
  print(f"File saved to: {args.output}")
  print(
      "Analyze with:  python3 tools/scripts/analyze_libtpu_timing.py"
      f" {args.output}"
  )
  print("=" * 75)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""verify_handoff_bucket_access.py - Pre-flight GCS connectivity/security probe.

Probes gs://<bucket>/ for read/write capability using a temporary canary file.
Verifies bucket exists, tests write/read/delete permissions, and inspects
whether publicAccessPrevention is enforced.

Usage:
  python3 verify_handoff_bucket_access.py --bucket=<bucket_name>
      [--storage_key=<key>]
"""

import argparse
from collections.abc import Sequence
import datetime
import json
import os
import subprocess
import sys
import time


def run_gcloud_storage(
    args: Sequence[str], timeout: int = 20
) -> tuple[int, str, str]:
  """Runs `gcloud storage <args>` without raising.

  Args:
    args: Arguments appended after `gcloud storage`.
    timeout: Timeout in seconds for the gcloud process.

  Returns:
    A (returncode, stdout, stderr) tuple; returncode is -1 and stderr holds
    the error text when gcloud could not be run or timed out.
  """
  cmd = ["gcloud", "storage", *args]
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr
  except (subprocess.SubprocessError, OSError) as e:
    return -1, "", str(e)


def verify_bucket_access(
    bucket_name: str, storage_key: str | None = None
) -> dict[str, object]:
  """Probes a bucket for existence, canary write/read, and public access.

  Args:
    bucket_name: Bucket name or gs:// URI.
    storage_key: Optional incident storage key; the canary is written under it,
      and a failed `buckets describe` is then only a warning.

  Returns:
    A report dict with the probe results, an overall status, and errors.
  """
  bucket = bucket_name.strip()
  if bucket.startswith("gs://"):
    bucket = bucket[5:]
  bucket = bucket.rstrip("/")

  report = {
      "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "bucket": bucket,
      "storage_key": storage_key,
      "bucket_exists": False,
      "write_accessible": False,
      "read_accessible": False,
      "public_access_prevention": "UNKNOWN",
      "status": "INITIALIZING",
      "errors": [],
  }

  # 1. Check if bucket exists and inspect config
  returncode, stdout, stderr = run_gcloud_storage(
      ["buckets", "describe", f"gs://{bucket}", "--format=json"]
  )
  if returncode == 0:
    report["bucket_exists"] = True
    bucket_description = None
    try:
      bucket_description = json.loads(stdout)
    except json.JSONDecodeError:
      report["public_access_prevention"] = "PARSE_ERROR"
    if isinstance(bucket_description, dict):
      iam_config = bucket_description.get("iamConfiguration")
      if not isinstance(iam_config, dict):
        iam_config = {}
      public_access_prevention = iam_config.get(
          "publicAccessPrevention"
      ) or bucket_description.get("publicAccessPrevention")
      report["public_access_prevention"] = (
          public_access_prevention or "unspecified"
      )
    elif report["public_access_prevention"] != "PARSE_ERROR":
      report["public_access_prevention"] = "PARSE_ERROR"
  else:
    if not storage_key:
      report["status"] = "BUCKET_NOT_FOUND_OR_FORBIDDEN"
      report["errors"].append(
          f"Failed to describe gs://{bucket}:"
          f" {stderr.strip() or stdout.strip()}"
      )
      return report
    report["errors"].append(
        f"Warning: Failed to describe gs://{bucket}:"
        f" {stderr.strip() or stdout.strip()}"
    )

  # 2. Test canary write & read
  canary_id = f"canary_{int(time.time())}_{os.getpid()}"
  if storage_key:
    canary_relative_path = f"{storage_key}/.canary_{canary_id}.txt"
  else:
    canary_relative_path = f".canary_{canary_id}.txt"
  canary_gcs_uri = f"gs://{bucket}/{canary_relative_path}"

  temp_canary_path = os.path.join(
      os.environ.get("TMPDIR", "/tmp"), f"{canary_id}.tmp"
  )
  try:
    with open(temp_canary_path, "w", encoding="utf-8") as f:
      f.write(f"handoff canary probe at {report['timestamp']}\n")

    # Upload
    write_returncode, write_stdout, write_stderr = run_gcloud_storage(
        ["cp", temp_canary_path, canary_gcs_uri]
    )
    if write_returncode == 0:
      report["write_accessible"] = True
      report["bucket_exists"] = True
    else:
      report["errors"].append(
          f"Canary write failed: {write_stderr.strip() or write_stdout.strip()}"
      )

    # Read / List check
    if report["write_accessible"]:
      read_returncode, read_stdout, read_stderr = run_gcloud_storage(
          ["cat", canary_gcs_uri]
      )
      if read_returncode == 0 and "handoff canary" in read_stdout:
        report["read_accessible"] = True
      else:
        report["errors"].append(
            f"Canary read failed: {read_stderr.strip() or read_stdout.strip()}"
        )

      # Delete canary
      run_gcloud_storage(["rm", canary_gcs_uri])

  finally:
    if os.path.exists(temp_canary_path):
      os.remove(temp_canary_path)

  # Evaluate overall status
  if (
      report["bucket_exists"]
      and report["write_accessible"]
      and report["read_accessible"]
  ):
    if str(report["public_access_prevention"]).lower() == "enforced":
      report["status"] = "PASSED_SECURE"
    else:
      report["status"] = "PASSED_PUBLIC_ACCESS_WARNING"
      if report["public_access_prevention"] != "UNKNOWN":
        report["errors"].append(
            "publicAccessPrevention is"
            f" '{report['public_access_prevention']}' (expected 'enforced')"
        )
      else:
        report["errors"].append(
            "publicAccessPrevention could not be verified via buckets"
            " describe (restricted bucket metadata IAM); object read/write"
            " verified."
        )
  elif not report["bucket_exists"]:
    report["status"] = "BUCKET_NOT_FOUND_OR_FORBIDDEN"
  else:
    report["status"] = "ACCESS_DENIED"

  return report


def main() -> None:
  """CLI entry point for probing GCS bucket connectivity and security."""
  parser = argparse.ArgumentParser(
      description="Pre-flight GCS Connectivity and Security Prober"
  )
  parser.add_argument("--bucket", required=True, help="Target GCS bucket name")
  parser.add_argument(
      "--storage_key", default=None, help="Incident storage key / directory"
  )
  parser.add_argument(
      "--output_file", default=None, help="Optional output JSON report path"
  )

  args = parser.parse_args()
  result = verify_bucket_access(args.bucket, args.storage_key)

  if args.output_file:
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

  print(json.dumps(result, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"status: {result['status']}")
  print(f"bucket: {result['bucket']}")
  print(f"bucket_exists: {result['bucket_exists']}")
  print(f"write_accessible: {result['write_accessible']}")
  print(f"read_accessible: {result['read_accessible']}")
  print(f"public_access_prevention: {result['public_access_prevention']}")

  if result["status"] not in ("PASSED_SECURE", "PASSED_PUBLIC_ACCESS_WARNING"):
    sys.exit(2)
  sys.exit(0)


if __name__ == "__main__":
  main()

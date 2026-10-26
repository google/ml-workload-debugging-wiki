#!/usr/bin/env python3
"""upload_handoff_package.py - Publishes the rendered handoff package to GCS.

generate-handoff-manifests, generate-two-level-failure-tables and
render-handoff-readmes write into one local staging tree that mirrors the
bucket layout (pass them the same --output_dir). This script uploads that tree
to gs://<bucket>/<storage_key>/.

It is all-or-nothing on content: before anything is copied, every staged file
is checked with findings_io.content_problem() and the required files are
checked for presence. If any file is empty, placeholder-only or invalid JSON,
nothing is uploaded and the script exits 1.

Usage:
  python3 upload_handoff_package.py --findings=<troubleshoot_findings.json>
      --staging_dir=<path> [--bucket=<name>] [--dry_run]
"""

import argparse
from collections.abc import Sequence
import json
import os
import subprocess
import sys

import findings_io as fio

# Uploaded by scaffold-handoff-bucket, not rendered into the staging tree.
_SCAFFOLDED = frozenset({"BUCKET_STRUCTURE.md"})


def collect_staged(staging_dir: str) -> list[str]:
  """Returns sorted relative POSIX paths for all files under `staging_dir`."""
  staged = []
  for root, _, files in os.walk(staging_dir):
    for filename in files:
      full_path = os.path.join(root, filename)
      staged.append(os.path.relpath(full_path, staging_dir).replace("\\", "/"))
  return sorted(staged)


def preflight(
    findings: dict[str, object], staging_dir: str, staged: Sequence[str]
) -> list[str]:
  """Returns content problems that block the upload."""
  problems = []
  present = set(staged)
  for relative_path in fio.required_package_files(fio.workload_dir(findings)):
    if relative_path not in _SCAFFOLDED and relative_path not in present:
      problems.append(f"{relative_path}: required file was not rendered")
  for relative_path in staged:
    with open(os.path.join(staging_dir, relative_path), "rb") as f:
      problem = fio.content_problem(relative_path, f.read())
    if problem:
      problems.append(f"{relative_path}: {problem}")
  return problems


def upload_package(
    findings: dict[str, object],
    staging_dir: str,
    bucket: str,
    dry_run: bool = False,
) -> dict[str, object]:
  """Uploads the staging tree to gs://<bucket>/<storage_key>/ all-or-nothing.

  Args:
    findings: troubleshoot_findings.json contents.
    staging_dir: Local package tree mirroring the bucket layout.
    bucket: Destination bucket name.
    dry_run: Only record the copy commands.

  Returns:
    A result dict whose status is UPLOADED, REFUSED or UPLOAD_FAILED.
  """
  storage_key = fio.storage_key(findings)
  result = {
      "status": "UPLOADED",
      "bucket": bucket,
      "storage_key": storage_key,
      "storage_root": f"gs://{bucket}/{storage_key}/",
      "uploaded_count": 0,
      "problems": [],
      "upload_records": [],
  }
  if not os.path.isdir(staging_dir):
    result["status"] = "REFUSED"
    result["problems"].append(f"staging directory not found: {staging_dir}")
    return result

  staged = collect_staged(staging_dir)
  result["problems"] = preflight(findings, staging_dir, staged)
  if result["problems"]:
    result["status"] = "REFUSED"
    return result

  active_config = os.environ.get("CLOUDSDK_ACTIVE_CONFIG_NAME")
  config_args = [f"--configuration={active_config}"] if active_config else []
  failures = 0
  for relative_path in staged:
    destination_uri = f"gs://{bucket}/{storage_key}/{relative_path}"
    cmd = [
        "gcloud",
        *config_args,
        "storage",
        "cp",
        os.path.join(staging_dir, relative_path),
        destination_uri,
    ]
    if dry_run:
      result["upload_records"].append(f"[DRY-RUN] {' '.join(cmd)}")
      continue
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
      result["uploaded_count"] += 1
      result["upload_records"].append(
          f"[OK] {relative_path} -> {destination_uri}"
      )
    else:
      failures += 1
      result["upload_records"].append(
          f"[FAIL] {relative_path}: {proc.stderr.strip()}"
      )
  if failures:
    result["status"] = "UPLOAD_FAILED"
  return result


def main() -> None:
  """CLI entry point for uploading a staged handoff package to GCS."""
  parser = argparse.ArgumentParser(
      description="Upload the rendered handoff package to GCS"
  )
  parser.add_argument(
      "--findings", required=True, help="troubleshoot_findings.json path"
  )
  parser.add_argument(
      "--staging_dir",
      required=True,
      help="Local package tree written by the manifest/table/README steps",
  )
  parser.add_argument("--bucket", default="", help="Optional bucket override")
  parser.add_argument("--dry_run", action="store_true", help="Print only")
  args = parser.parse_args()

  try:
    findings, bucket = fio.load_for_handoff(args.findings, args.bucket)
  except fio.FindingsError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

  staging_dir = args.staging_dir
  # The glue runner executes from _wiki/; accept workspace-relative paths too.
  if (
      not os.path.isdir(staging_dir)
      and staging_dir.startswith("_wiki/")
      and os.path.isdir(staging_dir[len("_wiki/") :])
  ):
    staging_dir = staging_dir[len("_wiki/") :]

  result = upload_package(findings, staging_dir, bucket, args.dry_run)
  print(json.dumps(result, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"status: {result['status']}")
  print(f"storage_root: {result['storage_root']}")
  print(f"uploaded_count: {result['uploaded_count']}")
  print(f"problem_count: {len(result['problems'])}")
  if result["status"] != "UPLOADED":
    sys.exit(1)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""verify_handoff_package.py - Post-publication content check for a handoff.

Lists gs://<bucket>/<storage_key>/ and fails unless the package holds real
content:
- every file in findings_io.required_package_files() is present,
- no file is 0 bytes or holds only a placeholder (NOT_PROVIDED, {}, ...),
- every .json file parses.

A listing failure is reported as VERIFY_FAILED, never as an empty bucket.

Usage:
  python3 verify_handoff_package.py --findings=<troubleshoot_findings.json>
      [--bucket=<name>] [--local_dir=<path>]
"""

import argparse
import json
import os
import subprocess
import sys

import findings_io as fio

# Files larger than this cannot be a bare placeholder, so only JSON files above
# it are downloaded (to parse them).
_PLACEHOLDER_MAX_BYTES = 64


class ListingError(Exception):
  """Raised when listing package objects fails."""


def list_objects(
    bucket: str, storage_key: str, local_dir: str | None = None
) -> dict[str, int]:
  """Returns {relative_path: size_bytes}; raises ListingError on failure."""
  objects = {}
  if local_dir:
    base = os.path.abspath(local_dir)
    if not os.path.isdir(base):
      raise ListingError(f"Local package directory not found: {base}")
    for root, _, files in os.walk(base):
      for filename in files:
        full_path = os.path.join(root, filename)
        objects[os.path.relpath(full_path, base).replace("\\", "/")] = (
            os.path.getsize(full_path)
        )
    return objects

  prefix = f"gs://{bucket}/{storage_key}/"
  cmd = ["gcloud", "storage", "ls", "--long", f"{prefix}**"]
  try:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60, check=False
    )
  except (subprocess.SubprocessError, OSError) as e:
    raise ListingError(f"{' '.join(cmd)} failed: {e}") from e
  if proc.returncode != 0:
    raise ListingError(
        f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()}"
    )
  for line in proc.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 3 and parts[-1].startswith(prefix) and parts[0].isdigit():
      objects[parts[-1][len(prefix) :]] = int(parts[0])
  return objects


def read_object(
    relative_path: str,
    bucket: str,
    storage_key: str,
    local_dir: str | None = None,
) -> bytes | None:
  """Returns the object's bytes, or None if it could not be read."""
  if local_dir:
    try:
      with open(os.path.join(local_dir, relative_path), "rb") as f:
        return f.read()
    except OSError:
      return None
  try:
    proc = subprocess.run(
        [
            "gcloud",
            "storage",
            "cat",
            f"gs://{bucket}/{storage_key}/{relative_path}",
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
  except (subprocess.SubprocessError, OSError):
    return None
  return proc.stdout if proc.returncode == 0 else None


def verify_package(
    findings: dict[str, object], bucket: str, local_dir: str | None = None
) -> dict[str, object]:
  """Checks that the published package holds real content.

  Args:
    findings: troubleshoot_findings.json contents.
    bucket: Destination bucket name.
    local_dir: Verify this local package tree instead of GCS.

  Returns:
    A result dict whose status is VERIFIED, INCOMPLETE or VERIFY_FAILED, with
    the problems found.
  """
  storage_key = fio.storage_key(findings)
  workload = fio.workload_dir(findings)
  result = {
      "status": "VERIFIED",
      "bucket": bucket,
      "storage_key": storage_key,
      "workload": workload,
      "storage_root": local_dir or f"gs://{bucket}/{storage_key}/",
      "file_count": 0,
      "problems": [],
  }
  if not storage_key or not workload:
    result["status"] = "VERIFY_FAILED"
    result["problems"].append(
        "identity.storage_key and identity.workload are required"
    )
    return result

  try:
    objects = list_objects(bucket, storage_key, local_dir)
  except ListingError as e:
    result["status"] = "VERIFY_FAILED"
    result["problems"].append(str(e))
    return result

  result["file_count"] = sum(
      1 for object_path in objects if os.path.basename(object_path) != ".keep"
  )
  problems = result["problems"]
  for relative_path in fio.required_package_files(workload):
    if relative_path not in objects:
      problems.append(f"{relative_path}: required file is missing")

  for relative_path, size in sorted(objects.items()):
    if os.path.basename(relative_path) == ".keep":
      continue
    if size == 0:
      problems.append(f"{relative_path}: file is empty (0 bytes)")
      continue
    if size > _PLACEHOLDER_MAX_BYTES and not relative_path.endswith(".json"):
      continue
    data = read_object(relative_path, bucket, storage_key, local_dir)
    if data is None:
      problems.append(f"{relative_path}: could not be read back")
      continue
    problem = fio.content_problem(relative_path, data)
    if problem:
      problems.append(f"{relative_path}: {problem}")

  if problems:
    result["status"] = "INCOMPLETE"
  return result


def main() -> None:
  """CLI entry point for verifying a published handoff package."""
  parser = argparse.ArgumentParser(
      description="Verify a published handoff package has real content"
  )
  parser.add_argument(
      "--findings", required=True, help="troubleshoot_findings.json path"
  )
  parser.add_argument("--bucket", default="", help="Optional bucket override")
  parser.add_argument(
      "--local_dir",
      default=None,
      help="Verify a local package tree instead of GCS",
  )
  args = parser.parse_args()

  try:
    findings = fio.load_findings(args.findings)
    bucket = (
        fio.gcs_bucket(findings, args.bucket)
        if args.local_dir
        else fio.require_bucket(findings, args.bucket)
    )
  except fio.FindingsError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

  local_dir = args.local_dir
  # The glue runner executes from _wiki/; accept workspace-relative paths too.
  if (
      local_dir
      and not os.path.isdir(local_dir)
      and local_dir.startswith("_wiki/")
      and os.path.isdir(local_dir[len("_wiki/") :])
  ):
    local_dir = local_dir[len("_wiki/") :]

  result = verify_package(findings, bucket, local_dir=local_dir)
  print(json.dumps(result, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"status: {result['status']}")
  print(f"storage_root: {result['storage_root']}")
  print(f"file_count: {result['file_count']}")
  print(f"problem_count: {len(result['problems'])}")
  if result["status"] != "VERIFIED":
    sys.exit(1)


if __name__ == "__main__":
  main()

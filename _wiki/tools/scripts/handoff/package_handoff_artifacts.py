#!/usr/bin/env python3
"""package_handoff_artifacts.py - Diagnostic artifact collector and stager.

Collects diagnostic scratch artifacts:
- crash_signatures.txt
- culprit_hosts.json
- workload_nodes.json
- vm_termination_events.json
- node_health_events.txt
- Orbax checkpoint barrier logs
- HLO textual dumps (automatically sanitized via redact_diagnostic_payload.py)

Stages them locally into a canonical handoff tree or uploads them directly to
GCS under jobs/<workload>/data/.
"""

import argparse
from collections.abc import Sequence
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

import convert_triage_to_findings
import findings_io as fio
import redact_diagnostic_payload

_MIN_JSON_ARTIFACT_BYTES = 10


def _triage_action(
    triage_data: dict[str, object] | None, name: str
) -> dict[str, object] | None:
  """Returns a /troubleshoot action record by name, or None if it never ran."""
  if not triage_data:
    return None
  for milestone in (
      "milestone_1_reconnaissance",
      "milestone_2_baseline_disruption_evidence",
  ):
    record = (triage_data.get(milestone) or {}).get(name)
    if isinstance(record, dict):
      return record
  for record in triage_data.get("actions_executed", []):
    if isinstance(record, dict) and record.get("action") == name:
      return record
  return None


def _counts_text(
    title: str, summary: dict[str, object], keys: tuple[str, ...]
) -> str:
  """Renders the counts an action reported; absent counts stay absent."""
  lines = [f"# {title}"]
  for key in keys:
    lines.append(f"{key}: {fio.md_cell(summary.get(key))}")
  return "\n".join(lines) + "\n"


def _stage_file(
    source_path: str,
    destination_path: str,
    kind: str,
    staging_dir: str,
    staged_files: list[dict[str, str]],
) -> None:
  """Copies a file into the staging directory and records its metadata."""
  if source_path != destination_path:
    shutil.copyfile(source_path, destination_path)
  relative_path = os.path.relpath(destination_path, staging_dir)
  staged_files.append({
      "path": relative_path,
      "kind": kind,
      "local_path": destination_path,
  })


def _find_scratch(scratch_dir: str, patterns: Sequence[str]) -> str | None:
  """Returns the newest scratch file matching any pattern, or None."""
  if not os.path.isdir(scratch_dir):
    return None
  for pattern in patterns:
    matches = glob.glob(os.path.join(scratch_dir, pattern))
    if matches:
      matches.sort(key=os.path.getmtime, reverse=True)
      return matches[0]
  return None


def _write_json(
    destination_path: str,
    payload: dict[str, object],
    kind: str,
    staging_dir: str,
    staged_files: list[dict[str, str]],
) -> None:
  """Writes a JSON dictionary into the staging tree and records it."""
  with open(destination_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  _stage_file(
      destination_path, destination_path, kind, staging_dir, staged_files
  )


def _write_text(
    destination_path: str,
    text: str,
    kind: str,
    staging_dir: str,
    staged_files: list[dict[str, str]],
) -> None:
  """Writes UTF-8 text into the staging tree and records it."""
  with open(destination_path, "w", encoding="utf-8") as f:
    f.write(text)
  _stage_file(
      destination_path, destination_path, kind, staging_dir, staged_files
  )


def package_artifacts(
    findings_path: str,
    scratch_dir: str,
    staging_dir: str,
    hlo_dir: str | None = None,
    orbax_dir: str | None = None,
    bucket: str | None = None,
    triage_json: str | None = None,
    workload: str | None = None,
    upload: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
  """Stages diagnostic artifacts into the canonical handoff tree.

  Artifacts are taken from scratch files when present, otherwise synthesized
  from the /troubleshoot triage JSON; anything with no source is reported as
  missing rather than written as a placeholder. The findings file is rewritten
  with the newly staged artifacts appended.

  Args:
    findings_path: Path to troubleshoot_findings.json (updated in place).
    scratch_dir: Directory holding diagnostic scratch dumps.
    staging_dir: Local root of the handoff tree to stage into.
    hlo_dir: Optional directory of raw HLO dumps to sanitize and stage.
    orbax_dir: Optional directory of Orbax checkpoint logs.
    bucket: Optional target GCS bucket override.
    triage_json: Optional path to the /troubleshoot triage JSON; discovered
      from the workload name when omitted.
    workload: Optional workload name overriding identity.workload.
    upload: Whether to upload staged files to GCS.
    dry_run: When uploading, only record the gcloud commands.

  Returns:
    A summary dict with status, staged and missing artifacts, and upload
    records.

  Raises:
    fio.FindingsError: The findings file is unusable, or upload is requested
      without identity.workload, identity.storage_key, or a bucket.
  """
  findings = fio.load_findings(findings_path)
  workload_name = str(
      workload
      or fio.get_raw(findings, "identity.workload")
      or "unknown_workload"
  )
  storage_key = fio.get_raw(findings, "identity.storage_key") or "unknown_key"
  if upload:
    # Never publish under a made-up prefix.
    if workload_name == "unknown_workload" or storage_key == "unknown_key":
      raise fio.FindingsError(
          "identity.workload and identity.storage_key are required to upload."
      )
    target_bucket = fio.require_bucket(findings, bucket)
  else:
    target_bucket = fio.gcs_bucket(findings, bucket)

  # Load triage data if available (explicit or discovered from scratch)
  triage_data = None
  triage_path = triage_json
  if not triage_path and workload_name != "unknown_workload":
    triage_path = convert_triage_to_findings.find_latest_triage_file(
        workload_name
    )
    if not triage_path and os.path.isdir(scratch_dir):
      candidates = sorted([
          os.path.join(scratch_dir, filename)
          for filename in os.listdir(scratch_dir)
          if filename.startswith(f"triage-{workload_name}")
          and filename.endswith(".json")
      ])
      triage_path = candidates[-1] if candidates else None

  if triage_path and os.path.isfile(triage_path):
    try:
      with open(triage_path, "r", encoding="utf-8") as f:
        triage_data = json.load(f)
    except (OSError, ValueError) as exc:
      print(
          f"[WARN] Failed to load triage JSON at {triage_path}: {exc}",
          file=sys.stderr,
      )

  # Canonical directories according to BUCKET_STRUCTURE.md
  job_staging_data = os.path.join(staging_dir, "jobs", workload_name, "data")
  baseline_dir = os.path.join(job_staging_data, "baseline")
  degraded_dir = os.path.join(job_staging_data, "degraded")
  degraded_xla_dir = os.path.join(degraded_dir, "xla_dumps")
  orbax_dest_dir = os.path.join(job_staging_data, "orbax")
  logs_dest_dir = os.path.join(job_staging_data, "logs")
  memory_dest_dir = os.path.join(job_staging_data, "memory")

  infra_staging_dir = os.path.join(staging_dir, "infrastructure")
  infra_reservation_dir = os.path.join(infra_staging_dir, "reservation")
  infra_subblocks_dir = os.path.join(infra_staging_dir, "subblocks")

  all_dirs = (
      baseline_dir,
      degraded_dir,
      degraded_xla_dir,
      orbax_dest_dir,
      logs_dest_dir,
      memory_dest_dir,
      infra_reservation_dir,
      infra_subblocks_dir,
  )
  for directory in all_dirs:
    os.makedirs(directory, exist_ok=True)

  staged_files: list[dict[str, str]] = []
  # Artifacts with no source are recorded as missing, never written as
  # placeholder files: an absent file is honest, "NOT_PROVIDED" is not data.
  missing = []

  # -------------------------------------------------------------
  # 1. Baseline Telemetry: jobs/<workload>/data/baseline/
  # -------------------------------------------------------------
  # A. workload_nodes.json
  target_nodes_file = os.path.join(baseline_dir, "workload_nodes.json")
  found_nodes = _find_scratch(
      scratch_dir,
      ("workload_nodes*.json", f"workload_nodes_{workload_name}*.json"),
  )
  nodes_action = _triage_action(triage_data, "resolve-workload-nodes")
  if found_nodes and os.path.getsize(found_nodes) > _MIN_JSON_ARTIFACT_BYTES:
    _stage_file(
        found_nodes, target_nodes_file, "baseline", staging_dir, staged_files
    )
  elif nodes_action and nodes_action.get("summary_fields"):
    nodes_summary = nodes_action["summary_fields"]
    _write_json(
        target_nodes_file,
        {
            "workload_name": workload_name,
            "total_nodes": nodes_summary.get("total_nodes"),
            "node_groups_count": nodes_summary.get("node_groups_count"),
            "status": nodes_summary.get("status"),
            "node_groups": nodes_summary.get("node_groups", {}),
        },
        "baseline",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("workload_nodes.json")

  # B. culprit_hosts.json
  target_culprit_file = os.path.join(baseline_dir, "culprit_hosts.json")
  found_culprit = _find_scratch(
      scratch_dir,
      ("culprit_hosts*.json", f"culprit_hosts_{workload_name}*.json"),
  )
  culprit_action = _triage_action(triage_data, "extract-culprit-hosts")
  culprit_payload = culprit_action and (
      culprit_action.get("reconciled_findings")
      or culprit_action.get("summary_fields")
  )
  if (
      found_culprit
      and os.path.getsize(found_culprit) > _MIN_JSON_ARTIFACT_BYTES
  ):
    _stage_file(
        found_culprit,
        target_culprit_file,
        "baseline",
        staging_dir,
        staged_files,
    )
  elif isinstance(culprit_payload, dict):
    _write_json(
        target_culprit_file,
        culprit_payload,
        "baseline",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("culprit_hosts.json")

  # C. vm_termination_events.json
  target_vm_file = os.path.join(baseline_dir, "vm_termination_events.json")
  found_vm = _find_scratch(
      scratch_dir,
      (
          "vm_termination_events*.json",
          f"vm_termination_events_{workload_name}*.json",
      ),
  )
  vm_action = _triage_action(triage_data, "query-vm-termination-events")
  if found_vm and os.path.getsize(found_vm) > _MIN_JSON_ARTIFACT_BYTES:
    _stage_file(found_vm, target_vm_file, "baseline", staging_dir, staged_files)
  elif vm_action and isinstance(vm_action.get("summary_fields"), dict):
    _write_json(
        target_vm_file,
        vm_action["summary_fields"],
        "baseline",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("vm_termination_events.json")

  # -------------------------------------------------------------
  # 2. Degraded Telemetry: jobs/<workload>/data/degraded/
  # -------------------------------------------------------------
  # A. crash_signatures.txt
  target_crash_file = os.path.join(degraded_dir, "crash_signatures.txt")
  found_crash = _find_scratch(
      scratch_dir,
      ("crash_signatures*.txt", f"crash_signatures_{workload_name}*.txt"),
  )
  crash_action = _triage_action(triage_data, "query-crash-signatures")
  if found_crash and os.path.getsize(found_crash) > 1:
    _stage_file(
        found_crash, target_crash_file, "degraded", staging_dir, staged_files
    )
  elif crash_action and isinstance(crash_action.get("summary_fields"), dict):
    _write_text(
        target_crash_file,
        _counts_text(
            f"Crash Signatures for {workload_name}",
            crash_action["summary_fields"],
            ("original_count", "deduplicated_count", "error_fatal_count"),
        ),
        "degraded",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("crash_signatures.txt")

  # B. xla_dumps/ (Collect and sanitize HLO dumps)
  if hlo_dir and os.path.isdir(hlo_dir):
    for root, _, files in os.walk(hlo_dir):
      for filename in files:
        if filename.endswith((".hlo", ".txt", ".ir")):
          raw_path = os.path.join(root, filename)
          clean_dest = os.path.join(degraded_xla_dir, filename)
          redact_diagnostic_payload.sanitize_file(raw_path, clean_dest)
          _stage_file(
              clean_dest, clean_dest, "hlo_dump", staging_dir, staged_files
          )

  # -------------------------------------------------------------
  # 3. Logs Telemetry: jobs/<workload>/data/logs/
  # -------------------------------------------------------------
  # A. troubleshoot_trace.md
  target_trace_file = os.path.join(logs_dest_dir, "troubleshoot_trace.md")
  found_trace = _find_scratch(
      scratch_dir,
      ("*trace*.md", "triage_paste_report.md", "*troubleshoot*.md"),
  )
  executed = [
      action_record
      for action_record in (triage_data or {}).get("actions_executed", [])
      if isinstance(action_record, dict)
  ]
  if found_trace and os.path.getsize(found_trace) > _MIN_JSON_ARTIFACT_BYTES:
    _stage_file(
        found_trace, target_trace_file, "logs", staging_dir, staged_files
    )
  elif executed:
    trace_content = (
        f"# Troubleshoot Trace: {workload_name}\n\n**Session**:"
        f" {fio.get(findings, 'source.troubleshoot_session')}\n**Triage"
        f" Window**: {fio.get(findings, 'windows.degraded.start')} to"
        f" {fio.get(findings, 'windows.degraded.end')}\n\n## Empirical Actions"
        " Executed\n"
    )
    for action_record in executed:
      trace_content += (
          f"- `{action_record.get('action')}`:"
          f" status={action_record.get('status')}"
          f" summary={json.dumps(action_record.get('summary_fields', {}))}\n"
      )
    _write_text(
        target_trace_file, trace_content, "logs", staging_dir, staged_files
    )
  else:
    missing.append("troubleshoot_trace.md")

  # B. node_health_events.txt
  target_health_file = os.path.join(logs_dest_dir, "node_health_events.txt")
  found_health = _find_scratch(
      scratch_dir,
      (
          "node_health_events*.txt",
          f"node_health_events_{workload_name}*.txt",
      ),
  )
  health_action = _triage_action(triage_data, "query-node-health-events")
  if found_health and os.path.getsize(found_health) > 1:
    _stage_file(
        found_health, target_health_file, "logs", staging_dir, staged_files
    )
  elif health_action and isinstance(health_action.get("summary_fields"), dict):
    _write_text(
        target_health_file,
        _counts_text(
            f"Node Health Events for {workload_name}",
            health_action["summary_fields"],
            ("original_count", "error_fatal_count"),
        ),
        "logs",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("node_health_events.txt")

  # C. raw_workload_logs.txt
  target_raw_logs = os.path.join(logs_dest_dir, "raw_workload_logs.txt")
  found_raw_logs = _find_scratch(
      scratch_dir,
      ("logs-*.txt", "raw_workload_logs*.txt", f"logs_{workload_name}*.txt"),
  )
  if found_raw_logs and os.path.getsize(found_raw_logs) > 1:
    _stage_file(
        found_raw_logs, target_raw_logs, "logs", staging_dir, staged_files
    )
  else:
    missing.append("raw_workload_logs.txt")

  # -------------------------------------------------------------
  # 4. Orbax Telemetry: jobs/<workload>/data/orbax/
  # -------------------------------------------------------------
  target_orbax_file = os.path.join(orbax_dest_dir, "orbax_checkpoint.log")
  found_orbax = _find_scratch(
      scratch_dir, ("orbax*.log", "checkpoint*.log", "*orbax*.txt")
  )
  if found_orbax and os.path.getsize(found_orbax) > 1:
    _stage_file(
        found_orbax, target_orbax_file, "orbax", staging_dir, staged_files
    )
  elif orbax_dir and os.path.isdir(orbax_dir):
    for root, _, files in os.walk(orbax_dir):
      for filename in files:
        raw_path = os.path.join(root, filename)
        if os.path.getsize(raw_path) == 0:
          continue
        dest_file = os.path.join(orbax_dest_dir, filename)
        shutil.copyfile(raw_path, dest_file)
        _stage_file(dest_file, dest_file, "orbax", staging_dir, staged_files)
  else:
    missing.append("orbax_checkpoint.log")

  # -------------------------------------------------------------
  # 5. Memory Telemetry: jobs/<workload>/data/memory/
  # -------------------------------------------------------------
  target_memory_file = os.path.join(memory_dest_dir, "hbm_memory.log")
  found_memory = _find_scratch(
      scratch_dir, ("hbm_memory*.log", "*memory*.txt", "*memory*.log")
  )
  if found_memory and os.path.getsize(found_memory) > 1:
    _stage_file(
        found_memory, target_memory_file, "memory", staging_dir, staged_files
    )
  else:
    missing.append("hbm_memory.log")

  # -------------------------------------------------------------
  # 6. Infrastructure Telemetry: infrastructure/reservation/ and
  #    infrastructure/subblocks/
  # -------------------------------------------------------------
  # Staged only from collected telemetry; identity fields alone would just
  # duplicate INCIDENT_MANIFEST.json.
  target_reservation_file = os.path.join(
      infra_reservation_dir, "reservation_status.json"
  )
  found_reservation = _find_scratch(
      scratch_dir, ("reservation_status*.json", "*reservation*.json")
  )
  if (
      found_reservation
      and os.path.getsize(found_reservation) > _MIN_JSON_ARTIFACT_BYTES
  ):
    _stage_file(
        found_reservation,
        target_reservation_file,
        "infrastructure",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("reservation_status.json")

  target_subblock_file = os.path.join(
      infra_subblocks_dir, "subblock_status.json"
  )
  found_subblock = _find_scratch(
      scratch_dir, ("subblock_status*.json", "*subblock*.json")
  )
  if (
      found_subblock
      and os.path.getsize(found_subblock) > _MIN_JSON_ARTIFACT_BYTES
  ):
    _stage_file(
        found_subblock,
        target_subblock_file,
        "infrastructure",
        staging_dir,
        staged_files,
    )
  else:
    missing.append("subblock_status.json")

  # -------------------------------------------------------------
  # 7. Update findings with newly packaged artifacts
  # -------------------------------------------------------------
  existing_paths = {artifact["path"] for artifact in fio.artifacts(findings)}
  new_artifacts = list(findings.get("artifacts", []))
  for staged_file in staged_files:
    if staged_file["path"] not in existing_paths:
      new_artifacts.append(
          {"path": staged_file["path"], "kind": staged_file["kind"]}
      )
  findings["artifacts"] = new_artifacts

  with open(findings_path, "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)

  # -------------------------------------------------------------
  # 8. Upload if requested (Strict BUCKET_STRUCTURE.md alignment)
  # -------------------------------------------------------------
  upload_records = []
  failures = 0
  if upload:
    # Handle gcloud active configuration
    active_config = os.environ.get("CLOUDSDK_ACTIVE_CONFIG_NAME")
    config_args = [f"--configuration={active_config}"] if active_config else []

    for staged_file in staged_files:
      # Invariant: staged_file['path'] is relative to staging root (e.g.
      # jobs/<workload>/data/... or infrastructure/...).
      # NEVER upload to data/jobs/
      with open(staged_file["local_path"], "rb") as f:
        problem = fio.content_problem(staged_file["path"], f.read())
      if problem:
        failures += 1
        upload_records.append(f"[REFUSED] {staged_file['path']}: {problem}")
        continue
      gcs_dest = f"gs://{target_bucket}/{storage_key}/{staged_file['path']}"
      cmd = (
          ["gcloud"]
          + config_args
          + ["storage", "cp", staged_file["local_path"], gcs_dest]
      )
      if dry_run:
        upload_records.append(f"[DRY-RUN] {' '.join(cmd)}")
      else:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
          upload_records.append(
              f"[OK] Uploaded {staged_file['path']} -> {gcs_dest}"
          )
        else:
          failures += 1
          upload_records.append(
              f"[FAIL] Upload failed for {staged_file['path']}:"
              f" {proc.stderr.strip()}"
          )

  if failures:
    status = "UPLOAD_FAILED"
  elif not staged_files:
    status = "NOTHING_TO_PACKAGE"
  else:
    status = "PACKAGED"
  return {
      "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "status": status,
      "workload": workload_name,
      "storage_key": storage_key,
      "bucket": target_bucket,
      "staged_count": len(staged_files),
      "staged_files": [staged_file["path"] for staged_file in staged_files],
      "missing_artifacts": missing,
      "upload_failures": failures,
      "upload_records": upload_records,
  }


def main() -> None:
  """CLI entry point for collecting and staging handoff artifacts."""
  parser = argparse.ArgumentParser(
      description="Package Diagnostic Artifacts for Handoff"
  )
  parser.add_argument(
      "--findings", required=True, help="Path to troubleshoot_findings.json"
  )
  parser.add_argument(
      "--scratch_dir",
      default=fio.WIKI_SCRATCH_DIR,
      help="Scratch directory with diagnostic dumps",
  )
  parser.add_argument(
      "--staging_dir", default=None, help="Local staging directory"
  )
  parser.add_argument(
      "--hlo_dir",
      default=None,
      help="Raw HLO dumps directory to sanitize and stage",
  )
  parser.add_argument(
      "--orbax_dir", default=None, help="Orbax checkpoint logs directory"
  )
  parser.add_argument(
      "--triage_json", default=None, help="Path to /troubleshoot triage JSON"
  )
  parser.add_argument(
      "--workload", default=None, help="Workload name for locating triage state"
  )
  parser.add_argument("--bucket", default=None, help="Target GCS bucket name")
  parser.add_argument(
      "--upload",
      dest="upload",
      action="store_true",
      default=True,
      help="Upload staged files to GCS (default)",
  )
  parser.add_argument(
      "--no_upload",
      dest="upload",
      action="store_false",
      help="Stage locally only",
  )
  parser.add_argument("--dry_run", action="store_true", help="Dry run upload")

  args = parser.parse_args()

  findings_path = args.findings
  if findings_path and not os.path.isfile(findings_path):
    if findings_path.startswith("_wiki/") and os.path.isfile(
        findings_path[len("_wiki/") :]
    ):
      findings_path = findings_path[len("_wiki/") :]

  triage_json = args.triage_json
  if triage_json and not os.path.isfile(triage_json):
    if triage_json.startswith("_wiki/") and os.path.isfile(
        triage_json[len("_wiki/") :]
    ):
      triage_json = triage_json[len("_wiki/") :]

  scratch_dir = args.scratch_dir
  if scratch_dir and not os.path.isdir(scratch_dir):
    if scratch_dir.startswith("_wiki/") and os.path.isdir(
        scratch_dir[len("_wiki/") :]
    ):
      scratch_dir = scratch_dir[len("_wiki/") :]

  staging = args.staging_dir
  if staging and staging.startswith("_wiki/") and not os.path.exists("_wiki"):
    staging = staging[len("_wiki/") :]
  if not staging:
    staging = os.path.join(os.environ.get("TMPDIR", "/tmp"), "handoff_staging")

  try:
    result = package_artifacts(
        findings_path=findings_path,
        scratch_dir=scratch_dir,
        staging_dir=staging,
        hlo_dir=args.hlo_dir,
        orbax_dir=args.orbax_dir,
        bucket=args.bucket,
        triage_json=triage_json,
        workload=args.workload,
        upload=args.upload,
        dry_run=args.dry_run,
    )
  except fio.FindingsError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)

  print(json.dumps(result, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"status: {result['status']}")
  print(f"workload: {result['workload']}")
  print(f"bucket: {result['bucket']}")
  print(f"staged_count: {result['staged_count']}")
  missing = ", ".join(result["missing_artifacts"]) or "none"
  print(f"missing_artifacts: {missing}")
  if result["status"] != "PACKAGED":
    sys.exit(1)


if __name__ == "__main__":
  main()

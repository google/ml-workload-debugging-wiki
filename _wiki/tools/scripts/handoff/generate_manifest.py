#!/usr/bin/env python3
"""generate_manifest.py - Generates JSON Schema v2.2 Manifests.

Supports both Incident-level (INCIDENT_MANIFEST.json) and
Job-level (JOB_MANIFEST.json) manifests.
Supports dynamic bucket parameterization via --bucket, identity.bucket
in findings, or $HANDOFF_BUCKET.

The ONLY data source is --findings=<troubleshoot_findings.json>.
"""

import argparse
import datetime
import json
import os
import sys

import findings_io as fio

GOOGLE_DOMAIN = "@google.com"


def validate_topology(
    accelerator_type: str | None,
    chip_count: int | None,
    slice_count: int | None,
) -> int | None:
  """Validates topology when data is present.

  Args:
    accelerator_type: Accelerator type; accepted for interface stability but
      not checked.
    chip_count: Total chip count, or None when unknown.
    slice_count: Slice count, or None when unknown.

  Returns:
    chips_per_slice, or None when chip_count or slice_count is missing.

  Raises:
    ValueError: The counts are non-positive, chip_count < slice_count, or
      chip_count is not evenly divisible by slice_count.
  """
  del accelerator_type
  if chip_count is None or slice_count is None:
    return None
  if chip_count <= 0 or slice_count <= 0:
    raise ValueError("chip_count and slice_count must be positive integers.")
  if chip_count < slice_count:
    raise ValueError(
        f"chip_count ({chip_count}) cannot be smaller than slice_count"
        f" ({slice_count})."
    )
  if chip_count % slice_count != 0:
    raise ValueError(
        f"chip_count ({chip_count}) must be evenly divisible by slice_count"
        f" ({slice_count})."
    )
  return chip_count // slice_count


def validate_ownership_emails(owners: dict[str, object]) -> None:
  """Validates domains only for owner values that are email addresses."""
  for key in ("google_support_lead", "google_ce_lead"):
    owner_email = owners.get(key)
    if (
        owner_email
        and "@" in str(owner_email)
        and not str(owner_email).endswith(GOOGLE_DOMAIN)
    ):
      raise ValueError(
          f"owners.{key} '{owner_email}' must belong to {GOOGLE_DOMAIN} domain."
      )
  for key in ("customer_dri", "customer_primary_investigator"):
    owner_email = owners.get(key)
    if (
        owner_email
        and "@" in str(owner_email)
        and str(owner_email).endswith(GOOGLE_DOMAIN)
    ):
      raise ValueError(
          f"owners.{key} '{owner_email}' is customer contact, must not be"
          f" {GOOGLE_DOMAIN}."
      )


def _owners(findings: dict[str, object]) -> dict[str, object]:
  return {
      key: fio.json_value(fio.get_raw(findings, f"owners.{key}"))
      for key in (
          "customer_dri",
          "customer_primary_investigator",
          "google_support_lead",
          "google_ce_lead",
      )
  }


def _source(findings: dict[str, object]) -> dict[str, object]:
  return {
      "troubleshoot_session": fio.json_value(
          fio.get_raw(findings, "source.troubleshoot_session")
      ),
      "report_path": fio.json_value(
          fio.get_raw(findings, "source.report_path")
      ),
      "extracted_at": fio.json_value(
          fio.get_raw(findings, "source.extracted_at")
      ),
  }


def _gcs(
    findings: dict[str, object],
    relative_path: str,
    bucket: str | None = None,
) -> str | None:
  return fio.gcs_path(findings, relative_path, bucket=bucket)


def build_incident_manifest(
    findings: dict[str, object], bucket: str | None = None
) -> dict[str, object]:
  """Constructs the canonical INCIDENT_MANIFEST.json conforming to Schema v2.2.

  Args:
    findings: Diagnostic findings dictionary conforming to Schema v2.2.
    bucket: Optional target GCS bucket override.

  Returns:
    Incident manifest dictionary adhering to Schema v2.2.
  """
  owners = _owners(findings)
  validate_ownership_emails(owners)

  now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
  storage_key = fio.json_value(fio.get_raw(findings, "identity.storage_key"))
  target_bucket = fio.gcs_bucket(findings, bucket)
  workload = fio.json_value(fio.get_raw(findings, "identity.workload"))
  canonical_category = fio.canonical_category(findings)

  jobs_map = {}
  if workload:
    job_prefix = f"jobs/{workload}"
    jobs_map[workload] = {
        "job_name": workload,
        "disruption_category": canonical_category,
        "manifest_path": _gcs(
            findings, f"{job_prefix}/JOB_MANIFEST.json", bucket=target_bucket
        ),
        "readme_path": _gcs(
            findings, f"{job_prefix}/README.md", bucket=target_bucket
        ),
        "findings_track": {
            "customer_latest": _gcs(
                findings,
                f"{job_prefix}/findings/customer/_customer_latest.md",
                bucket=target_bucket,
            ),
            "google_latest": _gcs(
                findings,
                f"{job_prefix}/findings/google/_google_latest.md",
                bucket=target_bucket,
            ),
        },
        "failure_tables": {
            "level_b_job_progression": _gcs(
                findings,
                f"{job_prefix}/failure_tables/Level_B_Job_Progression.md",
                bucket=target_bucket,
            ),
        },
    }

  return {
      "schema_version": "2.2.0",
      "manifest_type": "INCIDENT_MANIFEST",
      "generation_timestamp": now_utc,
      "source": _source(findings),
      "incident_metadata": {
          "storage_key": storage_key,
          "bucket": target_bucket,
          "case_id": fio.json_value(
              fio.get_raw(findings, "identity.case_id")
          ),
          "issue_id": fio.json_value(
              fio.get_raw(findings, "identity.issue_id")
          ),
          "issue_tracker_url": fio.issue_url(findings),
          "storage_root": fio.storage_root(findings, bucket=target_bucket),
          "storage_console_url": fio.console_url(
              findings, bucket=target_bucket
          ),
      },
      "cluster_specification": {
          "project_id": fio.json_value(
              fio.get_raw(findings, "identity.project_id")
          ),
          "project_number": fio.json_value(
              fio.get_raw(findings, "identity.project_number")
          ),
          "cluster_name": fio.json_value(
              fio.get_raw(findings, "identity.cluster")
          ),
          "zone": fio.json_value(fio.get_raw(findings, "identity.zone")),
          "reservation": fio.json_value(
              fio.get_raw(findings, "identity.reservation")
          ),
          "accelerator_type": fio.json_value(
              fio.get_raw(findings, "identity.accelerator_type")
          ),
      },
      "ownership": owners,
      "ownership_history": [{
          "timestamp": now_utc,
          "action": "INCIDENT_INITIALIZED",
          "actor": owners.get("google_support_lead"),
          "notes": (
              "Generated from troubleshoot session"
              f" {fio.get(findings, 'source.troubleshoot_session')}"
          ),
      }],
      "infrastructure_findings": {
          "level_a_table": _gcs(
              findings,
              "infrastructure/findings/_infra_latest.md",
              bucket=target_bucket,
          ),
          "level_a_legacy_mirror": _gcs(
              findings,
              "infrastructure/Level_A_Infrastructure.md",
              bucket=target_bucket,
          ),
          "subblocks_dir": _gcs(
              findings, "infrastructure/subblocks/", bucket=target_bucket
          ),
          "dcn_optical_dir": _gcs(
              findings, "infrastructure/dcn_optical/", bucket=target_bucket
          ),
          "reservation_dir": _gcs(
              findings, "infrastructure/reservation/", bucket=target_bucket
          ),
          "links_dir": _gcs(
              findings, "infrastructure/links/", bucket=target_bucket
          ),
      },
      "evidence_files": [
          {
              "path": artifact["path"],
              "gcs_uri": _gcs(findings, artifact["path"], bucket=target_bucket),
              "kind": artifact.get("kind"),
          }
          for artifact in fio.artifacts(findings)
      ],
      "jobs": jobs_map,
      "discrepancies": fio.get_list(findings, "discrepancies"),
  }


def build_job_manifest(
    findings: dict[str, object], bucket: str | None = None
) -> dict[str, object]:
  """Constructs the canonical JOB_MANIFEST.json conforming to Schema v2.2.

  Args:
    findings: Diagnostic findings dictionary conforming to Schema v2.2.
    bucket: Optional target GCS bucket override.

  Returns:
    Job manifest dictionary adhering to Schema v2.2.
  """
  owners = _owners(findings)
  validate_ownership_emails(owners)

  now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
  workload = fio.json_value(fio.get_raw(findings, "identity.workload"))
  target_bucket = fio.gcs_bucket(findings, bucket)
  job_prefix = f"jobs/{workload}" if workload else "jobs/NOT_PROVIDED"

  chip_count = fio.get_int(findings, "identity.chip_count")
  slice_count = fio.get_int(findings, "identity.slice_count")
  accelerator = fio.json_value(
      fio.get_raw(findings, "identity.accelerator_type")
  )
  chips_per_slice = validate_topology(accelerator, chip_count, slice_count)

  storage_key = fio.json_value(fio.get_raw(findings, "identity.storage_key"))

  artifacts_out = []
  for artifact in fio.artifacts(findings):
    artifact_path = artifact["path"]
    artifacts_out.append({
        "path": artifact_path,
        "gcs_uri": _gcs(findings, artifact_path, bucket=target_bucket),
        "kind": artifact.get("kind"),
    })

  return {
      "schema_version": "2.2.0",
      "manifest_type": "JOB_MANIFEST",
      "generation_timestamp": now_utc,
      "source": _source(findings),
      "job_metadata": {
          "job_name": workload,
          "storage_key": storage_key,
          "bucket": target_bucket,
          "case_id": fio.json_value(
              fio.get_raw(findings, "identity.case_id")
          ),
          "issue_id": fio.json_value(
              fio.get_raw(findings, "identity.issue_id")
          ),
          "disruption_category": fio.canonical_category(findings),
          "job_storage_root": _gcs(
              findings, f"{job_prefix}/", bucket=target_bucket
          ),
      },
      "workload_specification": {
          "project_id": fio.json_value(
              fio.get_raw(findings, "identity.project_id")
          ),
          "project_number": fio.json_value(
              fio.get_raw(findings, "identity.project_number")
          ),
          "cluster_name": fio.json_value(
              fio.get_raw(findings, "identity.cluster")
          ),
          "zone": fio.json_value(fio.get_raw(findings, "identity.zone")),
          "accelerator_type": accelerator,
          "chip_count": chip_count,
          "node_count": fio.get_int(findings, "identity.node_count"),
          "slice_count": slice_count,
          "chips_per_slice": chips_per_slice,
      },
      "system_info": {
          key: fio.json_value(fio.get_raw(findings, f"system_info.{key}"))
          for key in (
              "jax_version",
              "jaxlib_version",
              "libtpu_version",
              "python_version",
              "framework_version",
              "orbax_version",
              "environment",
              "workload_type",
          )
      },
      "storage_context": {
          "checkpoint_bucket": fio.json_value(
              fio.get_raw(findings, "storage.checkpoint_bucket")
          ),
          "dataset_bucket": fio.json_value(
              fio.get_raw(findings, "storage.dataset_bucket")
          ),
          "orbax_barrier_latency_sec": fio.json_value(
              fio.get_raw(findings, "storage.orbax_barrier_latency_sec")
          ),
      },
      "diagnostic_windows": {
          "baseline_window": {
              "start": fio.json_value(
                  fio.get_raw(findings, "windows.baseline.start")
              ),
              "end": fio.json_value(
                  fio.get_raw(findings, "windows.baseline.end")
              ),
              "mean_step_time_sec": fio.json_value(
                  fio.get_raw(findings, "windows.baseline.step_time_sec")
              ),
          },
          "degraded_window": {
              "start": fio.json_value(
                  fio.get_raw(findings, "windows.degraded.start")
              ),
              "end": fio.json_value(
                  fio.get_raw(findings, "windows.degraded.end")
              ),
              "mean_step_time_sec": fio.json_value(
                  fio.get_raw(findings, "windows.degraded.step_time_sec")
              ),
          },
      },
      "ownership": owners,
      "reproduction_attempts": fio.get_list(findings, "attempts"),
      "job_progress": fio.get_raw(findings, "job_progress"),
      "uploaded_artifacts": artifacts_out,
      "evidence_files": artifacts_out,
      "discrepancies": fio.get_list(findings, "discrepancies"),
  }


def main() -> None:
  """CLI entry point for generating incident and job manifests."""
  parser = argparse.ArgumentParser(
      description="Generate JSON Schema v2.2 Manifests"
  )
  parser.add_argument(
      "--findings", required=True, help="troubleshoot_findings.json path"
  )
  parser.add_argument(
      "--mode",
      choices=["incident", "job"],
      default="job",
      help="Manifest mode",
  )
  parser.add_argument(
      "--bucket", default="", help="Optional GCS bucket override"
  )
  parser.add_argument(
      "--output_file", default=None, help="Destination JSON file path"
  )
  parser.add_argument(
      "--output_dir", default=None, help="Destination directory for manifests"
  )
  args = parser.parse_args()

  if not args.output_file and not args.output_dir:
    print(
        "[ERROR] Either --output_file or --output_dir must be specified.",
        file=sys.stderr,
    )
    sys.exit(1)

  try:
    findings, target_bucket = fio.load_for_handoff(args.findings, args.bucket)
  except fio.FindingsError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)

  workload_dir = fio.workload_dir(findings)
  if not workload_dir:
    print(
        "[ERROR] identity.workload is missing; cannot place JOB_MANIFEST.json.",
        file=sys.stderr,
    )
    sys.exit(1)

  output_dir = args.output_dir
  if (
      output_dir
      and output_dir.startswith("_wiki/")
      and not os.path.exists("_wiki")
  ):
    output_dir = output_dir[len("_wiki/") :]

  if output_dir:
    # Generate both incident and job manifests
    incident_manifest = build_incident_manifest(findings, bucket=target_bucket)
    incident_path = os.path.join(output_dir, "INCIDENT_MANIFEST.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(incident_path, "w", encoding="utf-8") as f:
      json.dump(incident_manifest, f, indent=2)
    print(f"[OK] Generated incident manifest: {incident_path}")

    job_manifest = build_job_manifest(findings, bucket=target_bucket)
    job_dir = os.path.join(output_dir, "jobs", workload_dir)
    os.makedirs(job_dir, exist_ok=True)
    job_path = os.path.join(job_dir, "JOB_MANIFEST.json")
    with open(job_path, "w", encoding="utf-8") as f:
      json.dump(job_manifest, f, indent=2)
    print(f"[OK] Generated job manifest: {job_path}")

    print("\n--- SUMMARY REPORT (summary_fields) ---")
    print("status: GENERATED")
    print(f"output_dir: {output_dir}")
    return

  if args.mode == "incident":
    manifest = build_incident_manifest(findings, bucket=target_bucket)
    existing_manifest = None
    if os.path.isfile(args.output_file):
      try:
        with open(args.output_file, "r", encoding="utf-8") as f:
          existing_manifest = json.load(f)
      except (OSError, json.JSONDecodeError):
        # Ignore unreadable or malformed prior manifests and overwrite cleanly.
        existing_manifest = None
    if isinstance(existing_manifest, dict) and "jobs" in existing_manifest:
      merged_jobs = dict(existing_manifest["jobs"])
      merged_jobs.update(manifest.get("jobs", {}))
      manifest["jobs"] = merged_jobs
  else:
    manifest = build_job_manifest(findings, bucket=target_bucket)

  output_dir = os.path.dirname(os.path.abspath(args.output_file))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
  print(
      f"[OK] Generated {args.mode} manifest: {args.output_file} (target bucket:"
      f" gs://{target_bucket})"
  )
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print("status: GENERATED")
  print(f"output_file: {args.output_file}")


if __name__ == "__main__":
  main()

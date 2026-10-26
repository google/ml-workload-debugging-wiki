#!/usr/bin/env python3
"""generate_failure_table.py - Generates Level A and Level B failure tables.

Both tables adhere to the strict 2-level granularity standard and match the
/troubleshoot format:
- Level A (Infrastructure / Hardware / Host Events):
  Header: Storage Key, Partner Issue, Support Case, Cluster, Zone, Reservation,
  Generated, Status, Google Leads.
  Columns: Timestamp (UTC), GKE Node / Host, Quoted Log Line / Root Signature,
  Diagnostic Context / Failure Domain, Impacted Workload, Status, Log Ref,
  Linked Diagnostic Packet.
- Level B (Job Progression / Disruption):
  Header: Workload Name, JobSet Scale, Machine Type / Infra Spec, Baseline &
  Degraded Windows.
  Columns: Restart Attempt (very first column, e.g. Attempt 0, Attempt 1),
  Timestamp (UTC), Culprit Pod & Host, Root Signature / Mechanism, Downstream
  Impact & Failure Mode, Failure Domain, Recovery Outcome, Linked Diagnostic
  Packet.

Supports dynamic bucket parameterization via --bucket, identity.bucket in
findings, or $HANDOFF_BUCKET.
"""

import argparse
from collections.abc import Sequence
import datetime
import os
import sys

import findings_io as fio


def _bucket_text(bucket: str | None) -> str:
  return f"gs://{bucket}" if bucket else fio.NOT_PROVIDED


def _evidence_links(
    findings: dict[str, object],
    refs: Sequence[object] | None,
    bucket: str | None = None,
) -> str:
  del bucket  # Unused.
  paths = fio.filter_to_artifacts(findings, refs)
  if not paths:
    return "none linked"
  return "<br>".join(
      f"`{fio.md_cell(artifact_path)}`" for artifact_path in paths
  )


def _progress_cell(findings: dict[str, object], key: str) -> str:
  """Returns the formatted markdown cell for job_progress.<key>."""
  return fio.md_cell(fio.get_raw(findings, f"job_progress.{key}"))


def build_level_a(
    findings: dict[str, object],
    stage: str,
    table_type: str,
    timestamp_str: str,
    bucket: str | None = None,
) -> str:
  """Builds the Level A infrastructure failure table markdown.

  Args:
    findings: Parsed troubleshoot_findings.json.
    stage: "handoff" (pending Google correlation) or "verdict" (Google verdict
      recorded).
    table_type: "summary", "timeline", or anything else for both.
    timestamp_str: Generation timestamp rendered in the header.
    bucket: Optional target GCS bucket override.

  Returns:
    The rendered markdown table.
  """
  cluster = fio.get(findings, "identity.cluster")
  zone = fio.get(findings, "identity.zone")
  workload = fio.get(findings, "identity.workload")
  issue = fio.issue_ref(findings)
  category = fio.canonical_category(findings) or fio.NOT_PROVIDED
  target_bucket = fio.gcs_bucket(findings, bucket)

  if stage == "handoff":
    infra_status_str = "PENDING_GOOGLE_CORRELATION"
    summary_category = f"{category} (Candidate - Pending Triage)"
    summary_subsystem = "Pending Internal Correlation"
    summary_corroboration = "PENDING_GOOGLE_CORRELATION"
    summary_resolution = "Awaiting Google Support / CE Triage"
  else:
    infra_status_str = "GOOGLE_VERDICT_RECORDED"
    summary_category = category
    summary_subsystem = fio.get(findings, "google_verdict.subsystem")
    summary_corroboration = fio.get(findings, "google_verdict.corroboration")
    summary_resolution = fio.get(findings, "google_verdict.resolution")

  evidence = [
      evidence_row
      for evidence_row in fio.get_list(findings, "evidence_table")
      if isinstance(evidence_row, dict)
  ]
  nodes = sorted({
      fio.ascii_text(evidence_row["gke_node"])
      for evidence_row in evidence
      if not fio.is_missing(evidence_row.get("gke_node"))
  })
  if nodes:
    impacted = f"{len(nodes)} GKE node(s) cited in evidence table"
  else:
    impacted = fio.NOT_PROVIDED

  header = (
      "# Infrastructure Failure Table: GKE Cluster / Slice / Reservation\n"
      f"**Storage Key**: {fio.get(findings, 'identity.storage_key')}"
      f" | **Target Bucket**: `{_bucket_text(target_bucket)}`"
      f" | **Partner Issue**: {issue}"
      f" | **Support Case**: {fio.get(findings, 'identity.case_id')}\n"
      f"**Cluster**: {cluster} | **Zone**: {zone}"
      f" | **Reservation**: {fio.get(findings, 'identity.reservation')}\n"
      f"**Generated**: {timestamp_str} | **Status**: {infra_status_str}\n"
      "**Google Support Lead**:"
      f" {fio.get(findings, 'owners.google_support_lead')}"
      f" | **Google CE Lead**: {fio.get(findings, 'owners.google_ce_lead')}\n"
      "**Source**: Debug Wiki /troubleshoot session"
      f" {fio.get(findings, 'source.troubleshoot_session')}"
      " (customer-visible telemetry only)\n"
      "**Authorship**: Synthesized exclusively by Google Support"
      " using customer-visible GKE coordinates.\n"
      "**Division of Labor**: External customer provides empirical symptom"
      " reports; Google synthesizes Level A and Level B failure tables.\n"
  )

  summary = (
      "\n## Summary (Canonical Infrastructure Failure Table)\n"
      "| Storage Key | Partner Issue | Target Cluster / Zone"
      " | Component Category | Root Subsystem | Impacted GKE Nodes"
      " | Telemetry Corroboration | Resolution Action |\n"
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
      f"| {fio.md_cell(fio.storage_key(findings))} | {issue}"
      f" | {fio.md_cell(cluster)} ({fio.md_cell(zone)})"
      f" | {fio.md_cell(summary_category)} | {fio.md_cell(summary_subsystem)}"
      f" | {impacted} | {fio.md_cell(summary_corroboration)}"
      f" | {fio.md_cell(summary_resolution)} |\n"
  )

  rows = []
  for evidence_row in evidence:
    if fio.is_missing(evidence_row.get("observation")):
      observation = fio.NOT_PROVIDED
    else:
      observation = f"`{fio.md_cell(evidence_row.get('observation'))}`"
    if stage == "handoff":
      status = "PENDING_GOOGLE_CORRELATION"
    else:
      status = infra_status_str
    rows.append(
        "| {ts} | {worker} | `{node}` | {obs} | {ctx} | `{wl}` | {status}"
        " | {log_ref} | {packet} |".format(
            ts=fio.md_cell(evidence_row.get("timestamp")),
            worker=fio.md_cell(evidence_row.get("worker")),
            node=fio.md_cell(evidence_row.get("gke_node")),
            obs=observation,
            ctx=fio.md_cell(evidence_row.get("context")),
            wl=fio.md_cell(workload),
            status=status,
            log_ref=fio.md_cell(evidence_row.get("log_ref")),
            packet=_evidence_links(
                findings,
                evidence_row.get("artifact_refs"),
                bucket=target_bucket,
            ),
        )
    )
  if not rows:
    not_provided = fio.NOT_PROVIDED
    rows.append(
        f"| {not_provided} | {not_provided} | {not_provided} |"
        f" {not_provided} | {not_provided} | {not_provided}"
        f" | PENDING_GOOGLE_CORRELATION | {not_provided} | none linked |"
    )

  timeline = (
      "\n## Timeline (Customer-Sanitized Evidence from the Troubleshoot Run)\n"
      "| Timestamp (UTC) | Worker | GKE Node | Quoted Log Line"
      " | Diagnostic Context | Impacted Workload | Status | Log Ref"
      " | Linked Diagnostic Packet |\n"
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
      + "\n".join(rows)
      + "\n"
  )

  if table_type == "summary":
    return header + summary
  if table_type == "timeline":
    return header + timeline
  return header + summary + timeline


def build_level_b(
    findings: dict[str, object],
    table_type: str,
    timestamp_str: str,
    bucket: str | None = None,
) -> str:
  """Builds the Level B job progression (disruption) failure table markdown.

  Args:
    findings: Parsed troubleshoot_findings.json.
    table_type: "summary", "timeline", or anything else for both.
    timestamp_str: Generation timestamp; accepted for signature parity with
      build_level_a but not rendered.
    bucket: Optional target GCS bucket override.

  Returns:
    The rendered markdown table.
  """
  del timestamp_str  # Unused.
  workload = fio.get(findings, "identity.workload")
  chips = fio.get(findings, "identity.chip_count")
  slices = fio.get(findings, "identity.slice_count")
  nodes = fio.get(findings, "identity.node_count")
  accelerator_type = fio.get(findings, "identity.accelerator_type")
  scale = f"chips: {chips}, nodes: {nodes}, slices: {slices}"
  infra_spec = f"{accelerator_type} ({scale})"
  issue = fio.issue_ref(findings)
  target_bucket = fio.gcs_bucket(findings, bucket)
  degraded_step = fio.fmt_step(
      fio.get_raw(findings, "windows.degraded.step_time_sec")
  )
  baseline_step = fio.fmt_step(
      fio.get_raw(findings, "windows.baseline.step_time_sec")
  )

  header = (
      "# Master Disruption Failure Table: Workload Progression (Level B)\n"
      f"**Workload**: {workload} | **Scale**: {scale}"
      f" | **Machine Type / Infra Spec**: `{infra_spec}`\n"
      f"**Target Bucket**: `{_bucket_text(target_bucket)}`"
      f" | **Storage Key**: {fio.get(findings, 'identity.storage_key')}"
      f" | **Partner Issue**: {issue}\n"
      f"**Customer DRI**: {fio.get(findings, 'owners.customer_dri')}"
      " | **Customer Primary Investigator**:"
      f" {fio.get(findings, 'owners.customer_primary_investigator')}\n"
      f"**Degraded Window**: {fio.window_text(findings, 'degraded')}"
      f" (step time: {degraded_step})\n"
      f"**Baseline Window**: {fio.window_text(findings, 'baseline')}"
      f" (step time: {baseline_step})\n"
      "**Authorship**: Synthesized exclusively by Google Support"
      " adhering to `/troubleshoot` failure-table standards.\n"
      "**Attribution Invariant**: Culprit hosts are quoted from primary logs;"
      " never inferred from DATA_LOSS device coordinates.\n"
  )

  summary = (
      "\n## Summary (Canonical Job Progression Failure Table)\n"
      "| Workload ID | Machine Type / Infra Spec | Scheduled Steps"
      " | Completed Steps | Goodput Eff. (%) | Lost Compute Time"
      " | Primary Barrier Blocked | Partner Tracking Issue |\n"
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
      f"| {fio.md_cell(workload)} | `{fio.md_cell(infra_spec)}`"
      f" | {_progress_cell(findings, 'scheduled_steps')}"
      f" | {_progress_cell(findings, 'completed_steps')}"
      f" | {_progress_cell(findings, 'goodput')}"
      f" | {_progress_cell(findings, 'lost_time')}"
      f" | {_progress_cell(findings, 'primary_barrier')} | {issue} |\n"
  )

  # Level B master timeline conforms to /troubleshoot columns:
  # Restart Attempt, Timestamp (UTC), Culprit Pod & Host, Root Signature /
  # Mechanism, Downstream Impact & Failure Mode, Failure Domain, Recovery
  # Outcome, Linked Diagnostic Packet.
  rows = []
  attempts = fio.get_list(findings, "attempts")
  for attempt_record in attempts:
    if not isinstance(attempt_record, dict):
      continue
    attempt_number = attempt_record.get("attempt")
    if attempt_number is not None:
      attempt_label = f"**Attempt {attempt_number}**"
    else:
      attempt_label = "**Attempt ?**"
    timestamp = (
        attempt_record.get("start")
        or attempt_record.get("timestamp")
        or fio.NOT_PROVIDED
    )
    culprit = (
        attempt_record.get("culprit")
        or attempt_record.get("worker")
        or fio.NOT_PROVIDED
    )
    root_signature = (
        attempt_record.get("root_signature")
        or attempt_record.get("exit")
        or fio.NOT_PROVIDED
    )
    impact = (
        attempt_record.get("downstream_impact")
        or attempt_record.get("notes")
        or fio.NOT_PROVIDED
    )
    domain = (
        attempt_record.get("failure_domain")
        or attempt_record.get("case")
        or fio.NOT_PROVIDED
    )
    recovery = attempt_record.get("recovery_outcome") or fio.NOT_PROVIDED
    packet = _evidence_links(
        findings, attempt_record.get("artifact_refs"), bucket=target_bucket
    )

    rows.append(
        f"| {attempt_label} | {fio.md_cell(timestamp)}"
        f" | `{fio.md_cell(culprit)}` | `{fio.md_cell(root_signature)}`"
        f" | {fio.md_cell(impact)} | `{fio.md_cell(domain)}`"
        f" | {fio.md_cell(recovery)} | {packet} |"
    )

  if not rows:
    not_provided = fio.NOT_PROVIDED
    rows.append(
        f"| {not_provided} | {not_provided} | {not_provided} |"
        f" {not_provided} | {not_provided} | {not_provided} |"
        f" {not_provided} | none linked |"
    )

  timeline = (
      "\n## Master Disruption Failure Timeline\n"
      "| Restart Attempt | Timestamp (UTC) | Culprit Pod & Host"
      " | Root Signature / Mechanism | Downstream Impact & Failure Mode"
      " | Failure Domain | Recovery Outcome | Linked Diagnostic Packet |\n"
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
      + "\n".join(rows)
      + "\n"
  )

  if table_type == "summary":
    return header + summary
  if table_type == "timeline":
    return header + timeline
  return header + summary + timeline


def main() -> None:
  """CLI entry point for generating Level A and Level B failure tables."""
  default_output_dir = os.path.join(
      os.environ.get("TMPDIR", "/tmp"), "failure_tables"
  )
  parser = argparse.ArgumentParser(
      description="Generate Level A and Level B Failure Tables"
  )
  parser.add_argument(
      "--findings", required=True, help="troubleshoot_findings.json path"
  )
  parser.add_argument(
      "--stage",
      choices=["handoff", "verdict"],
      default="handoff",
      help="Handoff stage",
  )
  parser.add_argument(
      "--table_type",
      choices=["both", "summary", "timeline"],
      default="both",
      help="Table presentation type",
  )
  parser.add_argument(
      "--bucket", default="", help="Optional GCS bucket override"
  )
  parser.add_argument(
      "--output_dir", default=default_output_dir, help="Local output directory"
  )

  args = parser.parse_args()

  try:
    findings, target_bucket = fio.load_for_handoff(args.findings, args.bucket)
  except fio.FindingsError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)

  workload_dir = fio.workload_dir(findings)
  if not workload_dir:
    print(
        "[ERROR] identity.workload is missing; cannot place the Level B table.",
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

  infra_findings_dir = os.path.join(output_dir, "infrastructure", "findings")
  infra_google_findings_dir = os.path.join(infra_findings_dir, "google")
  legacy_infra_dir = os.path.join(output_dir, "infrastructure")

  job_findings_google_dir = os.path.join(
      output_dir, "jobs", workload_dir, "findings", "google"
  )
  legacy_job_tables_dir = os.path.join(
      output_dir, "jobs", workload_dir, "failure_tables"
  )

  os.makedirs(infra_findings_dir, exist_ok=True)
  os.makedirs(infra_google_findings_dir, exist_ok=True)
  os.makedirs(legacy_infra_dir, exist_ok=True)
  os.makedirs(job_findings_google_dir, exist_ok=True)
  os.makedirs(legacy_job_tables_dir, exist_ok=True)

  timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%dT%H:%M:%SZ"
  )
  date_slug = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%d_%H%M"
  )

  infra_table = build_level_a(
      findings,
      args.stage,
      args.table_type,
      timestamp_str,
      bucket=target_bucket,
  )
  job_table = build_level_b(
      findings, args.table_type, timestamp_str, bucket=target_bucket
  )

  infra_latest_path = os.path.join(infra_findings_dir, "_infra_latest.md")
  infra_legacy_path = os.path.join(
      legacy_infra_dir, "Level_A_Infrastructure.md"
  )
  google_infra_finding_path = os.path.join(
      infra_google_findings_dir,
      f"finding_{date_slug}_infrastructure_health.md",
  )

  job_latest_path = os.path.join(job_findings_google_dir, "_google_latest.md")
  job_legacy_path = os.path.join(
      legacy_job_tables_dir, "Level_B_Job_Progression.md"
  )
  google_job_finding_path = os.path.join(
      job_findings_google_dir, f"finding_{date_slug}_job_progression.md"
  )

  for path in (infra_latest_path, infra_legacy_path, google_infra_finding_path):
    with open(path, "w", encoding="utf-8") as f:
      f.write(infra_table)

  for path in (job_latest_path, job_legacy_path, google_job_finding_path):
    with open(path, "w", encoding="utf-8") as f:
      f.write(job_table)

  print(
      "[OK] Generated Level A Infrastructure Failure Table:"
      f" {infra_latest_path}"
  )
  print(f"[OK] Generated Level A Legacy Mirror: {infra_legacy_path}")
  print(
      "[OK] Generated Level B Job Progression Failure Table:"
      f" {job_latest_path}"
  )
  print(f"[OK] Generated Level B Legacy Mirror: {job_legacy_path}")
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print("status: GENERATED")
  print(f"output_file: {job_latest_path}")


if __name__ == "__main__":
  main()

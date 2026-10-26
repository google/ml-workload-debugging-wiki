#!/usr/bin/env python3
"""render_readme.py - Renders handoff markdown overviews from templates.

Supports dynamic bucket parameterization via --bucket, identity.bucket
in findings, or $HANDOFF_BUCKET.
"""

import argparse
import datetime
import os
import sys
import types

import findings_io as fio

TEMPLATES = types.MappingProxyType({
    "incident": "INCIDENT_OVERVIEW.template.md",
    "readme": "README.template.md",
    "root": "README.template.md",
    "job": "JOB_README.template.md",
})


def readme_values(
    findings: dict[str, object], bucket: str | None = None
) -> dict[str, object]:
  """Computes template substitution values from findings schema."""
  workload = fio.get_raw(findings, "identity.workload")
  job_prefix = f"jobs/{workload}/" if workload else None
  orbax = fio.get_raw(findings, "storage.orbax_barrier_latency_sec")
  target_bucket = fio.gcs_bucket(findings, bucket)

  failure_tables = [
      artifact_path
      for artifact_path in fio.artifact_paths(findings)
      if "failure_tables/" in artifact_path
      or artifact_path.endswith("_infra_latest.md")
      or artifact_path.endswith("_google_latest.md")
  ]
  rendered_tables = []
  for artifact_path in failure_tables:
    table_uri = (
        fio.gcs_path(findings, artifact_path, bucket=target_bucket)
        or artifact_path
    )
    rendered_tables.append(f"- `{table_uri}`")
  failure_tables_list = "\n".join(rendered_tables) or "- None uploaded"

  if workload:
    category = fio.canonical_category(findings) or fio.NOT_PROVIDED
    chips = fio.get(findings, "identity.chip_count")
    nodes = fio.get(findings, "identity.node_count")
    slices = fio.get(findings, "identity.slice_count")
    jobs_list = (
        f"- `{fio.ascii_text(workload)}`: Category `{category}`, "
        f"Chips `{chips}`, Nodes `{nodes}`, Slices `{slices}`"
    )
  else:
    jobs_list = f"- {fio.NOT_PROVIDED}"

  now_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%dT%H:%M:%SZ"
  )
  session_id = fio.get(findings, "source.troubleshoot_session")
  issue_ref = fio.issue_ref(findings)
  lead = fio.get(findings, "owners.google_support_lead")
  ownership = (
      f"- {now_utc}: Handoff package generated from "
      f"troubleshoot session {session_id}; partner issue {issue_ref}; "
      f"Google Support Lead {lead}"
  )

  job_storage = (
      fio.gcs_path(findings, job_prefix, bucket=target_bucket)
      if job_prefix
      else None
  ) or fio.NOT_PROVIDED

  job_artifacts = (
      fio.artifacts_md(findings, job_prefix, bucket=target_bucket)
      if job_prefix
      else "- None uploaded"
  )

  return {
      "ISSUE_REF": fio.issue_ref(findings),
      "ISSUE_URL": fio.issue_url(findings),
      "CASE_ID": fio.get(findings, "identity.case_id"),
      "STORAGE_KEY": fio.get(findings, "identity.storage_key"),
      "BUCKET_NAME": target_bucket,
      "STORAGE_ROOT": (
          fio.storage_root(findings, bucket=target_bucket) or fio.NOT_PROVIDED
      ),
      "JOB_STORAGE_ROOT": job_storage,
      "PROJECT": fio.get(findings, "identity.project_id"),
      "PROJECT_NUMBER": fio.get(findings, "identity.project_number"),
      "CLUSTER": fio.get(findings, "identity.cluster"),
      "ZONE": fio.get(findings, "identity.zone"),
      "CELL": fio.get(findings, "identity.zone"),
      "RESERVATION": fio.get(findings, "identity.reservation"),
      "WORKLOAD": fio.get(findings, "identity.workload"),
      "ISSUE_TYPE": fio.canonical_category(findings) or fio.NOT_PROVIDED,
      "ACCELERATOR_TYPE": fio.get(findings, "identity.accelerator_type"),
      "CHIP_COUNT": fio.get(findings, "identity.chip_count"),
      "NODE_COUNT": fio.get(findings, "identity.node_count"),
      "SLICE_COUNT": fio.get(findings, "identity.slice_count"),
      "CUSTOMER_DRI": fio.get(findings, "owners.customer_dri"),
      "PRIMARY_INVESTIGATOR": fio.get(
          findings, "owners.customer_primary_investigator"
      ),
      "GOOGLE_LEAD": fio.get(findings, "owners.google_support_lead"),
      "GOOGLE_CE": fio.get(findings, "owners.google_ce_lead"),
      "GOOD_WINDOW": fio.window_text(findings, "baseline"),
      "BAD_WINDOW": fio.window_text(findings, "degraded"),
      "BASELINE_STEP": fio.fmt_step(
          fio.get_raw(findings, "windows.baseline.step_time_sec")
      ),
      "DEGRADED_STEP": fio.fmt_step(
          fio.get_raw(findings, "windows.degraded.step_time_sec")
      ),
      "JAX_VERSION": fio.get(findings, "system_info.jax_version"),
      "JAXLIB_VERSION": fio.get(findings, "system_info.jaxlib_version"),
      "LIBTPU_VERSION": fio.get(findings, "system_info.libtpu_version"),
      "PYTHON_VERSION": fio.get(findings, "system_info.python_version"),
      "FRAMEWORK_VERSION": fio.get(findings, "system_info.framework_version"),
      "ORBAX_VERSION": fio.get(findings, "system_info.orbax_version"),
      "CHECKPOINT_BUCKET": fio.get(findings, "storage.checkpoint_bucket"),
      "DATASET_BUCKET": fio.get(findings, "storage.dataset_bucket"),
      "ORBAX_LATENCY": fio.fmt_step(orbax),
      "OWNERSHIP_HISTORY": ownership,
      "JOBS_LIST": jobs_list,
      "ARTIFACTS_LIST": fio.artifacts_md(findings, bucket=target_bucket),
      "JOB_ARTIFACTS_LIST": job_artifacts,
      "FAILURE_TABLES_LIST": failure_tables_list,
      "CUSTOMER_NEXT_STEPS": fio.next_steps_md(findings),
      "PROVES_LIST": fio.proves_md(findings),
      "CANNOT_PROVE_LIST": fio.cannot_prove_md(findings),
      "DISCREPANCIES": fio.discrepancies_md(findings),
      "TROUBLESHOOT_SOURCE": fio.get(findings, "source.troubleshoot_session"),
  }


def render_readme(
    findings: dict[str, object],
    mode: str,
    output_file: str,
    bucket: str | None = None,
) -> str:
  """Renders a handoff overview or README file from a template.

  Args:
    findings: Diagnostic findings dictionary conforming to Schema v2.2.
    mode: Template mode ('incident', 'job', 'readme', or 'root').
    output_file: Destination file path for rendered markdown.
    bucket: Optional target GCS bucket override.

  Returns:
    The rendered markdown string.

  Raises:
    FileNotFoundError: The template for the mode does not exist.
  """
  template_name = TEMPLATES.get(mode, "JOB_README.template.md")
  template_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "templates",
      template_name,
  )
  if not os.path.isfile(template_path):
    raise FileNotFoundError(f"Template not found: {template_path}")

  with open(template_path, "r", encoding="utf-8") as f:
    raw = f.read()

  values = readme_values(findings, bucket=bucket)
  rendered = fio.render_template(raw, values)

  output_dir = os.path.dirname(os.path.abspath(output_file))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output_file, "w", encoding="utf-8") as f:
    f.write(rendered)
  return rendered


def render_incident_overview(
    findings: dict[str, object],
    output_file: str,
    bucket: str | None = None,
) -> str:
  """Renders the top-level INCIDENT_OVERVIEW.md for a handoff package.

  Args:
    findings: Diagnostic findings dictionary conforming to Schema v2.2.
    output_file: Destination file path for rendered markdown.
    bucket: Optional target GCS bucket override.

  Returns:
    The rendered markdown string.
  """
  return render_readme(
      findings=findings,
      mode="incident",
      output_file=output_file,
      bucket=bucket,
  )


def main() -> None:
  """CLI entry point for rendering handoff README and overview files."""
  parser = argparse.ArgumentParser(description="Render overview markdown files")
  parser.add_argument(
      "--findings", required=True, help="troubleshoot_findings.json path"
  )
  parser.add_argument(
      "--mode",
      choices=["incident", "job", "readme", "root"],
      default="job",
  )
  parser.add_argument(
      "--bucket", default="", help="Optional GCS bucket override"
  )
  parser.add_argument(
      "--output_file", default=None, help="Output markdown path"
  )
  parser.add_argument(
      "--output_dir", default=None, help="Output directory for readmes"
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
  except fio.FindingsError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

  workload_dir = fio.workload_dir(findings)

  output_dir = args.output_dir
  if (
      output_dir
      and output_dir.startswith("_wiki/")
      and not os.path.exists("_wiki")
  ):
    output_dir = output_dir[len("_wiki/") :]

  output_file = args.output_file
  if (
      output_file
      and output_file.startswith("_wiki/")
      and not os.path.exists("_wiki")
  ):
    output_file = output_file[len("_wiki/") :]

  if output_dir:
    incident_path = os.path.join(output_dir, "INCIDENT_OVERVIEW.md")
    render_readme(findings, "incident", incident_path, bucket=target_bucket)
    print(f"[OK] Rendered incident overview to {incident_path}")

    root_path = os.path.join(output_dir, "README.md")
    render_readme(findings, "readme", root_path, bucket=target_bucket)
    print(f"[OK] Rendered root README to {root_path}")

    if workload_dir:
      # BUCKET_STRUCTURE.md: JOB_README.md is the workload context and
      # README.md mirrors it for web console browsing.
      job_dir = os.path.join(output_dir, "jobs", workload_dir)
      for name in ("JOB_README.md", "README.md"):
        job_path = os.path.join(job_dir, name)
        render_readme(findings, "job", job_path, bucket=target_bucket)
        print(f"[OK] Rendered job {name} to {job_path}")
    else:
      print(
          "[WARN] identity.workload is missing; job READMEs not rendered.",
          file=sys.stderr,
      )

    print("\n--- SUMMARY REPORT (summary_fields) ---")
    print("status: RENDERED")
    print(f"output_dir: {output_dir}")
    return

  render_readme(findings, args.mode, output_file, bucket=target_bucket)
  print(f"[OK] Rendered {args.mode} overview to {output_file}")
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print("status: RENDERED")
  print(f"output_file: {output_file}")


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Renders partner issue body / comment from troubleshoot_findings.json.

Structure follows the General JAX/TPU Partner Issue Template.

Supports dynamic bucket parameterization via --bucket, identity.bucket in
findings, or $HANDOFF_BUCKET.
"""

import argparse
import os
import sys
import types
from typing import Any

import findings_io as fio

TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "templates",
    "ISSUE_COMMENT.template.md",
)

CATEGORY_LABELS = (
    ("bug", "**Bug/Crash:** Runtime error, XLA error, Python exception."),
    ("memory", "**Memory/Resource:** OOM, HBM fragmentation, memory leak."),
    (
        "performance",
        "**Performance:** Compilation hang, slow execution, regression.",
    ),
    (
        "dependencies",
        "**Dependencies/Upgrade:** JAX/JAXlib, libtpu, and TPU versions.",
    ),
)
CANONICAL_TO_BOX = types.MappingProxyType({
    "category_1_step_time_latency": "performance",
    "category_2_memory_oom": "memory",
})

ARTIFACT_BOXES = (
    ("logs", "**Logs** (Full stack trace)"),
    ("hlo_dump", "**HLO/XLA dump** (Required for Compiler/Runtime errors)"),
    (
        "mosaic_dump",
        "**Mosaic dump** (Required for Pallas/Splash Attention issues)",
    ),
)


def category_checklist(findings: dict[str, Any]) -> str:
  """Renders the markdown category checklist for the issue template."""
  canonical = fio.canonical_category(findings)
  box = CANONICAL_TO_BOX.get(canonical)
  lines = []
  for key, label in CATEGORY_LABELS:
    lines.append(f"- [{'x' if box == key else ' '}] {label}")
  if canonical and box is None:
    lines.append(f"- [x] **Other:** {canonical}")
  else:
    lines.append("- [ ] **Other:** ____________________")
  return "\n".join(lines)


def _system_info_field(findings: dict[str, Any], key: str) -> str:
  """Reads a system/identity field with NOT_COLLECTED fallback."""
  return fio.get(findings, key, missing=fio.NOT_COLLECTED)


def _hypothesis_field(findings: dict[str, Any], key: str) -> str:
  """Reads a field under hypothesis.<key> from findings."""
  return fio.get(findings, f"hypothesis.{key}")


def accelerator_line(findings: dict[str, Any]) -> str:
  """Renders the single-line accelerator topology summary."""
  accelerator_type = _system_info_field(findings, "identity.accelerator_type")
  chip_count = _system_info_field(findings, "identity.chip_count")
  node_count = _system_info_field(findings, "identity.node_count")
  slice_count = _system_info_field(findings, "identity.slice_count")
  return (
      f"accelerator_type: {accelerator_type}; "
      f"chips: {chip_count}; "
      f"nodes: {node_count}; "
      f"slices: {slice_count}"
  )


def _field(row: dict[str, Any], key: str) -> str:
  """Formats a dictionary field as ASCII text or NOT_PROVIDED."""
  raw_value = row.get(key)
  if not fio.is_missing(raw_value):
    return fio.ascii_text(raw_value)
  return fio.NOT_PROVIDED


def attempts_list(findings: dict[str, Any]) -> str:
  """Renders the markdown bullet list of job restart attempts."""
  items = [
      attempt
      for attempt in fio.get_list(findings, "attempts")
      if isinstance(attempt, dict)
  ]
  if not items:
    return f"- {fio.NOT_PROVIDED}"
  lines = []
  for attempt in items:
    step_str = fio.fmt_step(attempt.get("step_time_sec"))
    lines.append(
        f"- Attempt {_field(attempt, 'attempt')} "
        f"(JobSet UID {_field(attempt, 'jobset_uid')}): "
        f"start {_field(attempt, 'start')}, end {_field(attempt, 'end')}, "
        f"step time {step_str}, exit: {_field(attempt, 'exit')}. "
        f"Notes: {_field(attempt, 'notes')}"
    )
  return "\n".join(lines)


def log_filters(findings: dict[str, Any]) -> str:
  """Formats the Cloud Logging filter blocks from findings."""
  items = [
      log_filter
      for log_filter in fio.get_list(findings, "log_filters")
      if isinstance(log_filter, dict)
  ]
  if not items:
    return f"- {fio.NOT_PROVIDED}"
  blocks = []
  for index, log_filter in enumerate(items, 1):
    label = _field(log_filter, "label")
    window = _field(log_filter, "window")
    count = _field(log_filter, "result_count")
    raw_filter = fio.ascii_text(log_filter.get("filter", fio.NOT_PROVIDED))
    blocks.append(
        f"Filter {index}: {label}\n"
        f"Window: {window} | Result count: {count}\n"
        f"```text\n{raw_filter}\n```"
    )
  return "\n\n".join(blocks)


def artifact_checklist(findings: dict[str, Any]) -> str:
  """Renders the uploaded artifact checklist for the issue template."""
  uploaded_kinds = {
      artifact.get("kind")
      for artifact in fio.artifacts(findings)
      if artifact.get("kind")
  }
  lines = []
  for key, label in ARTIFACT_BOXES:
    lines.append(f"- [{'x' if key in uploaded_kinds else ' '}] {label}")
  known_kinds = {kind for kind, _ in ARTIFACT_BOXES}
  other = [
      artifact
      for artifact in fio.artifacts(findings)
      if artifact.get("kind") not in known_kinds
  ]
  if other:
    other_paths = ", ".join(artifact["path"] for artifact in other)
    lines.append(f"- [x] **Other:** {other_paths}")
  else:
    lines.append("- [ ] **Other:** ____________________")
  return "\n".join(lines)


def artifact_locations(
    findings: dict[str, Any], bucket: str | None = None
) -> str:
  """Renders the bullet list mapping artifact paths to GCS URIs."""
  target_bucket = fio.gcs_bucket(findings, bucket)
  lines = []
  for artifact in fio.artifacts(findings):
    artifact_path = artifact["path"]
    gcs_uri = fio.gcs_path(findings, artifact_path, bucket=target_bucket)
    lines.append(f"- `{artifact_path}`: `{gcs_uri}`")
  return "\n".join(lines) if lines else f"- {fio.NOT_PROVIDED}"


def uploaded_files_bullet_list(
    findings: dict[str, Any], bucket: str | None = None
) -> str:
  """Renders the bullet list of uploaded GCS file paths."""
  target_bucket = fio.gcs_bucket(findings, bucket)
  items = [
      fio.gcs_path(findings, artifact_path, bucket=target_bucket)
      for artifact_path in fio.artifact_paths(findings)
  ]
  return fio.bullet_list(
      items, missing="No files recorded in findings.artifacts"
  )


def render_issue_body(
    findings: dict[str, Any],
    human_operator: str | None = None,
    bucket: str | None = None,
) -> str:
  """Renders the full partner issue body from findings."""
  target_bucket = fio.gcs_bucket(findings, bucket)

  human = human_operator or fio.get_raw(
      findings, "owners.customer_primary_investigator"
  )
  human_text = (
      fio.ascii_text(human) if not fio.is_missing(human) else fio.NOT_PROVIDED
  )

  infra_status = (
      "GOOGLE_VERDICT_RECORDED"
      if fio.get_raw(findings, "google_verdict")
      else "PENDING_GOOGLE_CORRELATION"
  )

  remediation_lines = []
  evidence = [
      evidence_row
      for evidence_row in fio.get_list(findings, "evidence_table")
      if isinstance(evidence_row, dict)
  ]
  culprit_nodes = sorted({
      fio.ascii_text(evidence_row["gke_node"])
      for evidence_row in evidence
      if not fio.is_missing(evidence_row.get("gke_node"))
  })
  if culprit_nodes:
    remediation_lines.append(
        "The following node(s) exhibited fatal interruption signatures:"
    )
    for node_name in culprit_nodes:
      remediation_lines.append(f"- Cordon node: `kubectl cordon {node_name}`")
      remediation_lines.append(
          "- Exclude node: Add taint or nodeSelector exclusion for"
          f" `{node_name}` in next JobSet attempt."
      )
  else:
    remediation_lines.append(
        "- No specific faulty nodes isolated for cordoning at this time."
    )
  remediation_block = "\n".join(remediation_lines)

  degraded_step = fio.fmt_step(
      fio.get_raw(findings, "windows.degraded.step_time_sec")
  )
  baseline_step = fio.fmt_step(
      fio.get_raw(findings, "windows.baseline.step_time_sec")
  )
  support_refs = fio.bullet_list(
      fio.get_list(findings, "hypothesis.support_refs")
  )
  gap_refs = fio.bullet_list(fio.get_list(findings, "hypothesis.gap_refs"))
  console_url = (
      fio.console_url(findings, bucket=target_bucket) or fio.NOT_PROVIDED
  )

  values = {
      "WORKLOAD_NAME": fio.get(findings, "identity.workload"),
      "ISSUE_REF": fio.issue_ref(findings),
      "CASE_ID": fio.get(findings, "identity.case_id"),
      "STORAGE_KEY": fio.get(findings, "identity.storage_key"),
      "TROUBLESHOOT_SOURCE": fio.get(findings, "source.troubleshoot_session"),
      "INFRA_CORRELATION_STATUS": infra_status,
      "CATEGORY_CHECKLIST": category_checklist(findings),
      "DISRUPTION_CATEGORY": (
          fio.canonical_category(findings) or fio.NOT_PROVIDED
      ),
      "JAX_VERSION": _system_info_field(findings, "system_info.jax_version"),
      "JAXLIB_VERSION": _system_info_field(
          findings, "system_info.jaxlib_version"
      ),
      "LIBTPU_VERSION": _system_info_field(
          findings, "system_info.libtpu_version"
      ),
      "ACCELERATOR": accelerator_line(findings),
      "ENVIRONMENT": _system_info_field(findings, "system_info.environment"),
      "PYTHON_VERSION": _system_info_field(
          findings, "system_info.python_version"
      ),
      "WORKLOAD_TYPE": _system_info_field(
          findings, "system_info.workload_type"
      ),
      "PROJECT_ID": _system_info_field(findings, "identity.project_id"),
      "PROJECT_NUMBER": _system_info_field(findings, "identity.project_number"),
      "CLUSTER": _system_info_field(findings, "identity.cluster"),
      "ZONE": _system_info_field(findings, "identity.zone"),
      "DEGRADED_WINDOW": fio.window_text(findings, "degraded"),
      "DEGRADED_STEP": degraded_step,
      "BASELINE_WINDOW": fio.window_text(findings, "baseline"),
      "BASELINE_STEP": baseline_step,
      "ATTEMPTS_LIST": attempts_list(findings),
      "LOG_FILTERS": log_filters(findings),
      "ARTIFACT_CHECKLIST": artifact_checklist(findings),
      "CUSTOMER_HYPOTHESIS": _hypothesis_field(findings, "text"),
      "HYPOTHESIS_AUTHOR": _hypothesis_field(findings, "author"),
      "HYPOTHESIS_CONFIDENCE": _hypothesis_field(findings, "confidence"),
      "HYPOTHESIS_SUPPORT_REFS": support_refs,
      "HYPOTHESIS_GAP_REFS": gap_refs,
      "HYPOTHESIS_FALSIFIER": _hypothesis_field(findings, "falsifier"),
      "PROVES_LIST": fio.proves_md(findings),
      "CANNOT_PROVE_LIST": fio.cannot_prove_md(findings),
      "QUESTIONS_FOR_GOOGLE": fio.questions_md(findings),
      "CUSTOMER_NEXT_STEPS": fio.next_steps_md(findings),
      "DISCREPANCIES": fio.discrepancies_md(findings),
      "STORAGE_CONSOLE_URL": console_url,
      "ARTIFACT_LOCATIONS": artifact_locations(findings, bucket=target_bucket),
      "GOOGLE_CE_LEAD": fio.get(findings, "owners.google_ce_lead"),
      "GOOGLE_SUPPORT_LEAD": fio.get(findings, "owners.google_support_lead"),
      "CUSTOMER_DRI": fio.get(findings, "owners.customer_dri"),
      "CUSTOMER_PRIMARY_INVESTIGATOR": fio.get(
          findings, "owners.customer_primary_investigator"
      ),
      "VERIFIED_HUMAN_OPERATOR": human_text,
      "ISSUE_URL": fio.issue_url(findings),
      "REMEDIATION_BLOCK": remediation_block,
      "UPLOADED_FILES_LIST": uploaded_files_bullet_list(
          findings, bucket=target_bucket
      ),
  }

  with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    raw = f.read()
  return fio.render_template(raw, values)


def print_readiness_scorecard(
    findings: dict[str, Any], bucket: str | None = None
) -> None:
  """Prints the pre-flight handoff readiness scorecard to stdout."""
  target_bucket = fio.gcs_bucket(findings, bucket)
  accelerator_type = fio.get(findings, "identity.accelerator_type")
  print("\n=======================================================")
  print("PRE-FLIGHT HANDOFF READINESS SCORECARD")
  print("=======================================================")
  # Report only what the findings record; nothing here is re-verified.
  node_count = fio.get(findings, "identity.node_count")
  print(f"1. Node Set Scoping: {node_count} node(s)")
  print(f"2. Accelerator Type: {accelerator_type}")
  hlo_count = len(fio.artifacts_of_kind(findings, "hlo_dump"))
  print(
      f"3. Redaction Gate: {hlo_count} HLO dump artifact(s) recorded"
      " (sanitized by package-handoff-artifacts)"
  )
  print(
      f"4. Bucket Security: gs://{target_bucket or fio.NOT_PROVIDED}"
      " (checked by verify-handoff-bucket-access)"
  )
  print(f"5. Partner Tracker Target: {fio.issue_url(findings)}")
  print("=======================================================\n")


def main() -> None:
  """CLI entry point for rendering partner issue body."""
  parser = argparse.ArgumentParser(description="Render partner issue body")
  parser.add_argument(
      "--findings", required=True, help="Path to troubleshoot_findings.json"
  )
  parser.add_argument(
      "--output_file",
      "--output",
      dest="output_file",
      required=True,
      help="Destination markdown path",
  )
  parser.add_argument(
      "--bucket", default="", help="Optional GCS bucket override"
  )
  parser.add_argument(
      "--operator_email", default=None, help="Verified human operator email"
  )
  args = parser.parse_args()

  try:
    findings, target_bucket = fio.load_for_handoff(args.findings, args.bucket)
  except fio.FindingsError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

  body = render_issue_body(
      findings, human_operator=args.operator_email, bucket=target_bucket
  )
  output_file = args.output_file
  if (
      output_file
      and output_file.startswith("_wiki/")
      and not os.path.exists("_wiki")
  ):
    output_file = output_file[len("_wiki/") :]

  output_dir = os.path.dirname(os.path.abspath(output_file))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output_file, "w", encoding="utf-8") as f:
    f.write(body)
  print_readiness_scorecard(findings, bucket=target_bucket)
  print(f"[OK] Rendered partner issue body to {output_file}")
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print("status: RENDERED")
  print(f"output_file: {args.output_file}")
  print(f"bucket: {target_bucket}")
  # Nothing is posted from here: there is no Partner Issue Tracker client in
  # this toolchain. The operator pastes the rendered body at this link.
  print(f"issue_url: {fio.issue_url(findings)}")


if __name__ == "__main__":
  main()

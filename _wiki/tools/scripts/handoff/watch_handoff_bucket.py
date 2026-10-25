#!/usr/bin/env python3
"""watch_handoff_bucket.py - GCS Bucket Change-Detection and Alerting Monitor.

Scans gs://<bucket>/<storage_key>/ across all diagnostic directories.
Maintains a lightweight hash/generation cache in the state file.

Supports Universal Symmetrical Monitoring:
- Automatically and symmetrically monitors the incident repository across all
tracks.
- If 1P Google updates are detected (in findings/google/,
infrastructure/findings/,
  or INCIDENT_MANIFEST.json), extracts:
    a) Verified Verdict / Status (e.g. VERIFIED_HARDWARE_FAULT on node
    gke-tpu-node-pool-xxxx).
    b) Supporting Evidence from 1P (quoted culprit host, hardware error
    registers, optical attenuation findings).
    c) Actionable Workarounds / Next Steps (kubectl cordon, recommended XLA
    flags, or verified Orbax checkpoint recovery paths).
- If 3P Customer updates are detected (in findings/customer/ or data/), reports
new customer telemetry, traces, and notes.
- If both or general files are updated, reports all changes cleanly in the alert
card.
"""

import argparse
from collections.abc import Sequence
import datetime
import json
import os
import re
import subprocess

import findings_io as fio

_INFRA_STATUS_RE = re.compile(r"\*\*Status\*\*:\s*([^\n|]+)")
_GKE_TPU_NODE_RE = re.compile(r"\b(gke-tpu-[a-zA-Z0-9-]+)\b")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$")
_NUMBERED_ITEM_RE = re.compile(r"^\d+\.")
_LIST_BULLET_PREFIX_RE = re.compile(r"^[-*\d.]+\s*")
_CHECKPOINT_GCS_URI_RE = re.compile(
    r"(gs://[^\s'\"`]+(?:checkpoint|ckpt)[^\s'\"`]*)",
    re.IGNORECASE,
)
_GCS_URI_RE = re.compile(r"(gs://[^\s'\"`]+)")


def _deduplicate_items(items: Sequence[str]) -> list[str]:
  """Deduplicates string items in order while stripping backticks and spaces."""
  seen = set()
  unique_items = []
  for item in items:
    normalized_item = item.strip().strip("`")
    if normalized_item and normalized_item not in seen:
      seen.add(normalized_item)
      unique_items.append(normalized_item)
  return unique_items


def list_gcs_objects(
    bucket: str, storage_key: str, local_dir: str | None = None
) -> dict[str, str]:
  """Returns a dictionary of {relative_path: signature}."""
  if local_dir:
    objects = {}
    base_path = os.path.abspath(local_dir)
    if not os.path.isdir(base_path):
      return {}
    for root, _, files in os.walk(base_path):
      for filename in sorted(files):
        full_path = os.path.join(root, filename)
        relative_path = os.path.relpath(full_path, base_path).replace("\\", "/")
        try:
          size = os.path.getsize(full_path)
          mtime = os.path.getmtime(full_path)
          objects[relative_path] = f"size:{size}_mtime:{mtime}"
        except OSError:
          # Skip files that vanish or become unreadable during directory walk.
          continue
    return objects

  prefix = f"gs://{bucket}/{storage_key}/"
  cmd = ["gcloud", "storage", "ls", "--long", f"{prefix}**"]
  try:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, check=False
    )
    if proc.returncode != 0:
      return {}
  except (subprocess.SubprocessError, OSError):
    # Return empty mapping when gcloud is unavailable or times out.
    return {}

  objects = {}
  for line in proc.stdout.splitlines():
    parts = line.strip().split()
    if len(parts) >= 3 and parts[-1].startswith("gs://"):
      uri = parts[-1]
      size = parts[0]
      mtime = parts[1] if len(parts) > 2 else ""
      relative_path = uri[len(prefix) :]
      objects[relative_path] = f"size:{size}_mtime:{mtime}"
  return objects


def is_1p_path(relative_path: str) -> bool:
  """Returns True if path belongs to 1P Google updates."""
  normalized_path = relative_path.replace("\\", "/")
  if "findings/google/" in normalized_path or normalized_path.endswith(
      "_google_latest.md"
  ):
    return True
  if (
      normalized_path == "infrastructure/findings/_infra_latest.md"
      or "infrastructure/findings/" in normalized_path
      or normalized_path.startswith("infrastructure/")
  ):
    return True
  if normalized_path == "INCIDENT_MANIFEST.json":
    return True
  return False


def is_3p_path(relative_path: str) -> bool:
  """Returns True if path belongs to 3P customer updates."""
  normalized_path = relative_path.replace("\\", "/")
  if "findings/customer/" in normalized_path or normalized_path.endswith(
      "_customer_latest.md"
  ):
    return True
  if "/data/" in normalized_path or normalized_path.startswith("data/"):
    return True
  return False


def read_artifact_text(
    relative_path: str,
    bucket: str,
    storage_key: str,
    local_dir: str | None = None,
) -> str:
  """Reads artifact text from local directory or GCS."""
  if local_dir:
    full_path = os.path.join(local_dir, relative_path)
    if os.path.isfile(full_path):
      try:
        with open(full_path, "r", encoding="utf-8") as f:
          return f.read()
      except (OSError, UnicodeDecodeError):
        return ""
    return ""

  uri = f"gs://{bucket}/{storage_key}/{relative_path}"
  try:
    proc = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode == 0:
      return proc.stdout
  except (subprocess.SubprocessError, OSError):
    # Return empty string when gcloud cat fails or times out.
    return ""
  return ""


def extract_1p_update_details(
    bucket: str,
    storage_key: str,
    current_objects: dict[str, str],
    modified_or_new: Sequence[str],
    local_dir: str | None = None,
) -> dict[str, object]:
  """Extracts verdict, evidence, workarounds, and checkpoint URI from files."""
  verdict = ""
  supporting_evidence = []
  workarounds_and_next_steps = []
  checkpoint_recovery_uri = ""

  # 1. Parse INCIDENT_MANIFEST.json
  manifest_text = read_artifact_text(
      "INCIDENT_MANIFEST.json", bucket, storage_key, local_dir
  )
  if manifest_text:
    manifest_data = None
    try:
      manifest_data = json.loads(manifest_text)
    except json.JSONDecodeError:
      manifest_data = None
    if isinstance(manifest_data, dict):
      history = manifest_data.get("ownership_history", [])
      if history and isinstance(history, list):
        for entry in reversed(history):
          if not isinstance(entry, dict):
            continue
          status_val = entry.get("status")
          notes_val = entry.get("notes", "")
          actor_val = entry.get("actor", "")
          if status_val and status_val != "PENDING_GOOGLE_CORRELATION":
            verdict = status_val
            if notes_val:
              supporting_evidence.append(
                  f"Manifest note ({actor_val}): {notes_val}"
              )
            break

  # 2. Parse infrastructure/findings/_infra_latest.md
  infra_text = read_artifact_text(
      "infrastructure/findings/_infra_latest.md", bucket, storage_key, local_dir
  )
  if infra_text:
    status_match = _INFRA_STATUS_RE.search(infra_text)
    if status_match:
      infra_status = status_match.group(1).strip()
      if infra_status and infra_status != "PENDING_GOOGLE_CORRELATION":
        if not verdict:
          verdict = infra_status

    node_matches = _GKE_TPU_NODE_RE.findall(infra_text)
    if node_matches and verdict and "node" not in verdict.lower():
      verdict = f"{verdict} on node {node_matches[0]}"

    table_lines = [
        line
        for line in infra_text.splitlines()
        if line.strip().startswith("|")
        and not line.strip().startswith("| :---")
    ]
    for line in table_lines[1:]:
      columns = [cell.strip() for cell in line.split("|")[1:-1]]
      if len(columns) >= 8:
        subsystem, corroboration, resolution_action = (
            columns[4],
            columns[6],
            columns[7],
        )
        if (
            corroboration
            and corroboration != "PENDING_GOOGLE_CORRELATION"
            and corroboration != "NOT_PROVIDED"
        ):
          supporting_evidence.append(
              f"Infrastructure corroboration: {corroboration}"
              f" (Subsystem: {subsystem})"
          )
        if resolution_action and resolution_action not in (
            "Awaiting Google Support / CE Triage",
            "NOT_PROVIDED",
            "",
        ):
          workarounds_and_next_steps.append(
              f"Resolution action: {resolution_action}"
          )

  # 3. Parse findings/google/*.md (especially _google_latest.md)
  modified_or_new_set = set(modified_or_new)
  ranked_finding_paths = sorted(
      (
          0 if file_path in modified_or_new_set else 1,
          0 if file_path.endswith("_google_latest.md") else 1,
          file_path,
      )
      for file_path in current_objects
      if "findings/google/" in file_path and file_path.endswith(".md")
  )
  google_finding_paths = [file_path for _, _, file_path in ranked_finding_paths]

  # Files are sorted newest-first (modified, then _google_latest.md), so the
  # first file that supplies a verdict or checkpoint URI wins; older
  # finding_*.md files must not overwrite it.
  verdict_from_findings = False
  for finding_path in google_finding_paths:
    finding_text = read_artifact_text(
        finding_path, bucket, storage_key, local_dir
    )
    if not finding_text:
      continue

    sections = {}
    current_section = None
    current_lines = []
    for line in finding_text.splitlines():
      heading_match = _MARKDOWN_HEADING_RE.search(line.strip())
      if heading_match:
        if current_section:
          sections[current_section] = current_lines
        current_section = heading_match.group(1).strip().lower()
        current_lines = []
      else:
        if current_section:
          current_lines.append(line)
    if current_section:
      sections[current_section] = current_lines

    # Extract verdict from section
    for section_name, lines in sections.items():
      if verdict_from_findings:
        break
      if any(
          keyword in section_name
          for keyword in ("verdict", "status", "root cause")
      ):
        non_empty = [
            line.strip().lstrip("-*0123456789. `").rstrip("`")
            for line in lines
            if line.strip()
        ]
        if non_empty:
          verdict = non_empty[0]
          verdict_from_findings = True
          break

    # Extract supporting evidence from section
    for section_name, lines in sections.items():
      if any(
          keyword in section_name
          for keyword in ("evidence", "corroborat", "finding", "symptom")
      ):
        for line in lines:
          cleaned = line.strip()
          if cleaned.startswith(("-", "*")) or _NUMBERED_ITEM_RE.search(
              cleaned
          ):
            item = _LIST_BULLET_PREFIX_RE.sub("", cleaned).strip()
            if item:
              supporting_evidence.append(item)
          elif cleaned and not cleaned.startswith("#"):
            supporting_evidence.append(cleaned)

    # Extract workarounds from section
    for section_name, lines in sections.items():
      if any(
          keyword in section_name
          for keyword in (
              "workaround",
              "next step",
              "recommend",
              "action",
              "mitigat",
          )
      ):
        for line in lines:
          cleaned = line.strip()
          if cleaned.startswith(("-", "*")) or _NUMBERED_ITEM_RE.search(
              cleaned
          ):
            item = _LIST_BULLET_PREFIX_RE.sub("", cleaned).strip()
            if item:
              workarounds_and_next_steps.append(item)
          elif cleaned and not cleaned.startswith("#"):
            workarounds_and_next_steps.append(cleaned)

    # Checkpoint recovery URI
    checkpoint_match = _CHECKPOINT_GCS_URI_RE.search(finding_text)
    if checkpoint_recovery_uri:
      pass  # Already taken from a newer finding file.
    elif checkpoint_match:
      checkpoint_recovery_uri = checkpoint_match.group(1)
    else:
      for line in finding_text.splitlines():
        if "checkpoint" in line.lower() or "orbax" in line.lower():
          uri_match = _GCS_URI_RE.search(line)
          if uri_match:
            checkpoint_recovery_uri = uri_match.group(1)
            break

    # Fallback line scanning if sections not present
    if not supporting_evidence:
      for line in finding_text.splitlines():
        cleaned_line = line.strip()
        if any(
            keyword in cleaned_line.lower()
            for keyword in (
                "culprit host",
                "culprit node",
                "register",
                "tpu_support_",
                "optical attenuation",
                "parity error",
                "link degraded",
            )
        ):
          supporting_evidence.append(
              _LIST_BULLET_PREFIX_RE.sub("", cleaned_line)
          )

    if not workarounds_and_next_steps:
      for line in finding_text.splitlines():
        cleaned_line = line.strip()
        if any(
            keyword in cleaned_line.lower()
            for keyword in (
                "kubectl cordon",
                "xla",
                "checkpoint recovery",
                "cordon node",
                "workaround",
            )
        ):
          workarounds_and_next_steps.append(
              _LIST_BULLET_PREFIX_RE.sub("", cleaned_line)
          )

  supporting_evidence = _deduplicate_items(supporting_evidence)
  workarounds_and_next_steps = _deduplicate_items(workarounds_and_next_steps)

  if not verdict:
    verdict = "VERIFIED_1P_UPDATE"

  return {
      "verdict": verdict,
      "supporting_evidence": supporting_evidence,
      "workarounds_and_next_steps": workarounds_and_next_steps,
      "checkpoint_recovery_uri": checkpoint_recovery_uri,
  }


def format_1p_alert_card(
    storage_key: str,
    bucket: str,
    verdict: str,
    supporting_evidence: Sequence[str],
    workarounds_and_next_steps: Sequence[str],
    checkpoint_recovery_uri: str = "",
) -> str:
  """Formats a prominent terminal/chat alert card for 1P Google updates."""
  lines = [
      "=" * 80,
      f"🚨 [ALERT: 1P GOOGLE UPDATE DETECTED] Incident: {storage_key}",
      f"Repository: gs://{bucket}/{storage_key}/",
      "=" * 80,
      "VERIFIED VERDICT / STATUS:",
      f"  {verdict if verdict else 'GOOGLE_UPDATE_RECORDED'}",
      "",
      "SUPPORTING EVIDENCE FROM 1P:",
  ]
  if supporting_evidence:
    for item in supporting_evidence:
      lines.append(f"  • {item}")
  else:
    lines.append("  (No supporting evidence extracted)")

  lines.extend([
      "",
      "ACTIONABLE WORKAROUNDS & NEXT STEPS:",
  ])
  if workarounds_and_next_steps:
    for item in workarounds_and_next_steps:
      lines.append(f"  • {item}")
  else:
    lines.append("  (No actionable workarounds specified)")

  if checkpoint_recovery_uri:
    lines.extend([
        "",
        "CHECKPOINT RECOVERY PATH:",
        f"  • {checkpoint_recovery_uri}",
    ])

  lines.append("=" * 80)
  return "\n".join(lines)


def format_3p_alert_card(
    storage_key: str,
    bucket: str,
    new_files: Sequence[str],
    modified_files: Sequence[str],
) -> str:
  """Formats an alert card for 3P customer updates."""
  lines = [
      "=" * 80,
      f"📥 [ALERT: 3P CUSTOMER UPDATE DETECTED] Incident: {storage_key}",
      f"Repository: gs://{bucket}/{storage_key}/",
      "=" * 80,
      f"New Customer Artifacts ({len(new_files)}):",
  ]
  for file_path in new_files[:10]:
    lines.append(f"  • {file_path}")
  if len(new_files) > 10:
    lines.append(f"  ... and {len(new_files) - 10} more")

  if modified_files:
    lines.extend([
        "",
        f"Modified Customer Artifacts ({len(modified_files)}):",
    ])
    for file_path in modified_files[:10]:
      lines.append(f"  • {file_path}")
    if len(modified_files) > 10:
      lines.append(f"  ... and {len(modified_files) - 10} more")

  lines.extend([
      "",
      "Action: Review uploaded telemetry with `/handoff --resume` or",
      " inspect scratch traces.",
      "=" * 80,
  ])
  return "\n".join(lines)


def scan_bucket_changes(
    bucket: str,
    storage_key: str,
    state_file: str,
    auto_watch: bool = True,
    init_baseline: bool = False,
    local_dir: str | None = None,
) -> dict[str, object]:
  """Scans target GCS repository and reports updates across all tracks."""
  now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
  prior_state = {}
  state_existed = os.path.isfile(state_file)

  if state_existed:
    loaded_state = None
    try:
      with open(state_file, "r", encoding="utf-8") as f:
        loaded_state = json.load(f)
    except (OSError, json.JSONDecodeError):
      loaded_state = None
    if isinstance(loaded_state, dict):
      prior_objects = loaded_state.get("objects")
      if isinstance(prior_objects, dict):
        prior_state = prior_objects

  current_objects = list_gcs_objects(bucket, storage_key, local_dir=local_dir)

  all_new_files = []
  all_modified_files = []

  for path, signature in current_objects.items():
    if path not in prior_state:
      all_new_files.append(path)
    elif prior_state[path] != signature:
      all_modified_files.append(path)

  # Partition by tracks
  first_party_new = [
      file_path for file_path in all_new_files if is_1p_path(file_path)
  ]
  first_party_modified = [
      file_path for file_path in all_modified_files if is_1p_path(file_path)
  ]
  third_party_new = [
      file_path for file_path in all_new_files if is_3p_path(file_path)
  ]
  third_party_modified = [
      file_path for file_path in all_modified_files if is_3p_path(file_path)
  ]

  verdict = ""
  supporting_evidence = []
  workarounds_and_next_steps = []
  checkpoint_recovery_uri = ""
  alert_message = ""

  if not state_existed or init_baseline:
    status = "INITIALIZED"
    summary_message = (
        f"Baseline snapshot initialized with {len(current_objects)} object(s)"
        f" in gs://{bucket}/{storage_key}/"
    )
    reported_new = []
    reported_modified = []
  else:
    if (first_party_new or first_party_modified) and (
        third_party_new or third_party_modified
    ):
      status = "UPDATES_DETECTED"
      first_party_count = len(first_party_new) + len(first_party_modified)
      third_party_count = len(third_party_new) + len(third_party_modified)
      summary_message = (
          f"Both 1P Google ({first_party_count} file(s)) and 3P"
          f" Customer ({third_party_count} file(s)) updates detected"
      )
      reported_new = all_new_files
      reported_modified = all_modified_files
      details = extract_1p_update_details(
          bucket,
          storage_key,
          current_objects,
          first_party_new + first_party_modified,
          local_dir,
      )
      verdict = details["verdict"]
      supporting_evidence = details["supporting_evidence"]
      workarounds_and_next_steps = details["workarounds_and_next_steps"]
      checkpoint_recovery_uri = details["checkpoint_recovery_uri"]
      first_party_card = format_1p_alert_card(
          storage_key=storage_key,
          bucket=bucket,
          verdict=verdict,
          supporting_evidence=supporting_evidence,
          workarounds_and_next_steps=workarounds_and_next_steps,
          checkpoint_recovery_uri=checkpoint_recovery_uri,
      )
      third_party_card = format_3p_alert_card(
          storage_key=storage_key,
          bucket=bucket,
          new_files=third_party_new,
          modified_files=third_party_modified,
      )
      alert_message = f"{first_party_card}\n\n{third_party_card}"
    elif first_party_new or first_party_modified:
      status = "1P_UPDATE_DETECTED"
      summary_message = (
          f"1P Google updates detected: {len(first_party_new)} new file(s),"
          f" {len(first_party_modified)} modified file(s)"
      )
      reported_new = all_new_files
      reported_modified = all_modified_files
      details = extract_1p_update_details(
          bucket,
          storage_key,
          current_objects,
          first_party_new + first_party_modified,
          local_dir,
      )
      verdict = details["verdict"]
      supporting_evidence = details["supporting_evidence"]
      workarounds_and_next_steps = details["workarounds_and_next_steps"]
      checkpoint_recovery_uri = details["checkpoint_recovery_uri"]
      alert_message = format_1p_alert_card(
          storage_key=storage_key,
          bucket=bucket,
          verdict=verdict,
          supporting_evidence=supporting_evidence,
          workarounds_and_next_steps=workarounds_and_next_steps,
          checkpoint_recovery_uri=checkpoint_recovery_uri,
      )
    elif third_party_new or third_party_modified:
      status = "3P_UPDATE_DETECTED"
      summary_message = (
          f"3P customer updates detected: {len(third_party_new)} new file(s),"
          f" {len(third_party_modified)} modified file(s)"
      )
      reported_new = all_new_files
      reported_modified = all_modified_files
      alert_message = format_3p_alert_card(
          storage_key=storage_key,
          bucket=bucket,
          new_files=third_party_new,
          modified_files=third_party_modified,
      )
    elif all_new_files or all_modified_files:
      status = "NEW_UPDATES_DETECTED"
      summary_message = (
          f"{len(all_new_files)} new file(s),"
          f" {len(all_modified_files)} modified file(s)"
      )
      reported_new = all_new_files
      reported_modified = all_modified_files
    else:
      status = "NO_CHANGE"
      summary_message = f"No updates detected in gs://{bucket}/{storage_key}/"
      reported_new = []
      reported_modified = []

  # Save updated state
  output_dir = os.path.dirname(os.path.abspath(state_file))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  state_data = {
      "last_checked_utc": now_utc,
      "bucket": bucket,
      "storage_key": storage_key,
      "auto_watch": auto_watch,
      "objects": current_objects,
  }
  with open(state_file, "w", encoding="utf-8") as f:
    json.dump(state_data, f, indent=2)

  return {
      "status": status,
      "bucket": bucket,
      "storage_key": storage_key,
      "auto_watch": auto_watch,
      "last_checked_utc": now_utc,
      "new_files": reported_new,
      "modified_files": reported_modified,
      "summary": summary_message,
      "verdict": verdict,
      "supporting_evidence": supporting_evidence,
      "workarounds_and_next_steps": workarounds_and_next_steps,
      "checkpoint_recovery_uri": checkpoint_recovery_uri,
      "alert_message": alert_message,
  }


def main() -> None:
  """CLI entry point for monitoring handoff bucket updates across tracks."""
  parser = argparse.ArgumentParser(
      description=(
          "Monitor Handoff Bucket for Updates Universally Across All Tracks"
      )
  )
  parser.add_argument("--bucket", required=True, help="Target GCS bucket name")
  parser.add_argument(
      "--storage_key", required=True, help="Incident storage key"
  )
  parser.add_argument(
      "--state_file", default=None, help="State cache file path"
  )
  parser.add_argument(
      "--local_dir",
      default=None,
      help="Optional local directory for mock testing or local monitoring",
  )
  parser.add_argument(
      "--init_baseline",
      action="store_true",
      help="Force initialize baseline snapshot",
  )
  parser.add_argument(
      "--no_auto_watch", action="store_true", help="Disable default auto_watch"
  )
  parser.add_argument(
      "--perspective",
      default="auto",
      choices=["1p", "3p", "auto"],
      help="Monitoring perspective (1p, 3p, or auto)",
  )
  parser.add_argument(
      "--check_once",
      action="store_true",
      default=True,
      help="Run single poll",
  )

  args = parser.parse_args()

  bucket = args.bucket.replace("gs://", "").rstrip("/")
  state_path = args.state_file
  if not state_path:
    # Resolved from this file, not the cwd: the glue runner executes from
    # _wiki/, where "_wiki/scratch" would create a nested _wiki/_wiki/.
    state_path = os.path.join(
        fio.WIKI_SCRATCH_DIR, f"watch_{args.storage_key}_{bucket}.state.json"
    )

  report = scan_bucket_changes(
      bucket=bucket,
      storage_key=args.storage_key,
      state_file=state_path,
      auto_watch=not args.no_auto_watch,
      init_baseline=args.init_baseline,
      local_dir=args.local_dir,
  )

  print(json.dumps(report, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"status: {report['status']}")
  print(f"summary: {report['summary']}")
  print(f"verdict: {report.get('verdict', '')}")
  print(
      "supporting_evidence:"
      f" {json.dumps(report.get('supporting_evidence', []))}"
  )
  print(
      "workarounds_and_next_steps:"
      f" {json.dumps(report.get('workarounds_and_next_steps', []))}"
  )
  print(f"new_files_count: {len(report['new_files'])}")
  print(f"modified_files_count: {len(report['modified_files'])}")
  print(f"last_checked_utc: {report['last_checked_utc']}")

  if report.get("alert_message"):
    print("\n" + report["alert_message"])


if __name__ == "__main__":
  main()

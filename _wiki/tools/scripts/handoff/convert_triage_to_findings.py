#!/usr/bin/env python3
"""convert_triage_to_findings.py - Bridges triage state into findings.

Converts diagnostic state output from /troubleshoot
(e.g. scratch/triage-*.json) into canonical troubleshoot_findings.json
conforming to Schema v2.2.
"""

import argparse
import datetime
import json
import os
import sys

import findings_io as fio


def convert_triage_to_findings(
    triage_data: dict[str, object],
    case_id: str,
    storage_key: str | None = None,
    issue_id: str | None = None,
    bucket: str | None = None,
    session_id: str | None = None,
    report_path: str | None = None,
) -> dict[str, object]:
  """Converts /troubleshoot triage state into canonical findings Schema v2.2.

  Args:
    triage_data: Raw triage state dictionary produced by /troubleshoot.
    case_id: Support case identifier.
    storage_key: Optional storage key override; defaults to case_id when None.
    issue_id: Optional partner issue ID; recorded as None when
      omitted.
    bucket: Optional target GCS bucket override; resolved via fio.gcs_bucket
      when None.
    session_id: Optional troubleshoot session identifier; falls back to
      triage_data["timestamp"] when None.
    report_path: Optional path to the markdown triage report; recorded as None
      when omitted.

  Returns:
    A canonical troubleshoot_findings dictionary conforming to Schema v2.2.
  """
  now_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%dT%H:%M:%SZ"
  )

  workload = triage_data.get("workload") or triage_data.get("workload_name")
  project_id = triage_data.get("project_id")
  cluster = triage_data.get("cluster") or triage_data.get("cluster_name")
  zone = triage_data.get("zone") or triage_data.get("location")
  reservation = triage_data.get("reservation")

  resolved_storage_key = storage_key or case_id

  milestone_1 = triage_data.get("milestone_1_reconnaissance", {})
  milestone_2 = triage_data.get("milestone_2_baseline_disruption_evidence", {})
  summary = dict(triage_data.get("triage_summary", {}))
  invariant_assessment = triage_data.get("invariant_assessment", {})

  actions_list = triage_data.get("actions_executed", [])
  actions_by_name = {}
  for item in actions_list:
    if isinstance(item, dict) and "action" in item:
      actions_by_name[item["action"]] = item

  if not milestone_2 and actions_by_name:
    milestone_2 = actions_by_name

  accelerator_info = triage_data.get("accelerator_platform", {})
  accelerator_type = (
      accelerator_info.get("accelerator_type")
      or accelerator_info.get("type")
      or accelerator_info.get("machine_type")
      or triage_data.get("accelerator_type")
  )
  scale_info = triage_data.get("scale", {})
  triage_window = triage_data.get("triage_window", {})
  root_cause_analysis = triage_data.get("root_cause_analysis", {})

  node_summary = milestone_2.get("resolve-workload-nodes", {}).get(
      "summary_fields", {}
  )
  node_count = (
      node_summary.get("total_nodes")
      or scale_info.get("node_count")
      or triage_data.get("node_count")
  )
  slice_count = (
      node_summary.get("node_groups_count")
      or (
          len(node_summary.get("node_groups", {}))
          if node_summary.get("node_groups")
          else None
      )
      or scale_info.get("slice_count")
      or triage_data.get("slice_count")
  )
  chip_count = (
      scale_info.get("chip_count")
      or triage_data.get("chip_count")
      or (
          node_count * 4
          if node_count and accelerator_type and "4t" in accelerator_type
          else None
      )
  )

  if root_cause_analysis.get("summary") and not summary.get("root_cause"):
    summary["root_cause"] = root_cause_analysis.get("summary")

  # Formulate P-list (proves)
  proves = []
  prove_index = 1
  if summary.get("root_cause"):
    proves.append({
        "id": f"P{prove_index}",
        "text": f"Triage established: {summary.get('root_cause')}",
        "evidence_refs": ["triage_summary.root_cause"],
    })
    prove_index += 1
  if summary.get("culprit_attribution"):
    proves.append({
        "id": f"P{prove_index}",
        "text": (
            f"Culprit attribution state: {summary.get('culprit_attribution')}"
        ),
        "evidence_refs": ["extract-culprit-hosts"],
    })
    prove_index += 1
  for invariant_key, invariant_status in invariant_assessment.items():
    if "COMPLIANT" in invariant_status:
      proves.append({
          "id": f"P{prove_index}",
          "text": (
              f"{invariant_key.upper()} adherence verified: {invariant_status}"
          ),
          "evidence_refs": [invariant_key],
      })
      prove_index += 1

  # Formulate C-list (cannot_prove)
  cannot_prove = []
  cannot_prove_index = 1
  if node_summary.get("status") == "INCOMPLETE":
    cannot_prove.append({
        "id": f"C{cannot_prove_index}",
        "text": (
            "Complete node set could not be verified due to telemetry query"
            " boundaries or missing permissions."
        ),
    })
    cannot_prove_index += 1
  if "INCONCLUSIVE" in str(summary.get("culprit_attribution", "")).upper():
    cannot_prove.append({
        "id": f"C{cannot_prove_index}",
        "text": (
            "Internal hardware root cause cannot be established without"
            " correlated Google-internal telemetry."
        ),
    })
    cannot_prove_index += 1

  if not cannot_prove:
    cannot_prove.append({
        "id": "C1",
        "text": (
            "Customer-side telemetry cannot observe internal optical fabric or"
            " host kernel logs."
        ),
    })

  # Formulate questions for google
  questions = []
  for cannot_prove_item in cannot_prove:
    questions.append({
        "id": f"Q{cannot_prove_item['id'][1:]}",
        "c_ref": cannot_prove_item["id"],
        "owner": "Google Support Lead",
        "question": (
            "Can Google Support correlate internal telemetry to resolve"
            f" {cannot_prove_item['text']}?"
        ),
        "if_yes": "Identifies internal root cause.",
        "if_no": "Awaiting further reproduction.",
    })

  # Evidence table
  evidence_table = []
  culprit_action = milestone_2.get("extract-culprit-hosts", {})
  culprit_summary = culprit_action.get("summary_fields", {})
  reconciled = culprit_action.get("reconciled_findings", {})
  by_node = culprit_summary.get("primary_incidents_by_node", {})
  for culprit_node, count in by_node.items():
    evidence_table.append({
        # The action reports per-node counts, not event times.
        "timestamp": None,
        "worker": None,
        "gke_node": culprit_node,
        "observation": (
            f"Identified in {count} primary incident(s) from application logs"
        ),
        "context": "Primary log correlation",
        "log_ref": "extract-culprit-hosts",
        "artifact_refs": [],
    })
  if not evidence_table and reconciled.get("affected_nodes"):
    for node in reconciled["affected_nodes"]:
      evidence_table.append({
          "timestamp": None,
          "worker": None,
          "gke_node": node,
          "observation": (
              "Listed as affected node; primary culprit:"
              f" {reconciled.get('primary_culprit') or fio.NOT_PROVIDED}"
          ),
          "context": "extract-culprit-hosts reconciled affected_nodes",
          "log_ref": "extract-culprit-hosts",
          "artifact_refs": [],
      })

  target_bucket = fio.gcs_bucket(override_bucket=bucket)

  hypothesis = None
  if summary.get("recommendation"):
    hypothesis = {
        "text": summary.get("recommendation"),
        "author": "AI Diagnostic Triage",
        "confidence": "Medium",
        "support_refs": [prove_item["id"] for prove_item in proves],
        "gap_refs": [
            cannot_prove_item["id"] for cannot_prove_item in cannot_prove
        ],
        "falsifier": "Resolved upon internal correlation.",
    }

  next_steps = []
  if summary.get("recommendation"):
    next_steps.append({
        "c_ref": cannot_prove[0]["id"] if cannot_prove else None,
        "owner": "Customer Primary Investigator",
        "step": summary.get("recommendation"),
    })

  findings = {
      "source": {
          "troubleshoot_session": session_id or triage_data.get("timestamp"),
          "report_path": report_path,
          "extracted_at": now_utc,
      },
      "identity": {
          "case_id": str(case_id),
          "storage_key": str(resolved_storage_key),
          "issue_id": str(issue_id) if issue_id else None,
          "bucket": target_bucket,
          "project_id": project_id,
          "project_number": None,
          "cluster": cluster,
          "zone": zone,
          "reservation": reservation,
          "workload": workload,
          "accelerator_type": accelerator_type,
          "chip_count": chip_count,
          "node_count": node_count,
          "slice_count": slice_count,
      },
      "category": triage_data.get("category"),
      "windows": {
          "degraded": {
              "start": (
                  triage_window.get("start")
                  or triage_data.get("start")
                  or milestone_2.get("resolve-workload-nodes", {}).get("start")
              ),
              "end": (
                  triage_window.get("end")
                  or triage_data.get("end")
                  or milestone_2.get("resolve-workload-nodes", {}).get("end")
              ),
              "step_time_sec": None,
          },
          "baseline": {
              "start": None,
              "end": None,
              "step_time_sec": None,
          },
      },
      "proves": proves,
      "cannot_prove": cannot_prove,
      "evidence_table": evidence_table,
      "attempts": triage_data.get("attempts", []),
      "job_progress": None,
      "google_verdict": None,
      "storage": None,
      "actions_run": [
          {
              "name": action_name,
              "execution_mode": action_data.get("execution_mode", "command"),
          }
          for action_name, action_data in {
              **milestone_1,
              **milestone_2,
          }.items()
      ],
      "hypothesis": hypothesis,
      "questions_for_google": questions,
      "customer_next_steps": next_steps,
      "artifacts": [
          {"path": "INCIDENT_MANIFEST.json", "kind": "manifest"},
          {"path": f"jobs/{workload}/JOB_MANIFEST.json", "kind": "manifest"}
          if workload
          else None,
      ],
      "system_info": None,
      "owners": {
          "customer_dri": None,
          "customer_primary_investigator": None,
          "google_support_lead": os.environ.get("GOOGLE_SUPPORT_LEAD") or None,
          "google_ce_lead": os.environ.get("GOOGLE_CE_LEAD") or None,
      },
      "discrepancies": [],
  }

  findings["artifacts"] = [
      artifact for artifact in findings["artifacts"] if artifact is not None
  ]
  return findings


def find_latest_triage_file(workload: str) -> str | None:
  """Finds the most recent triage state file for a workload.

  Args:
    workload: Workload identifier to search for.

  Returns:
    Path to the latest matching triage JSON file, or None if not found.
  """
  candidates = []
  search_dirs = [
      os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "..", "..", "scratch"
      ),
      os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "..",
          "..",
          "..",
          "_wiki",
          "scratch",
      ),
      os.path.join(os.getcwd(), "_wiki", "scratch"),
      os.path.join(os.getcwd(), "scratch"),
  ]
  seen_dirs = set()
  for search_dir in search_dirs:
    normalized_dir = os.path.abspath(search_dir)
    if normalized_dir in seen_dirs or not os.path.isdir(normalized_dir):
      continue
    seen_dirs.add(normalized_dir)
    for filename in os.listdir(normalized_dir):
      if filename.startswith(f"triage-{workload}") and filename.endswith(
          ".json"
      ):
        full_path = os.path.join(normalized_dir, filename)
        if os.path.isfile(full_path):
          candidates.append((os.path.getmtime(full_path), full_path))

  if not candidates:
    return None
  candidates.sort(reverse=True)
  return candidates[0][1]


def main() -> None:
  """CLI entry point for converting triage state JSON into findings JSON."""
  parser = argparse.ArgumentParser(
      description=(
          "Convert /troubleshoot triage state into troubleshoot_findings.json"
      )
  )
  parser.add_argument(
      "--triage_json", default=None, help="Path to input triage state JSON file"
  )
  parser.add_argument(
      "--workload",
      default=None,
      help=(
          "Target workload name (used to locate triage JSON if --triage_json"
          " omitted)"
      ),
  )
  parser.add_argument("--case_id", required=True, help="Support case ID")
  parser.add_argument(
      "--storage_key",
      default=None,
      help="Storage key override (defaults to case_id)",
  )
  parser.add_argument(
      "--issue_id", default=None, help="Partner Issue ID"
  )
  parser.add_argument("--bucket", default=None, help="Target GCS bucket name")
  parser.add_argument(
      "--session_id", default=None, help="Troubleshoot session ID"
  )
  parser.add_argument(
      "--report_path", default=None, help="Path to markdown triage report"
  )
  parser.add_argument(
      "--output_file",
      required=True,
      help="Destination troubleshoot_findings.json path",
  )

  args = parser.parse_args()

  try:
    fio.require_bucket(override_bucket=args.bucket)
  except fio.FindingsError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)

  triage_json_path = args.triage_json
  if not triage_json_path:
    if args.workload:
      triage_json_path = find_latest_triage_file(args.workload)
      if not triage_json_path:
        print(
            f"[ERROR] No triage state JSON found for workload '{args.workload}'"
            " in _wiki/scratch/. You must execute `/troubleshoot"
            f" {args.workload}` first to gather empirical findings before"
            " running /handoff.",
            file=sys.stderr,
        )
        sys.exit(1)
      print(
          "[INFO] Automatically selected most recent triage JSON for"
          f" '{args.workload}': {triage_json_path}"
      )
    else:
      print(
          "[ERROR] Either --triage_json or --workload must be specified.",
          file=sys.stderr,
      )
      sys.exit(1)

  if triage_json_path and not os.path.isfile(triage_json_path):
    if triage_json_path.startswith("_wiki/"):
      stripped = triage_json_path[len("_wiki/") :]
      if os.path.isfile(stripped):
        triage_json_path = stripped
    elif os.path.isfile(os.path.join("_wiki", triage_json_path)):
      triage_json_path = os.path.join("_wiki", triage_json_path)

  if not os.path.isfile(triage_json_path):
    print(
        f"[ERROR] Input triage JSON not found: {triage_json_path}",
        file=sys.stderr,
    )
    sys.exit(1)

  with open(triage_json_path, "r", encoding="utf-8") as f:
    triage_data = json.load(f)

  findings = convert_triage_to_findings(
      triage_data=triage_data,
      case_id=args.case_id,
      storage_key=args.storage_key,
      issue_id=args.issue_id,
      bucket=args.bucket,
      session_id=args.session_id,
      report_path=args.report_path,
  )

  output_file = args.output_file
  if output_file.startswith("_wiki/") and not os.path.exists("_wiki"):
    output_file = output_file[len("_wiki/") :]

  output_dir = os.path.dirname(os.path.abspath(output_file))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)

  print(f"[OK] Successfully converted triage state into {output_file}")
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print("status: CONVERTED")
  print(f"workload: {findings['identity']['workload']}")
  print(f"case_id: {findings['identity']['case_id']}")
  print(f"bucket: {findings['identity']['bucket']}")
  print(f"proves_count: {len(findings['proves'])}")
  print(f"cannot_prove_count: {len(findings['cannot_prove'])}")


if __name__ == "__main__":
  main()

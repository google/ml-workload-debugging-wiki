---
title: Workload Diagnostic Handoff Tool
type: tool
internal_only: false
tags: [triage, storage, analysis, gcp]
sources:
  - playbooks/workload-diagnostic-handoff.md
  - concepts/failure-tables.md
created: 2026-10-01
updated: 2026-10-05
---

<!-- mdformat off -->

# Workload Diagnostic Handoff Tool

## Overview
The **Workload Diagnostic Handoff Tool** (`_wiki/tools/scripts/handoff/`) provides an automated, secure diagnostic handoff boundary between Google Support / Customer Engineering and partner ML engineering teams for Cloud TPU workloads.

It orchestrates:

1. **Empirical Troubleshooting & Reconnaissance**: Automatically executes `/troubleshoot` via [[workload-disruption-orchestration]] to investigate disruption signals, quote culprit hosts, and produce the raw triage state JSON before packaging.
2. **Dynamic Bucket Provisioning**: Canonical multi-job directory scaffolding at `gs://<bucket>/<storage_key>/` with automatic generation of `BUCKET_STRUCTURE.md`, rendered by [[scaffold-handoff-bucket]] from the template `_wiki/tools/scripts/handoff/templates/BUCKET_STRUCTURE.template.md`.
3. **Schema v2.2 Manifest Generation**: Generates machine-readable `INCIDENT_MANIFEST.json` and `JOB_MANIFEST.json`.
4. **Strict Two-Level Failure Tables**: Generates Level A Infrastructure Failure Tables and Level B Unified Disruption Failure Tables conforming to `/troubleshoot` schemas.
5. **Proprietary HLO Redaction**: Strips tensor names, dimensions, and customer source paths while preserving collective communication mathematical structures.
6. **Human-in-the-Loop Safety Gate**: Blocks autonomous cordoning or replacement of TPU infrastructure without verified human operator sign-off.
7. **Partner Issue Tracker Integration**: Renders structured updates for the customer's Partner Issue Tracker component and template.
8. **Content-Verified Publication**: Uploads the rendered package all-or-nothing via [[upload-handoff-package]] and confirms with [[verify-handoff-package]] that every required file in the bucket has real content (no 0-byte, placeholder-only, or invalid JSON files).
9. **Bi-Directional Symmetrical Bucket Monitoring**: Symmetrically monitors target GCS repository for customer telemetry uploads (1P perspective) or Google verdicts, concrete empirical evidence, and actionable workarounds (3P perspective).

---

## Tool Scripts Catalog

| Script | Purpose |
| :--- | :--- |
| `findings_io.py` | Core loader, validator, and GCS path accessor. Resolves the bucket from `--bucket`, `identity.bucket` or `$HANDOFF_BUCKET`; there is no default bucket. Defines the required package files and the content check shared by upload and verify. |
| `scaffold_bucket.sh` | Provisions the incident structure on GCS (`BUCKET_STRUCTURE.md` rendered from `templates/BUCKET_STRUCTURE.template.md`, support case link). Never pre-creates empty directory markers (`.keep`), manifests, READMEs or tables; directories in GCS are created on-demand as real files are uploaded. |
| `BUCKET_STRUCTURE.md` | Readable copy of the template with `<bucket>` / `<storage_key>` placeholders. Not used by any script; `handoff_test.py` fails if it drifts from the template, so edit the template and regenerate the copy. |
| `generate_manifest.py` | Generates `INCIDENT_MANIFEST.json` and `JOB_MANIFEST.json` with null-propagation. |
| `generate_failure_table.py` | Generates Level A (Infrastructure) and Level B (Job Progression) failure tables. |
| `render_readme.py` / `generate_readme.sh` | Renders human-readable overviews (`INCIDENT_OVERVIEW.md`, `README.md`, `JOB_README.md`). |
| `redact_diagnostic_payload.py` | Redacts proprietary strings and shapes from textual HLO graphs and stack traces. |
| `render_issue_body.py` | Formats comments conforming to the Partner Issue Tracker template and prints the issue link. Does not post; the operator pastes the comment. |
| `convert_triage_to_findings.py` | Bridges `/troubleshoot` diagnostic state JSON into `troubleshoot_findings.json`. |
| `verify_handoff_bucket_access.py` | Pre-flight security prober testing bucket write/read access and public access prevention. |
| `package_handoff_artifacts.py` | Gathers scratch traces, sanitizes HLOs, and uploads them under `jobs/<workload>/data/` (upload on by default). Telemetry with no source is listed in `missing_artifacts`, never written as a placeholder. |
| `upload_handoff_package.py` | Uploads the shared generator staging tree to `gs://<bucket>/<storage_key>/`; refuses everything if a required file is missing or any file is empty, placeholder-only, or invalid JSON. |
| `verify_handoff_package.py` | Lists the published bucket and returns `VERIFIED`, `INCOMPLETE`, or `VERIFY_FAILED` (listing failed). |
| `watch_handoff_bucket.py` | Symmetrically monitors GCS repository for customer uploads (1P) or Google verdicts, evidence, and workarounds (3P). |

---

## Usage and Commands

### 1. Pre-flight Bucket Access Check

```bash
python3 _wiki/tools/scripts/handoff/verify_handoff_bucket_access.py \
  --bucket="<your-debug-bucket>" \
  --storage_key="<case-id>"
```

### 2. Scaffold GCS Directory

```bash
_wiki/tools/scripts/handoff/scaffold_bucket.sh <case-id> \
  --bucket="<your-debug-bucket>" \
  --job="my-workload-run1"
```

### 3. Render, Publish, and Verify the Package

The three generators share one `--output_dir`, which then mirrors the bucket
layout; the upload step publishes that directory and the verify step checks the
bucket.

```bash
OUT="_wiki/scratch/handoff_<case-id>"
for script in generate_manifest.py generate_failure_table.py render_readme.py; do
  python3 "_wiki/tools/scripts/handoff/${script}" \
    --findings="_wiki/scratch/troubleshoot_findings.json" \
    --bucket="<your-debug-bucket>" \
    --output_dir="${OUT}"
done
python3 _wiki/tools/scripts/handoff/upload_handoff_package.py \
  --findings="_wiki/scratch/troubleshoot_findings.json" \
  --staging_dir="${OUT}" --dry_run   # drop --dry_run to publish
python3 _wiki/tools/scripts/handoff/verify_handoff_package.py \
  --findings="_wiki/scratch/troubleshoot_findings.json"
```

### 5. Monitor Bucket for Updates (Universal Symmetrical)

```bash
python3 _wiki/tools/scripts/handoff/watch_handoff_bucket.py \
  --bucket="<your-debug-bucket>" \
  --storage_key="<case-id>"
```

---

## Troubleshooting and Application

### Common Scenarios

1. **Access Denied During Scaffolding**:
   If `scaffold_bucket.sh` or `verify_handoff_bucket_access.py` reports `403 Access Denied`, verify active credentials via `gcloud auth list` and ensure caller identity possesses `roles/storage.objectAdmin` on the destination bucket.
2. **`verify-handoff-package` returns `INCOMPLETE`**:
   Each problem names the file and the reason (missing, 0 bytes, placeholder-only, invalid JSON). Re-run the generator that owns the file into the same `--output_dir`, then [[upload-handoff-package]] and verify again. `VERIFY_FAILED` means the bucket listing itself failed (usually permissions), not that the bucket is empty.
3. **Missing Customer Coordinates**:
   When converting `/troubleshoot` findings, if node counts or topology are omitted in telemetry, `findings_io.py` propagates `null` to prevent hallucinating cluster specs.

---

## Connections

- Workflow: [[workload-diagnostic-handoff]]
- Playbook: [[workload-diagnostic-handoff]]
- Concepts: [[failure-tables]], [[cloud-storage]]
<!-- mdformat on -->

---
title: Workload Diagnostic Handoff Playbook
type: playbook
internal_only: false
tags: [triage, methodology, training, analysis, storage]
sources:
  - concepts/failure-tables.md
  - playbooks/workload-disruption-orchestration.md
created: 2026-10-01
updated: 2026-10-04
contested: false
contradictions: []
---


# Workload Diagnostic Handoff Playbook

## Overview

The **Workload Diagnostic Handoff Playbook** establishes the standard operating
procedure for bi-directional diagnostic evidence exchange, failure table
synchronization, and issue tracker coordination between Customer ML Engineering
teams and Google Cloud Support / CE on large-scale Cloud TPU multi-slice
clusters.

It codifies three operational paths:

-   **Path A: Prepare & Hand Off**: Translates completed `/troubleshoot`
    findings into a secure, structured GCS diagnostic package, stages failure
    tables, and stages a partner issue tracker comment for the operator to post.

--------------------------------------------------------------------------------

## Core Mission Invariants

1.  **Single Source of Truth**: Every manifest, table, and overview is rendered
    exclusively from `troubleshoot_findings.json` (schema:
    `templates/troubleshoot_findings.schema.json`). Scripts contain zero example
    data defaults. Missing values propagate as `NOT PROVIDED` (markdown) or
    `null` (JSON). Generators exit non-zero when the findings fail validation; a
    missing artifact is reported as missing, never written as a placeholder
    file.
2.  **Dynamic Bucket Parameterization**: Destination buckets depend on the
    customer project (e.g. `gs://<your-debug-bucket>/`,
    `gs://example-debug-bucket/`). All tools accept `--bucket=<name>`, check
    `identity.bucket`, or read `$HANDOFF_BUCKET`. If not specified, operators
    must be prompted for the customer project's GCS bucket.
3.  **Canonical Directory Dictionary (`BUCKET_STRUCTURE.md`)**: Every handoff
    root automatically includes `BUCKET_STRUCTURE.md`, rendered by
    [[scaffold-handoff-bucket]] from the template
    `_wiki/tools/scripts/handoff/templates/BUCKET_STRUCTURE.template.md`,
    providing an exhaustive directory layout and ownership dictionary.
4.  **Strict Two-Level Failure Table Standard**:
    -   **Level A (Infrastructure / Hardware / Host Events)**: Maintained by
        Google in `infrastructure/findings/_infra_latest.md`. Focuses on
        physical hosts, VM terminations, DCN/optical events, and host
        maintenance errors. Initial status at handoff is strictly
        `PENDING_GOOGLE_CORRELATION`.
    -   **Level B (Job Progression / Disruption Timeline)**: Maintained in
        `jobs/<workload>/findings/google/_google_latest.md`. Follows the
        `/troubleshoot` schema, featuring `Restart Attempt` as the very first
        column, followed by `Timestamp (UTC)`, `Culprit Pod & Host`,
        `Root Signature / Mechanism`, `Downstream Impact & Failure Mode`,
        `Failure Domain`, `Recovery Outcome`, and `Linked Diagnostic Packet`.
    -   **Division of Diagnostic Labor**: External customers provide raw
        empirical traces in `jobs/<workload>/data/` and customer reports in
        `findings/customer/`; **Google synthesizes Level A and Level B failure
        tables**.
5.  **Proprietary HLO Redaction**: All compiler dumps and HLO graphs must be
    sanitized via
    [[redact-diagnostic-artifacts]](`redact_diagnostic_payload.py`) before
    publication. Proprietary variable names, vocabulary dimensions, and customer
    source paths are stripped while mathematical op topologies (`all-gather`,
    `reduce-scatter`, `custom-call`) are preserved.
6.  **Partner Issue Tracker Targeting**: All issue tracker operations target the
    customer's designated **Partner Issue Tracker** component and template
    (`identity.component_id` / `identity.template_id` or environment
    configuration). Never file customer-facing diagnostic packages in unapproved
    internal-only components.
7.  **Universal Symmetrical Monitoring Loop**: By default, completing `/handoff`
    arms background monitoring via [[monitor-handoff-bucket]], taking a baseline
    snapshot and establishing continuous polling across all diagnostic tracks:
    -   **Google Updates (1P)**: Monitors `jobs/<workload>/findings/google/`
        (specifically `_google_latest.md`),
        `infrastructure/findings/_infra_latest.md`, and `INCIDENT_MANIFEST.json`
        for verified Google verdicts, concrete empirical evidence (e.g. culprit
        host, physical error registers, optical attenuation), and actionable
        workarounds/next steps (e.g. `kubectl cordon <node>`, recommended XLA
        flags, or verified Orbax checkpoint recovery paths).
    -   **Customer Updates (3P)**: Monitors `jobs/<workload>/data/` and
        `findings/customer/` for customer-uploaded traces, reproduction scripts,
        and updated notes.
    -   Symmetrically alerts on new findings across both sides without requiring
        manual perspective flags.

--------------------------------------------------------------------------------

## Diagnostic Steps and Triage Workflow

### Path A: Prepare & Hand Off (Initial Investigator)

1.  **Execute `/troubleshoot` & Gather Findings**: Run
    [[workload-disruption-orchestration]] against the target workload to
    investigate disruption mechanisms, quote culprit hosts from primary logs,
    resolve node scopes, and emit the triage state JSON
    (`_wiki/scratch/triage-<workload>-<timestamp>.json`). If pre-gathered
    findings are supplied via `--findings`, verify schema compliance.
2.  **Verify Bucket Connectivity & Security**: Run
    [[verify-handoff-bucket-access]] to ensure the destination GCS bucket
    exists, is write-accessible, and enforces
    `publicAccessPrevention: enforced`.
3.  **Convert Triage State**: Run [[convert-triage-to-findings]] to translate
    `/troubleshoot` outputs (`_wiki/scratch/triage-*.json`) into
    `troubleshoot_findings.json`.
4.  **Scaffold Repository**: Run [[scaffold-handoff-bucket]] to create the
    incident folder tree and upload `BUCKET_STRUCTURE.md`. It never pre-creates
    manifests, READMEs or failure tables.
5.  **Package & Redact Artifacts**: Run [[package-handoff-artifacts]] to gather
    logs, sanitize HLO graphs with [[redact-diagnostic-artifacts]], and upload
    them to `jobs/<workload>/data/`. Its `missing_artifacts` lists telemetry
    that had no source.
6.  **Generate Manifests & Tables**: Run [[generate-handoff-manifests]],
    [[generate-two-level-failure-tables]], and [[render-handoff-readmes]] with
    the **same** `--output_dir` (e.g. `_wiki/scratch/handoff_<storage_key>/`),
    which then mirrors the bucket layout.
7.  **Publish the Package**: Run [[upload-handoff-package]] on that directory.
    It uploads nothing if a required file is missing or any file is empty,
    placeholder-only, or invalid JSON.
8.  **Verify Bucket Content**: Run [[verify-handoff-package]]. It must return
    `VERIFIED`; on `INCOMPLETE` or `VERIFY_FAILED`, stop and fix the listed
    problems before continuing.
9.  **Stage Partner Issue Tracker Update**: Run [[post-partner-issue-comment]]
    to render the comment and print the issue link for the operator to paste (it
    does not post automatically), only after presenting the formatted update to
    the operator and receiving explicit approval (`requires_approval`: the glue
    runner refuses it until re-run with `--approved`).
10. **Arm Background Monitor**: Execute [[monitor-handoff-bucket]] to initialize
    the baseline change-detection state.

--------------------------------------------------------------------------------

## Mitigations, Resolution and Gotchas

### Operational Gotchas

-   **Hardcoded Buckets**: Never assume a shared default bucket. Customers
    provision isolated projects with custom debugging buckets. No script has a
    default bucket; anything that writes to GCS fails without one.
-   **Empty Packages**: A successful scaffold is not a published package. Only a
    `VERIFIED` result from [[verify-handoff-package]] means the bucket holds
    real content.
-   **Unverified Node Attribution**: Never cite device coordinates as the
    culprit node; quote the host named in Pathways logs.
-   **Premature node replacement**: Do not drain or isolate nodes without
    4-point verification and human TPU-oncall approval.

--------------------------------------------------------------------------------

## Connections

-   Workflow: [[workload-diagnostic-handoff]]
-   Actions:
    -   [[verify-handoff-bucket-access]]
    -   [[convert-triage-to-findings]]
    -   [[scaffold-handoff-bucket]]
    -   [[package-handoff-artifacts]]
    -   [[generate-handoff-manifests]]
    -   [[generate-two-level-failure-tables]]
    -   [[render-handoff-readmes]]
    -   [[upload-handoff-package]]
    -   [[verify-handoff-package]]
    -   [[redact-diagnostic-artifacts]]
    -   [[post-partner-issue-comment]]
    -   [[monitor-handoff-bucket]]
-   Tool Reference: [[workload-diagnostic-handoff-tool]]

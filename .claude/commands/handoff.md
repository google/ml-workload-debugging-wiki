
# `/handoff` — Cloud TPU Workload Diagnostic Handoff

Claude Code counterpart of the Antigravity `handoff` skill
(`_agents/skills/handoff/SKILL.md`).

## Overview

Automates the diagnostic exchange protocol between Partner ML Engineering and
Google Cloud Support / CE for Cloud TPU workloads, following the
`[[workload-diagnostic-handoff]]` workflow and playbook.

> [!IMPORTANT] **DYNAMIC BUCKET PARAMETERIZATION**: Destination buckets depend
> on the customer project (e.g. `gs://<your-debug-bucket>/`,
> `gs://example-debug-bucket/`). If the target GCS bucket is not provided in the
> prompt or findings file, you **MUST prompt the operator to specify the
> customer project's GCS bucket** before proceeding with scaffolding or uploads.

> [!IMPORTANT] **DEFAULT BUCKET MONITORING**: Completing `/handoff` takes a
> baseline snapshot and arms background change-detection via
> [[monitor-handoff-bucket]] by default. Pass `--no-watch` to disable.

--------------------------------------------------------------------------------

## Commands

### 1. Prepare & Hand Off

```bash
/handoff <workload_name> [case_id] [--bucket=<bucket_name>] [--findings=<path>]
```

-   **Step 1: Execute `/troubleshoot` & Gather Findings**: Unless
    `--findings=<path>` is explicitly provided with pre-verified findings,
    `/handoff` automatically starts by executing `/troubleshoot <workload_name>`
    (following `[[workload-disruption-orchestration]]` Milestones 1 & 2:
    `validate-workload`, `resolve-workload-nodes`, `query-crash-signatures`,
    `extract-culprit-hosts`, `query-vm-termination-events`,
    `query-node-health-events`, `resolve-workload-infra-specs`, and failure
    table generation) to gather empirical findings and output the triage state
    JSON (`_wiki/scratch/triage-<workload>-<timestamp>.json`).
-   **Step 2: Verify Bucket Connectivity & Security**: Prompts for the GCS
    bucket if not specified, then verifies bucket connectivity and public access
    prevention via [[verify-handoff-bucket-access]].
-   **Step 3: Convert Triage State**: Converts triage state via
    [[convert-triage-to-findings]] into canonical `troubleshoot_findings.json`.
-   **Step 4: Scaffold GCS Bucket**: Scaffolds `gs://<bucket>/<storage_key>/`
    via [[scaffold-handoff-bucket]], which renders the bucket layout template
    `_wiki/tools/scripts/handoff/templates/BUCKET_STRUCTURE.template.md` into
    `BUCKET_STRUCTURE.md` at the incident root — the dictionary of where each
    file belongs in the bucket.
-   **Step 5: Package & Redact Artifacts**: Sanitizes HLO dumps via
    [[redact-diagnostic-artifacts]] and uploads scratch telemetry via
    [[package-handoff-artifacts]](upload is on by default). Report its
    `missing_artifacts`: telemetry with no source is listed, never written as a
    placeholder file.
-   **Step 6: Generate Manifests & Failure Tables**: Generates JSON Schema v2.2
    manifests via [[generate-handoff-manifests]] and synthesizes Two-Level
    Failure Tables via [[generate-two-level-failure-tables]] adhering to
    `/troubleshoot` schema. Give every generator the **same** `output_dir` (e.g.
    `_wiki/scratch/handoff_<storage_key>/`); it then mirrors the bucket layout.
    A generator that exits 1 (invalid findings, no bucket) stops the handoff.
-   **Step 7: Render Readmes**: Renders `INCIDENT_OVERVIEW.md`, `README.md`,
    `jobs/<workload>/JOB_README.md` and `jobs/<workload>/README.md` via
    [[render-handoff-readmes]] into the same `output_dir`.
-   **Step 8: Publish the Package**: Uploads that directory via
    [[upload-handoff-package]]. It uploads nothing (`REFUSED`) if a required
    file is missing or any file is empty, placeholder-only, or invalid JSON;
    pass `dry_run: true` to see what would be copied.
-   **Step 9: Verify Bucket Content**: Runs [[verify-handoff-package]], which
    must return `VERIFIED`. On `INCOMPLETE` or `VERIFY_FAILED`, stop, report the
    listed problems, and do not continue to Step 10.
-   **Step 10: Stage Partner Issue Tracker Comment**: Stages the Partner Issue
    Tracker comment (using the target component and template configured for the
    partner project) via [[post-partner-issue-comment]]: renders the comment and
    prints the issue link. **Nothing is posted automatically** — the operator
    pastes it. **Requires explicit operator approval** — show the comment and
    wait for a yes before running it.
-   **Step 11: Arm Baseline Bucket Monitoring**: Initializes baseline monitoring
    state via [[monitor-handoff-bucket]].

### 3. Check for Updates

```bash
/handoff --watch <storage_key> [--bucket=<bucket_name>]
```

-   **Universal Symmetrical Monitoring Loop** via [[monitor-handoff-bucket]]:
    -   Naturally unified: polls the repository symmetrically across all tracks
        without needing any perspective flags.
    -   If 1P Google updates are detected (`findings/google/`,
        `infrastructure/findings/_infra_latest.md`, and
        `INCIDENT_MANIFEST.json`), immediately alerts with:
        -   **Verified Verdict / Status**: (e.g. `VERIFIED_HARDWARE_FAULT` on
            node `gke-tpu-node-pool-xxxx`).
        -   **Supporting Empirical Evidence**: Quoted culprit host, hardware
            error registers (e.g. TPU Support parity errors), and optical attenuation
            measurements.
        -   **Actionable Workarounds & Mitigation Paths**: Exact commands (e.g.
            `kubectl cordon <node>`), recommended XLA flags, and verified Orbax
            checkpoint recovery paths (`gs://...`).
    -   If 3P Customer updates are detected (`findings/customer/` or `data/`),
        immediately alerts on new customer telemetry, traces, and notes.
    -   If both or general files are updated, reports all changes cleanly in the
        alert card.

--------------------------------------------------------------------------------

## Execution

Every handoff Action is `execution_mode: "command"`. Per the delegation
invariant below, the coordinator does not run them itself: `practitioner` runs
each one through the glue runner:

```bash
python3 _wiki/tools/scripts/python_glue_runner.py \
    <action_name> '<parameters_json>'
```

-   [[post-partner-issue-comment]] declare `requires_approval`. The glue runner
    refuses them (exit code 4) and prints an `Awaiting user approval ...` line.
    The coordinator relays it to the operator, and only after explicit approval
    does the practitioner re-run the same command with `--approved` appended.
-   [[upload-handoff-package]] and [[package-handoff-artifacts]] write to the
    customer bucket. Use `dry_run: true` / `upload: false` when testing.
-   [[monitor-handoff-bucket]] polls once per invocation. To keep watching, the
    practitioner re-runs it with the `Bash` tool's `run_in_background`, not a
    blocking loop.

--------------------------------------------------------------------------------

## Grounding & Invariants

-   Grounding: Follow [[debugging-mindset]] and [[failure-tables]].
-   Two-Level Standard: Level A (Infrastructure / Hardware) vs Level B (Job
    Progression). The external customer provides empirical traces; Google
    synthesizes failure tables.

--------------------------------------------------------------------------------

## Main Coordinator Delegation Invariant

The main coordinator agent acts strictly as a high-level orchestrator and has
**zero authorization to execute standalone shell commands, direct script
invocations, log queries, file edits, or GCS uploads directly**.

All execution MUST be delegated to the designated specialized wiki subagents via
the `Agent` tool (`subagent_type`; `SendMessage` to continue one):

-   **Workflow & Playbook Routing**: Delegate to `router`.
-   **Diagnostic & Action Execution**: Delegate to `practitioner` (for
    `/troubleshoot` milestones, Cloud Logging, GCS bucket scaffolding, empirical
    packaging, uploads, and background monitoring).
-   **Hypothesis & Gap Evaluation**: Delegate to `advisor`.
-   **Adversarial Verification & Audit**: Delegate to `reviewer` (for the
    `ai-verify-failure-attribution` prompt task before publication).
-   **Code, Script & Schema Edits**: Delegate to `maintainer`.

The main agent manages user approvals, coordinates subagent dispatches, and
presents final synthesized reports.

--------------------------------------------------------------------------------

## Connections

-   Antigravity counterpart: `_agents/skills/handoff/SKILL.md`
-   `_wiki/workflows/workload-diagnostic-handoff.json`
-   `_wiki/playbooks/workload-diagnostic-handoff.md`
-   `_wiki/tools/workload-diagnostic-handoff-tool.md`
-   `_wiki/tools/scripts/handoff/templates/BUCKET_STRUCTURE.template.md`

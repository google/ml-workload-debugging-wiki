
# `/troubleshoot` Command

Multi-agent troubleshooting command equivalent to `/troubleshoot`.

## Description

Executes an end-to-end multi-agent diagnostic sequence for a target ML workload
ID by invoking specialized subagents in sequence: Router, Practitioner,
Reviewer, and Advisor. Operates strictly over public tools and data sources
(Cloud Logging, GKE cluster events, Kubelet/Node logs) without requiring
Google-internal permissions or tools.

## Public Surface & Scope Boundary

This command is strictly public-facing:

-   Operates only on public telemetry sources: Cloud Logging, Kubernetes/GKE
    cluster events, Kubelet/Node logs, and public Cloud TPU APIs.
-   **MUST NOT** execute or recommend any action, tool, or workflow marked
    "internal-only". If a symptom or workflow has `internal_steps` or references
    internal actions, `/troubleshoot` skips them and uses only public
    alternatives (e.g., `lookup-workload-timestamps`,
    `isolate-node-via-rr-label` via `node-isolation-and-rr-api`).

## Command Usage

`/troubleshoot <workload_id>`

> [!IMPORTANT] **MUST read first:** `_wiki/playbooks/rca-invariants.md` (access
> preflight -> `NO_DATA`, earliest error across all pods, barriers are victims,
> workload scoping, verdict outcomes). It overrides any step below;
> `_wiki/playbooks/debugging-mindset.md` is optional background.

## Subagent Invocation Sequence

This command coordinates the specialized subagents defined in `.claude/agents/`:

1.  **Router** (`.claude/agents/router.md`): Log triage and symptom matching
    system instructions.
2.  **Practitioner** (`.claude/agents/practitioner.md`): Action and workflow
    execution instructions via
    `python3 _wiki/tools/scripts/python_glue_runner.py`.
3.  **Reviewer** (`.claude/agents/reviewer.md`): Schema review and LOG.md
    compliance guidelines.
4.  **Advisor** (`.claude/agents/advisor.md`): Query resolution and routing
    guidance.

--------------------------------------------------------------------------------

## Action Execution Modes

Read `execution_mode` on every Action before running it. The three modes
dispatch differently and an Action is not necessarily a shell command.

| `execution_mode` | Naming   | Dispatch in Claude Code                   |
| :--------------- | :------- | :---------------------------------------- |
| `command` | any | `Bash`: `python3 _wiki/tools/scripts/python_glue_runner.py <action> '<params_json>'` |
| `ai` | **`ai-` prefix** | The `Agent` tool with interpolated prompt |
| `manual`         | any      | Surface `instruction`; do not automate    |

For an `ai-` Action: read its `prompt` verbatim, interpolate every declared
parameter (both `${name}` and `<name>` styles), then dispatch it as a subagent
task, honouring the Action's `model`, `batch_size`, and `max_concurrent_workers`
hints. Never route an `ai-` Action through the glue runner, and never hand-roll
an equivalent analysis in place of the prompt — if it cannot be dispatched, say
so.

--------------------------------------------------------------------------------

## Investigation Invariants

**These apply to every `/troubleshoot` run, regardless of which workflow is
selected.** They are hard rules derived from real misdiagnoses; the canonical
list with worked examples is
`_wiki/playbooks/workload-disruption-orchestration.md`.

1.  **Never infer a culprit host from device coordinates.**
    `DATA_LOSS: Device device(N,...,slice=K) is unplaced` shows where the
    *symptom surfaced*; the slice index is frequently not the slice that died.
    Pathways names the real host literally
    (`lost connection to peer at http://machine/<node>`). Quote it — run
    `extract-culprit-hosts`.
2.  **Never assert a fleet-wide negative over an unverified node set.** Run
    `resolve-workload-nodes` and confirm `status: COMPLETE` before any "no
    hardware event" conclusion.
3.  **Start every causal chain at the first process death**, not the first
    reported error. Client-side `DATA_LOSS` / `PlacementDataLoss` / `ValueError`
    / `JaxRuntimeError` are almost always downstream.
4.  **Read `si_code` on every signal.** `0` = `SI_USER` = sent via `kill()` by
    another process, not a CPU fault. `-6` with `from PID <self>` =
    self-`abort()` teardown, not an independent fault.
5.  **Preemption wording is not preemption evidence.**
    `EXIT_TYPE_QUICKLY_DURING_PREEMPTION` and
    `"Resource manager is being preempted"` are emitted for any SIGTERM.
    Corroborate with an audit event on this workload's own nodes. **Kubelet
    eviction also sends SIGTERM** — before writing `VM termination` or
    `preempted`, check `reason=Evicted` / `ephemeral-storage` / `DiskPressure`.
    A missing exit code alongside SIGTERM wording points at eviction, not at an
    unknown failure. See [[false-preemption-sigterm]] and
    [[missing-exit-codes-eviction]].
6.  **Restart timelines are drafts.** `job-disruption-timeline` both over-counts
    (duplicate records at identical timestamps; clean exit logged as a restart)
    and under-counts (it cannot see disruptions the workload recovered from).
    Reconcile against the crash sweep.
7.  **Recovered disruptions are findings.** A crash the workload survived proves
    the fault is survivable and reclassifies failed recoveries as recovery-path
    defects.
8.  **Never conclude from truncated output.** Anything read through
    `head`/`tail`, or any query where `original_count == limit`, must be
    re-verified unbounded.
9.  **Never infer platform identity.** Accelerator type, topology, chips per
    host and cluster must be quoted from log entries for this workload
    timestamped inside the investigated window. Current cluster state
    (`kubectl`, node pools, `resolve-cluster-metadata`) is not evidence. Never
    take them from names, `INDEX.md`, or case studies.
10. **Report only what ran — never invent Action IDs.** Every entry in the
    `"Actions Executed"` table MUST match an existing `_wiki/actions/<id>.json`
    file whose `python_glue_runner.py` or `ai-` subagent call actually
    completed. Never invent hyphenated action names (e.g.
    `diagnose-step-progression`, `check-worker-skew`,
    `check-inspection-session`) for ad-hoc queries or scratch scripts; record
    ad-hoc queries separately with their exact command. Do not write or execute
    custom `.py` scripts under `_wiki/scratch/`. Copy log quotes from tool
    output, never retype them.
11. **Dynamic Attempt Discovery.** When an operator requests the latest N
    restart attempts or when investigating multi-attempt disruptions, NEVER
    assume the restart attempt count is static or bounded by an
    initial/arbitrary window. Dynamically query
    `labels."k8s-pod/jobset_sigs_k8s_io/restart-attempt"` across the unclipped
    workload lifespan (or inspect JobSet status) to discover all distinct
    attempt indices [0..M], and select the latest N attempts in reverse
    chronological order. Discover the true unclipped workload lifetime boundary
    (`first_occurrence` to `last_occurrence`) before time-slicing disruption
    windows.

--------------------------------------------------------------------------------

## Fast-Path CLI Cheat-Sheet (No Action JSON File Inspection Needed)

Practitioner MUST invoke actions directly via `python_glue_runner.py` without
wasting turns viewing `_wiki/actions/*.json` schemas first:

```bash
# 1. Cloud Logging 2-Second Preflight Probe (trips circuit breaker on exit 77 ACCESS_DENIED)
python3 _wiki/tools/scripts/python_glue_runner.py validate-workload \
  '{"workload": "<workload>", "project_id": "<project_id>"}'

# 2. Single Workload-Wide Timestamp & Attempt Discovery
python3 _wiki/tools/scripts/python_glue_runner.py lookup-workload-timestamps \
  '{"jobset_name": "<workload>", "project_id": "<project_id>", "output_file": "scratch/timestamps_<workload>.json"}'

# 3. Full-Window Crash Signatures & Disruption Windows
python3 _wiki/tools/scripts/python_glue_runner.py query-crash-signatures \
  '{"workload": "<workload>", "project_id": "<project_id>", "start": "<start_utc>", "end": "<end_utc>"}'
python3 _wiki/tools/scripts/python_glue_runner.py locate-disruption-windows \
  '{"workload": "<workload>", "project_id": "<project_id>", "start": "<start_utc>", "end": "<end_utc>"}'

# 4. Parallel Per-Restart-Attempt Deep-Dive (executes all K attempts concurrently via ThreadPoolExecutor)
python3 _wiki/tools/scripts/python_glue_runner.py find-earliest-failure \
  '{"workload": "<workload>", "project_id": "<project_id>"}' --attempts-json=@scratch/timestamps_<workload>.json
python3 _wiki/tools/scripts/python_glue_runner.py extract-culprit-hosts \
  '{"workload": "<workload>", "project_id": "<project_id>"}' --attempts-json=@scratch/timestamps_<workload>.json
```

--------------------------------------------------------------------------------

## Milestone Execution Sequence (Milestones 1 to 3)

Execute Milestones 1 to 3 for the target workload ID:

### Milestone 1: Reconnaissance & Parallel Dispatch (Router)

-   **Multi-Workload Turn-1 Parallel Fan-Out:** When the user asks to
    investigate $N > 1$ workloads (or multiple independent incidents), the
    coordinator MUST spawn $N$ **Practitioner** subagents **in parallel in Turn
    1** alongside **Router** — one **Practitioner** per workload — so all $N$
    workloads execute their Milestone 1 & 2 diagnostics concurrently rather than
    waiting for **Router** to finish reading files first. Even for a single
    workload (`N = 1`), spawn **Router** and **Practitioner** concurrently in
    Turn 1.
-   **Pre-flight Auth & Access Check (`gcloud-login`):** Before querying
    telemetry or cluster logs, verify gcloud authentication and Application
    Default Credentials (ADC) validity by executing `gcloud-login` with
    `check_only=true`. If either credential token is invalid or expired, request
    user approval to run `gcloud-login` to authenticate and refresh credentials.
-   **2-Second Cloud Logging Preflight Gate, Circuit Breaker (Exit Code 77) &
    `NO_DATA` Stop Rule**: Run `validate-workload` to establish state
    (`INITIALIZED` / `FAILED_STARTUP` / `WORKLOAD_NOT_FOUND`) and recover
    `cluster_name` / `location`. Diagnostic scripts exit **77**
    (`ACCESS_DENIED`) or **75** (`QUERY_FAILED`) when Cloud Logging refuses or
    fails a query. When exit **77** occurs (or log queries fail with
    `PERMISSION_DENIED`, pending IAM approval `tickets/...`,
    `logging.logViews.access`, or `invalid_auth`), `python_glue_runner.py` trips
    the project-scoped circuit breaker
    (`_wiki/scratch/.cloud_logging_access_denied_cache.json`) and fast-fails
    subsequent Cloud Logging actions in `< 50ms`. **STOP immediately** and
    report verdict `NO_DATA` (`ACCESS_DENIED`) with the exact error; do **not**
    execute Milestone 2 / 2a actions (`resolve-workload-nodes`,
    `query-vm-termination-events`, etc.) when log access is blocked.
-   **Phase A — Dynamic Attempt Discovery, Lookback Questionnaire & Lifetime Boundary**:
    If a specific time range (`start`/`end`) or `lookback_days` is not already
    specified by the user, present the action's questionnaire to prompt the user
    how far back to analyze (default: 1 day; options: `1` day (default), `3`
    days, `7` days, `14` days, `30` days, or custom input).
    Before time-slicing disruption windows, discover the true unclipped workload
    lifetime boundary (`first_occurrence` to `last_occurrence`) and all distinct
    attempt indices `[0..M]` in a single pass using `lookup-workload-timestamps`
    (passing `--lookback_days=<N>` when specified, or selecting the latest N
    attempts in reverse chronological order via `--latest_attempts=N` when requested).
    Note: `lookup-workload-timestamps` queries the coordinator/head pod (`-0-0-`,
    `-pathways-head`, or `-pwhd`) rather than scanning logs from every worker host
    across all slices.
-   Automatically retrieve application and telemetry logs for the target
    workload (using `query-workload-logs` or
    `ai-triage-classify-and-mask-logs`).
-   **Always run `query-crash-signatures` over the FULL run window, passing
    explicit `start` and `end`.**
-   **Locate disruption windows (`locate-disruption-windows`).**
-   Match error signatures against the wiki symptom catalog in `INDEX.md` and
    `symptoms/`.

### Milestone 2: Direct Workload Diagnostics & Per-Attempt Parallel Fan-Out (Router & Practitioner & Advisor)

**Disk-backed state passing.** The Practitioner must save diagnostic payloads to
`_wiki/scratch/triage-<workload>-<ts>.json` and pass only the summary and the
artifact file path to Reviewer and Advisor. Do not pass large log payloads
inline in inter-agent messages.

-   **Phase B — Parallel Per-Restart-Attempt Investigation:** Once Phase A
    enumerates the $K$ restart attempts in the target window, **NEVER
    investigate Attempt $0, 1, \dots, K-1$ in a serial loop**. Pass
    `--attempts-json=@scratch/timestamps_<workload>.json` (or
    `"attempts": [...]` in `params`) to `python_glue_runner.py` so per-attempt
    Cloud Logging queries (`find-earliest-failure`, `extract-culprit-hosts`,
    `query-vm-termination-events`, `query-node-health-events`,
    `query-workload-logs`) run concurrently across all $K$ attempts via
    `ThreadPoolExecutor`, while preserving cross-attempt correlation.
-   Invoke **Advisor** (`.claude/agents/advisor.md`) if multiple symptoms or
    workflows match to resolve ambiguity and guide workflow selection.
-   Invoke **Practitioner** (`.claude/agents/practitioner.md`) to interpret
    JSON/YAML workflow schemas.
    -   Dispatch `execution_mode: "ai"` actions (the `ai-` prefixed ones) as
        **subagent prompt tasks**, NOT through the glue runner. See
        `.claude/agents/practitioner.md`.
-   Prompt for user approval before executing any state-mutating actions
    (`requires_approval: true`).
-   Post action completion summary and `summary_fields` metrics.

#### Milestone 2a: Baseline Disruption Evidence

(mandatory when a disruption occurred)

Run whenever Milestone 1 found readable logs with crash signatures, restarts, or
a non-zero exit — **regardless of which symptom-specific workflow was
selected**. Skip for `NO_DATA`. For `FAILED_STARTUP`, `WORKLOAD_NOT_FOUND`,
pending Kueue workloads (`allows to fit only N out of M slice(s)`), or multi-run
JobSet series where some runs never schedule and others crash, route to
`workload-run-forensics` and format the output with
`run-forensics-report-template`. (Note: Kueue and JobSet are optional on GKE
clusters; if the cluster schedules Pods or Jobs directly without Kueue/JobSet
CRDs, skip Kueue-specific queue actions and scope log queries by Pod/Job
prefix.)

1.  `resolve-workload-nodes` — establish authoritative node scope. Confirm
    `status: COMPLETE`; if `INCOMPLETE`, downstream hardware verdicts are
    inconclusive, not negative. Emits two scoping clauses, which are **not**
    interchangeable: `node_filter_clause` (`protoPayload.resourceName`) for
    audit logs, and `node_name_filter_clause` (`resource.labels.node_name`) for
    `k8s_node` queries. Each field exists only on its own log shape.
2.  `extract-culprit-hosts` — quote culprit hosts from Pathways' own logs and
    identify repeat offenders (parallelized across attempts via
    `--attempts-json`). Node Replacement is decided on distinct **primary**
    incidents (classified by reporting breadth), never on raw event counts: a
    single host death is reported by every surviving peer at once.
3.  `query-vm-termination-events` — deterministic, **node-scoped** audit query.
    Pass the `node_filter_clause` from step 1; an unscoped query returns other
    workloads' preemptions and invites a false preemption verdict.
4.  `query-node-health-events` — in-node evidence (kubelet,
    node-problem-detector, TPU device plugin). Pass `node_name_filter_clause`; a
    `k8s_node` query carries no pod_name label, so unscoped it returns other
    workloads' events on other clusters. A clean audit log alongside in-node
    worker kills is a real and common pattern.
5.  `ai-hw-analyzer` — interpret the above plus in-node evidence. Must return a
    **scoped** verdict naming what was checked (e.g.
    `No VM Termination (all 65 nodes across 4 node groups, 07:45-10:05)`), never
    a bare `No VM Termination`.
6.  `resolve-workload-infra-specs` — resolve workload infrastructure specs
    (machine type, accelerator type, chips per host, topology) from node
    metadata or telemetry fallback to populate the Primary Failure Table.

For a full disruption investigation (restart loops, reshards, multi-cycle
failures), prefer the complete `workload-disruption-orchestration` workflow,
which additionally runs `job-disruption-timeline`, `ai-app-analyzer`,
`pod-analyzer`, `resolve-workload-infra-specs`, and `ai-generate-failure-table`
with a reconciliation step.

### Milestone 3: Final Compliance Review & Audit (Reviewer)

-   Invoke **Reviewer** (`.claude/agents/reviewer.md`).
-   **Run `ai-verify-failure-attribution` before publishing any failure table or
    culprit attribution.** This is a gate, not an optional step: it re-derives
    every attribution from primary logs and flags claims that were inferred
    rather than quoted. Do not publish until it returns `APPROVED`.
-   Perform final compliance review and audit on `_wiki/LOG.md` and diagnostic
    artifacts.
-   Verify schema adherence, taxonomy accuracy, and link integrity.
-   **Prompt Handoff Follow-up (`prompt-handoff-followup`)**: At the conclusion
    of triage, prompt the operator with an interactive questionnaire:
    "Troubleshooting analysis complete. Would you like to package the findings,
    failure table, and diagnostic telemetry for follow-up via `/handoff`?" If
    confirmed, invoke `/handoff` with
    `--workload=${workload} --project_id=${project_id} --cluster=${cluster}`.

--------------------------------------------------------------------------------

## Grounding

Every recommendation must cite the wiki action, workflow, or page that backs it,
and the citation must apply to *this* platform — GKE-managed TPUs use
`isolate-node-via-rr-label` via `node-isolation-and-rr-api`. Anything the wiki does
not support must be labelled `unsupported - operator judgement` rather than
presented as documented practice. Never invent action, parameter, host, or file
names. State which log line supports each claim, and mark values that were
derived rather than read. A query that hit its entry cap
(`original_count == limit`) proves nothing negative.

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

-   Antigravity counterpart: `_agents/skills/troubleshoot/SKILL.md`
-   `_wiki/playbooks/workload-disruption-orchestration.md`
-   `_wiki/workflows/workload-disruption-orchestration.json`
-   Subagents: `.claude/agents/{router,practitioner,advisor,reviewer}.md`

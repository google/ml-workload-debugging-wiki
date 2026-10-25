---
title: The Expert Debugging Mindset
type: playbook
internal_only: false
tags: [training]
created: 2026-05-14
updated: 2026-10-03
---


## Overview

Optional background. The mandatory rules (access preflight, NO_DATA, earliest
error across all pods, barriers are victims, workload scoping, verdicts) are in
[[rca-invariants]]; read that first.

Debugging complex ML workloads at scale requires more than just following a
checklist. It requires a specific mindset characterized by rigor, persistence,
and skepticism. This page outlines the attitude and practices that distinguish
an expert debugger from a novice.

## Core Principles of the Mindset

### 1. Do Not Settle for "Good Enough"

Don't stop at downstream symptoms (e.g., "Shutdown barrier timed out"). Push to
find the trigger.

### 2. Find Pride in the Answer

Provide a definitive, actionable diagnosis that enables a developer to fix the
problem without ambiguity.

### 3. Look for the Missing Data (Silent Outliers)

Account for all workers. Workers that died silently or stopped logging without
exit codes are often the earliest failures.

### 4. Verify Error Uniqueness (Isolate Red Herrings)

Actively test if a severe-looking error is a red herring:

*   **Worker Outlier**: Is the error unique to failing workers? If all workers
    log it, it is likely noise.
*   **Success Baseline**: Does the error occur in healthy runs? If so, it is not
    the cause.
*   **Temporal Test**: Ensure the error preceded the teardown. Downstream errors
    are consequences, not causes. (e.g., if a software compilation crash occurs
    before a GKE eviction event, the software crash is the cause, and the
    eviction is a downstream event possibly triggered by core dumps filling the
    disk; refer to [[gotchas-kubernetes-gke]] Gotcha 9).
*   **Reporter vs. Culprit**: The node broadcasting cancellation is the
    reporter. The culprit is often a node that stalled silently earlier.
*   **Sibling Logs**: Compare with 2-3 other workers to verify if it is an
    outlier.

### 5. Corroborate with Multiple Signals

Cross-reference application logs, GKE events, and hardware status. Critical
conditions are often logged as `INFO` or `WARNING` rather than `ERROR`.

### 6. Scope Broadly Before Diving Deep

Analyze high-level signals (timestamps, exit codes, events) across the system
before analyzing stack traces or code. Use Failure Tables to structure findings.

### 7. Triage across the Stack

Map application symptoms to lower-level events (evictions, preemptions). See
[[layered-debugging-framework]] for stack mapping.

*   **Identify Checkpoint Coordinators**: In distributed JAX/Orbax, storage
    directory operations (such as folder deletions) are executed exclusively by
    worker 0 (JAX process zero). Secondary workers simply experience barrier
    timeouts. Always name worker 0 / JAX process zero as the blocked
    coordinator.

## How to Practice This

*   **Fill out Failure Tables thoroughly**: Document steps taken and defend your
    reasoning.
*   **Use strict time windows**: Avoid missing early exception messages by
    checking logs in chronological order.
*   **Verify Timestamps**: Always verify that the timestamp of any
    error/exception log matches the incident timeline (i.e. immediately
    preceding the exit code log). Do not diagnose based on stale logs or
    transient errors from earlier in the time window (refer to
    [[gotchas-cloud-logging]] Gotcha 11).
*   **Identify Cascade Aborts**: If you see `CoordinationServiceAgent::SetError`
    in the stack trace of an aborted process (SIGABRT/134), it is a cascade
    symptom. Look for the worker that failed *first* (earliest timestamp or
    eviction event) rather than blaming the check failure (refer to
    [[exit-codes]] under 134).
*   **Check both Worker 0s**: Always check both GKE Worker 0 and JAX Process 0.
*   **Triage Exit Codes Carefully**: Refer to [[exit-codes]] to understand the
    meaning of exit codes (like 134, 143, 137). When triaging, always use the
    sorted earliest-failure query to avoid missing truncated early failures in
    large clusters. Refer to [[gotchas-cloud-logging]](Gotcha 10) for details.
*   **Always Read Workflow Playbooks First**: Do not skip to querying logs.
    Before running any commands, find and read the relevant workflow playbook
    (e.g., [[general-ml-debugging]] or [[hang-diagnosis-methodology]]) to ensure
    you follow the structured diagnostic steps and avoid common pitfalls.
*   **Follow the Wiki Guidance**: Do not diagnose issues in isolation. When you
    see infrastructure events (like `Evicted` or `ephemeral-storage` limits),
    check INDEX.md and read the related playbooks/symptoms (like
    [[stopiteration-disk-full]]) to understand the correct causal relationship
    of events.
*   **Adhere to Reporting Formats**: When writing the final diagnosis JSON (e.g.
    `run_1_output.json`), strictly follow the requested schema (specifically
    `culprit_physical_machine` and NOT `culprit_machine`). Ensure all fields
    (`root_cause`, `culprit_pod`, `culprit_host`, `culprit_physical_machine`,
    `mitigation_action`) are present.
*   **Accurate Failure Table Categorization**: Classify failures correctly in
    Failure Tables:
    *   `Infrastructure / Eviction`: Ephemeral storage exhaustion, GKE
        evictions, node-level resource limits.
    *   `Configuration`: Configuration errors, invalid flags, missing
        parameters, pyconfig/ValueError initialization errors.
    *   `Distributed Hang`: JAX collective hangs, sflag timeouts, coordination
        service aborts (SIGABRT/134) during execution.
    *   `SW`: Other software bugs, JAX/XLA assertion failures, compiler errors,
        framework crashes.
*   **Close the Loop**: Update failure tables or case studies immediately upon
    finding evidence to avoid documentation gaps.

## Connections

-   **Playbooks**:
    -   [[debugging-flowchart]]
    -   [[debugging-gotchas]]
-   **Concepts**:
    -   [[exit-codes]]
    -   [[failure-tables]]
-   **Tools**:
    -   [[cloud-logging]]

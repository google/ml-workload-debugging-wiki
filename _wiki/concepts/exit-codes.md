---
title: Exit Codes
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-06-11
updated: 2026-10-20
---

When debugging a complex ML workload failure, exit codes returned by workers and
pods are crucial for identifying the root cause. This page documents the
interpretation and recommended actions for common exit codes.

## Exit Code Reference

### 134: SIGABRT

-   **Meaning**: System-side crash or JAX coordination service abort.
-   **JAX Coordination Timeout**: If the logs show
    `DEADLINE_EXCEEDED: Barrier timed out` or `Shutdown barrier has failed`,
    this means the JAX process aborted because one or more workers failed to
    reach a synchronization barrier. The exit code is 134, but this is a
    *symptom* of a failure elsewhere, not the root cause.
-   **Coordination Service SetError Abort**: If the logs show
    `Fatal Python error: Aborted` or `*** Check failure stack trace: ***` with
    `CoordinationServiceAgent::SetError` in the stack trace, this indicates the
    worker aborted because the JAX coordination service notified it that
    *another* worker has failed or exited. This is a cascade abort (symptom),
    not the root cause.
-   **Action**:
    1.  **Identify Cascade Aborts**: If `CoordinationServiceAgent::SetError` is
        present, do NOT assume the check failure is the root cause.
    2.  **Look for Earliest Evictions/Failures**: Query GKE events for `Evicted`
        or `low on resource` around the abort timestamp (note that eviction
        events can sometimes be logged with a delay of 1-3 minutes in GKE logs).
    3.  **Triage Chronologically**: If evictions occurred close to the abort
        time, categorize the failure as `Resource: Eviction` (due to
        ephemeral-storage or memory) rather than `Software`.
    4.  If no evictions occurred, identify the timestamp of the barrier timeout
        or abort, and query logs across all pods for a window *before* that
        timestamp to find the earliest worker process that crashed or hung
        (searching for `Traceback`, `Error`, `Exception`, or `429`).

### 137: SIGKILL

-   **Meaning**: 128 + 9: the process received SIGKILL. It cannot log anything
    itself.
-   **Causes**:
    1.  Host memory: the kernel OOM-killer (container memory limit or node RAM),
        e.g. host-RAM leaks in the software stack (see [[known-issues]]).
    2.  The kubelet's SIGKILL after the termination grace period (the process
        ignored SIGTERM / was still exiting).
-   **Not TPU HBM OOM**: HBM exhaustion is reported by the runtime as an error
    (`RESOURCE_EXHAUSTED`, `TpuOOM`), not by a kernel kill.
-   **Action**:
    -   Check the container's terminated reason (`OOMKilled` vs `Error`) and
        node events.
    -   Investigate memory usage patterns.
    -   Apply known mitigations for memory leaks if applicable.

### 143: SIGTERM

-   **Meaning**: Termination signal. Can be caused by:
    1.  **Kubernetes Eviction**: GKE evicts the pod due to resource limits
        (specifically `ephemeral-storage` exhaustion or `memory` OOM). This is a
        common root cause for false data exhaustion hangs. Refer to
        [[stopiteration-disk-full]] and [[missing-exit-codes-eviction]].
    2.  Maintenance event (expected every ~5 days on average).
    3.  Forced shutdown of node pool due to automated hardware issue detection.
    4.  Manual user shutdown (`kubectl delete jobset`).
    5.  Error in another node pool causing GKE to terminate the workload.
    6.  Coordinator Success: In JAX/MaxText, when the coordinator (Process 0)
        exits 0, GKE shuts down the remaining workers via SIGTERM.
    7.  **Watchdog Stalls / Checkpoint Hangs**: Exit code 143 (SIGTERM) can be a
        lagging indicator of an undetected node-level stall or deep checkpoint
        hang (which paralyzes the node proxy, causing the control plane to scale
        down the frozen node).
-   **Action**:
    -   **Check for Checkpoint Hangs**: Check for preceding `step-timebomb`
        watchdog logs or thread tracebacks on the specific node (especially
        slice 0, worker 0) before assuming a clean scale-down.
    -   **Verify Eviction**: Check Kubernetes events or Cloud Logging for
        `reason="Evicted"` or messages containing
        `low on resource: ephemeral-storage`.
    -   Check GCE system events for host termination or maintenance events.
    -   Check Kubelet logs for shutdown reasons.
    -   If caused by another node pool, find that node pool and investigate its
        logs.
    -   If the coordinator exited with 0, this is expected behavior and not a
        failure.

### 42: UNKNOWN ERROR / UNRECOVERABLE_ERROR

-   **Meaning**: Often relates to MegaScale/RPC timeouts (see [[megascale-hang]]
    for details).
    -   **MXLA Hang**: Collective DCN transfer timeout (default ~5 mins).
    -   **RPC Timeout**: RPC request cannot complete within deadline (default
        ~60s).
-   **Causes**:
    1.  Hardware issue/Interruption.
    2.  Bad machine requiring host isolation/replacement (look for repeat
        offenders).
    3.  Network contention (e.g., GCS checkpointing in parallel).
-   **Action**:
    -   Check if error is between specific machine pairs (`task-src->task-dst`).
    -   In newer libtpu versions, logs show GKE instance names for both sender
        and receiver.
    -   If a specific machine is a repeat offender across two or more distinct
        primary incidents, isolate/replace it -- see [[failure-tables]] for how
        primary incidents are counted.

### 1: User Program Error / Configuration Crash

-   **Meaning**: Error originating directly from user code (e.g., Python typo,
    module import error, configuration schema violation, or XLA compiler issue
    triggered by code).
-   **Startup Crash**: If multiple pods exit with `EXIT_CODE=1` almost
    immediately after startup (typically <5 minutes), this indicates a
    configuration or launch error. Refer to [[workload-startup-failures]].
-   **XPK Trailing Space Bug**: A common reason for Exit Code 1 in MaxText runs
    via XPK is trailing whitespace in command line arguments (e.g.,
    `--args="run_name=foo "`), which results in a `ValueError: '' not in [...]`
    crash.
-   **Action**:
    1.  Check early worker logs (first 60 seconds) to find the Python stack
        trace.
    2.  Review the command-line entrypoint arguments for formatting errors or
        extra spaces.
    3.  Refer to [[workload-startup-crash]] for details on typical startup error
        messages.

### 0: Success (or Rare Hardware Issue)

-   **Meaning**: Normally indicates success. However, in extremely rare cases
    (e.g., swapped PCIe cables), a failure might incorrectly report exit code 0.
-   **Action**: If behavior is illogical despite exit code 0, suspect rare
    hardware issues and gather more data points.

## Connections

-   **Workflows**:
    -   [[exit-code-multi-pod-analysis]]
-   **Concepts**:
    -   [[multi-pod-exit-code-analysis]]
-   **Symptoms**:
    -   [[megascale-hang]]

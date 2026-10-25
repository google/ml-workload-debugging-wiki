---
title: Hang Diagnosis Methodology
type: playbook
internal_only: false
tags:
- tpu
- hardware_fault
created: 2026-06-11
updated: 2026-10-20
---


## Definition

**Hang Diagnosis Methodology** outlines the step-by-step process of diagnosing
and resolving hangs in TPU ML workloads. A hang occurs when the workload is
stuck without making progress, which can be caused by software stalls, network
issues, or hardware faults.

Because hangs can cascade up the stack, diagnosis follows the
[[layered-debugging-framework]] to isolate issues starting from the Application
Layer down to the Hardware Layer. While the MXLA runtime detects DCN collective
stalls (`--megascale_graph_hang_threshold`; default timeout is **5 minutes**, or
**1 minute** for collectives classified as within-launch collectives), DCN-level
detection alone misses single-slice hangs, simultaneous data-input stalls across
all workers, host-offload / MPMD stalls, or synchronous post-step checkpoint
stalls; wrapping the training step in an application-level step watchdog
(`with health_watchdog(step, timeout_sec=5 * 60):` or
[[step-timebomb-watchdog]]) provides top-level coverage across all six primary
hang classes (`DATA_INPUT_STALL`, `BAD_TPU_CHIP`, broken ICI links,
`NETWORKING_ISSUE`, `DIFFERENT_MODULE` / `FINGERPRINT_MISMATCH`, and blocking
user business logic).

## Diagnosis Steps

1.  **Identify the Hang (Application Layer)**: Confirm if the job is actually
    hung or just slow (e.g., loss values not updating in container logs). For
    proactive detection across single-slice and non-DCN stalls, wrap the
    training step in an application-level step watchdog
    (`health_watchdog` or [[step-timebomb-watchdog]]). When a watchdog
    triggers, inspect pull-style state across the four core subsystems:

    -   **Data Input**: Check PjRt / JAX input arrays and host-to-device (H2D)
        transfer queues for unresolved or unready input buffers blocking TPU
        execution (`DATA_INPUT_STALL`, `pc == 0`) via
        `[[analyze-libtpu-timing]]`.
    -   **TPU Execution**: Inspect per-chip TPU program counter (`pc` and
        `tag:pc`), active HLO instruction, running TPU program fingerprint
        (`DIFFERENT_MODULE` / `FINGERPRINT_MISMATCH` via
        `[[inspect-hlo-dumps]]`), `launch_id` alignment across workers, and any
        inflight host-offloaded XLA CPU instructions/thunks.
    -   **ICI Collectives**: Inspect per-chip ICI link state,
        `Wait timeout on sflag` signatures, and delay wave propagation
        (`[[query-ici-logs]]`, `[[analyze-ici-wave]]`).
    -   **MXLA Collectives**: Inspect DCN `Send`/`Recv` errors (`NETWORK_SEND`,
        `NETWORK_RECEIVE`), inflight `Send` requests and uncompleted launch
        graph actions (`D2H_START`, `H2D_START`, `WAIT_BARRIER`), and recent
        sent/received buffer history (`[[debugging-megascale-hangs]]`).

2.  **Check Automated Analysis & Dual Coordinator/Worker Monitoring**: Verify if
    automated tools (like RapidEye) have classified the hang reason
    (`UNRECOVERABLE_ERROR`, `DIFFERENT_MODULE`, `FINGERPRINT_MISMATCH`,
    `DATA_INPUT_STALL`, `BAD_TPU_CHIP`, or `NETWORKING_ISSUE`). Hang monitoring
    relies on both **coordinator reporting** (aggregated root-cause
    classification) and **per-worker reporting** (tracking how frequently a
    specific host or worker ID triggers or reports a hang for fleet-level
    correlation). Hang analyzers aggregate errors across all workers even when
    the on-node `libtpu` is older.

3.  **Locate Suspect Machines (Workload / GCE / Hardware Layers)**:

    -   **GKE/Cloud**: Check for pod evictions or GCE VM
        preemptions/terminations.
    -   **cluster/Internal**: Inspect the cluster Pods View or cluster capacity
        structures (see Cluster Infrastructure) for broken TPUs.

4.  **Log Inspection (Coordinator First)**:

    -   **Step 4a. Check Coordinator Logs**: Always start by checking logs of
        GKE Worker 0 (typically worker `job0-0` or head pod `pathways-head-0-0`)
        or JAX Process 0. Look for primary crash errors or
        `Megascale detects a hang` / `HANG_DETECTED` messages. Compare
        coordinator digests with worker log traces if the on-node `libtpu`
        predates recent hang aggregation heuristics.
    -   **Step 4b. Parse Coordinator Hang Digest**:
        -   If the digest identifies a clear cause (e.g.,
            `Potential cause: BAD_TPU_CHIP` with `pc != 0`, `DATA_INPUT_STALL`
            with `pc == 0`, or OOM), proceed to remediation.
        -   If the digest reports `Unknown` or
            `cannot determine the root cause`, proceed to worker log sweeps
            (Step 5).
        -   *Warning*: Ignore worker cancellation warnings (e.g.,
            `cancelled due to MXLA error on another device`) at this stage; they
            are downstream cascades.

5.  **TPU State Analysis & Launch Alignment (Worker Sweep)**:

    -   If the coordinator could not determine the root cause, query all worker
        logs for `"Wait timeout on sflag"` and `"Sflag wait timeout on op"`
        messages using [[diagnose-sflag-hang]] or [[analyze-sflag-timeouts]].
        -   **CRITICAL**: You MUST use chronological ascending sorting
            (`--order=ASC`) and an appropriate `--freshness` window when
            querying sflags to ensure you capture the earliest timeouts. Sorting
            descending (default) will return teardown timeouts from downstream
            cascades, causing you to identify the wrong culprit.
        -   *Warning*: Do NOT filter by `severity>=ERROR` when querying for
            `"Wait timeout on sflag"` as JAX logs these collective timeouts as
            `INFO`. Refer to [[gotchas-cloud-logging]] Gotcha 9.
    -   **CRITICAL**: You MUST query and compare both TensorCore (TC) and
        SparseCore (SC) wait timeouts (`tag:pc` sequencer halts). Refer to
        [[tpu-state-analysis-methodology]] for details.
    -   **Isolate Both Minority Groups and Silent/Missing Pods**:
        -   **Minority `sflag` / `tag:pc` Groups (`< 20%` of majority)**: Group
            the timeouts by HLO operation or SparseCore `tag:pc` and flag
            minority outlier groups (`< 20%` of the majority group count).
        -   **Silent / Missing Pods**: Compare the set of pods that emitted
            `Wait timeout on sflag` against the expected `<slice>-<worker>` pod
            topology. A worker that crashed, hung in host code, or stalled
            earlier often fails to emit any `sflag` timeout logs at all (the
            silent culprit).
    -   **Cross-Reference Launch Alignment (`[[analyze-libtpu-timing]]`)**: When
        `sflag` counts appear uniform across all reporting pods, run
        [[analyze-libtpu-timing]] to inspect `Enqueueing program` and
        `last_finished_run_id updated` timestamps per `launch_id` across workers
        to isolate late-enqueue host stragglers (`DATA_INPUT_STALL`), device
        execution stragglers, or workers missing from the final `launch_id`.
    -   Retrieve the GKE node name (`[[get-node-from-pod-logs]]`) for the
        outlier or silent culprit pod.

## Remediation & Mitigations

Restart the workload for transient issues, or isolate/repair the identified
physical host using [[node-isolation-methodology]].

-   **Restart precondition**: the workload owner restarts only after the
    previous attempt has 0 pods left (none `Terminating`, no `FailedKillPod`
    events). A restart while old TPU containers still run can form the new slice
    on TPUs they still hold; follow [[kueue-stuck-termination-recovery]] and
    [[gotchas-kubernetes-gke]] instead.

## Reporting Format (CRITICAL)

When you reach a diagnosis for a hang, you MUST write a JSON summary of your
findings to the file `runs/run_1_output.json` (relative to the active test case
directory). The JSON must follow this exact schema:

```json
{
  "root_cause": "brief explanation",
  "culprit_pod": "pod-name",
  "culprit_host": "host-name-or-na",
  "culprit_physical_machine": "physical-host-name-or-na",
  "mitigation_action": "action-page-or-na"
}
```

*   **culprit_physical_machine**: You MUST resolve this using `telemetry` (e.g.,
    `host-xxxx`). Do not leave it blank or use the GKE node name.
*   **mitigation_action**: Specify the markdown file name (e.g., Node
    Replacement or `report-host-as-faulty.md`) from the wiki that describes the
    mitigation.

## Connections

-   **Actions**:
    -   [[analyze-ici-wave]]
    -   [[analyze-libtpu-timing]]
    -   [[diagnose-sflag-hang]]
    -   [[inspect-hlo-dumps]]
    -   [[query-ici-logs]]
    -   [[run-single-slice-hardware-test]]
-   **Playbooks**:
    -   [[debugging-megascale-hangs]]
    -   [[gotchas-cloud-logging]]
    -   [[gotchas-kubernetes-gke]]
    -   [[kueue-stuck-termination-recovery]]
    -   [[layered-debugging-framework]]
    -   [[node-isolation-methodology]]
    -   [[tpu-state-analysis-methodology]]
-   **Symptoms**:
    -   [[step-timebomb-watchdog]]
-   **Tools**:
    -   [[analyze-libtpu-timing-tool]]
    -   [[cloud-logging]]

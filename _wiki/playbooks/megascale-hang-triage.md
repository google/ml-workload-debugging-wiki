---
title: Megascale Hang Triage
description: "Triage methodology for diagnosing and resolving Megascale multislice hangs across 1P and 3P environments using error aggregator digests and HLO inspection."
type: playbook
internal_only: false
tags: [tpu, training, timeout, hardware_fault, triage, mxla, collective_communication]
created: 2026-09-28
updated: 2026-10-20
---


# Megascale Hang Triage

## Overview

In multi-slice distributed training on [[cloud-tpu]] systems, [[megascale]]
manages Data Center Network (DCN) collective communications. When a collective
communication operation (such as `AllReduce`, `ReduceScatter`, or `AllGather`)
fails to complete within the stale threshold (default 5 minutes, configured via
`--megascale_graph_hang_threshold`), the runtime triggers a hang detection
mechanism.

Because all nodes synchronize at collective communication barriers, a fault on a
single chip, host, or network link stalls the entire job. Megascale acts as the
detection point and aggregates worker execution states to diagnose the root
cause.

This playbook establishes the standard operating procedure for locating,
diagnosing, and remediating Megascale hangs across both public Cloud (3P) and
Google-internal (1P) environments.

--------------------------------------------------------------------------------

## Prerequisites & Workload Flags

To ensure actionable telemetry when a hang occurs, configure workloads with the
following prerequisites:

1.  **JAX Version**: Use **JAX >= 0.6** and ensure JAX distributed service is
    enabled. This version includes runtime logging that reliably identifies
    worker status.
2.  **Program Execution & Error Aggregator Logging**: Set
    `--vmodule=real_program_continuator=1` to enable verbose logging for TPU
    program execution status, and set
    `--vmodule=error_aggregator=2 --alsologtostderr` on the coordinator to emit
    detailed Megascale error aggregator digests to standard error logs.
3.  **HLO Dumps & `launch_id` Correlation**: Launch workloads with the
    `--xla_dump_to` flag enabled:

    ```bash
    rm -rf /tmp/xla_dump/
    export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump/"
    ```

    Capturing HLO dumps allows immediate diffing of compiler artifacts when
    investigating module mismatches, as well as correlating a failing Megascale
    `launch_id=X` from logs to the exact HLO collective operation by searching
    the HLO dumps for `send`/`recv` instructions carrying
    `frontend_attributes={..._xla_megascale_launch_id="X"...}` or
    `channel_id=X`. Refer to [[inspect-hlo-dumps]].

--------------------------------------------------------------------------------

## Locating the Hang

When Megascale detects a stall, worker nodes report their execution states to a
central coordinator. The coordinator aggregates these states and logs the
definitive diagnostic payload:

### 1. Coordinator Log Locations

-   **Pathways**: Check the **`pathways-head`** pod / `resource_manager`
    container logs (e.g., `-pathways-head-0-0-`).
-   **McJAX (Multislice JAX)**: Check the **`worker-job0-0`** pod logs (task 0
    of slice 0, the MXLA Coordinator).

### 2. Search Signatures (Cloud Logging)

In Google Cloud Logging, search for:

```text
textPayload =~ "Megascale detects a hang" OR textPayload =~ "Megascale HANG_DETECTED"
```

--------------------------------------------------------------------------------

## Diagnostic Decision Tree: The 7 Categories

The Megascale error aggregator parses TPU program counters (`tag:pc`), launch
IDs, and HLO module metadata across all workers to categorize the hang into one
of 7 causes:

```
                                 [ Megascale Hang Detected ]
                                              |
                     +------------------------+------------------------+
                     |                                                 |
         [ Digest Cause Identified ]                               [ UNKNOWN ]
                     |                                                 |
     +---------------+---------------+                                 v
     |               |               |                   [ TPU State Analysis ]
     v               v               v                   - Query sflag timeouts
[BAD_TPU_CHIP] [NETWORKING]  [DIFFERENT_MODULE]          - Find unnotified sflag
[BAD_SC_CHIP]  [   ISSUE  ]  [FINGERPRINT_MISMATCH]      - [[tpu-state-analysis-
     |               |       [DATA_INPUT_STALL]             methodology]]
     |               |       [UNRECOVERABLE_ERROR]
     v               v               v
(Check recidivism) (Check links) (Inspect HLO / logs)
```

### 1. Bad TPU Chip (`BAD_TPU_CHIP` or `BAD_SC_CHIP`)

-   **Digest Signature**:

    ```text
    Potential cause: Bad TPU chips (or Bad Sparse Core)
    Potential culprit workers: <worker>:<host>
    ```
-   **Analysis**:

    -   The error aggregator isolated a worker whose TensorCore or SparseCore
        was halted on a unique `tag:pc` state while other workers waited.
    -   **Recidivism Rule**: An isolated occurrence can occasionally be caused
        by an XLA compiler bug or custom Pallas kernel issue. However, if the
        **same host appears more than 3 times** across multiple hang events, the
        TPU hardware on that host is confirmed faulty.
-   **Remediation**:

    -   **3P**: Isolate the GKE node using [[isolate-node-via-rr-label]]:

    ```bash
    kubectl label nodes ${NODE_NAME} cloud.google.com/fault-behavior=SDC
    ```

### 2. Networking Issues (`NETWORKING_ISSUE`)

-   **Digest Signature**:

    ```text
    Potential cause: Networking issue
    Potential culprit workers: <worker_src>:<host_src>, <worker_dst>:<host_dst>
    ```
-   **Analysis**:

    -   Collective transfers between the listed source and destination hosts
        stalled or timed out.
    -   If multi-link localization is available: if a single host appears across
        multiple failing communication pairs, that host's NIC or physical cable
        is the culprit.
-   **Remediation**:

    -   Verify node VPC network health, packet drop counters, and switch health.
    -   In 3P, isolate degraded nodes.

### 3. Different Modules (`DIFFERENT_MODULE`)

-   **Digest Signature**:

    ```text
    Potential cause: Different module
    TPU states:
      Module: jit_loss_and_grad ...
      Module: jit_optimizer_apply ...
    ```
-   **Analysis**:

    -   TPU workers are executing different HLO modules simultaneously.
    -   **Process 0 Divergence**: Check JAX process 0 (`pod-0-0`). Training
        pipelines often execute auxiliary tasks (such as evaluation loops or
        checkpoint saves via Orbax) exclusively on process 0, causing other
        workers to hang at the subsequent step barrier.
    -   Alternatively, unaffected workers may have completed a step and queued
        the next module while a subset remained stuck on the previous one.
-   **Remediation**:

    -   Verify that all worker processes run identical code branches.
    -   Replace raw collective barriers with non-blocking coordination or
        `jax.distributed.wait_at_barrier`.

### 4. Fingerprint Mismatch (`FINGERPRINT_MISMATCH`)

-   **Digest Signature**:

    ```text
    Potential cause: Fingerprint mismatch
    TPU states:
      Module: <module_name> Fingerprint: <hash_1>
      Module: <module_name> Fingerprint: <hash_2>
    ```
-   **Analysis**:

    -   All workers launched the same function name, but the compiled XLA
        executable fingerprints differ across workers.
    -   Indicates non-deterministic JAX tracing, inconsistent environment
        variables, or a compiler optimization bug.
-   **Remediation**:

    -   Retrieve HLO dumps (`--xla_dump_to`) from both sample workers.
    -   Diff the HLO text files using [[inspect-hlo-dumps]] to isolate diverging
        compiler instructions.

### 5. Data Input Stall (`DATA_INPUT_STALL`)

-   **Digest Signature**:

    ```text
    Potential cause: Data input stall
    Potential culprit workers: <worker>:<host>
    ```
-   **Analysis**:

    -   Workers initialized the computation graph but stalled waiting for input
        tensors. The outlier TPU typically exhibits a Program Counter of `PC=0`
        or `no HLO mapping` with `tag=1`.
-   **Remediation**:

    -   Inspect data loader pipeline logs on the culprit hosts.
    -   Verify storage accessibility, GCS 429/503 rate limiting, and disk
        throughput.

### 6. Unrecoverable Error (`UNRECOVERABLE_ERROR`)

-   **Digest Signature**:

    ```text
    Potential cause: Unrecoverable error
    Some workers have halted with an unrecoverable error:
      <worker>:<host> : {error_message}
    ```
-   **Analysis**:

    -   Worker process encountered a fatal host-side runtime failure, such as
        host-to-device (H2D) or device-to-host (D2H) data transfer errors
        (`d2h_transfer_latency`).
-   **Remediation**:

    -   Check the worker's application logs for fatal stack traces or driver
        errors.

### 7. Unknown Error (`UNKNOWN_CAUSE` / `Unknown`)

-   **Digest Signature**:

    ```text
    Potential cause: Unknown
    Megascale detects a hang but cannot determine the root cause.
    ```
-   **Analysis**:

    -   The error aggregator could not identify an outlier worker from the TPU
        program counter states.
-   **Remediation**:

    -   Bypass the digest and initiate **TPU State Analysis**:
    *   Query worker logs for `"Wait timeout on sflag"` messages using
        [[analyze-sflag-timeouts]].
    *   Identify the silent outlier worker that failed to notify the
        synchronization flag.
    *   Refer to [[tpu-state-analysis-methodology]] for detailed instruction tag
        analysis.

--------------------------------------------------------------------------------

## Remediation Procedures & Mitigations

When a Megascale hang is diagnosed:

1.  **Bad TPU Hardware / Host Issues**:

    -   **3P (Cloud GKE)**: Isolate the node via [[isolate-node-via-rr-label]]:
        `kubectl label nodes <NODE_NAME> cloud.google.com/fault-behavior=SDC`.

2.  **Software / Compiler Fingerprint Mismatch**:

    -   Collect and diff HLO dumps (`--xla_dump_to`) using
        [[inspect-hlo-dumps]].

3.  **Data Input Stall**:

    -   Address input pipeline bottlenecks, GCS throttling, or data loader
        deadlocks.

4.  **Unknown Stalls**:

    -   Fall back to sync flag analysis via [[analyze-sflag-timeouts]] and
        [[tpu-state-analysis-methodology]].

--------------------------------------------------------------------------------

## Connections

-   **Symptoms**:
    -   [[megascale-hang]]
    -   [[wait-timeout-on-sflag]]
    -   [[slow-step-time]]
-   **Playbooks**:

    -   [[node-isolation-methodology]]
    -   [[tpu-state-analysis-methodology]]
    -   [[debugging-mindset]]

-   **Workflows**:

    -   [[debugging-megascale-hangs]]
    -   [[tpu-state-analysis]]
    -   [[libtpu-timing-outlier-analysis]]

-   **Actions**:

    -   [[isolate-node-via-rr-label]]
    -   [[inspect-hlo-dumps]]
    -   [[analyze-sflag-timeouts]]

---
title: ICI Straggler Triage
description: Triage methodology for detecting and localizing degraded or abnormally
  slow ICI links using active `ici-sdc-test` testing and XLA delay wave propagation
  analysis.
type: playbook
internal_only: false
tags:
- tpu
- hardware_fault
- timeout
- mxla
- collective_communication
created: 2026-09-26
updated: 2026-10-13
---


# ICI Straggler Triage

## Overview

Inter-Chip Interconnect (ICI) links connect Cloud TPU chips directly within
cubes and across Superpod topologies (including across Optical Circuit Switches,
OCS) to execute low-latency collective operations (e.g., `AllReduce`,
`ReduceScatter`, `AllGather`). When an ICI link or OCS path experiences hardware
degradation, packet loss, or high latency without failing outright, it acts as
an **ICI straggler** (fail-slow link). Because distributed training synchronizes
across all chips at collective barriers, the entire cluster throttles to the
throughput of the slowest link.

Standard hang detection flags (like
`--xla_tpu_debug_sflag_wait_timeout_ms=180000`) operate with long ~3-minute
timeouts designed to avoid false alarms during compilation or transient
scheduling hiccups. Consequently, millisecond-level link stragglers do not
trigger hard crashes; instead, they cause severe step time regressions and MFU
collapses.

This playbook documents the diagnostic methodology for triaging and isolating
ICI stragglers using **`ici-sdc-test` & Delay Wave Propagation Analysis**.

--------------------------------------------------------------------------------

## 1. `ici-sdc-test` & Delay Wave Propagation Triage

### Wave Propagation Mechanics

When a degraded ICI link is exercised via synthetic collective communication
patterns (`ici-sdc-test`):

-   **S0 Cores (`SEND` Opcode Stall)**: The TPU cores directly attached to the
    degraded ICI link are designated **S0 cores**. At iteration $N$, an S0 core
    directly connected to the slow link experiences a wait timeout (`> 5ms`) on
    its `SEND` synchronization flag (`sflag`), proving that the DMA transfer
    failed to leave the chip within the configured timeout window.
-   **Cascade to Neighbors ($S1, S2, \dots$ on `RECV-DONE`)**: As the slow data
    transfer fails to complete in time, neighbor cores that depend on data from
    S0 cannot advance their pipeline stages and stall on `RECV-DONE` sync flags
    at subsequent iterations $N + k$.
-   **Iteration Delta ($\Delta \text{iteration}$)**: In a pipeline with depth
    $D = 13$, the delay wavefront propagates with an iteration delta of $\Delta
    \text{iteration} = 14$. Thus, S1 cores observe timeouts at iteration $N +
    14$, S2 at $N + 28$, and so on.
-   **Root Culprit Localization ($N_{\min}$)**: By aggregating timeout
    iterations across all cores in the slice, the core exhibiting the **minimum
    iteration number** ($N_{\min}$) waiting on a `SEND` opcode identifies the
    root **S0 core** directly attached to the physical link or OCS failure.

### Required XLA / `LIBTPU_INIT_ARGS` Flags

Configure the workload with the following 11 XLA flags to run `ici-sdc-test` at
program start with a `5ms` `sflag` wait timeout and log recorder enabled:

```text
xla_tpu_enable_sdc_checker = true
xla_tpu_sdc_check_repeat_count = 0
xla_tpu_sdc_check_halt_on_detection = false

xla_tpu_ici_sdc_test_max_distance = 1
xla_tpu_ici_sdc_test_pipeline_depth = 13
xla_tpu_ici_sdc_test_buffer_size_chunks = 256
xla_tpu_ici_sdc_test_packet_size_chunks = 256
xla_tpu_ici_sdc_test_iterations = 5000
xla_tpu_ici_sdc_test_run_on_program_start = true

xla_tpu_enable_log_recorder = true
xla_tpu_ici_sdc_test_sflag_wait_timeout_ms = 5
```

### Log Signatures & Operational Caveats (#13, #14`)

-   **Log Signatures**:
    -   `XLA_SDC_CHECKER`: Captures `ici-sdc-test` summary events and `> 5ms`
        link wait timeouts.
    -   `ICI transfer wait`: Captures per-core logger timeout details including
        iteration number, `sflag` ID, chip coordinates, and opcode (`SEND` vs.
        `RECV-DONE`).
-   **Operational Caveats**:
    -   **Step-Time & Compilation Overhead**: Running 5,000 iterations on every
        step start adds step-time and compilation overhead (e.g., increasing
        step time from ~3.46s to ~7.38s). Advise operators to either enable the
        flags briefly during a diagnostic run to capture `XLA_SDC_CHECKER` /
        `ICI transfer wait` logs and then remove them, or trigger `ici-sdc-test`
        on a periodic cadence (e.g., 1 step out of 100).
    -   **Zero DCN/MXLA Interference**: `ici-sdc-test` is unaffected by MXLA DCN
        traffic because no DCN communication occurs while `ici-sdc-test`
        executes.
    -   **Sub-5ms Fail-Slow Tuning**: If a degraded link stalls for `< 5ms`,
        lower `xla_tpu_ici_sdc_test_sflag_wait_timeout_ms` or increase
        `xla_tpu_ici_sdc_test_buffer_size_chunks` /
        `xla_tpu_ici_sdc_test_packet_size_chunks`.

### Diagnostic Execution Workflow

1.  **Trigger Diagnostic Run**: Dispatch `ici-sdc-test` across the suspected
    slice or nodepool using the 11 XLA flags above.
2.  **Query ICI Diagnostic Logs**: Execute [[query-ici-logs]] to pull
    `XLA_SDC_CHECKER`, `ICI transfer wait`, and `Wait timeout on sflag`
    telemetry.
3.  **Analyze Delay Wave Propagation**: Execute [[analyze-ici-wave]] to extract
    the iteration number, core ID, chip coordinates, and opcode (`SEND` vs.
    `RECV-DONE`) of each timeout.
4.  **Isolate S0 & Map Host**: Identify the chip(s) with minimum iteration
    $N_{\min}$ stalling on `SEND` sync flags. Map the culprit pod/chip to its
    GKE node via [[get-node-from-pod-logs]].
5.  **Decommission Bad Node**: Quarantine the host via
    `[[isolate-node-via-rr-label]]` (see `[[node-isolation-methodology]]`) or
    trigger hardware repair via [[repair-in-place]].

--------------------------------------------------------------------------------

## 2. Remediation Actions & Telemetry Surfacing Channels

| Environment / Segment | Telemetry Callback & Surfacing Mechanism | Log / Artifact Access |
| :-------------------- | :------------------- | :-------------------- |

| **Cloud Customers** | **Debug callbacks** | GKE Cloud Logging | : :
(`debugger callbacks`, : ([[query-ici-logs]]), : : : `libtpu.sdk.debug`) : MXLA
telemetry, offline : : : integrated into the : callback files : : : **MXLA
coordinator : : : : runtime** : : | **Custom XLA | Documented XLA tools and |
Customer-managed log | : Collectors** : callback integration : collectors and
offline : : : guidelines for building : wave analyzers : : : equivalent
straggler : : : : callback pipelines : :

--------------------------------------------------------------------------------

## Connections

-   **Workflows**:
    -   [[ici-straggler-triage]]
    -   [[tpu-state-analysis]]
    -   [[mxla-hardware-triage]]
-   **Actions**:
    -   [[query-ici-logs]]
    -   [[analyze-ici-wave]]
    -   [[analyze-sflag-timeouts]]
    -   [[diagnose-sflag-hang]]
-   **Playbooks**:
    -   [[multi-chip-hang-investigation]]
    -   [[hardware-failure-modes]]
    -   [[node-isolation-methodology]]
-   **Symptoms**:
    -   [[wait-timeout-on-sflag]]
    -   [[slow-step-time]]
    -   [[megascale-hang]]
-   **Concepts**:
    -   [[ici-errors]]
    -   [[multi-chip-hangs]]
    -   [[megascale]]

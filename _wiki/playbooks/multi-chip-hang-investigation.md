---
title: Multiprong Approach to Multi-chip Hangs
type: playbook
internal_only: false
tags: [tpu, pathways, hangs, methodology]
created: 2026-05-13
updated: 2026-10-15
---

## Overview

Prongs for finding the culprit chip or machine when one stalled chip hangs a job
of thousands of chips.

## Investigation Prongs

1.  **Pattern detection:** keep a failure table (timestamps, restart attempts,
    errors, instance names, node pools) and intersect the machines in bad runs
    versus good runs.
2.  **TPU state and timeout logs:** find which TensorCore/SparseCore is uniquely
    stuck. Enable:

    ```bash
    --xla_tpu_enable_log_recorder=true
    --xla_tpu_enable_sc_log_recorder=true
    --xla_tpu_debug_sc_sflag_wait_timeout_ms=150000
    --xla_tpu_debug_sflag_wait_timeout_ms=150000
    --xla_tpu_debug_sflag_wait_shalt_on_detection=false
    --xla_tpu_enable_progress_tracker=16
    ```

    -   TensorCore: search `Wait timeout on sflag`; note the last completed HLO
        and the timing-out step; group by "latest value seen" to find the stuck
        subset.
    -   SparseCore: use physical location, chip ID (e.g. SC1) and program count
        (`tag:pc`); group to find unique programs stuck on specific chips.
    -   Map physical locations and ICI coordinates to GKE instances via
        application logs.
3.  **Libtpu metrics / SDK:** look for per-worker outliers in `tensorcore_util`,
    `core_throttle_signal`, `duty_cycle`, `hbm_util`, `buffer_transfer_latency`,
    `collective_e2e_latency`, `ici_slow_signal`, `hlo_exec_timing`,
    `hlo_queue_size`.
4.  **Xprof across profiles:** follow barrier cores and long collectives to the
    slow hosts (e.g. a grid view of worker completion).
5.  **Log analysis:** find culprit workers in application logs, then check host
    logs for machine errors.

## Mitigation: Bisection

Rerun the failing workload on the suspect GKE instances and split them into
subgroups to find the smallest reproducing set:

-   GKE [[superslicing-subslicing]]: run on node pools of different sizes
    without reprovisioning.
-   HLO-based (not supported in cloud yet): build an HLO module with controlled
    replica groups to run unique work per device.
-   Pathways-based: run JAX programs on custom device meshes (e.g. subslices).

## Connections

-   **Playbooks**:
    -   [[multi-workload-triage]]
    -   [[tpu-state-analysis-methodology]]
-   **Symptoms**:
    -   [[wait-timeout-on-sflag]]
-   **Concepts**:
    -   [[multi-chip-hangs]]
    -   [[superslicing-subslicing]]
-   **Entities**:
    -   [[libtpu]]
    -   [[tensorcore]]

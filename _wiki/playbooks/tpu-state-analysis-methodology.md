---
title: TPU State Analysis Methodology
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-06-11
updated: 2026-09-15
---

## Definition

**TPU State Analysis Methodology** describes how to identify root culprit
machines causing multi-chip hangs by analyzing TPU log registers, TensorCore
states, and SparseCore instruction tags when execution flags time out.

## Diagnostic XLA Flags

To capture precise state on hang timeouts:

-   `--xla_tpu_enable_log_recorder=true`
-   `--xla_tpu_enable_sc_log_recorder=true`
-   `--xla_tpu_debug_sc_sflag_wait_timeout_ms=150000`
-   `--xla_tpu_debug_sflag_wait_timeout_ms=150000`
-   `--xla_tpu_debug_sflag_wait_shalt_on_detection=false`
-   `--xla_tpu_enable_progress_tracker=16`

## Wait Timeout on sflag Analysis

-   **Symptom Log**: `[chip_TC0]: Wait timeout on sflag
    {[#allocation33477]} 'sparse-core-wait' ... hlo: all-reduce.1418...`
-   **Verify Both TC and SC**:
    -   **TensorCore (TC)**: Focuses on compute and synchronization. Stalls here
        often look like waiting for SparseCore (e.g., `sparse-core-wait`).
    -   **SparseCore (SC)**: Focuses on embedding and sparse data transfer. Look
        for `sc_tpu` logs and check `DmaWaitTag`.
    -   **CRITICAL**: You MUST query and compare both TC and SC logs to confirm
        the source of the stall. If TC logs show a wait timeout on
        `sparse-core-wait`, you MUST verify the corresponding SC logs on the
        suspect host to identify the specific SparseCore stall reason (e.g.
        `sync_wait` or `dma_wait`).
    -   **CRITICAL**: Always query state logs with chronological ascending
        sorting (`--order=ASC`). If you use descending sorting (default), the
        query limit will cut off the earliest logs, and you will only see late
        timeouts from the teardown phase, leading to diagnosing the wrong
        culprit.
-   **Grouping**: Group stuck cores by their last completed HLO and current
    value. The minority group stuck on chronologically earlier operations (e.g.
    `all-reduce.X` vs. `reduce-scatter.Y`) are the culprits.
-   **SparseCore sequencer halts**: Look for
    `Accelerator device halted prematurely` or
    `Node halted unexpectedly at tag:pc`. Grouping by the program counter
    (`tag:pc`) helps locate the primary hardware or instruction failure point.

## Mapping Physical IDs

Physical coordinates or hardware labels (e.g., `tpu740`) logged during halts
must be mapped back to GKE VM instances to trigger node node replacement or
repair behaviors.

## Connections

-   **Workflows**:
    -   [[tpu-state-analysis]]
-   **Playbooks**:
    -   [[hang-diagnosis-methodology]]
    -   [[node-isolation-methodology]]
-   **Concepts**:
    -   [[multi-chip-hangs]]
-   **Entities**:
    -   [[sparsecore]]
-   **Tools**:
    -   [[cloud-logging]]

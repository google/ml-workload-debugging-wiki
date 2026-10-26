---
title: SparseCore
description: The TPU on-chip core handling embeddings, sparse data transfer, and offloaded operations, featuring mismatch detection and bundle profiling tools.
type: entity
internal_only: false
tags: [tpu, pathways, hardware_fault, hangs, analysis, triage]
created: 2026-09-15
updated: 2026-10-15
---


# SparseCore

## Overview

The **SparseCore (SC)** is the dataflow processor on a TPU chip responsible for
accelerating sparse operations, large embedding lookups (Large Embedding Models
/ LEM), and sparse data transfer (DMA), working alongside the dense-compute
[[tensorcore]]. It matters disproportionately when debugging: a single stalled
SparseCore blocks its chip, and because collectives are synchronous, the whole
job hangs behind it.

Additionally, the XLA compiler can offload compute operations (`sort`, `gather`,
`scatter`, `copy`) and collective operations (`all-gather`, `all-reduce`,
`reduce-scatter`) to SparseCore to run in parallel with TensorCore.

## Technical Specifications

-   **SparseCores per Chip by Generation**:

    -   **TPU7x (Ironwood)**: **4 SparseCores per chip** (2 SparseCores per
        chiplet alongside 1 TensorCore and 96 GB HBM per chiplet).

    -   **TPU v6e (Trillium)**: **2 SparseCores per chip**.

    -   **TPU v5p**: **4 SparseCores per chip**.

    -   **TPU v5e**: **0 SparseCores** (no SparseCore hardware).

-   **Log Identification**: Multiple SparseCores per chip, identified in logs as
    `SC0`, `SC1`, ... alongside a physical location, e.g.
    `5x10x9_SC1: [Physical location: tpu740:pe0:3]`.

-   **Execution State & Compiler Flags**: Execution state is exposed through the
    sequencer program counter (`tag:pc SparseCoreSequencer:0:0x407`) and the DMA
    wait tag (`DmaWaitTag`). XLA controls SparseCore offloading via TPU XLA
    flags, and SparseCore milestones are logged when verbose TPU logging is
    enabled (`real_program_continuator=1` via [[tpu-logging-flags]]).

-   **Vector Processing Unit (VPU)**: Sublanes matching vector shapes (typically
    unrolled to `[8]` or `[8 * n]`), supporting custom kernels via Pallas and
    Mosaic IR.

## Diagnostic and Profiling Tools Integration

### SparseCore Mismatch Detector

When ops are offloaded to SparseCore, the **SparseCore Mismatch Detector**
creates an equivalent TensorCore shadow operation executing the same operands
and compares output tensors via `all_close`. If outputs diverge, the system logs
an error (`Found Mismatch in offloaded op <op_name>:`) or halts execution.

-   Activated via `--xla_tpu_enable_sparse_core_mismatch_detector=true` and
    `--xla_tpu_enable_log_recorder=true`.

-   Detailed in [[sparsecore-debugging-triage]].

### Assertion Level Control

SparseCore hardware and driver invariants are checked at runtime using
`--xla_sc_assert_level=[prod|san|san-lite|no]`, controlling safety assertions
across memory bounds, DMAs, continuations, streams, vector masks, and radix
sort.

## Common Failure Modes

-   **Sflag wait timeout** on `sc_tpu.dma_wait` or `sc_tpu.sync_wait`: the
    SparseCore never completed its operation, so the waiting [[tensorcore]]
    reports `sparse-core-wait`. See [[wait-timeout-on-sflag]].

-   **Sequencer halt**: `Accelerator device halted prematurely` /
    `Node halted unexpectedly at tag:pc`. Group by `tag:pc` to find the unique
    stuck program rather than the thousands of cores waiting on it.

-   **Offload Mismatch Divergence**: Numerical or structural divergence during
    op offloading caused by XLA compiler bugs. Mitigated immediately by
    disabling offloading flags
    (`--xla_tpu_enable_offloading_sort_to_sparsecore=false`, etc.) to run the op
    on TensorCore.

-   **Hardware fault confirmed by repair**: in a 2048-chip Pathways hang, 8190
    of 8192 SparseCore logs were stuck on the same `all-gather`, and the two
    outliers on one chip localised a genuinely faulty machine that was
    subsequently sent for repair. A 16-slice startup hang isolated the same way,
    via PC counters after scaling down to single slices.

## Connections

-   **Workflows**:
    -   [[sdc-and-numerics-triage]]
    -   [[tpu-state-analysis]]
-   **Playbooks**:
    -   [[sparsecore-debugging-triage]]
    -   [[pallas-mosaic-kernel-debugging]]
    -   [[tpu-state-analysis-methodology]]
    -   [[data-non-determinism-triage]]
    -   [[on-device-debugging-triage]]
-   **Actions**:
    -   [[analyze-sparsecore-offload]]
    -   [[analyze-sflag-timeouts]]
    -   [[inspect-hlo-dumps]]
    -   [[query-workload-logs]]
-   **Symptoms**:
    -   [[mxla-hardware-fault]]
    -   [[wait-timeout-on-sflag]]
    -   [[sdc-checksum-mismatch]]
-   **Concepts**:
    -   [[multi-chip-hangs]]
    -   [[tpu-logging-flags]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[tensorcore]]
    -   [[xla]]

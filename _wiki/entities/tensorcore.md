---
title: TensorCore
description: The TPU on-chip dense compute and synchronization core, and the unit
  whose logs carry program lifecycle timing.
type: entity
internal_only: false
tags: [tpu, pathways, training, analysis]
created: 2026-09-15
updated: 2026-10-15
---

# TensorCore

## Overview

The **TensorCore (TC)** is the dense compute and synchronization unit of a TPU
chip. Each TensorCore comprises one or more Matrix Multiply Units (MXUs), a
Vector Unit (activations, softmax, reductions), and a Scalar Unit (control flow,
addressing, and `sflag` synchronization). It executes the compiled [[xla]] HLO
program and waits on synchronization flags (`sflag`) for peers, making its logs
the primary timeline for "what was this chip doing when progress stopped".

## Technical Specifications

-   **TensorCores & MXUs per Chip Across Generations**:

    -   **TPU7x (Ironwood)**: **2 TensorCores per chip** in a **dual-chiplet**
        architecture (1 TensorCore, 2 SparseCores, and 96 GB dedicated HBM per
        chiplet, connected by a die-to-die interface 6x faster than 1D ICI).
        Uses `256x256` MXU systolic arrays. Exposed to JAX/PyTorch as two
        separate devices per chip (`TC0`, `TC1`), so a 4-chip host worker emits
        8 enqueue and 8 completion lines per program execution cycle.

    -   **TPU v6e (Trillium)**: **1 TensorCore per chip**, containing **2 MXUs
        of `256x256`** multiply-accumulators and 32 GB HBM.

    -   **TPU v5p & TPU v4**: **2 TensorCores per chip** (often operated in
        fused **MegaCore** mode sharing a unified 95 GiB or 32 GiB HBM space),
        with **4 MXUs of `128x128`** multiply-accumulators per TensorCore.

    -   **TPU v5e**: **1 TensorCore per chip** with **4 MXUs of `128x128`** and
        16 GB HBM.

-   **Precision & Tiling**: MXU systolic arrays multiply `bfloat16` (or
    `fp8`/`int8`) inputs and accumulate in `FP32`. To avoid zero-padding
    underutilization or HBM OOMs, batch or feature dimensions should align to
    the MXU tile size (`128` or `256`) and multiples of `8` for the memory
    subsystem.

-   **Log Identification**: Logs identify a core by mesh coordinate plus core
    index, e.g. `3x2x2_TC1` or `5x15x3_TC0` — the coordinate is the chip
    position in the ICI topology, the suffix the core.

-   **Lifecycle Events**: Per-step lifecycle events
    (`Enqueueing program ... (launch_id=...)` and
    `last_finished_run_id updated to ...`) are emitted by [[libtpu]]; see
    [[libtpu-program-lifecycle]] for the field semantics.

## Common Failure Modes

-   **`Wait timeout on sflag`**: the TensorCore timed out waiting for another
    core or chip. Group the timeouts by the `hlo:` field — the minority stuck on
    an *earlier* collective are the stragglers, not the majority. See
    [[wait-timeout-on-sflag]].

-   **Waiting on `sparse-core-wait`**: the stall is downstream of the
    [[sparsecore]] on the same host; TC logs alone will misattribute it, so both
    must be queried.

-   **Intra-host TensorCore skew**: `TC0` and `TC1` on the same chip diverging
    in dispatch gap points at host-side thread contention rather than a device
    fault — see [[libtpu-enqueue-timing-analysis]].

## Connections

-   **Actions**:
    -   [[analyze-libtpu-timing]]
-   **Playbooks**:
    -   [[tpu-state-analysis-methodology]]
-   **Symptoms**:
    -   [[wait-timeout-on-sflag]]
-   **Concepts**:
    -   [[libtpu-program-lifecycle]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[libtpu]]
    -   [[sparsecore]]
    -   [[xla]]

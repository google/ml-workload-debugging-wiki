---
title: XLA
description: Accelerated Linear Algebra compiler infrastructure.
type: entity
internal_only: false
tags: [training, pathways, tpu, gpu, hardware_fault]
created: 2026-07-30
updated: 2026-10-15
---

## Overview

XLA (Accelerated Linear Algebra) is a just-in-time compiler that takes the High
Level Optimizer (HLO) graph emitted by [[jax]], PyTorch/XLA, or TensorFlow and
compiles linear algebra, loss, and gradient computations into target machine
code for CPU, GPU, and [[cloud-tpu]].

## Technical Specifications: TPU Tiling, Shapes, and Padding

-   **MXU Systolic Array Tiling**: XLA tiles matrix multiplications into blocks
    matching the [[tensorcore]] MXU systolic array (`128x128` on TPU `v4`,
    `v5e`, and `v5p`; `256x256` on `v6e` and `TPU7x`) and aligns memory
    subsystem accesses to multiples of `8`. Unaligned tensor layouts force
    memory-bound `reshape` operations before tiling.

-   **Zero-Padding & OOM Risk**: When batch or feature dimensions cannot fill an
    MXU tile, XLA zero-pads tensors to a multiple of `128` (or `256`). Excessive
    padding underutilizes the TPU core and inflates on-chip memory / HBM usage,
    which can trigger `RESOURCE_EXHAUSTED` OOM errors. Inspect padding overhead
    using the `op_profile` tool in XProf.

-   **Static Shapes**: XLA compiles the graph for the first batch's tensor
    shapes; dynamic shapes that vary across steps trigger costly recompilations.

## Compilation in MCJAX vs Pathways & Failure Modes

-   **HLO Module Fingerprints**: Each compiled HLO module has a fingerprint that
    [[libtpu]] logs on every program launch (`Enqueueing program ...`).
    Differing fingerprints across workers in [[mcjax]] indicate inconsistent
    compilation (`[[inspect-hlo-dumps]]`).

-   **Pathways Proxy Compilation**: In [[pathways]], XLA compilation runs inside
    the `pathways-proxy` container rather than the user container, and compiled
    executables are cached in `--gcs_scratch_location`. XLA compiler flags (such
    as `--xla_dump_to`) must be passed to `pathways-proxy` args rather than
    `LIBTPU_INIT_ARGS` on the user container
    (`[[pathways-debugging-principles]]`).

## Connections

-   **Actions**:
    -   [[inspect-hlo-dumps]]
-   **Playbooks**:
    -   [[pathways-debugging-principles]]
-   **Concepts**:
    -   [[pathways]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   GPU
    -   [[jax]]
    -   [[libtpu]]
    -   [[tensorcore]]

---
title: HBM Out Of Memory (HBM OOM / Error Code E1000)
description: Accelerator High Bandwidth Memory (HBM) capacity exhaustion during XLA compilation or runtime execution on TPUs, emitting RESOURCE_EXHAUSTED errors.
type: symptom
internal_only: false
tags: [tpu, out_of_memory, training]
created: 2026-09-28
updated: 2026-09-28
---


# HBM Out Of Memory (HBM OOM / Error Code E1000)

## Overview

High Bandwidth Memory (HBM) Out Of Memory occurs when the aggregate memory
required for TPU execution exceeds the physical HBM capacity of the TPU device
(e.g., 16 GiB on TPU v4/v5e, 32 GiB on TPU v6e, or 96 GiB on TPU v5p).

HBM capacity is managed strictly by the XLA compiler and TPU runtime for:

-   **Program inputs and outputs**: Model weights, training batches, and
    optimizer states.
-   **TPU temporaries**: Dynamic memory required for intermediate activations,
    gradients, and fusion scratch buffers.
-   **Compiled binaries**: Machine code for both TensorCore (TC) and SparseCore
    (SC).
-   **Constants and system overhead**: Embedded HLO IR constants and reserved
    runtime space.

HBM exhaustion can occur either **statically at compile time** (Error code
`E1000`) when XLA verifies that allocations exceed device capacity, or
**dynamically at runtime** during tensor allocation.

## Log Signatures

### 1. Runtime HBM Allocation Failure

Emitted by the XLA runtime when attempting to allocate a device buffer that
exceeds remaining HBM:

```
XlaRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=107374182400) would exceed memory (size=17179869184) :: #allocation7 [shape = 'u8[327680,327680]{1,0:T(8,128)(4,1)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of xor_convert_fusion@{}'] :: <no-hlo-instruction>
```

### 2. Compile-Time HBM OOM with Component Breakdown (TC vs SC)

Emitted when the compiler detects that the combined TensorCore and SparseCore
static requirements exceed total HBM:

```
RESOURCE_EXHAUSTED: TPU TensorCore Hbm usage: 34.82G, SparseCore Hbm usage 174.10G, exceeding available bytes: 95.74G
```

### 3. Compile-Time HBM OOM (Permanent Compiler Error)

Emitted when aggregate allocations exceed device capacity during the
scheduling/allocation pass:

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G.
```

## Verification Steps & Distinctions

### 1. Confirm the Memory Space

Examine the error traceback in worker application logs:

-   Look for `space=hbm` or `memory space hbm`.
-   If the error contains `space=hbm` or mentions `TensorCore Hbm usage` /
    `SparseCore Hbm usage`, the issue is definitively accelerator HBM
    exhaustion.

### 2. Distinguish from Host RAM Coordinator OOM

It is critical to distinguish HBM OOM from TPU Coordinator Host RAM OOM
([[tpu-coordinator-oom]]):

| Dimension | HBM OOM (`symptoms/hbm-oom.md`) | TPU Coordinator OOM (`symptoms/tpu-coordinator-oom.md`) |
| :---------- | :--------------------------- | :------------------------------------- |
| **Physical Memory** | Accelerator Device HBM (16 GiB – 96 GiB per TPU chip) | Host VM System RAM (typically 128 GiB – 512+ GiB per host) |
| **Crash Mechanism** | Handled exception: `jax.errors.XlaRuntimeError` or XLA compilation failure | Unhandled kernel kill: Linux OOM killer / Kubelet eviction |
| **Exit Code** | Process exits with standard failure code (typically `Exit Code 1`) | Process terminated abruptly with `Exit Code 137` (`SIGKILL` / `OOMKilled`) |
| **Log Signature** | `RESOURCE_EXHAUSTED: Allocation ... space=hbm` | `Container killed by OOM`, `Killed process ... (python3)` |
| **Root Cause** | Tensor shapes, batch size, activation memory, missing sharding, remat settings | Host memory leak (e.g. `--megascale_grpc_enable_xor_tracer`), data loaders, CPU queue buildup |

### 3. Profile Peak Memory with XProf

To identify which operations and tensors are consuming HBM:

-   Trace the application with `jax.profiler.trace("/tmp/xprof")`.
-   Open **Memory Viewer** in [[xprof]]:
    `xprof --logdir=/tmp/xprof/ --port=6006`.
-   Inspect the *HLO Ops at Peak Memory Allocation* to identify whether the peak
    is driven by parameters, un-sharded tensors, or intermediate temporaries.

## Playbooks & Root Causes

Follow the step-by-step triage workflow in [[hbm-oom-triage]]. Root causes
generally fall into three categories:

1.  **TensorCore vs SparseCore Imbalance**: Embedding tables consume excessive
    HBM stack or heap on SparseCore. Mitigated via
    `--xla_sc_num_serialized_tables_to_optimize_hbm`, reduced
    `maximum_parallel_iterations`, 32B (8-float) embedding table alignment, and
    mod-sharding embedding tables across chips.
2.  **Unexpectedly Large Allocations**: Leftover `jax.debug.print` calls forcing
    host transfers and breaking fusion, or missing sharding annotations causing
    XLA to default to full tensor replication on each chip.
3.  **Aggregate Allocation Exceeded**: Model parameters and activation
    footprints exceed physical capacity. Mitigated via micro-batching, gradient
    checkpointing (`jax.checkpoint`), mixed precision (`bfloat16`), dimension
    alignment to multiples of 128, and compiler flags such as
    `` (disables packing
    two `bf16` values into an `s32` register, saving ~1 GB of HBM on a 70B LLM
    at a ~1–2% step time cost).

## Connections

-   **Workflows**:
    -   [[hbm-oom-triage]]
    -   [[general-ml-debugging]]
-   **Actions**:
    -   [[analyze-hbm-oom]]
    -   [[query-workload-logs]]
    -   [[inspect-hlo-dumps]]
-   **Playbooks**:
    -   [[hbm-oom-triage]]
    -   [[layered-debugging-framework]]
-   **Symptoms**:
    -   [[tpu-coordinator-oom]]
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[ecc-errors]]
-   **Entities**:

    -   [[cloud-tpu]]
    -   [[tensorcore]]
    -   [[sparsecore]]
    -   [[jax]]
    -   [[xla]]

    -   [[xprof]]

    -   [[cloud-logging]]

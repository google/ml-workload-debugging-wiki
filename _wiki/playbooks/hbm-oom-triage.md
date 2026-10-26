---
title: HBM Out Of Memory (HBM OOM) Triage Playbook
description: Playbook for investigating and resolving TPU High Bandwidth Memory (HBM) exhaustion using XProf Memory Viewer, buffer analysis, and optimization scenarios.
type: playbook
internal_only: false
tags: [tpu, out_of_memory, training]
created: 2026-09-28
updated: 2026-09-28
---


# HBM Out Of Memory (HBM OOM) Triage Playbook

## Overview

High Bandwidth Memory (HBM) Out Of Memory (OOM) errors occur when a workload
requires more accelerator device memory than physically available on the TPU
chip. On Cloud TPUs, HBM capacity ranges from 16 GiB (v4, v5e) to 32 GiB (v6e)
and 96 GiB (v5p).

When an HBM OOM occurs, the system emits `RESOURCE_EXHAUSTED` with error code
`E1000` (Compile Time HBM OOM) or runtime allocation errors. This playbook
provides a systematic methodology using the **XProf Memory Viewer tool** to
visualize memory allocations, identify peak usage contributors, and apply
targeted mitigations.

### Capturing an XProf Profile

To generate an XProf profile for an OOMing program:

```python
import jax
from jax import random
import jax.numpy as jnp

# Trace using the exception-safe context manager
@jax.profiler.trace("/tmp/xprof")
@jax.jit
def train_step(state, batch):
  # Computation that approaches or hits HBM capacity
  ...
```

Start the XProf server locally:

```bash
pip install xprof
xprof --logdir=/tmp/xprof/ --port=6006
```

Open `http://localhost:6006`, select **Memory Viewer** from the Tools menu, and
select **HBM** in the Memory Types dropdown.

## Diagnostic Steps

### 1. Classify the Failure from Logs

Inspect the error logs to categorize the HBM OOM into one of three triage
scenarios:

-   **Scenario 1: Explicit TC vs SC breakdown**: The log reports `TPU TensorCore
    Hbm usage: X, SparseCore Hbm usage Y, exceeding available bytes: Z`. The
    combined static usage exceeds the HBM limit.
-   **Scenario 2: Single unexpectedly large allocation**: The log reports
    `Ran out of memory in memory space hbm` and lists one or more allocations
    individually consuming $>50\%$ of physical HBM.
-   **Scenario 3: Aggregate allocations exceeded**: The log reports
    `Ran out of memory in memory space hbm` but no single allocation is
    anomalously large. The sum of normal program buffers exceeds capacity.

--------------------------------------------------------------------------------

### 2. Analyze Peak Memory Allocation in XProf

In the XProf Memory Viewer, navigate to the **HLO Ops at Peak Memory
Allocation** section. Examine the three buffer charts at the peak memory usage
point:

1.  **Program Inputs and Outputs**: Training batches, model weights, optimizer
    states.
2.  **TensorCore and SparseCore Temporaries**: Dynamic memory required for
    intermediate activations, gradients, and fusion scratch spaces.

Hover over the buffer charts to inspect:

-   **Shape and Data Type**: Check whether tensors are in `float32` rather than
    `bfloat16`.
-   **Allocation Type**: Identifies whether the buffer is an input, temporary,
    or output.
-   **Padding Information**: Identifies memory inflated due to dimension
    alignment.

--------------------------------------------------------------------------------

### 3. Scenario-Specific Diagnosis

#### Scenario 1. Balance TensorCore and SparseCore Usage

If the error breaks down usage between TC and SC:

-   **High SparseCore Usage**:
    -   *Stack usage*: HBM stack scales with `feature_width`,
        `max_unique_nz_per_row`, and `logical_replica_count`. Tune
        `--xla_sc_num_serialized_tables_to_optimize_hbm` to serialize table
        processing at the cost of parallelism.
    -   *Padding overhead*: SparseCore aligns embedding tables to 32 bytes (8
        floats). Tables with small feature widths (< 8 floats) suffer heavy
        padding waste.
    -   *Heap usage*: High values of `maximum_parallel_iterations` prefetch
        excess input data into HBM heap. Lower this value to free memory.
    -   *Verify sharding*: Ensure embedding tables are mod-sharded across all
        chips.
-   **High TensorCore Usage**:
    -   Proceed to Scenario 2 and Scenario 3.
-   **Balanced**:
    -   If the sum of TC and SC usage exceeds capacity, lower usage across both
        components.

#### Scenario 2. Diagnose Unexpectedly Large Allocations

If individual allocations exceed $50\%$ of HBM, the root cause is almost always
software configuration rather than hardware limits:

-   **Debug Artifacts**: Look for leftover `jax.debug.print()` calls. In
    large-scale training, `jax.debug.print` forces XLA to materialize full
    un-fused tensors in HBM for CPU transfer, spiking peak usage. Remove all
    debugging print calls.
-   **Mesh Shape & Missing Sharding**: Check if tensors are replicated across
    all chips. If sharding annotations are omitted or invalid, XLA defaults to
    **replication**, attempting to fit the entire global tensor onto every
    individual chip.

#### Scenario 3. Diagnose Aggregate Allocations

When peak usage is accumulated across hundreds of normal-sized buffers:

-   Identify if the peak occurs during the forward pass (batch size /
    activations) or backward pass (gradient accumulation / activation
    checkpointing).
-   Check tensor dimension alignment against TPU hardware tiles (multiples of
    128 for minor-most dimension, 8 for second-minor).

## Mitigations & Resolutions

### 1. Configuration Adjustments

-   **Micro-Batching**: Split each batch into $n$ micro-batches. Process forward
    and backward passes sequentially per micro-batch, then accumulate gradients
    before updating weights. For original batch size $M$, intermediate
    activation memory drops to $M/n$. *Trade-offs*: Increases step time due to
    repeated passes; monitor convergence if batch statistics change.
-   **Buffer Donation (`donate_argnums`)**: When passing input arrays that are
    overwritten by outputs of identical shape and dtype (e.g. optimizer states
    or model parameters in an update step), designate them via
    `jax.jit(fn, donate_argnums=(...))`. This reuses input buffers in place.
-   **Mixed Precision (`bfloat16`)**: Cast model weights, activations, and
    optimizer states from `float32` to `bfloat16` where numerically safe. This
    immediately halves buffer memory footprints.

### 2. Architecture and Sharding Optimizations

-   **Advanced Sharding**: Transition from 1D data parallelism to
    multi-dimensional tensor or pipeline parallelism. Specify explicit JAX
    sharding constraints (`jax.lax.with_sharding_constraint`) on large
    intermediates.
-   **Scale Topology**: Increase the total chip count to shard large weight and
    activation matrices across a wider mesh.
-   **Host Offloading**: Offload large optimizer states or activations to host
    CPU DRAM using JAX memory spaces. *Caution*: High host-to-device PCIe
    transfer latency can severely degrade step time.

### 3. Tensor Padding and Dimension Alignment

XLA TPU layout engines pad array dimensions to multiples of 128 (minor-most
dimension) and 8 (second-minor dimension):

-   **Audit Shapes**: In Memory Viewer, hover over large buffers to check
    padding overhead. For example, shape `(129, 1024)` is padded to
    `(256, 1024)`, wasting nearly 50% of allocated HBM.
-   **Align Dimensions**: Adjust batch sizes, hidden dimensions, and projection
    matrices to multiples of 128.

### 4. XLA Rematerialization & Compiler Flags

If the model is slightly above HBM capacity, configure gradient checkpointing or
tune XLA rematerialization flags to trade compute cycles for memory:

-   **Manual Rematerialization**: Apply `jax.checkpoint` (`jax.remat`) around
    transformer layers to recompute activations during the backward pass instead
    of holding them in HBM.
-   **XLA Remat & Memory Compiler Flags**: Pass the following flags via the
    `XLA_FLAGS` environment variable:

| Flag                                                     | Description       | Trade-Off    |
| :------------------------------------------------------- | :---------------- | :----------- |
| `` | Disables packing two `bf16` values into an `s32` register, which can otherwise increase peak live buffer sizes | Saves ~1 GB of HBM on a 70B LLM at a ~1–2% step time cost |
| `--xla_tpu_max_hbm_size_mib=<size>` | Sets the target HBM budget for the rematerialization pass | Forces compiler to fit within a constrained budget |
| `--xla_tpu_rematerialization_algo=PEAK_PRIORITY` | Focuses rematerialization at peak allocation points | More aggressive memory reduction |
| `--xla_tpu_rematerialization_max_block_size_limit=32` | Maximum instructions per rematerialized block | Increases compilation time |
| `--xla_tpu_rematerialization_block_effort_factor=10.0` | Search effort factor for rematerialization blocks | Slower compilation, deeper memory savings |
| `--xla_tpu_pre_fusion_remat=true` | Adds rematerialization pass prior to instruction fusion | Increases compile time; check numerical stability |

--------------------------------------------------------------------------------

### Summary of Interventions

| Intervention | Numerically Safe? | Potential Gains | Telltale Signs in XProf |
| :------------------- | :-------------- | :-------------- | :---------------- |
| **Tensor Sharding** | Yes | Massive (up to 256x) | Single replicated tensor in Memory Viewer much larger than other allocations |
| **Reducing Batch Size** | **No** (alters training dynamics; requires LR retuning) | Massive (up to 1000x) | Temporaries failing to allocate during gradient (`JVP`) calculations; many batch-size-shaped tensors |
| **Micro-Batching** | Yes | Massive (linear in $n$) | Peak memory dominated by batch-sized activation temporaries during backward pass |
| **bfloat16 Precision** | Moderate risk | ~2x reduction | Memory Viewer shows large tensors allocated as `float32` |
| **Rematerialization (`jax.checkpoint`)** | Yes | Large (~50% activations) | Long activation buffer lifetimes persisting across forward pass into backward pass |
| **Buffer Donation (`donate_argnums`)** | Yes | Marginal (~1-5%) | Input parameters duplicated alongside identical-shaped output buffers |
| **Dimension Alignment (128x)** | **No** (changes model width/shapes) | Variable (up to 50% per tensor) | Memory Viewer cards report high padding percentage on non-aligned shapes |
| **Host Offloading** | Yes (numerically), No (performance) | Marginal (~3x CPU RAM) | Massive optimizer states held in HBM throughout forward and backward passes |

## Connections

-   **Workflows**:
    -   [[hbm-oom-triage]]
    -   [[general-ml-debugging]]
-   **Actions**:
    -   [[analyze-hbm-oom]]
    -   [[query-workload-logs]]
    -   [[inspect-hlo-dumps]]
-   **Playbooks**:
    -   [[layered-debugging-framework]]
    -   [[general-ml-debugging]]
-   **Symptoms**:
    -   [[hbm-oom]]
    -   [[tpu-coordinator-oom]]
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[ecc-errors]]
    -   [[known-issues]]
-   **Entities**:

    -   [[cloud-tpu]]
    -   [[tensorcore]]
    -   [[sparsecore]]
    -   [[jax]]
    -   [[xla]]

    -   [[xprof]]

    -   [[cloud-logging]]

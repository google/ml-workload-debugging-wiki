---
title: "Pallas and Mosaic Kernel Debugging Playbook"
description: "Architecture, compilation pipeline, custom kernel design patterns, and debugging practices for JAX Pallas and Mosaic IR on TensorCore and SparseCore."
type: playbook
internal_only: false
tags: [tpu, analysis, triage, methodology]
created: 2026-09-28
updated: 2026-09-28
---


# Pallas and Mosaic Kernel Debugging Playbook

## Overview

In TPU programming, the standard XLA compiler provides automatic optimization
and operator fusion, combining multiple operations into unified kernels to
minimize memory bandwidth overhead. However, certain high-performance
architectures (such as FlashAttention, custom attention variations, ragged
fusions, or specialized communication-compute overlap) demand fine-grained,
manual control over the underlying TPU memory hierarchy and vector registers.

[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) is JAX's custom
kernel language, providing a high-level, portable programming abstraction across
accelerators (TPU and GPU). On Google Cloud TPUs, Pallas lowers directly to
**Mosaic**, an MLIR-based intermediate representation (IR) and optimizing
compiler that acts as an "escape hatch" from standard XLA HLO to Low-Level
Optimizer (LLO) machine code.

This playbook provides compiler engineers and TPU kernel developers with an
in-depth reference for:

1.  Understanding the lowering pipeline from Python JAX/Pallas down to MLIR LLO.
2.  Navigating custom kernel design trade-offs vs automatic XLA compilation.
3.  Managing memref and vector layouts on TensorCore and SparseCore.
4.  Enabling diagnostic flags, bounds checks, and layout inference passes.

--------------------------------------------------------------------------------

## Architecture & Lowering Pipeline

The compilation stack bridges high-level user code with TPU hardware execution
through a multi-stage lowering process:

$$\text{JAX Pallas Python API} \longrightarrow \text{Jaxpr} \longrightarrow \text{Mosaic IR (MLIR)} \longrightarrow \text{MLIR LLO} \longrightarrow \text{LLO Binary}$$

```
+--------------------------------------------------------------+
|                   JAX Pallas Python Code                     |
|         (@pl.pallas_call, @pl.kernel, pl.when, etc.)         |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                   JAX Program (Jaxpr)                        |
+--------------------------------------------------------------+
                               |
                               v (TPU Lowering)
+--------------------------------------------------------------+
|                     Mosaic MLIR IR                           |
|  - Standard MLIR: builtin, arith, func, math, memref, scf,   |
|                   vector                                     |
|  - TPU Dialect:   tpu (DMAs, semaphores, hardware vregs)     |
+--------------------------------------------------------------+
                               |
                               v (Mosaic Compiler Passes)
+--------------------------------------------------------------+
|                        MLIR LLO                              |
|           (Pipelined loop via PipelineEmitter)               |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                 LLO Device Executable Binary                 |
+--------------------------------------------------------------+
```

### MLIR Dialects in Mosaic IR

The input to Mosaic is an MLIR `FuncOp` annotated with Mosaic-specific
attributes. Mosaic IR synthesizes standard LLVM/MLIR dialects with a custom
`tpu` dialect:

-   **`vector` dialect**: Represents logical values that reside in physical
    vector registers (`vregs`). Internally, Mosaic operates on $8 \times 128$
    vector tiles to match hardware sublanes.
-   **`memref` dialect**: Represents buffers and arrays allocated across
    physical memory spaces:
    -   `VMEM`: On-chip Vector Memory (high bandwidth, low latency, limited
        capacity per core).
    -   `SMEM`: Scalar Memory for index computation and control registers.
    -   `HBM`: High Bandwidth Memory (device global memory).
-   **`builtin` dialect**: Scalar types representing values living in scalar
    registers (`sregs`).
-   **`arith`, `math`, `scf`, `func` dialects**: Standard arithmetic
    expressions, control flow (`scf.for`, `scf.if`), and functional boundaries.
-   **`tpu` dialect**: Custom TPU dialect handling hardware-specific primitives
    not expressible in upstream MLIR: asynchronous direct memory access (DMA),
    semaphores, barrier signals, and core selection.
-   **StableHLO input**: Limited StableHLO input can be ingested into Mosaic
    using the `--jax_mosaic_allow_hlo` flag (via
    `jax.config.update('jax_mosaic_allow_hlo', True)`), which converts
    `stablehlo` ops to `linalg` and subsequently to `vector`.

--------------------------------------------------------------------------------

## Custom Kernel Design vs XLA Automatic Compilation

Relying on standard XLA compilation versus authoring a Pallas/Mosaic custom
kernel involves fundamental trade-offs:

| Aspect | Automatic XLA Compilation | Pallas / Mosaic Custom Kernel |
| :------------------- | :------------------------ | :---------------------- |
| **Optimization Philosophy** | Automatic operator fusion, buffer assignment, and scheduling. | Explicit developer control over execution, scheduling, and buffering. |
| **Memory Hierarchy** | Automatic management of HBM-to-VMEM transfers via compiler heuristics. | Explicit allocation and staging between HBM, VMEM, SMEM, and scratchpad. |
| **Double-Buffering & DMA** | Compiler-generated asynchronous copies. | Developer-orchestrated pipelined DMAs and semaphore signalling. |
| **Tiling & Layouts** | Compiler automatically permutes minor/major physical dimensions. | Manual innermost dimension layout; no automatic transposes. |
| **Use Cases** | Standard dense layers, elementwise fusions, transformers, convolutions. | FlashAttention, ragged sequence processing, custom quantizations, SparseCore pipelines. |

### When to Use Pallas/Mosaic

1.  **Memory Hierarchy Control**: When XLA emits redundant round-trips to HBM or
    fails to keep intermediate tensors resident in VMEM across large fused
    subgraphs.
2.  **FlashAttention and Fused Attention**: Bypassing materialization of
    intermediate $N \times N$ attention matrices in HBM by tiling queries, keys,
    and values into VMEM tiles and accumulating softmax statistics in-register.
3.  **Custom Operator Fusion**: Fusing non-standard patterns that XLA's fusion
    heuristics decline to group (e.g. complex dynamic masking, custom
    non-linearities, or combined scatter-reductions).

--------------------------------------------------------------------------------

## Mosaic TensorCore: Memory Layouts and Compiler Passes

### Memref Layouts

A memref layout specifies how data is physically arranged in memory (`VMEM`,
`SMEM`, or `HBM`). It is parameterized by:

-   **Tiling**: Chunk-padded formatting dictated by the 2D nature of TPU vector
    registers ($8 \times 128$).
-   **Tile Strides**: Slicing step sizes. For example,
    `#tpu.tiled<(8,128), [4, 1]>` indicates skipping 4 tiles along the major
    dimension and 1 tile along the minor dimension when navigating the array.

> [!IMPORTANT] **No Automatic Dimension Transposition**: Unlike standard XLA
> HLO, Mosaic memref layouts **do not** feature physical-to-logical dimension
> mapping. The Mosaic compiler will not automatically transpose arrays to
> optimize access patterns. Because TPU hardware natively processes innermost
> dimensions in $(8, 128)$ vector blocks, the developer **must place contiguous,
> compute-heavy feature dimensions at the very end of the array shape**.

### Vector Layouts

Vector layouts define how logical multi-dimensional values are mapped into
physical `vregs`:

-   **Element Bitwidth**: Bitwidth of the primitive type (e.g., 32 bits for
    `f32` / `s32`).
-   **Tiling**: Based on XLA tiling (e.g., $(8, 128)$).
-   **Sublane Packing**: TPU physical `vreg` slots are 32 bits wide. Narrower
    data types pack elements vertically into each sublane slot:
    -   `bf16`: Packs 2 elements vertically $\rightarrow 16 \times 128$ elements
        per `vreg`.
    -   `i8`: Packs 4 elements vertically $\rightarrow 32 \times 128$ elements
        per `vreg`.
-   **2D Offset & Replication**: Specified as `{row_offset, col_offset}` or `*`
    (replicated across the dimension).
-   **Implicit Dimension**: Adds a dimension of size 1 to 1D shapes at position
    `-1` (minor) or `-2` (second-minor).
-   MLIR representation example: `#tpu.vpad<"32,{0,0},(8,128),-1">`.

### TensorCore Compilation Passes

The Mosaic TensorCore compiler pipeline executes the following sequential
passes:

1.  **Linalg Vectorization Pass**: Converts incoming `stablehlo` ops to `linalg`
    and then `vector` (enabled via
    `jax.config.update('jax_mosaic_allow_hlo', True)`).
2.  **Debug Assert Insertion Pass**: Injects on-device assertions for dynamic
    indices in `load`, `store`, and `slice` operations:

    ```bash
    --xla_mosaic_on_device_checks=bounds
    ```
3.  **Infer Memref Layout Pass**: Computes required memref tiling attributes and
    tile strides to match hardware vector requirements:

    ```bash
    --infer_memref_layout
    ```

    *(To suppress MLIR static layout checks during experimentation:
    `tpu.erase_memref_layout`)*.
4.  **Infer Vector Layout Pass**: Traverses the graph from inputs to outputs,
    propagating compatible vector layouts and resolving branch discrepancies
    (e.g., in `scf.if`).
5.  **Apply Vector Layout Pass**:

    -   Unrolls multi-dimensional vectors and vector operations into native $8
        \times 128$ `vreg`-sized chunks.
    -   Inserts explicit relayout and permutation operations when a producer's
        output layout conflicts with a consumer's input layout.
6.  **Lower to LLO Pass and Custom Call Emitter**:

    -   Expands `vreg`-sized vector operations into MLIR LLO instructions and
        generates the outer pipelined loop via XLA's `PipelineEmitter`.

--------------------------------------------------------------------------------

## Mosaic SparseCore (SC)

Mosaic also supports custom kernel execution on TPU SparseCore (SC), the
specialized accelerator for embeddings and sparse memory access.

### Key Differences Between TensorCore and SparseCore Mosaic

1.  **Subcore Selection**: SparseCore contains scalar and vector subcores. A
    kernel `FuncOp` must be annotated with `#tpu.core_type`:

    -   `#tpu.core_type<sc_vector_subcore>`
    -   `#tpu.core_type<sc_scalar_subcore>` In Pallas Python code, target
        SparseCore subcores via `compiler_params`:

    ```python
    import jax.experimental.pallas.tpu as pltpu
    import jax.experimental.pallas.tpu_sc as plsc

    compiler_params = pltpu.CompilerParams(
        kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE
    )
    ```
2.  **Absence of Vector Layout Inference**: Unlike TensorCore, Mosaic SC **does
    not** execute "Infer Vector Layout" or "Apply Vector Layout" passes.
    Consequently:

    -   Kernel inputs must be **pre-unrolled** by the author.
    -   All vector shapes must directly match the physical number of lanes in
        the SC VPU (e.g., `[8]` for 32-bit types, or `[8 * n]` for bitpacked
        types).
3.  **Pipelined Loop Emission**: Mosaic SC lowers to SparseCore's intermediate
    representation (MLO) using a fixed-window pipeline emitter.
4.  **Dialect Extensions**: While sharing the core `tpu` dialect for DMAs and
    semaphores, SC supports indirect HBM access, gather-scatter indexing, and
    dedicated scratchpad memory spaces.

--------------------------------------------------------------------------------

## Diagnostic Procedures and Debugging Steps

### 1. Pallas API Stability vs Direct Mosaic IR

> [!TIP] **Recommendation**: Always prioritize writing kernels using the
> **Pallas Python API** rather than directly targeting Mosaic MLIR. Pallas
> provides a stable, high-level programming model that abstracts hardware
> generation differences. Mosaic IR is an internal compiler intermediate
> representation subject to syntax and dialect changes across JAX/XLA releases.
> Any kernel expressible in Mosaic is designed to be expressible in Pallas.

### 2. On-Device Bounds Checking

Unchecked dynamic indexing in Pallas kernels frequently causes silent data
corruptions or sudden hardware halts
(`tag:pc TensorCoreSequencer:... BoundsCheck`). Enable runtime bounds checks
during debugging:

```bash
export LIBTPU_INIT_ARGS="--xla_mosaic_on_device_checks=bounds"
```

When an out-of-bounds dynamic slice or load occurs, the check halts device
execution immediately and reports the failing coordinate and allocation bounds,
preventing silent memory stomps.

## Mitigations, Workarounds & Best Practices

### 1. Scratchpad Allocation and VMEM Sizing

-   **Explicit Scratchpad & VMEM Management**: Unlike automatic XLA compilation,
    custom kernels require explicit management of VMEM, SMEM, and scratchpad
    memory spaces. Over-allocating tiles or failing to release temporary buffers
    will trigger compile-time memory exhaustion or runtime tile allocation
    faults.
-   **Double-Buffering Overhead**: Pipelining DMAs with `@pltpu.emit_pipeline`
    or double-buffering requires allocating twice the tile size in VMEM. Verify
    that `tile_size * 2 + scratchpad <= total_vmem`.

### 2. Inspecting Lowered Code

When a Pallas kernel produces unexpected numerical output or fails to compile:

1.  **Inspect Jaxpr**: Print the staged JAX program:

    ```python
    jaxpr = jax.make_jaxpr(pallas_fn)(*args)
    print(jaxpr)
    ```
2.  **Dump Mosaic MLIR**: Capture intermediate MLIR module before and after
    layout passes:

    ```bash
    export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
    ```
3.  **Verify Layout Annotations**: Inspect the generated MLIR `FuncOp` for
    `#tpu.tiled` and `#tpu.vpad` attributes to confirm innermost dimensions
    match hardware tile alignment without excessive relayout overhead.

--------------------------------------------------------------------------------

## Connections

-   **Playbooks**:
    -   [[sparsecore-debugging-triage]]
    -   [[on-device-debugging-triage]]
    -   [[data-non-determinism-triage]]
    -   [[numerics-error-triage]]
    -   [[debugging-mindset]]
-   **Symptoms**:
    -   [[sdc-checksum-mismatch]]
    -   [[wait-timeout-on-sflag]]
    -   [[hbm-oom]]
-   **Concepts**:
    -   [[libtpu-program-lifecycle]]
    -   [[multi-chip-hangs]]
-   **Entities**:
    -   [[jax]]
    -   [[xla]]
    -   [[tensorcore]]
    -   [[sparsecore]]
    -   [[cloud-tpu]]
-   **Tools**:
    -   [[xprof]]

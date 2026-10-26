---
title: "HLO and LLO Dump Analysis Playbook"
description: "Comprehensive guide for configuring, generating, and analyzing XLA High-Level Optimizer (HLO) and Low-Level Optimizer (LLO) dumps, compiler pipeline passes, bundle formatting, and fingerprint hashing on Cloud TPUs."
type: playbook
internal_only: false
tags: [tpu, analysis, triage, methodology]
created: 2026-09-28
updated: 2026-09-28
---


# HLO and LLO Dump Analysis Playbook

## Overview

In the Cloud TPU software stack, user code written in frameworks such as JAX or
TensorFlow is traced and compiled by **OpenXLA** into accelerator machine
binaries. This compilation pipeline spans two primary optimization layers:

1.  **High-Level Optimizer (HLO)**: A hardware-agnostic intermediate
    representation (IR) representing computation graphs. HLO transformations
    optimize mathematical operations, perform operator fusions, infer memory
    buffer assignments, and determine tensor layout representations.
2.  **Low-Level Optimizer (LLO)**: The lowest-level compiler layer in the
    TPU-specific XLA backend. LLO lowers optimized HLO computations to physical
    TPU silicon execution graphs, executing target-specific instruction
    selection, vector/scalar register allocation, dependency analysis, and
    bundle packing for the TPU VPU (Vector Processing Unit) and SPU (Scalar
    Processing Unit).

Capturing HLO and LLO dumps provides exact visibility into graph transformations
across each compiler pass. These dumps are critical for:

-   Root-causing compile-time failures, assertion crashes, and compiler hangs
    (see [[compile-time-crash-reporting]]).
-   Diagnosing numerical divergences or precision bugs between software releases
    (see [[numerics-error-triage]]).
-   Detecting compilation non-determinism across distributed TPU workers (see
    [[megascale-hang-triage]] and [[data-non-determinism-triage]]).
-   Verifying register allocation pressure and bundle packing density.

--------------------------------------------------------------------------------

## Diagnostic Analysis & Pipeline Stage Breakdown

The XLA compilation pipeline transforms high-level framework operations into
native TPU instruction bundles across distinct stages:

```
               +---------------------------------------------+
               | User Framework Graph (JAX Jaxpr / TF Graph)  |
               +---------------------------------------------+
                                      |
                                      v
+===========================================================================+
| HIGH-LEVEL OPTIMIZER (HLO) PIPELINE                                       |
|                                                                           |
| 1. High-Level Optimizations & Algebraic Rewrites                         |
|    - Constant folding, dead code elimination, broadcast simplifications   |
|    - Output: module_####.<name>.before_optimizations.txt                  |
|                                                                           |
| 2. Operator Fusion & Target-Independent Transformations                   |
|    - Loop fusions, input fusions, multi-output fusions                    |
|                                                                           |
| 3. Layout Assignment & Memory Space Assignment (MSA)                      |
|    - Tiling selection, physical memory allocation, live-range analysis    |
|    - Output: module_####.<name>.after_optimizations.txt                   |
+===========================================================================+
                                      |
                                      v
+===========================================================================+
| LOW-LEVEL OPTIMIZER (LLO) PIPELINE (TPU Backend)                          |
|                                                                           |
| 4. Instruction Selection & Initial LLO IR Emitters                        |
|    - Auto MXU assignment, loop inversion, sub-word SIMD expansion        |
|    - Output: *-01-original.txt                                            |
|                                                                           |
| 5. Low-Level Optimization Passes                                          |
|    - Copy propagation (CP), Global Value Numbering (GVN), DCE             |
|    - Bounds check assertions, constant materializer, CSSA lowering        |
|                                                                           |
| 6. Bundle Packing & Scheduling (Pre-RA)                                   |
|    - Dependency graph construction, instruction bundle grouping           |
|    - Output: *-23-packed-bundles-pre-ra.txt                               |
|                                                                           |
| 7. Register Allocation (RA)                                               |
|    - Scalar/vector register pressure analysis, spill code generation      |
|    - Output: *-29-post-ra.txt                                             |
|                                                                           |
| 8. Post-RA Bundle Packing & Final Machine Code Generation                 |
|    - Empty bundle elimination / vnop/vdelay insertion                     |
|    - Output: *-40-final_bundles.txt                                       |
+===========================================================================+
```

--------------------------------------------------------------------------------

## Configuring HLO Dumps

### Enabling HLO Dumps via Environment Variables

HLO dumping can be enabled either via `XLA_FLAGS` or through `LIBTPU_INIT_ARGS`:

```bash
# Via XLA_FLAGS:
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"

# Or via LIBTPU_INIT_ARGS:
export LIBTPU_INIT_ARGS="--xla_dump_to=/tmp/xla_dump"
```

To emit both human-readable text and binary protobuf representations (useful for
automated tools and bug submissions):

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
```

In interactive environments like Google Colab or Jupyter notebooks, set the
environment variable prior to importing JAX or initializing the runtime:

```python
import os
os.environ['XLA_FLAGS'] = "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
import jax
```

### Extracting HLO from XProf

In addition to flag-based file dumps, the [[xprof]]
[Graph Viewer tool](https://openxla.org/xprof/graph_viewer) can visualize the
unoptimized HLO graph directly from a captured profile session and extract the
relevant HLO text in various formats.

### Core HLO Dump Files

HLO dumps generate chronological text files prefixed by module index:

| Filename Suffix              | Description           | Primary Use Case      |
| :--------------------------- | :-------------------- | :-------------------- |
| `*.before_optimizations.txt` | Raw, unoptimized HLO graph translated directly from the high-level framework. | Verify whether user framework math/logic was lowered correctly. |
| `*.after_optimizations.txt` | Fully optimized HLO graph immediately prior to backend lowering. | Inspect operator fusions, layout assignments, and memory allocations. |
| `*buffer-assignment.txt` | Detailed layout of allocated memory buffers and alias relationships. | Troubleshoot buffer sharing and memory overhead. |
| `*memory-usage-report.txt` | Static memory footprint summary across HBM and shared memory spaces. | Diagnose compile-time memory exhaustion ([[hbm-oom-triage]]). |
| `*tpu_comp_env.txt` | Exact compiler flags, build targets, and environment settings. | Mandatory artifact for compiler bug reproduction. |

--------------------------------------------------------------------------------

## Configuring LLO Dumps

The Low Level Optimizer (LLO) is responsible for physical TPU instruction
scheduling, register allocation, and hardware bundle generation.

> [!IMPORTANT] The `--xla_jf_dump_to` flag controls LLO dumping and **MUST** be
> passed through `LIBTPU_INIT_ARGS`. Passing it via `XLA_FLAGS` will be ignored
> by the TPU runtime.

### Enabling Basic LLO Dumps

```bash
export LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_dump"
```

### LLO Pass Files & Artifacts

Dumping LLO produces numbered pipeline stages reflecting transformations:

-   `*-01-original.txt`: Initial unoptimized LLO IR generated directly by
    backend emitters.
-   `*-09-post-MXU-assigner.txt`: Instructions after Matrix Multiply Unit (MXU)
    allocation.
-   `*-12-post-bounds-check.txt`: Insertion of bounds-check assertions (see
    [[on-device-debugging-triage]]).
-   `*-21-register-pressure.txt`: Register pressure analysis across scalar and
    vector registers.
-   `*-23-packed-bundles-pre-ra.txt`: Instruction bundles packed prior to
    register allocation.
-   `*-29-post-ra.txt`: Linearized LLO post register allocation.
-   `*-36-packed-bundles-post-ra.txt`: Bundles packed with concrete register
    assignments.
-   `*-40-final_bundles.txt`: Final silicon-ready instruction stream with empty
    bundles removed or padded with target-specific delays (`vnop`/`vdelay`).

--------------------------------------------------------------------------------

## LLO Fingerprint Hashing for Compilation Divergence

When running multi-worker TPU workloads, different workers must execute
identical program binaries. If compilation diverges across hosts, workers stall
indefinitely at collective communication barriers, causing [[megascale-hang]] or
[[sdc-checksum-mismatch]].

LLO Fingerprints are generated by recursively hashing all semantic components of
instructions across compiler passes:

### Configuration Flags

```bash
export LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_dump \
  --xla_jf_dump_category_filter=fingerprint \
  --xla_jf_dump_extended_fingerprint=true \
  --xla_jf_dump_limit_fingerprints=true"
```

-   `--xla_jf_dump_category_filter=fingerprint`: Restricts dump output
    specifically to fingerprint hashes.
-   `--xla_jf_dump_extended_fingerprint=true`: Enables "show your work" verbose
    mode, logging intermediate hashes per instruction block to isolate the exact
    instruction where hashes diverge.
-   `--xla_jf_dump_limit_fingerprints=true`: Restricts calculation to 1–2 passes
    at the beginning and conclusion of LLO compilation, minimizing compile-time
    overhead.

--------------------------------------------------------------------------------

## Bundle Formatting: Single-Line vs. Multi-Line

TPU VPU/SPU hardware executes instruction bundles containing parallel operations
issued concurrently. LLO dump bundle formatting can be controlled based on
diagnostic goals:

### Default: Single-Line Delimited Bundles

By default, instructions within a bundle are printed on a single line separated
by `;`:

```
 0x13   : > { %s43 = sshrl.u32 %s12724, 10;  %s12725 = smul.u32 3072, %s33 }
 0x14   : > { %p40 = scmp.ge.s32.totalorder %s39, 5;  %s45 = sshrl.u32 %s12725, 10 }
```

-   **Advantage**: Bundle count corresponds directly to line count. Line length
    provides an immediate visual proxy for bundle packing density and
    instruction-level parallelism (ILP).

### Multi-Line Bundles (`--xla_tpu_instructions_on_separate_lines`)

To format each instruction on an individual indented line:

```bash
export LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_dump --xla_tpu_instructions_on_separate_lines=true"
```

Emitted formatting:

```
 0x13   : > { %s43 = sshrl.u32 %s12724, 10
               %s12725 = smul.u32 3072, %s33 }
 0x14   : > { %p40 = scmp.ge.s32.totalorder %s39, 5
               %s45 = sshrl.u32 %s12725, 10 }
```

-   **Advantage**: Eliminates horizontal scrolling and line wrapping in
    terminals, simplifying searches for specific registers, opcodes, or
    operands.

--------------------------------------------------------------------------------

## Category Filters and Compile-Time Memory Management

Certain compiler dumping utilities format entire IR representations in host RAM
before writing to disk. For large multi-gigabyte models, generating unrestricted
dumps across dozens of modules can trigger host compile-time Out of Memory (OOM)
crashes.

Use `--xla_jf_dump_category_filter` with a comma-separated list of target
categories to limit output:

```bash
export LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_dump --xla_jf_dump_category_filter=original,codegen"
```

| Category      | Description                                                 |
| :------------ | :---------------------------------------------------------- |
| `original` | Limits dumps to unoptimized initial LLO IR from emitters before optimization passes. |
| `codegen` | Emits only final code generation artifacts (packed bundles, post-RA). |
| `fingerprint` | Emits LLO fingerprint hashes for compiler divergence testing. |
| `program`     | Outputs specific executable program metadata.               |
| `msa`         | Emits Memory Space Assignment allocation reports.           |

--------------------------------------------------------------------------------

## Common Gotchas & Mitigations

1.  **`LIBTPU_INIT_ARGS` vs. `XLA_FLAGS`**: LLO flags like `--xla_jf_dump_to`
    must be passed via `LIBTPU_INIT_ARGS`. Passing them via `XLA_FLAGS` is
    silently ignored by the TPU LLO backend.
2.  **Disk Saturation & Host OOM**: Dumping every HLO and LLO pass across large
    models generates thousands of files and gigabytes of text, which can
    saturate `/tmp` or trigger host RAM exhaustion. Mitigate by scoping dumps
    with `--xla_dump_hlo_pass_re` or using category filters
    (`--xla_jf_dump_category_filter=fingerprint`).
3.  **Container Ephemerality**: On GKE TPU pods, local `/tmp` directories are
    destroyed when the pod restarts. Ensure dumps are copied to Google Cloud
    Storage (GCS) using [[inspect-hlo-dumps]].

--------------------------------------------------------------------------------

## Connections

-   **Playbooks**:
    -   [[compile-time-crash-reporting]]
    -   [[numerics-error-triage]]
    -   [[on-device-debugging-triage]]
    -   [[megascale-hang-triage]]
    -   [[pallas-mosaic-kernel-debugging]]
-   **Actions**:
    -   [[inspect-hlo-dumps]]

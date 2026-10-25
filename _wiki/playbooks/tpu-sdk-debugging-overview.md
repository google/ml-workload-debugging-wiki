---
title: "TPU SDK Debugging Overview and Reference Guide"
description: "Comprehensive reference guide and navigation index for the 11-module Cloud TPU SDK debugging curriculum, connecting high-level JAX troubleshooting down to low-level hardware diagnostics."
type: playbook
internal_only: false
tags: [tpu, triage, methodology, analysis]
created: 2026-09-28
updated: 2026-10-09
---


# TPU SDK Debugging Overview and Reference Guide

## Overview

Debugging large-scale machine learning workloads on Cloud TPUs requires
navigating multiple layers of abstraction—from high-level Python and JAX
primitives down to XLA compiler intermediate representations, collective
networking fabrics, and bare-metal accelerator silicon.

Both JAX and the TPU SDK provide an extensive suite of diagnostic flags, tracing
utilities, and telemetry frameworks. Selecting the appropriate diagnostic tool
is a critical engineering decision: using overly heavy instrumentation across
thousands of chips can perturb execution timing or trigger Out-of-Memory
conditions, while insufficient logging leaves silent hangs unresolvable.

This primary guide serves as the central navigation index connecting all 11
modules of the Cloud TPU SDK debugging series, organizing triage workflows by
failure domain.

--------------------------------------------------------------------------------

## 11-Module Curriculum Navigation Matrix

The table below maps the modular TPU SDK curriculum to dedicated wiki playbooks,
concepts, and primary diagnostic tools:

| Module   | Topic            | Primary Diagnostic Tools & Flags      | Dedicated Wiki Playbook            |
| :------- | :--------------- | :------------------------------------ | :--------------------------------- |
| **Module 00** | TPU SDK Overview & JAX Debugging | JAX config flags (`jax_debug_nans`, `jax_disable_jit`), JAX docs | [[tpu-sdk-debugging-overview]] |
| **Module 01** | Megascale Error Handling & Hangs | `--megascale_graph_hang_threshold`, Error Aggregator | [[megascale-hang-triage]] |
| **Module 02** | Megascale Stragglers | Latency metrics, Node Replacement R&R | [[megascale-performance-tuning]] |
| **Module 03** | Megascale Performance Tuning | XProf trace viewer, MapDMA buffer shortage analysis | [[megascale-performance-tuning]] |
| **Module 04** | OOM & HBM OOM (Error Code 1000) | XProf Memory Viewer, static memory reports, remat | [[hbm-oom-triage]] |
| **Module 05** | TPU On-Device Debugging | Core dumps (`--xla_tpu_dump_core_on_halt=true`), HPPT, sync flag timeouts, LLO bounds check | [[on-device-debugging-triage]] |
| **Module 06** | Reporting JAX/XLA Compiler Crashes | C++ stack traces, minimal reproducers, HLO dumps | [[compile-time-crash-reporting]] |
| **Module 07** | Pallas & Mosaic Kernel Debugging | Mosaic IR lowering, VMem layouts, `@pltpu.emit_pipeline`, MLIR passes | [[pallas-mosaic-kernel-debugging]] |
| **Module 08** | Data Non-Determinism & SDC | XLA SDC Checker (XSC), repeat checksums, NaNs/Infs localization | [[data-non-determinism-triage]] |
| **Module 09** | Numerics-Related Errors | `hlo-diff` semantic diffing, `hlo-isolation`, host-assisted logging | [[numerics-error-triage]] |
| **Module 10** | SparseCore Debugging Tools | SparseCore Mismatch Detector, Perfetto trace visualizer, SC assert levels | [[sparsecore-debugging-triage]] |
| **Module 11** | HLO & LLO Dumps Generation | `--xla_dump_to`, `--xla_jf_dump_to`, LLO fingerprint hashing | [[hlo-llo-dump-analysis]] |

--------------------------------------------------------------------------------

## Triage Hierarchy: Choosing the Right Debugging Level

When investigating an anomalous workload, follow this progressive triage
hierarchy:

```
[Workload Failure / Anomaly]
             |
             +---> Is it a Python / Graph Tracing Error?
             |        |
             |        +-> YES: Module 00 (JAX Debugging) & Module 06 (Compiler Crashes)
             |
             +---> Is it a Compilation-Time Memory or Graph Failure?
             |        |
             |        +-> YES: Module 04 (HBM OOM 1000) & Module 11 (HLO/LLO Dumps)
             |
             +---> Is it a Multi-Worker Collective Hang or Stall?
             |        |
             |        +-> YES: Module 01 (Megascale Hangs) & Module 05 (Sync Flag Timeouts)
             |
             +---> Is the Model Running but Performing Abnormally Slow?
             |        |
             |        +-> YES: Module 02 (Stragglers) & Module 03 (Megascale Performance)
             |
             +---> Did the Chip Halt with a Fatal Driver/Hardware Signal?
             |        |
             |        +-> YES: Module 05 (On-Device Core Dumps & HPPT)
             |
             +---> Is Output Mathematically Inconsistent or Diverging?
             |        |
             |        +-> YES: Module 08 (SDC / Non-Determinism) & Module 09 (Numerics Diff)
             |
             +---> Is Custom Kernel Code (Pallas/Mosaic) Failing?
             |        |
             |        +-> YES: Module 07 (Pallas & Mosaic IR)
             |
             +---> Is Embedding / Sparse Acceleration Suspect?
                      |
                      +-> YES: Module 10 (SparseCore Tools & Mismatch Detector)
```

--------------------------------------------------------------------------------

## Detailed Module Overviews

### Module 00: Overview & General JAX Debugging

When starting an investigation, high-level JAX debugging APIs provide fast,
non-invasive visibility before applying heavy compiler or runtime tracing:

-   Inspect Python exceptions and Jaxpr traces using `jax.make_jaxpr`.
-   Isolate NaN/Inf generation early using `jax.debug.print` or
    `jax.config.update("jax_debug_nans", True)`.
-   Eliminate compilation artifacts temporarily to test eager Python semantics
    via `jax.config.update("jax_disable_jit", True)`.
-   Consult the official JAX debugging documentation:
    [JAX Debugging](https://docs.jax.dev/en/latest/debugging.html).

### Module 01: Megascale Error Handling & Hangs

Multi-slice TPU training leverages Megascale to execute DCN collective
communications. A stall on a single chip halts the global barrier:

-   Use the Megascale Error Aggregator digestto identify the first-stalled
    worker.
-   Tune hang thresholds via `--megascale_graph_hang_threshold`.
-   Detailed playbook: [[megascale-hang-triage]].

### Module 02: Megascale Stragglers

A straggler is an outlier host that computes or transmits slower than peers
without crashing, degrading cluster goodput:

-   Track per-host collective step times and RPC latencies.
-   Isolate network fabric degradations from host compute bottlenecks.
-   Apply fleet-level node quarantining via [[node-isolation-methodology]].
-   Detailed playbook: [[megascale-performance-tuning]].

### Module 03: Megascale Performance Tuning

Investigating step-dependent, consistent, or periodic performance slowdowns:

-   Capture and analyze XProf traces to inspect DCN communication timelines
    against TPU compute phases.
-   Identify steady-state MapDMA buffer pool exhaustion and configure host
    buffer headroom.
-   Detailed playbook: [[megascale-performance-tuning]].

### Module 04: OOM & HBM OOM (Error Code 1000)

Exhaustion of on-chip High Bandwidth Memory (HBM) during compilation or
execution:

-   Interpret `RESOURCE_EXHAUSTED` error code `E1000` (Compile Time HBM OOM).
-   Utilize the XProf Memory Viewer to inspect peak memory allocation points,
    live ranges, and buffer fragmentation.
-   Apply targeted remediations: activation rematerialization, batch size
    adjustment, and layout tuning.
-   Detailed playbook: [[hbm-oom-triage]].

### Module 05: TPU On-Device Debugging

Hardware execution halts, bounds check violations, and firmware assertion
failures:

-   Capture silicon state on failure with `--xla_tpu_dump_core_on_halt=true`,
    `--tpu_core_dump_arch_states`, and `--tpu_core_dump_directory=/path/to/dir`.
-   Analyze scalar vs vector PC drift using
    `--xla_tpu_debug_add_sfence_after_hlo=true`.
-   Track progress across execution phases with Hierarchical Program Progress
    Tracking (HPPT).
-   Localize bounds check violations with LLO Bounds Check tooling
    (`--xla_jf_bounds_check=true`).
-   Detailed playbook: [[on-device-debugging-triage]].

### Module 06: Reporting JAX/XLA Compiler Crashes

Escalating internal compiler crashes, unhandled C++ exceptions, or graph
lowering failures:

-   Capture complete console output and C++ stack traces identifying the failing
    compiler pass.
-   Construct minimal, self-contained standalone reproducers.
-   Package HLO dump artifacts and `tpu_comp_env.txt` for compiler development
    teams.
-   Detailed playbook: [[compile-time-crash-reporting]].

### Module 07: Pallas & Mosaic Kernel Debugging

Writing and optimizing custom accelerator kernels that bypass standard HLO
transformations:

-   Understand lowering from JAX Pallas down to MLIR Mosaic IR and LLO machine
    code.
-   Manage memory spaces explicitly across VMem, Shared Memory (SMEM), and HBM.
-   Manage vector layouts, `#tpu.tiled` memrefs, sublane packing, and SparseCore
    pipeline emitters.
-   Detailed playbook: [[pallas-mosaic-kernel-debugging]].

### Module 08: Data Non-Determinism & SDC

Investigating mathematical divergences across runs, unexpected NaNs/Infs, or
silent corruption:

-   Deploy the XLA Silent Data Corruptions Checker (XSC) to run repeated
    execution validation.
-   Detect arithmetic differences across identical inputs without explicit
    driver crashes.
-   Detailed playbook: [[data-non-determinism-triage]].

### Module 09: Numerics-Related Errors

Isolating subtle numerical divergence between software versions, frameworks, or
precision formats:

-   Compare compiled HLO graphs semantically using `hlo-diff`.
-   Bisect intermediate HLO execution using `hlo-isolation`.
-   Verify TPU floating-point precision quirks: bfloat16, subnormal
    flush-to-zero, e4m3b11, e4m3b7, and multi-pass matrix multiplication
    emulation.
-   Detailed playbook: [[numerics-error-triage]].

### Module 10: SparseCore Debugging Tools

Verifying operations offloaded from TensorCore to SparseCore (embeddings,
collectives, sorting):

-   Execute shadow-execution verification with the SparseCore Mismatch Detector.
-   Inspect LLVM execution profiles and interactive Perfetto trace bundles.
-   Control assertion enforcement via `--xla_sc_assert_level`.
-   Detailed playbook: [[sparsecore-debugging-triage]].

### Module 11: HLO & LLO Dump Generation

Generating and interpreting compiler intermediate representations:

-   Configure HLO dumping via
    `XLA_FLAGS="--xla_dump_to=... --xla_dump_hlo_as_text"`.
-   Configure LLO dumping via `LIBTPU_INIT_ARGS="--xla_jf_dump_to=..."`.
-   Isolate compilation non-determinism across workers using LLO Fingerprint
    hashing.
-   Adjust instruction bundle formatting (single-line vs multi-line).
-   Detailed playbook: [[hlo-llo-dump-analysis]].

--------------------------------------------------------------------------------

## Common Gotchas & Mitigations

1.  **Jumping Directly to Complex Diagnostics**: Starting triage with low-level
    LLO dumps or XProf before verifying basic Python stack traces and GKE pod
    exit codes creates unnecessary overhead. Always begin at Level 0
    (Framework/JAX) and work downward.
2.  **Missing Redaction**: Shared diagnostic artifacts or escalations
    must ensure that all proprietary workload names and project IDs are replaced
    with sanitized placeholders.
3.  **Environment Variable Scope**: Remember that XLA flags (`XLA_FLAGS`)
    configure high-level compiler behavior, while low-level LLO and TPU runtime
    settings must be passed through `LIBTPU_INIT_ARGS`.

--------------------------------------------------------------------------------

## Connections

-   **Workflows**:
    -   [[debugging-megascale-hangs]]
    -   [[libtpu-timing-outlier-analysis]]
    -   [[hbm-oom-triage]]
    -   [[sdc-and-numerics-triage]]
    -   [[tpu-state-analysis]]
-   **Actions**:
    -   [[analyze-hbm-oom]]
    -   [[triage-sdc-and-numerics]]
    -   [[analyze-sparsecore-offload]]
    -   [[inspect-hlo-dumps]]
-   **Playbooks**:
    -   [[debugging-mindset]]
    -   [[layered-debugging-framework]]
    -   [[hang-diagnosis-methodology]]
    -   [[megascale-hang-triage]]
    -   [[megascale-performance-tuning]]
    -   [[hbm-oom-triage]]
    -   [[on-device-debugging-triage]]
    -   [[compile-time-crash-reporting]]
    -   [[pallas-mosaic-kernel-debugging]]
    -   [[data-non-determinism-triage]]
    -   [[numerics-error-triage]]
    -   [[sparsecore-debugging-triage]]
    -   [[hlo-llo-dump-analysis]]

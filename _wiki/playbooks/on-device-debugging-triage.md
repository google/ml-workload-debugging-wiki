---
title: "TPU On-Device Debugging and Crash Triage"
description: "Heuristic playbook for diagnosing on-device TPU crashes, hangs, core dump register states, hierarchical progress tracking, and LLO bounds check failures."
type: playbook
internal_only: false
tags: [tpu, hangs, triage, methodology, timeout]
created: 2026-09-28
updated: 2026-09-28
---


# TPU On-Device Debugging and Crash Triage

## Overview

When TPU programs fail during on-device execution, failures typically present as
sudden hardware fatal errors, compiler assertion halts, or multi-chip
distributed hangs. Debugging on-device accelerator execution proceeds through
three phases:

1.  **Localization**: Isolating the failure to a specific culprit HLO, custom
    kernel, or low-level instruction (LLO) on a specific core.
2.  **Debugging**: Examining architectural state, core dump registers,
    synchronization flags, or dereference calculations to understand the root
    cause.
3.  **Fix / Workaround**: Correcting tensor bounds, restructuring kernels,
    altering compiler optimization passes, or applying workarounds.

Localization is usually the most challenging step in large slices. This playbook
provides standard diagnostic procedures, compiler flags, and triage
methodologies for on-device investigations.

## Diagnostic Steps and On-Device Triage

### 1. Classify the Failure Signature

Identify whether the issue is a crash or a hang from the initial error logs:

-   **Hardware Fatal Error**: The TPU firmware reports an unrecoverable user or
    hardware error. Immediate action: collect and inspect core dumps across
    chips.
-   **XLA Generated Assertion**: Execution stopped at a compiler-inserted
    assertion (e.g. bounds check violation). Immediate action: inspect sequencer
    `tag:pc` and bounds check IDs.
-   **Single-Slice or Multi-Slice Hang**: Cores stop progressing while waiting
    for communication, DMA completion, or barrier synchronization. Immediate
    action: configure sync flag wait timeouts and progress tracking, and use
    TPUZ Monitoring to inspect live runtime status across all TPU cores attached
    to a host.

### 2. Core Dump Analysis and Architectural State

When an unrecoverable hardware fatal error occurs, the TPU firmware
automatically collects and serializes the chip's architectural state into a
binary core dump when `--xla_tpu_dump_core_on_halt=true` is enabled. Configure
core dump output and architectural state collection with:

```bash
--xla_tpu_dump_core_on_halt=true
--tpu_core_dump_arch_states=TPU_CORE_DUMP_ARCH_STATE_TC,TPU_CORE_DUMP_ARCH_STATE_SCS
--tpu_core_dump_directory=/path/to/dir
```

The core dump captures instruction memory (IMEM), scalar and vector registers
(`sregs`, `pregs`, `dregs` on SparseCore), shared memory (Smem), and
synchronization flags (`sflags`).

-   **Anomalies Section**: Inspect reported anomalies. Errors tagged `[USER]`
    indicate software-induced hardware faults (such as out-of-bounds memory
    indexing or sync flag counter overflow), whereas `[HARDWARE]` points to
    physical faults.

-   **Scalar PC Location (`tag:pc`)**: Identifies the overlay number and
    instruction memory (IMEM) offset at or immediately following the last
    executed instruction bundle.

-   **Scalar vs. Vector PC Drift**: Because the scalar issue unit runs ahead of
    the vector issue unit, the recorded scalar PC may point to an HLO
    immediately following the culprit. To enforce strict HLO-boundary
    synchronization and eliminate drift, run with:

    ```bash
    --xla_tpu_debug_add_sfence_after_hlo=true
    ```

-   **SparseCore State**: On architectures with SparseCore, sequencer and tile
    states are dumped independently. Inspect the SC sequencer PC to determine if
    stalls originated in embedding or DMA routines.

### 3. Sync Flag Wait Timeout Hierarchy

When a TPU program hangs, cores are usually blocked waiting on Direct Memory
Access (DMA) transfers or collective barrier sync flags.

#### Basic Configuration

Enable sync flag timeout detection (recommended threshold: 180,000 ms / 3
minutes):

```bash
--xla_tpu_debug_sflag_wait_timeout_ms=180000
--xla_tpu_debug_sc_sflag_wait_timeout_ms=180000
```

To halt the program immediately upon detecting a timeout:

```bash
--xla_tpu_debug_sflag_wait_shalt_on_detection=true
```

#### Disambiguating False Positives and Cascades

In multi-core workloads, early launch barriers or collective propagation can
trigger spurious timeout reports on downstream cores:

-   Run with halting disabled and log recorders enabled:

    ```bash
    --xla_tpu_debug_sflag_wait_shalt_on_detection=false
    --xla_tpu_enable_log_recorder=true
    --xla_tpu_enable_sc_log_recorder=true
    ```
-   **Filter Flag Categories**: Scope instrumentation to specific
    synchronization paths using:

    ```bash
    --xla_tpu_sflag_busy_wait_filter=local|remote|all|barriers
    ```
-   **Barrier vs. DMA Waits**: If some chips timeout on barriers while others
    timeout on normal DMAs, the DMA timeouts are primary and the barrier waits
    are secondary cascades. Map the reported HLOs against the program control
    flow graph to find the earliest stalled operation.

### 4. Hierarchical Program Progress Tracking (HPPT)

Because different cores take different branches in conditionals and execute
varying loop iterations, simple monotonic counters cannot be correlated across
cores. The Hierarchical Program Progress Tracker (HPPT) maintains a hierarchy of
`{iteration, counter}` pairs across macro-level control flow structures.

#### Macro Progress Tracking & Heartbeat

Enable HPPT with hierarchical depth levels (minimum 8) and configure a periodic
heartbeat:

```bash
--xla_tpu_enable_progress_tracker=8
--xla_tpu_progress_tracker_heartbeat='0.5'
```

-   When a hang occurs, the core stuck at the **lowest depth level** is the
    primary culprit.
-   Cores reporting timeouts at higher depth levels are secondary victims
    blocked on collective dependencies.

#### Fine-Grained LLO Progress Tracking

To track intra-kernel communication points (such as barrier waits and remote DMA
enqueues) within collective HLOs:

```bash
--xla_tpu_llo_progress_tracker=true
```

The LLO tracker pushes an LLO boundary marker (`0x7FFFFFFF` / `2147483647`) onto
the tracker stack, followed by intra-kernel loop levels and a 32-bit `meta_id`
(e.g., `HPPT_DMA_GENERAL`) identifying the active low-level primitive.

### 5. LLO Runtime Bounds Checking

Memory corruption, buffer overruns, and invalid memory dereferences in vector
load/store instructions or DMA transfers cause unexpected halts. The LLO
compiler provides runtime bounds checking:

#### Enable Bounds Checking

Enable runtime bounds checking across all supported memory operations:

```bash
--xla_jf_bounds_check=true
```

For strided DMA operations:

```bash
--xla_jf_bounds_check_stride=true
```

*Static Analysis Limitations*: LLO bounds checking relies on static dataflow
analysis to determine candidate buffer allocations for each pointer register.
When pointer arithmetic involves raw integer-to-pointer casts (`kIntToPtr`) or
complex indirect addressing that defeats static allocation tracking, bounds
checks may not be emitted or may require inspecting
`--vmodule=deepsea_compiler_base=3` logs.

#### Diagnosing Bounds Check Halts

When an out-of-bounds access occurs, the sequencer halts with an annotated
message:

```
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:336:0x840 (from TensorCoreSequencer:336:0xda98): BoundsCheck 5114 [deref of %s4242] for %4249 = dma.hbm_to_vmem ...
```

To log exact dereference addresses, candidate allocation bases, and limit
calculations:

```bash
--xla_tpu_enable_log_recorder=true
--xla_tpu_impure_bounds_check_log_failures=5114
```

Increase compiler diagnostic verbosity if needed:

```bash
--vmodule=deepsea_compiler_base=3 --xla_jf_debug_level=2
```

#### Limiting Bounds Checking Overhead

Full bounds checking roughly doubles instruction bundle count and can trigger
host or HBM OOM. Limit checking using a RangeSpec:

-   By opcode class: `--xla_jf_bounds_check=DMA`, `--xla_jf_bounds_check=VMEM`,
    or `--xla_jf_bounds_check=HBM`.
-   By exclusion: `--xla_jf_bounds_check=-vst.msk` (excludes masked vector
    stores).
-   By clause ranges: `--xla_jf_bounds_check=100:200,-140:160`.

### 6. Asynchronous Runtime Diagnostics (`libtpu.sdk.debug` Callbacks)

In `libtpu >= 0.0.46`, Python workloads can register an asynchronous debug
callback via `libtpu.sdk.debug.register_debug_callback(callback_fn)` (and clean
up with `libtpu.sdk.debug.unregister_debug_callback(callback_id)`) to receive
structured runtime events (`payload` and `metadata` mappings) directly from
`libtpu` without parsing stderr logs:

-   **`STALL_EVENT`** (enabled by default): Emitted by
    `TpuDiagnosticCoordinator` ~10s after a stall trigger (`source` enum:
    `PJRT_WATCHDOG=1`, `HAL_WATCHDOG=2`, `SFLAG_TIMEOUT=3`, `CORE_STALL=4`,
    `HARDWARE_WEDGE=5`, or `0` for manual trigger), providing per-core
    `core_states` (`sequencer_info` with `pc` and `hlo_location`) and
    `serialized_artifact` bytes.
-   **`SFLAG_TIMEOUT`**: Triggered when `is_sflag_timeout: True` (requires
    `--xla_tpu_debug_sflag_wait_timeout_ms > 0`,
    `--xla_tpu_debug_sflag_wait_shalt_on_detection=false`, and
    `--xla_tpu_enable_log_recorder=true`).
-   **`SC_MISMATCH`**, **`SDC_NAN_INF`**, **`SDC_OUTPUT_MISMATCH`**,
    **`HLO_LOGGER`**, **`FUSION_DEBUG`**, and **`CORE_DUMP`**: Deliver runtime
    tensor literals (`payload["literal"]`), `deduplicated_names`, and core dump
    register values (`core_dump_tc_values`, `core_dump_sc_values`).

### 7. Automated TPU Hang Detection (`TpuDiagnosticCoordinator`)

In `libtpu >= 0.0.46`, each TPU host runs a `TpuDiagnosticCoordinator` fed by
three watchdogs:

1.  **PJRT Event Tracker (`PendingEventLogger`)**: Enabled by default (5-minute
    threshold via `--pjrt_tpu_event_tracker_slow_operation_threshold=5m`).
2.  **Driver Watchdog (`--tpu_driver_callback_watchdog_timeout`)**: Monitors HAL
    driver callbacks (disabled by default on Cloud TPU VMs).
3.  **LLO Sync Flag Watchdog (`--xla_tpu_debug_sflag_wait_timeout_ms`)**:
    Hardware timer baked into compiled sync flag waits.

When triggered, the coordinator waits 10s
(`--xla_tpu_diagnostic_aggregation_window=10s`), harvests per-core state and up
to 5 active programs, writes `tpu_stall_summary_*.pb`,
`tpu_compiler_metadata_*.pb`, and optional `tpu_core_dump_*.bin`
(`--xla_tpu_diagnostic_collect_core_dumps=true`) to
`--xla_tpu_impure_diagnostic_artifacts_dir` (default `/tmp` or a `gs://` bucket
for multi-host Pods), and enters a 15-minute cooldown
(`--xla_tpu_diagnostic_cooldown_duration=15m`).

## Mitigations and Resolution Strategies

1.  **Kernel and HLO Buffer Adjustments**: If LLO bounds checking reveals an
    access exceeding `alloc_limit_words`, correct the slice indexing, grid
    tiling, or pad shapes in the upstream kernel definition (e.g. Pallas or
    custom call).
2.  **Compiler Optimization Barriers**: When optimization passes generate
    invalid fusions or instruction reorderings that cause race conditions,
    insert `jax.lax.optimization_barrier` between operations to inhibit fusion.
    See [[compile-time-crash-reporting]].
3.  **Hardware Isolation**: If core dump anomalies show unrecoverable
    `[HARDWARE]` error codes without software bounds violations, quarantine the
    faulty host or node using [[isolate-node-via-rr-label]] on GKE.

## Connections

-   **Workflows**:
    -   [[tpu-state-analysis]]
    -   [[ici-straggler-triage]]
-   **Symptoms**:
    -   [[wait-timeout-on-sflag]]
    -   [[worker-hardware-fatal-error]]
    -   [[megascale-hang]]
-   **Playbooks**:
    -   [[compile-time-crash-reporting]]
    -   [[hang-diagnosis-methodology]]
    -   [[multi-chip-hang-investigation]]
    -   [[tpu-state-analysis-methodology]]
-   **Actions**:
    -   [[diagnose-sflag-hang]]
    -   [[inspect-hlo-dumps]]
    -   [[analyze-sflag-timeouts]]
-   **Entities**:
    -   [[tensorcore]]
    -   [[sparsecore]]
    -   [[xla]]
    -   [[cloud-tpu]]

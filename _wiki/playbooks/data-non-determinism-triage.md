---
title: "Data Non-Determinism and SDC Triage Playbook"
description: "Heuristic playbook for detecting, localizing, and diagnosing data non-determinism, silent data corruptions (SDC), unexpected NaNs/Infs, and SparseCore vs TensorCore divergences using XSC and mismatch detectors."
type: playbook
internal_only: false
tags: [tpu, hardware_fault, triage, methodology, analysis]
created: 2026-09-28
updated: 2026-09-28
---


# Data Non-Determinism and SDC Triage Playbook

## Overview

When TPU programs execute without fatal driver crashes or hangs but produce
non-deterministic outputs across identical inputs, unexpected NaNs or
Infinities, or mathematical divergence between hardware engines, the issue is
typically classified as either **Silent Data Corruption (SDC)** or an **XLA
compiler / kernel bug**.

Investigating data non-determinism and SDC proceeds through three main phases:

1.  **Localization**: Isolating the divergence or corruption to a specific
    High-Level Optimizer (HLO) instruction, fusion computation, or custom kernel
    (e.g. Pallas) on a specific core.
2.  **Debugging**: Determining whether the failure signature stems from physical
    hardware faults (bad chips) or software/compiler defects, and examining
    inputs, outputs, and intermediate states.
3.  **Fix / Mitigation**: Replacing faulty hardware slices, adjusting model
    precision, disabling problematic compiler fusions, or modifying custom
    kernels.

Localization is typically the most difficult stage. This playbook details the
configuration and usage of the **XLA Silent Data Corruptions Checker (XSC)** and
the **SparseCore / TensorCore Mismatch Detector**.

## Diagnostic Steps and Triage Workflows

### 1. Identify the Primary Symptom

Categorize the observed numerical anomaly into one of three primary failure
categories:

-   **Data Non-Determinism**: Observed program metrics (such as loss, weights,
    or activations) diverge across multiple identical runs on identical inputs,
    or XSC reports an output checksum mismatch.
-   **Unexpected NaNs / Infs**: Floating-point NaN (Not a Number) or Infinity
    values appear during forward/backward training passes or within intermediate
    layer outputs.
-   **SparseCore / TensorCore Divergence**: Numerical deviations appear only
    when SparseCore offloading is active for collective communication or
    embedding operations.

--------------------------------------------------------------------------------

### 2. XLA Silent Data Corruptions Checker (XSC)

**Silent Data Corruption (SDC)** is an event where the result of an operation on
a TPU is corrupted—that is, the output does not conform to the inputs and
operation performed. There are two types of SDCs:

-   **Non-deterministic SDC**: Execution of the exact same instructions on the
    same data yields different outputs across runs (e.g.,
    `2 + 2 (sometimes) != 4`).
-   **Deterministic SDC**: The result is skewed consistently in the same way
    (e.g., `2 + 2 (always) == 42`).

**XSC** is a dedicated compiler instrumentation framework within XLA that
instruments HLO operations on the accelerator to execute repeatedly and validate
outputs via checksums (catching both deterministic and non-deterministic
corruptions) or scan for invalid floating-point values (`NaN`/`Inf`).

#### Basic Configuration: HLO Re-Run and Checksums

Enable XSC instrumentation on TensorCore by passing the following compiler flag
to the TPU worker environment:

```bash
--xla_tpu_enable_sdc_checker=true
--xla_tpu_sdc_check_input_identical=true
```

By default, XSC instruments non-trivial HLOs to execute twice ($N=1$ additional
repeat) and asserts that output data checksums match across runs. Setting
`--xla_tpu_sdc_check_input_identical=true` also verifies that the inputs to the
HLO remain identical across repeated executions, catching in-place input buffer
mutations or unintended aliasing.

> [!WARNING] **Performance Overhead**: XSC introduces significant execution
> overhead (typically 1.8x to 2.5x slowdown on standard models). It is intended
> as an offline diagnostic tool or for isolated diagnostic runs of a few
> training steps, rather than continuous production training.

#### Increasing Repeat Count

Rare or intermittent hardware bit-flips and race conditions may require
increasing the number of re-runs ($N > 1$):

```bash
--xla_tpu_sdc_check_repeat_count=N
```

*Note*: When $N > 0$, XSC cannot instrument asynchronous operations (e.g.
`all-reduce-start` / `all-reduce-done`) because asynchronous operation handles
cannot be initiated or completed multiple times.

#### Silent Reporting Mode

By default, XSC immediately halts program execution upon the first detected SDC
event. To collect multiple SDC instances across the entire execution graph and
study distribution patterns, enable silent reporting:

```bash
--xla_tpu_sdc_check_halt_on_detection=false
```

When set to `false`, the TPU worker logs the SDC event details to standard logs
and continues program execution.

#### Fine-Grained HLO Filtering

To avoid instrumenting the entire HLO graph or to narrow down suspect fusions,
apply regex filtering:

```bash
--xla_tpu_sdc_checker_extra_filter="<filter_expression>"
```

Filter syntax:

-   Clauses are separated by `;`.
-   Allowlist rules start with `+` (or no prefix); denylist rules start with
    `-`.
-   Rules are evaluated case-insensitively in order.
-   `*` matches all HLO names; `tlp` represents Top-Level Program outside HLOs.
-   Example: `--xla_tpu_sdc_checker_extra_filter="all-reduce.*;-"` (instruments
    only AllReduce operations).
-   Example:
    `--xla_tpu_sdc_checker_extra_filter="-all-reduce.123;all-reduce.*;-"`
    (instruments all AllReduces except `all-reduce.123`).

#### Disambiguating Deduplicated HLOs

When XLA detects identical subcomputations, it compiles them once and reuses the
emitted code across all instances. When an SDC occurs, the log references all
deduplicated HLO names. To force unique compilation and isolate the exact
failing instance:

```bash
# Force unique compilation for all HLOs (may increase compile time and binary size):
--xla_jf_hlo_deduplication_all_unique=true

# Or selectively disable deduplication for specific suspect fusions:
--xla_jf_hlo_deduplication_all_unique_filter='fusion25.*'
```

#### Handling Input/Output Aliasing and Checksum Algorithms

-   **Aliasing**: XSC does not copy inputs prior to re-execution and assumes
    input buffers are unmodified. For HLOs where inputs and outputs share
    physical memory buffers, disable aliasing:

    ```bash
    --xla_tpu_disable_aliasing_extra_filter="<regex>"
    ```
-   **Checksum Algorithm**: XSC defaults to a fast bitwise `xor` algorithm. For
    high-entropy tensors where XOR may miss subtle multi-bit compensations,
    select SipHash:

    ```bash
    --xla_tpu_sdc_checker_checksum_algo='siphash13'
    ```
-   **Double-Checksumming**: To verify that memory buffers are not corrupted
    concurrently during checksum calculation itself, enable:

    ```bash
    --xla_tpu_sdc_checker_double_check_checksums=true
    --xla_tpu_sdc_check_inputs=true
    --xla_tpu_sdc_check_repeat_count=0
    ```

--------------------------------------------------------------------------------

### 3. Root Cause Analysis: Bad Hardware vs. XLA Compiler Bugs

When SDC events or checksum mismatches are logged, analyze the **spatial
locality** and **operation locality** across workers:

| Diagnostic Dimension | Physical Hardware Fault (Bad Chip) | XLA Compiler / Kernel Bug |
| :------------------- | :-------------------------- | :---------------------- |
| **Chip Locality** | Concentrated on a single chip or a specific TensorCore (`e.g. 0x0x0_TC0` or `3x3x1_TC0`). | Distributed uniformly across multiple chips and hosts in the TPU slice. |
| **Instruction Locality** | Occurs across diverse, unrelated HLO operations (fusions, convolutions, dynamic slices) on the same core. | Occurs consistently on the exact same HLO instruction or specific fusion pattern across different cores. |
| **Reproducibility** | Intermittent or depends on thermal/voltage load on the specific physical accelerator. | Deterministic given identical input data and compiler optimization flags. |
| **Recommended Action** | Quarantine and replace the faulty node via [[node-isolation-methodology]]. | Isolate the minimal HLO reproducer, file an XLA defect, or apply optimization barriers. |

--------------------------------------------------------------------------------

### 4. Debugging NaNs and Infs with XSC

XSC provides specialized floating-point verification to pinpoint where
non-finite numbers first enter the computation graph:

#### Configuration for NaN/Inf Detection

```bash
--xla_tpu_enable_sdc_checker=true
--xla_tpu_sdc_check_repeat_count=0
--xla_tpu_sdc_check_halt_on_detection=false
--xla_tpu_sdc_checker_enable_nan_checking=true
```

#### Isolating Root-Cause NaNs from False Positives

NaN debugging frequently encounters false positives: situations where NaNs or
Infs are intentionally generated by intermediate operations and subsequently
masked or discarded before user visibility:

1.  **Bitcasts and Padding**: A tensor is bitcast or padded with Infs/NaNs in
    out-of-bounds regions, processed by a subgraph, and later sliced back down
    to valid boundaries.
2.  **Dynamic Slices**: An intermediate tensor contains Inf padding for
    numerical masking (e.g. attention masks), which is normal.
3.  **Ragged Kernels**: Custom Pallas kernels intentionally leave implicit
    padding uninitialized.

**Techniques to Filter False Positives**:

-   **Track Inputs**: Check HLO inputs to detect if NaNs were already present
    upstream:

    ```bash
    --xla_tpu_sdc_check_inputs=true
    ```
-   **Report New NaNs Only**: Restrict reporting strictly to HLOs that produce
    NaNs/Infs on outputs when inputs were clean:

    ```bash
    --xla_tpu_sdc_check_inputs=true
    --xla_tpu_sdc_checker_new_nans_only=true
    ```

    *(Caution: If an HLO failed to properly mask out padding NaNs from its
    input, this filter will omit it since NaNs were already present on input).*
-   **Poison Uninitialized Padding**: To catch operations reading uninitialized
    memory regions, force buffer padding to NaN (or a custom integer
    bit-pattern):

    ```bash
    --xla_tpu_sdc_checker_poison_padding=true
    --xla_tpu_sdc_checker_poison_padding_value=... # optional, NaNs by default
    ```
-   **Disable Custom Calls**: Custom kernels that intentionally emit
    uninitialized padding can be excluded:

    ```bash
    --xla_tpu_sdc_checker_enable_tpu_custom_calls=false
    ```

--------------------------------------------------------------------------------

### 5. SparseCore / TensorCore Mismatch Detector

When numerical instability or data corruption appears only when SparseCore
offloading is active (e.g. embedding tables or offloaded collective
communication), use the **SparseCore / TensorCore Mismatch Detector**.

The detector duplicates the operation across both SparseCore (SC) and TensorCore
(TC), compares the resulting tensors using `jax.numpy.allclose` ($|a - b| \le
\text{atol} + \text{rtol} \times |b|$), and reports discrepancies.

#### Supported Operations

-   **Collectives**: `all-reduce`, `all-gather`, `reduce-scatter`
-   **Compute Operations**: `sort`, `gather`/`scatter` fusions

#### Enabling the Detector

You must enable the LLO log recorder in conjunction with the mismatch detector:

```bash
--xla_tpu_enable_sparse_core_mismatch_detector=true
--xla_tpu_enable_log_recorder=true
```

#### Operational Flags

-   **Target Specific Operations**:

    ```bash
    --xla_tpu_sparse_core_mismatch_detector_on_specified_ops='all-gather'
    ```
-   **Immediate Halt vs. Silent Reporting**:

    ```bash
    # Crash immediately upon first mismatch (identifies first offending instruction):
    --xla_tpu_sparse_core_mismatch_detector_crash_on_mismatch=true

    # Exhaustive logging without crashing:
    --xla_tpu_sparse_core_mismatch_detector_crash_on_mismatch=false
    ```
-   **Tolerances**: For non-arithmetic ops like `all-gather`, tolerances default
    to zero (bit-exact matching). For arithmetic operations, specify tolerances:

    ```bash
    --xla_tpu_sparse_core_mismatch_detector_atol=1e-5
    --xla_tpu_sparse_core_mismatch_detector_rtol=1e-4
    ```

#### SparseCore HLO Re-Run

TensorCore instrumentation cannot re-run `*-start` or `*-done` handles of
offloaded collectives. To enable native HLO re-run directly on SparseCore:

```bash
--xla_tpu_enable_sc_sdc_checker=true
--xla_tpu_enable_sc_sdc_checker_loop_iteration_count=N
```

--------------------------------------------------------------------------------

## Mitigations and Resolution Strategies

1.  **Hardware Isolation**: When SDC events concentrate on a single chip or host
    across multiple HLO types, the root cause is physical hardware failure.
    Quarantine and replace the node using [[node-isolation-methodology]].
2.  **Compiler Optimization Barriers**: When SDCs occur across chips on
    identical fusions, insert `jax.lax.optimization_barrier` around suspect
    fusions to prevent problematic compiler transformations.
3.  **Pallas Kernel Corrections**: If custom kernels produce NaNs or
    non-deterministic uninitialized memory reads, explicitly zero output buffers
    or adjust tensor padding bounds.

--------------------------------------------------------------------------------

## Compiler Flags Quick Reference

Flag                                                        | Default | Description
:---------------------------------------------------------- | :------ | :----------
`--xla_tpu_enable_sdc_checker`                              | `false` | Enables XSC instrumentation on TensorCore.
`--xla_tpu_sdc_check_repeat_count`                          | `1`     | Number of additional HLO executions (0 disables re-runs).
`--xla_tpu_sdc_check_halt_on_detection`                     | `true`  | When `false`, logs SDC events silently without halting execution.
`--xla_tpu_sdc_checker_extra_filter`                        | `""`    | Regex filter defining which HLOs are instrumented.
`--xla_jf_hlo_deduplication_all_unique`                     | `false` | Disables HLO deduplication across the entire module.
`--xla_jf_hlo_deduplication_all_unique_filter`              | `""`    | Disables deduplication for HLOs matching regex.
`--xla_tpu_sdc_checker_enable_nan_checking`                 | `false` | Enables checking HLO outputs for NaNs and Infs.
`--xla_tpu_sdc_check_inputs`                                | `false` | Enables checking HLO inputs for checksums or NaNs/Infs.
`--xla_tpu_sdc_checker_new_nans_only`                       | `false` | Reports only HLOs where output has NaNs but inputs were finite.
`--xla_tpu_sdc_checker_poison_padding`                      | `false` | Fills memory padding with NaNs to catch uninitialized reads.
`--xla_tpu_sdc_checker_poison_padding_value`                | `NaN`   | Custom value used to pad output buffers when poison padding is enabled.
`--xla_tpu_enable_sparse_core_mismatch_detector`            | `false` | Enables SC vs TC redundant execution and comparison.
`--xla_tpu_enable_log_recorder`                             | `false` | Enables LLO log recorder (required for mismatch detector).
`--xla_tpu_sparse_core_mismatch_detector_on_specified_ops`  | `""`    | Filters which operations are checked by SC/TC mismatch detector.
`--xla_tpu_sparse_core_mismatch_detector_crash_on_mismatch` | `false` | When `true`, halts immediately on first SC/TC mismatch.
`--xla_tpu_enable_sc_sdc_checker`                           | `false` | Enables HLO re-run checker directly on SparseCore.

## Connections

-   **Workflows**:
    -   [[sdc-and-numerics-triage]]
    -   [[general-ml-debugging]]
-   **Actions**:
    -   [[triage-sdc-and-numerics]]
    -   [[analyze-sparsecore-offload]]
    -   [[inspect-hlo-dumps]]
    -   [[query-workload-logs]]
-   **Playbooks**:

    -   [[on-device-debugging-triage]]
    -   [[compile-time-crash-reporting]]
    -   [[numerics-error-triage]]
    -   [[sparsecore-debugging-triage]]
    -   [[hardware-failure-modes]]
    -   [[node-isolation-methodology]]

-   **Symptoms**:

    -   [[sdc-checksum-mismatch]]
    -   [[mxla-hardware-fault]]
    -   [[wait-timeout-on-sflag]]

-   **Concepts**:

    -   [[ecc-errors]]
    -   [[tpu-logging-flags]]

-   **Entities**:

    -   [[cloud-tpu]]
    -   [[tensorcore]]
    -   [[sparsecore]]
    -   [[xla]]
    -   [[jax]]

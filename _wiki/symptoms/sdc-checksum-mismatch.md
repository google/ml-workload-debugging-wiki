---
title: "SDC Checksum Mismatch"
description: "Symptom signature for on-device Silent Data Corruption (SDC) events, checksum mismatches during repeated HLO execution, and SparseCore vs TensorCore divergence."
type: symptom
internal_only: false
tags: [tpu, hardware_fault, triage, analysis]
created: 2026-09-28
updated: 2026-09-28
---


# SDC Checksum Mismatch

## Overview

The **SDC Checksum Mismatch** symptom manifests when on-device accelerator
computation produces inconsistent or corrupt numerical data across repeated
executions on identical inputs. Unlike hardware execution halts, ECC panics, or
memory bounds check violations that immediately crash the TPU driver, Silent
Data Corruption (SDC) proceeds without a hardware exception unless detected by
the **XLA Silent Data Corruptions Checker (XSC)** or the **SparseCore /
TensorCore Mismatch Detector**.

When instrumented with XSC or mismatch detection flags, the TPU runtime halts or
logs an explicit SDC event fingerprint detailing the physical host, chip
coordinate, TensorCore/SparseCore ID, and the culprit HLO instruction.

## Log Signatures

### 1. XSC Checksum Mismatch Signature

When repeated HLO execution yields different output bit patterns:

```
[XLA_SDC_CHECKER] chip [3x3x1_TC0] host [node-15] ::> OUTPUT data checksum does not match :: fusion.435, HLO ::
```

Or when on-device halt is triggered:

```
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:7:0x27d4 (from TensorCoreSequencer:7:0xa88c): SDC event :: OUTPUT data checksum does not match :
```

Key signature fields:

-   `host [node-15]`: Physical machine host identifier within the slice.
-   `OUTPUT data checksum does not match`: The authoritative SDC event
    fingerprint.
-   `fusion.435`: The specific HLO instruction or fusion where the checksum
    diverged.
-   `chip [3x3x1_TC0]`: Topological coordinates of the chip and the specific
    TensorCore execution engine.

### 2. XSC NaN / Inf Detection Signature

When non-finite values are detected by XSC:

```
[XLA_SDC_CHECKER] chip [0x0x0_TC0] host [worker-4] ::> NaN/Inf detected in OUTPUT :: %dot.14 = f32[128,128] dot(%lhs, %rhs)
```

### 3. SparseCore / TensorCore Mismatch Signature

When SparseCore offloading is active and SC output diverges from TensorCore
reference execution:

```
Found Mismatch in offloaded op all-gather.2.cloned.call-start: atol=0 rtol=0 max_abs_diff=1.4285e-03
```

## Verification Steps

1.  **Verify if SDC Checker is Active**: Confirm whether
    `--xla_tpu_enable_sdc_checker=true` (and optionally
    `--xla_tpu_sdc_check_input_identical=true` to also verify that HLO inputs
    remain identical across repeated executions and catch in-place input buffer
    mutations) was supplied in the job launch flags. If uninstrumented, SDC may
    surface downstream as loss divergence, NaN model weights, or evaluation
    score collapse.
2.  **Determine Hardware vs. Compiler Locality**:
    -   Query all worker logs for `OUTPUT data checksum does not match`.
    -   **Hardware Fault (Bad Chip)**: All SDC events originate from the same
        chip (e.g. `3x3x1_TC0`) or host across varied, unrelated HLO
        instructions (fusions, convolutions, elementwise ops).
    -   **XLA Compiler Bug**: SDC events are distributed across multiple
        different chips or hosts in the cluster, but always target the same HLO
        instruction or fusion name.
3.  **Inspect Deduplicated HLOs**: If the log lists multiple deduplicated HLO
    candidates, recompile with `--xla_jf_hlo_deduplication_all_unique=true` or
    filter by suspect fusion to pinpoint the single offending instruction.
4.  **Distinguish Primary SDC from False Positives**: If NaNs/Infs are logged,
    check whether `--xla_tpu_sdc_checker_new_nans_only=true` was enabled to
    confirm the HLO created the NaN rather than propagating upstream padding.

## Playbooks & Root Causes

-   **Root Causes**:
    -   **Physical Hardware Fault**: Marginal transistor, HBM read bit-flip, or
        timing degradation on a specific TPU core causing non-deterministic
        arithmetic execution.
    -   **XLA Compiler Optimization Bug**: Aggressive instruction fusion,
        vectorization bug, or uninitialized memory register reuse in
        compiler-generated LLO code.
    -   **SparseCore Offload Divergence**: Precision mismatch or collective
        communication logic bug between SparseCore sequencer and TensorCore.
-   **Playbook Guidance**:
    -   Follow [[data-non-determinism-triage]] for flag configuration, false
        positive filtering, and repeated execution triage.
    -   If isolated to a physical bad chip, isolate and decommission the node
        using [[node-isolation-methodology]].
    -   If isolated to a compiler bug, follow [[compile-time-crash-reporting]]
        or [[numerics-error-triage]] to gather HLO dumps and construct a minimal
        reproducer.

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
    -   [[data-non-determinism-triage]]
    -   [[numerics-error-triage]]
    -   [[sparsecore-debugging-triage]]
    -   [[on-device-debugging-triage]]
    -   [[hardware-failure-modes]]
    -   [[node-isolation-methodology]]
-   **Symptoms**:
    -   [[mxla-hardware-fault]]
    -   [[slow-step-time]]
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

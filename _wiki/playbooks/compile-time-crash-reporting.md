---
title: "Reporting JAX and XLA Compile-Time Crashes on TPU"
description: "Standard operating procedure for isolating, reporting, and escalating JAX/XLA compiler crashes to the XLA:TPU team, capturing C++ stack traces, minimal reproducers, HLO dumps, and applying optimization barriers."
type: playbook
internal_only: false
tags: [tpu, triage, methodology]
created: 2026-09-28
updated: 2026-09-28
---


# Reporting JAX and XLA Compile-Time Crashes on TPU

## Overview

Compile-time crashes occur during graph tracing, lowering, or XLA optimization
passes before program binaries execute on TPU hardware. These issues typically
present as unhandled Python exceptions wrapping C++ compiler errors (such as
`INTERNAL`, `Check failed:`, segment violations, or assertion aborts inside
compiler passes).

Resolving compiler crashes requires systematic information gathering so compiler
teams can reproduce the defect independently of proprietary models, customer
data, or complex cluster environments.

## Diagnostic and Reporting Procedures

### 1. Capture Full Error Output and C++ Stack Trace

When compilation aborts, capture the complete console output from the root
process (typically JAX Process 0 or Worker 0):

-   **Full Traceback**: Capture stdout and stderr in their entirety. The C++
    stack trace identifies the specific HLO pass, lowering pipeline, or module
    transformation running at the point of failure (e.g., instruction fusion,
    layout assignment, or memory scheduling).
-   Save the raw output to a text file for attachment.

### 2. Generate HLO Dumps (Highest Priority)

XLA operates on the High Level Optimizer (HLO) intermediate representation. An
HLO dump allows compiler engineers to reproduce compiler behavior directly
without requiring access to the original JAX source code or model weights.

Configure environment variables before launching execution:

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
```

The output directory generates HLO files for each compilation stage along with a
compiler environment file (`*tpu_comp_env.txt`).

**Cloud TPU Storage Considerations**:

-   On Cloud TPU VMs or GKE pods, configure remote cloud storage (e.g., GCS
    bucket via [[cloud-storage]]) or tar and download the dump directory before
    worker teardown.
-   Ensure the `tpu_comp_env.txt` file is included in the dump bundle, as it
    records active compilation flags and target hardware architectures.

### 3. Construct a Minimal Reproducer

Whenever feasible, build a standalone, self-contained Python script reproducing
the crash:

-   Isolate the specific `@jax.jit`, `jax.pmap`, or transformation that triggers
    the compiler failure.
-   Strip external dataset dependencies, substituting dummy arrays using
    `jax.random` or static shape mocks.
-   Minimize input shapes and tensor dimensions while preserving the failure
    mode.

A minimal reproducer dramatically accelerates triage and fix verification.

### 4. File the Crash Report

Package the gathered artifacts:

1.  **Description**: Concise summary of the failure, framework versions (JAX,
    jaxlib, libtpu), and accelerator generation.
2.  **Full Console Log**: Complete error message and C++ stack trace.
3.  **HLO Dump Directory**: Compressed tarball or storage bucket link containing
    HLO passes and `tpu_comp_env.txt`.
4.  **Minimal Reproducer**: Standalone script triggering the bug.

**Issue Destination**:

-   For open-source JAX and Cloud TPU users: file an issue on
    [google/jax GitHub Issues](https://github.com/google/jax/issues) or
    [openxla/xla GitHub Issues](https://github.com/openxla/xla/issues).

### 5. Advanced Analysis Tools (Optional)

Before escalating, advanced users can further localize problematic instructions:

-   **HLO Instruction Isolation**: OpenXLA instruction isolation tools accept an
    HLO graph and run individual instructions across comparative backends (such
    as TPU vs. `HloEvaluator`), dumping isolated instructions causing mismatches
    or exceptions.

## Workarounds and Mitigations

### Optimization Barriers (`jax.lax.optimization_barrier`)

Many compiler crashes stem from complex interactions between specific
optimization passes (such as aggressive instruction fusion, expression
reordering, or memory layout rewrites).

An optimization barrier (`OptimizationBarrier`) prevents the compiler from
moving operations across the barrier boundary, inhibiting fusions and
reassociations:

```python
import jax
import jax.numpy as jnp

# Strategic placement of optimization barrier
x = suspect_computation(a, b)
x = jax.lax.optimization_barrier(x)
y = downstream_computation(x)
```

**Diagnostic and Mitigation Value**:

-   If inserting an optimization barrier resolves the crash, this proves the bug
    resides in a specific fusion or scheduling pass rather than upstream graph
    construction.
-   Documenting that an optimization barrier bypasses the crash provides
    critical diagnostic evidence for compiler engineers and acts as a temporary
    unblocking workaround while a permanent compiler fix is implemented.

## Connections

-   **Symptoms**:
    -   [[hbm-oom]]
    -   [[workload-startup-crash]]
-   **Playbooks**:
    -   [[on-device-debugging-triage]]
    -   [[debugging-gotchas]]
    -   [[general-ml-debugging]]
-   **Actions**:
    -   [[inspect-hlo-dumps]]
-   **Entities**:
    -   [[xla]]
    -   [[jax]]
    -   [[cloud-tpu]]
    -   [[cloud-storage]]

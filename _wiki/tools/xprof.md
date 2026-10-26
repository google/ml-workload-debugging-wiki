---
title: XProf (XLA Profiler)
description: Accelerator profiling and visualization suite for XLA, JAX, and PyTorch, featuring Trace Viewer and Memory Viewer for TPU and GPU execution.
type: tool
internal_only: false
tags: [tpu, gpu, observability, training]
created: 2026-09-28
updated: 2026-09-28
---


# XProf (XLA Profiler)

## Overview

**XProf** (XLA Profiler) is the core diagnostic and performance profiling suite
for accelerator workloads (TPUs and GPUs) running JAX, PyTorch/XLA, and
TensorFlow. It allows developers and operators to visualize execution timelines,
inspect hardware utilization, profile memory buffer lifespans, and isolate
performance bottlenecks across distributed clusters.

XProf provides two primary diagnostic tools for ML workload triage:

1.  **Trace Viewer**: Displays synchronized timelines of host threads, runtime
    operations, DMA transfers, and device core execution.

2.  **Memory Viewer**: Visualizes High Bandwidth Memory (HBM) utilization at
    peak allocation points, breaking down buffers by inputs, outputs, and
    temporaries.

## Usage & Commands

### 1. Capturing a Profile

In JAX workloads, capture traces programmatically using the `jax.profiler.trace`
context manager:

```python
import jax

# Exception-safe profiling context manager
with jax.profiler.trace("/tmp/xprof"):
  step_result = train_step(state, batch)
```

> [!TIP] Always prefer `jax.profiler.trace` over `start_trace`/`stop_trace`
> because the context manager guarantees profile finalization and flush even if
> the enclosed code encounters an exception or OOM error.

### 2. Launching the XProf Server

Install the standalone Python package and start the visualization dashboard:

```bash
pip install xprof
xprof --logdir=/tmp/xprof/ --port=6006
```

Open a browser and navigate to `http://localhost:6006` (or set up port
forwarding if working in remote cloud environments).

## Troubleshooting Scenarios

### 1. Steady-State MapDMA Buffer Shortage Analysis

In large-scale Megascale workloads experiencing unexplained step-time
degradation:

-   Open **Trace Viewer** in XProf.
-   Search for the worker timeline or search specifically for the string
    `MapDmaBuffer`.
-   Verify whether `MapDmaBuffer` calls occur during steady-state execution
    steps. If present, the premapped host buffer pool is undersized.
-   Inspect the Megascale "Communication Transport" slices for fallback memory
    copies.
-   Refer to [[megascale-performance-tuning]] for remediation steps.

### 2. Accelerator HBM Out Of Memory (OOM) Triage

When an application fails with `RESOURCE_EXHAUSTED` (Error Code `E1000`):

-   Open **Memory Viewer** in XProf and select **HBM** under Memory Types.
-   Inspect the **HLO Ops at Peak Memory Allocation** chart to identify which
    tensors dominate peak memory.
-   Hover over buffer cards to evaluate dimension padding overhead (such as
    non-multiple-of-128 shapes).
-   Refer to [[hbm-oom-triage]] for mitigation strategies including
    micro-batching, sharding, and rematerialization.

## Connections

-   **Workflows**:
    -   [[hbm-oom-triage]]
    -   [[libtpu-timing-outlier-analysis]]
    -   [[general-ml-debugging]]
-   **Actions**:
    -   [[analyze-hbm-oom]]
    -   [[analyze-libtpu-timing]]
    -   [[inspect-hlo-dumps]]
    -   [[query-workload-logs]]
-   **Playbooks**:
    -   [[megascale-performance-tuning]]
    -   [[hbm-oom-triage]]
    -   [[multi-chip-hang-investigation]]
-   **Symptoms**:
    -   [[slow-step-time]]
    -   [[hbm-oom]]
-   **Concepts**:
    -   [[megascale]]
-   **Entities**:

    -   [[cloud-tpu]]
    -   [[tensorcore]]
    -   [[sparsecore]]
    -   [[jax]]
    -   [[xla]]

    -   [[cloud-logging]]

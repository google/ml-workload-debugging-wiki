---
title: Megascale Performance Tuning and MapDMA Buffer Triage
description: Playbook for diagnosing Megascale performance degradation, steady-state MapDMA buffer shortages, network transfer latency vs compute latency, and RPC latency timelines using XProf.
type: playbook
internal_only: false
tags: [tpu, training, hangs, networking, mxla]
created: 2026-09-28
updated: 2026-09-28
---


# Megascale Performance Tuning and MapDMA Buffer Triage

## Overview

In large-scale TPU training using [[megascale]] for collective communication,
performance degradation typically presents in one of three patterns:

1.  **Step-dependent slowness**: Sharp or gradual slowdown after a specific
    number of steps or restart attempts.
2.  **Consistent slowness**: Throughput is uniformly degraded compared to an
    established baseline.
3.  **Periodic slowness**: Recurring latency spikes occurring at regular
    intervals (e.g. coinciding with logging, checkpointing, or barrier
    synchronization).

When investigating slowness, first eliminate basic causes by checking compute
latencies on CPU, PCIe host-to-device transfers, and network slowness via logs.
If the collective end-to-end latency is larger than expected while individual
compute kernels appear normal, the bottleneck is typically host resource
contention or network communication stalls. XProf profiling is the definitive
tool to isolate the root cause.

## Diagnostic Steps

### 1. Capture an XProf Session

Capture an execution trace during a representative slow step window using JAX's
profiler:

```python
import jax

# Use the context manager form for exception safety
with jax.profiler.trace("/tmp/xprof"):
  # Execute several representative training steps
  train_step(batch)
```

Launch the XProf server to inspect the profile:

```bash
xprof --logdir=/tmp/xprof/ --port=6006
```

Navigate to `http://localhost:6006` and open the **Trace Viewer** tool.

### 2. Check for Shortage of Mapped DMA Buffers

The Megascale XLA runtime registers host memory buffers before they can be used
for DMA transfers to and from the TPU devices. This registration is intended to
occur **once**, early during process initialization.

-   In the XProf Trace Viewer, zoom into the timeline during **steady-state**
    execution (after the first few steps).
-   Search for the worker timeline and look for `MapDmaBuffer` calls.
-   *Tip*: Search for the exact worker hostname or filter specifically by the
    string `MapDmaBuffer`.
-   **Diagnosis**: Seeing `MapDmaBuffer` calls during steady-state execution
    confirms a buffer shortage. The runtime is forced to allocate and register
    host memory dynamically on the fly, introducing synchronous pipeline
    bubbles.

### 3. Check for Memory Copies during Network Transfers

Megascale XLA network transfers are designed to be zero-copy. When premapped DMA
buffer capacity is insufficient:

-   Inspect the Megascale **Communication Transport** traces in the Trace
    Viewer.
-   Look for unexpected memory copy operations (`memcpy` / buffer copy events)
    accompanying communication operations.
-   **Diagnosis**: The presence of memory copies in the communication transport
    indicates that data could not be placed directly into premapped DMA memory,
    degrading throughput.

### 4. Network and Collective Slack Analysis

Megascale provides an open-source
[Network Analysis Colab notebook](https://github.com/openxla/xla/blob/main/xla/megascale/tools/network_analysis_oss.ipynb)
(`network_analysis_oss.ipynb`) to analyze network performance from an XProf
trace:

-   **Transfer Latencies & Sizes**: Examine transfer latencies to identify
    potential network slowdowns or host slowness, and inspect transfer sizes to
    verify whether the workload uses a smaller number of larger transfers rather
    than a high volume of small transfers.
-   **Pending / Bursty Collectives**: Determine whether the workload produces
    bursty collectives or constantly maintains a large queue of pending
    collectives.
-   **Hardware Isolation via `{source, destination}` Pairs**: Examine
    `{source, destination}` worker pairs to identify possible bad network
    hardware on individual hosts.
-   **Small Collective Slack**: Inspect `recv-done` op traces. If `recv-done`
    ops exhibit long durations, or if collectives show zero or near-zero
    [slack time](https://cloud.google.com/tpu/docs/troubleshooting/troubleshoot-multislice#slack_time),
    the workload fails to overlap computation with communication.
-   **Bandwidth Demand**: Use the Network Analysis Colab to generate a network
    throughput timeline. If throughput exhibits consistently high usage
    throughout the step or specific regions with large spikes, the model is
    bandwidth-bound in those regions of the step function.
-   **RPC Latency Timeline**: If network bandwidth is not saturated but step
    time remains slow, use
    [Network Analysis](https://openxla.org/xla/megascale/debugging_workflow#network_analysis)
    to generate the RPC latency timeline. Look for constant or sporadic tail
    latencies (e.g., individual RPCs spiking to 200ms+). Sporadic high tail
    latencies point to MXLA RPC contention or TCP socket configuration issues
    rather than raw bandwidth saturation.

## Mitigations & Resolutions

### 1. Increase Premapped DMA Memory Region

If steady-state `MapDmaBuffer` calls or communication transport memory copies
are detected in XProf:

-   Increase the size of the premapped DMA memory region by raising the value of
    `--megascale_grpc_premap_memory_bytes`:

    ```bash
    --megascale_grpc_premap_memory_bytes=<larger_byte_size>
    ```
-   Restart the training job and capture a verification trace to confirm that
    `MapDmaBuffer` calls and transport memory copies no longer appear during
    steady-state steps.

### 2. Mitigate Low Collective Slack & Bandwidth Saturation

-   **Check Collective Slack**: Models with many collectives exhibiting low
    slack will experience bandwidth-bound regions.
-   **Batch Size per Replica**: In data-parallel configurations, confirm that
    each local replica has a sufficient batch size to sustain compute execution
    while background communication transfers complete.
-   **Model Structure & Sharding**: Examine model structure and data sharding to
    increase computation/communication overlap.
-   **Confirm Network Settings**: Verify that container and host network
    settings are optimized.

### 3. Confirm Optimal Container TCP Configurations

High RPC latencies in Cloud environments are often caused by suboptimal TCP
configuration. Confirm whether all TCP parameters are configured properly within
the workload container. If any TCP parameters are not correctly configured,
consult the Google Cloud ML Compute Services (CMCS) team for the recommended
container and host TCP settings.

## Connections

-   **Workflows**:
    -   [[libtpu-timing-outlier-analysis]]
-   **Actions**:
    -   [[query-workload-logs]]
    -   [[analyze-libtpu-timing]]
-   **Playbooks**:
    -   [[layered-debugging-framework]]
    -   [[multi-chip-hang-investigation]]
-   **Symptoms**:
    -   [[slow-step-time]]
    -   [[megascale-hang]]
-   **Concepts**:
    -   [[megascale]]
    -   [[goodput]]
-   **Entities**:

    -   [[cloud-tpu]]
    -   [[libtpu]]

    -   [[xprof]]

    -   [[cloud-logging]]

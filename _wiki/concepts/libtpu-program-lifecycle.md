---
title: libtpu Program Lifecycle and Continuation Queue
description: Architecture and mechanics of libtpu program queuing, execution lifecycle,
  HLO fingerprints, launch IDs, and dual-interval timing analysis.
type: concept
internal_only: false
tags: [tpu, training, analysis, logging, observability, metrics]
created: 2026-09-01
updated: 2026-10-03
---

## Overview

`libtpu.so` runs each compiled executable through an async **Continuation
Queue** and logs its enqueue and finish. These events are the basis for
isolating step-time regressions, straggler hosts, collective deadlocks and
host-device imbalance.

## Architecture & Design

### 1. Log Anatomy & Identifiers

A typical lifecycle loop emits two primary log signatures for each TensorCore
(TC):

```text
INFO <timestamp> [<container/pod>] <mesh_coord>_TC<id>: Enqueueing program <prog_id>: <hlo_fingerprint> to continuation queue (launch_id=<launch_id>)
INFO <timestamp> [<container/pod>] <mesh_coord>_TC<id>: last_finished_run_id updated to <prog_id>, remaining programs=<count>
```

Key fields include:

-   **Timestamp**: Nanosecond-precision ISO 8601 timestamps emitted by the host
    logging system.
-   **Mesh Coordinate and TensorCore ID** (`<mesh_coord>_TC<id>`): Reflects the
    physical or logical chip coordinates (e.g., `3x2x2_TC1`, `2x2x2_TC0`).
    Modern TPU architectures (such as Ironwood / Trillium) feature two
    TensorCores (`TC0` and `TC1`) per TPU chip. Consequently, a 4-chip host
    worker emits 8 distinct enqueue and completion log lines per program
    execution cycle.
-   **Program ID** (`<prog_id>`): A sequential execution counter tracking
    scheduled programs on each core. Different cores on the same host may start
    at differing initial offsets if auxiliary setup programs ran during
    initialization, but all active cores increment synchronously (+1 per step).
-   **HLO Fingerprint** (`<hlo_fingerprint>`): A 64-character hexadecimal hash
    representing the compiled XLA HLO module. Identical workloads across
    homogeneous workers must execute the exact same fingerprint.
-   **Launch ID** (`<launch_id>`): A monotonically increasing launch identifier
    assigned by the host runtime for each submission batch.
-   **Remaining Programs** (`remaining programs=<count>`): The depth of queued
    computations waiting in the device continuation queue. A value of `0`
    indicates that the queue has drained and the accelerator will become idle if
    the host does not immediately enqueue the next program.

### 2. The Two Critical Timing Intervals

Analyzing libtpu execution requires decomposing the timeline into two distinct
intervals:

```
[ Enqueue Prog N ] ------------------------> [ Finish Prog N ] ----------> [ Enqueue Prog N+1 ]
|<------------ T_exec (Device) ------------>|<--- T_dispatch_gap (Host) -->|
```

1.  **Device Execution Duration ($T_{exec}$)**:
    $$T_{exec} = T_{finish}(N) - T_{enqueue}(N)$$

    -   Represents the time spent executing on the TPU hardware, including
        kernel compute time, local memory transfers, and synchronous Inter-Chip
        Interconnect (ICI) / Data Center Network (DCN) collective communications
        (such as all-reduce or all-gather).

2.  **Host Dispatch Gap ($T_{dispatch\_gap}$)**:
    $$T_{dispatch\_gap} = T_{enqueue}(N+1) - T_{finish}(N)$$

    -   Represents the inter-step replenishment delay on the host CPU. This
        includes Python runtime dispatch, input data pipeline loading (reading
        batches from storage or GCS), gradient accumulation logic, host
        synchronization barriers, and checkpoint I/O operations.
    -   In healthy, well-pipelined training runs, $T_{dispatch\_gap}$ is minimal
        (typically tens of milliseconds).

## Operational Impact & Debugging Significance

### The "Inverted Execution Time" Paradox in Distributed Workloads

In distributed synchronous training, all workers synchronize at collective
communication barriers within the compiled program. If one worker experiences a
host-side bottleneck (e.g., slow data preprocessing or GCS read stall) and
enqueues late:

1.  **The Straggler Worker**: Has a very large $T_{dispatch\_gap}$ (e.g.,
    $1.5\text{s}$), but once it finally enqueues, its device execution
    $T_{exec}$ is short (e.g., $3.8\text{s}$) because all other workers are
    already waiting at the barrier.
2.  **The Healthy Workers**: Have tiny $T_{dispatch\_gap}$ values (e.g.,
    $50\text{ms}$), but their device execution $T_{exec}$ appears elongated
    (e.g., $5.2\text{s}$) because they entered the kernel early and blocked at
    the ICI barrier waiting for the straggler.

**Debugging Rule**: When diagnosing distributed step slowdowns, always sort
workers by **Host Dispatch Gap ($T_{dispatch\_gap}$)** or **Enqueue Timestamp
($T_{enqueue}$)** to locate the root cause host, rather than assuming workers
with the longest $T_{exec}$ are at fault.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[trace-program-execution-times]]
-   **Symptoms**:
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[megascale]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[libtpu]]
    -   [[tensorcore]]

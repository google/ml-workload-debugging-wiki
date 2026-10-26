---
title: Wait timeout on sflag
description: "TPU core times out while waiting for a synchronization flag (sflag) to be set, indicating a hang or severe ICI straggler."
type: symptom
internal_only: false
tags: [tpu, timeout]
created: 2026-05-13
updated: 2026-09-28
---


## Description

The log message `Wait timeout on sflag` indicates that a TPU core (typically
TensorCore) has timed out while waiting for a synchronization flag (`sflag`) to
be set by another process or core (often SparseCore or another chip). This is a
strong indicator of a hang or severe slowdown.

## Example Error Message/Log

```
[chip_TC0]: Wait timeout on sflag {[#allocation33477]} 'sparse-core-wait' for value {1} after 60000 milliseconds, latest value seen {0}, hlo: all-reduce.1418.cloned.1.call-done
```

*   `chip_TC0`: Represents the chip location in the ICI topology (5x15x3)
    and TensorCore 0.
*   `sparse-core-wait`: The name or context of the wait operation.

*   `value {1}`: The expected value of the flag.

*   `latest value seen {0}`: The actual value, showing it hasn't changed.

*   `hlo: all-reduce.1418.cloned.1.call-done`: The last completed HLO operation
    before the wait.

### SparseCore Companion Log

You should broadly check for SparseCore logs if they exist. Specifically, the
sparse core can be used in many cases (including collectives, sparse core wait,
and other operations), so you must investigate sparse core logs on the same host
regardless of whether the TensorCore is specifically waiting on a
`sparse-core-wait`:

```
5x10x9_SC1: [Physical location: tpu740:pe0:3] generic::internal: Program or fatal error occurred; computation may be invalid: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 1 halted unexpectedly at tag:pc SparseCoreSequencer:0:0x407 (from SparseCoreSequencer:0:0x447): Sflag wait timeout on op: sc_tpu.dma_wait sflag(%alloca_12 : memref<1xi32, 204>) [%c0] granules(%110)
```

**Breakdown of the SparseCore log:**

*   `5x10x9`: The ICI topology location of this chip (x, y, z).

*   `SC1`: The specific SparseCore chip (e.g., SparseCore 1).

*   `tpu740`: The physical chip ID based on its location within its pod.

*   `tag:pc SparseCoreSequencer:0:0x407`: The program counter ID indicating
    exactly what operation is currently running on the chip.

*   `Sflag wait timeout on op: sc_tpu.dma_wait`: The specific operation the
    sparse core is blocked on. A key part of looking at this is checking what op
    the sparse core is waiting on. If this op name is unique compared to other
    ones, that should be investigated here, similar to how you look at Tensor
    Core logs.

Group SparseCore logs by core (`SC0`, `SC1`, etc.), by `tag:pc`, and by the op
they are waiting on to identify the unique program stuck on a specific chip.
This helps locate the primary hardware or instruction failure point before the
hang propagates.

## On-Device Halt Signatures and Bounds Check Failures

When execution halts due to an on-device check failure or synchronization
timeout, the TPU runtime logs a fatal error citing the specific sequencer core
and program counter (`tag:pc`):

```
0x0x0_SC0: [Physical location: worker-cell:pe0:0] generic::internal: Program or fatal error occurred; computation may be invalid: Assertion args: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc SparseCoreSequencer:0:0xf2 (from SparseCoreSequencer:0:0x132): Sflag wait timeout on op {sc_tpu.dma_wait sflag(%alloca_1 : memref<1xi32, #sc_tpu.memory_space<sflag>>) [%c0] granules(%c16)} after 1 milliseconds.
```

Similarly, LLO bounds checker assertions halt the sequencer when memory
operations (such as DMA transfers between HBM and VMEM) violate static or
dynamic buffer limits:

```
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:336:0x840 (from TensorCoreSequencer:336:0xda98): BoundsCheck 5114 [deref of %s4242] for %4249 = dma.hbm_to_vmem [thread:$0] /*hbm=*/%s4242, /*size_in_granules=*/65536, /*vmem=*/%s4244
```

Key signatures to inspect:

-   `tag:pc SparseCoreSequencer:0:0xf2`: The exact sequencer instruction address
    where the core was halted.

-   `BoundsCheck <ID>`: Denotes an LLO bounds violation, identifying the
    dereferencing register (`%s4242`) and the offending memory opcode.

## Diagnostic and Verification Steps

1.  **Enable Core Dump Capture**: When an unrecoverable on-device halt or
    hardware check-failure occurs, TPU firmware collects binary core dumps when
    `--xla_tpu_dump_core_on_halt=true` is enabled. Configure core dump output by
    setting:

    ```bash
    --xla_tpu_dump_core_on_halt=true
    --tpu_core_dump_arch_states=TPU_CORE_DUMP_ARCH_STATE_TC,TPU_CORE_DUMP_ARCH_STATE_SCS
    --tpu_core_dump_directory=/path/to/dir
    ```

    This captures the architectural state of the halted TPU (including
    instruction memory, scalar/vector registers `sregs`/`pregs`/`dregs`, Smem,
    and sync flag states) for symbolization and post-mortem analysis.

2.  **Track Macro Execution with HPPT Progress Tracker & Heartbeat**: Correlate
    sflag timeouts against loop and conditional progress by enabling the
    Hierarchical Program Progress Tracker (HPPT) with a heartbeat interval:

    ```bash
    --xla_tpu_enable_progress_tracker=8
    --xla_tpu_progress_tracker_heartbeat='0.5'
    ```

    The periodic heartbeat logs the monotonic execution counters
    `{iteration, counter}` across cores, exposing which chip stalled at the
    lowest loop depth before timeout waves propagate.

3.  **Enable Fine-Grained LLO Progress Tracking**: To disambiguate macro HLO
    hang locations from intra-kernel communication stalls (such as remote DMA
    enqueues or barrier waits), activate the low-level progress tracker:

    ```bash
    --xla_tpu_llo_progress_tracker=true
    ```

4.  **Isolate Bounds Check Violations**: If an on-device halt indicates a bounds
    check failure, enable runtime bounds checking and detailed dereference
    logging for the reported ID:

    ```bash
    --xla_jf_bounds_check=true
    --xla_tpu_impure_bounds_check_log_failures=true
    ```

## Potential Causes

*   **ICI Link Degradation (Straggler)**: An abnormally slow ICI link causes an
    S0 core to stall on a `SEND` sync flag, propagating delay waves to neighbor
    chips.
*   **SparseCore Hang**: The SparseCore failed to complete its operation and set
    the flag.
*   **Deadlock**: Circular wait condition between cores.

*   **Hardware Failure**: A core halted unexpectedly due to a hardware error.

*   **Stalled Host Process (Checkpoint Hang)**: `sflag` network timeouts are
    frequently *secondary* casualties of a stalled host process (like a deep
    checkpoint hang) that prevents the node proxy from responding to health
    checks. In distributed checkpointing (e.g. Orbax), GCS checkpoint folder
    deletion delays or rate limits stall worker 0 / JAX process zero (the
    designated coordinator), while worker processes 1..N wait and trigger
    secondary barrier timeouts.

## Related Workflows and Playbooks

*   [[tpu-state-analysis-methodology]]: This workflow uses these logs to
    identify culprit machines.

*   [[ici-straggler-triage]]: Diagnostic SOP workflow for detecting and
    localizing degraded ICI links using delay wave analysis.

*   [[on-device-debugging-triage]]: Playbook for on-device crash analysis, core
    dumps, HPPT progress tracking, and LLO bounds check failures.

## Connections

-   **Workflows**:
    -   [[tpu-state-analysis]]
    -   [[ici-straggler-triage]]
-   **Actions**:
    -   [[analyze-sflag-timeouts]]
    -   [[query-ici-logs]]
    -   [[analyze-ici-wave]]
-   **Playbooks**:
    -   [[multi-chip-hang-investigation]]
    -   [[on-device-debugging-triage]]
    -   [[tpu-state-analysis-methodology]]
    -   [[ici-straggler-triage]]
-   **Concepts**:
    -   [[multi-chip-hangs]]
    -   [[ici-errors]]
-   **Entities**:
    -   [[sparsecore]]
    -   [[tensorcore]]

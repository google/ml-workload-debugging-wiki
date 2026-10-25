---
title: "SparseCore Debugging and Mismatch Triage Playbook"
description: "Diagnostic playbook for isolating SparseCore offloading errors, numerical mismatches, LLVM bundle profiling, and assert level verification."
type: playbook
internal_only: false
tags: [tpu, hardware_fault, triage, methodology, analysis]
created: 2026-09-28
updated: 2026-09-28
---


# SparseCore Debugging and Mismatch Triage Playbook

## Overview

On Google Cloud TPUs, the **SparseCore (SC)** is a dedicated accelerator engine
optimized for embedding lookups, sparse tensor gathering, and
memory-bandwidth-bound operations, operating in parallel with the dense-compute
**TensorCore (TC)**. The XLA compiler can automatically push (offload) selected
operations—including collective communication (`all-gather`, `all-reduce`,
`reduce-scatter`) and compute ops (`gather`, `scatter`, `sort`, `copy`)—from
TensorCore to SparseCore.

While offloading frees TensorCore execution cycles for dense matrix
multiplication, bugs in XLA offload lowering or SparseCore code generation can
introduce subtle numerical drift, functional divergence, or execution stalls.

This playbook documents:

1.  The **SparseCore Mismatch Detector**: detecting, isolating, and validating
    divergences between SparseCore and TensorCore implementations.
2.  **Profiling Tools**: analyzing SparseCore execution bundles, generating LLVM
    execution profiles, and visualizing interactive Perfetto traces.
3.  **Compiler & Diagnostic Flags**: setting assertion verbosity
    (`--xla_sc_assert_level`) and controlling operation offloading.
4.  **Triage & Remediation**: working around compiler offload bugs to unblock
    active training jobs.

--------------------------------------------------------------------------------

## SparseCore Mismatch Detector

### How the Mismatch Detector Works

The SparseCore Mismatch Detector operates on a shadow-execution principle: For
each operation offloaded to SparseCore, the XLA compiler synthesizes a
functionally equivalent TensorCore operation that takes the identical input
operands and executes concurrently on the TensorCore thread. The framework
compares the results of both engines elementwise using an `all_close`
verification pass.

If the results diverge beyond defined numerical tolerances, the system emits an
explicit log signature or halts execution immediately.

```
                  +---------------------------+
                  |    Input Operands (%x)    |
                  +---------------------------+
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
    +----------------------+          +----------------------+
    |  SparseCore Engine   |          |  TensorCore Engine   |
    | (Async Offloaded Op) |          | (Shadow Baseline Op) |
    +----------------------+          +----------------------+
               |                                 |
               +----------------+----------------+
                                |
                                v
               +---------------------------------+
               |     all_close Verification      |
               | (abs(a-b) <= atol + rtol*abs(b))|
               +---------------------------------+
                                |
                     +----------+----------+
                     |                     |
                     v [Match]             v [Mismatch]
               [Continue]             xla-tpu-log / Crash:
                                      "Found Mismatch in offloaded op..."
```

### Sample HLO Transformation

To observe how the tool instruments the High-Level Optimizer (HLO) computation,
consider an `all-gather` collective offload:

**Original Offloaded HLO:** `%color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)}
broadcast(%convert), dimensions={} %constant = s32[8]{0:T(128)} constant({0, 1,
1, -1, 1, 0, 0, -1}) %all-gather.2.cloned.call-start =
((f32[8,256,256]{2,1,0:T(8,128)}, s32[8]{0:T(128)}), f32[16,256,256]{2,1,0},
u32[]) async-start(%color_operand.1, %constant),
async_execution_thread="sparsecore", calls=%async_computation,
metadata={op_type="AllGather" op_name="ag0"} %all-gather.2.cloned.call-done =
f32[16,256,256]{2,1,0} async-done(%all-gather.2.cloned.call-start),
metadata={op_type="AllGather" op_name="ag0"}`

**Instrumented HLO (Mismatch Detector Active):** `%color_operand.1 =
f32[8,256,256]{2,1,0:T(8,128)} broadcast(%convert), dimensions={} %all-gather =
f32[16,256,256]{2,1,0} all-gather(%color_operand.1), replica_groups={{0,1}},
dimensions={0} %constant = s32[8]{0:T(128)} constant({0, 1, 1, -1, 1, 0, 0, -1})
%all-gather.2.cloned.call-start = ((f32[8,256,256]{2,1,0:T(8,128)},
s32[8]{0:T(128)}), f32[16,256,256]{2,1,0}, u32[]) async-start(%color_operand.1,
%constant), async_execution_thread="sparsecore", calls=%async_computation,
metadata={op_type="AllGather" op_name="ag0"} %all-gather.2.cloned.call-done =
f32[16,256,256]{2,1,0} async-done(%all-gather.2.cloned.call-start),
metadata={op_type="AllGather" op_name="ag0"} %call.3 = pred[] call(%all-gather,
%all-gather.2.cloned.call-done, %constant.1, %constant.2), to_apply=%all_close
%not.2 = pred[] not(%call.3) %custom-call.8 = token[] custom-call(%not.2,
%call.3, %call.3), custom_call_target="xla-tpu-log",
custom_call_has_side_effect=true,
backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"log_config":{"severity":"LOG_INFO","tag":"Found
Mismatch in offloaded op all-gather.2.cloned.call-start:
","extra_args":[],"opcode":"LOG_OPCODE_MESSAGE"}}`

Notice that the transformed HLO introduces:

1.  A baseline TensorCore operation (`%all-gather`).
2.  An elementwise comparator calling `%all_close` (`%call.3`).
3.  A logging custom-call (`custom_call_target="xla-tpu-log"`) triggered
    whenever `%not.2` is true.

### Evaluating Comparison Results & Edge Cases

1.  **Exact vs Approximate Matching**:
    -   **Non-computational operations** (`all-gather`, `sort`, `copy`): Expect
        an **exact match**. Any bitwise divergence indicates a bug.
    -   **Computational operations** (`all-reduce`, `reduce-scatter`,
        `scatter`): Floating-point arithmetic differences across hardware
        architectures and summation order require tolerance checks:
        $$\text{abs}(a - b) \le \text{atol} + \text{rtol} \times \text{abs}(b)$$
2.  **Handling NaNs & Infs**:
    -   Positions where both SparseCore and TensorCore produce `NaN` are
        excluded from mismatch flagging (since floating-point `NaN != NaN`).
    -   Positions where both engines produce identical signed infinities
        ($+\infty$ or $-\infty$) are excluded.
3.  **Sort Stability**:
    -   SparseCore implements a **stable** radix sort (preserving relative order
        of equivalent elements).
    -   TensorCore implements an **unstable** bitonic sort.
    -   To avoid false positive mismatch alerts during `sort` verification, the
        TensorCore comparator stabilizes its sort using an auxiliary `iota`
        index tiebreaker.

--------------------------------------------------------------------------------

## Configuration Flags

### Mismatch Detector Flags

| Flag                                                      | Default | Description           |
| :-------------------------------------------------------- | :------ | :-------------------- |
| `xla_tpu_enable_sparse_core_mismatch_detector` | `false` | **Required**. Activates shadow execution and comparison on offloaded ops. |
| `xla_tpu_enable_log_recorder` | `false` | **Required**. Enables the LLO logger to capture runtime trace messages emitted by the detector. |
| `xla_tpu_sparse_core_mismatch_detector_on_specified_ops` | `""` | Optional comma-separated filter to restrict checks (e.g. `'all-gather,sort'`). |
| `xla_tpu_sparse_core_mismatch_detector_crash_on_mismatch` | `false` | If `true`, halts immediately on first mismatch; if `false`, performs exhaustive search and logs all mismatches. |
| `xla_tpu_sparse_core_mismatch_detector_atol` | `0.0` | Absolute numerical tolerance for floating-point comparisons. |
| `xla_tpu_sparse_core_mismatch_detector_rtol` | `0.0` | Relative numerical tolerance for floating-point comparisons. |

### Offloading Control Flags

To test or isolate individual operations, toggle offloading per operation
family:

```bash
# Toggle compute op offloading
--xla_tpu_enable_offloading_sort_to_sparsecore=true|false
--xla_tpu_enable_offloading_scatter_to_sparsecore=true|false
--xla_tpu_offload_gather_to_sparsecore=true|false

# Toggle collective communication offloading
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true|false
```

### Running with JAX & PyTorch

**For JAX on Cloud TPU:**

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_enable_sparse_core_mismatch_detector=true \
                         --xla_tpu_enable_log_recorder=true \
                         --xla_tpu_enable_offloading_sort_to_sparsecore=true \
                         --xla_tpu_enable_offloading_scatter_to_sparsecore=true \
                         --xla_tpu_offload_gather_to_sparsecore=true"

# Enable HLO text dumps to correlate mismatches with HLO graph names
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
mkdir -p /tmp/xla_dump

python3 your_jax_model.py
```

**For PyTorch / XLA:**

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text \
                  --xla_tpu_enable_sparse_core_mismatch_detector=true \
                  --xla_tpu_enable_log_recorder=true \
                  --xla_tpu_enable_offloading_sort_to_sparsecore=true \
                  --xla_tpu_enable_offloading_scatter_to_sparsecore=true \
                  --xla_tpu_offload_gather_to_sparsecore=true"
mkdir -p /tmp/xla_dump

python3 your_pytorch_model.py
```

### Interpreting Mismatch Logs

When a discrepancy occurs, search logs for:

```
Found Mismatch in offloaded op <op_instance_name>: ...
```

Example:

```
Found Mismatch in offloaded op all-gather.2.cloned.call-start: Mismatch detected between TensorCore baseline and SparseCore offload
```

--------------------------------------------------------------------------------

## SparseCore Assertion Levels (`--xla_sc_assert_level`)

The `--xla_sc_assert_level` compiler flag governs runtime verification verbosity
on SparseCore hardware.

### Aliases

Alias                 | Description
:-------------------- | :----------
`prod`                | **Default**. Enables `always` (critical asserts only). Minimal performance overhead.
`san`                 | Enables all asserts across all categories. High diagnostic coverage.
`san-lite`            | Balanced profile: `always,streams,dma,alltoall,runids`.
`vector-loads-stores` | Enables `vector-loads,vector-stores`.
`all-loads-stores`    | Enables `vector-loads,vector-stores,scalar-loads,scalar-stores`.
`no`                  | Completely disables all assertions.

### Fine-Grained Assertion Categories

Categories can be combined via comma-separated list (e.g.
`--xla_sc_assert_level=bounds,dma,syncflags`):

| Category        | Description                                    |
| :-------------- | :--------------------------------------------- |
| `alltoall` | Asserts for all-to-all communication operations. |
| `always` | Critical invariants and hardware safety assertions. |
| `bounds`        | Memory buffer and tile bounds checks.          |
| `checksums` | Validates buffer checksums (used with `--xla_tpu_buffer_contents_sanitizer_config`). |
| `continuations` | Asserts for async task launching from TensorCore to SparseCore. |
| `csrs` | Compressed Sparse Row format checks for embedding tables. |
| `dma` | DMA alignment and transfer boundary validation. |
| `masks` | Vector mask integrity (e.g. ensuring contiguous active lanes). |
| `overlays` | Manages sequencer programs exceeding instruction memory capacity. |
| `radixsort`     | Radix sort emitter and stability checks.       |
| `runids` | Enforces sequential execution ordering of SparseCore programs. |
| `scalar-loads`  | Bounds checks on scalar memory reads.          |
| `scalar-stores` | Bounds checks on scalar memory writes.         |
| `syncflags` | Verifies synchronization flags are within bounds and reset to expected state. |
| `streams`       | Stream execution and ordering checks.          |
| `vector-loads`  | Bounds checks on vector memory reads.          |
| `vector-stores` | Bounds checks on vector memory writes.         |

--------------------------------------------------------------------------------

## Diagnostic Triage Steps & Escalation Workflow

```
[Workload Fails / Numerics Diverge with SparseCore Offload]
                        |
                        v
        [Step 1: Enable Mismatch Detector]
   --xla_tpu_enable_sparse_core_mismatch_detector=true
   --xla_tpu_enable_log_recorder=true
                        |
                        v
        [Step 2: Inspect Application Logs]
   Match: "Found Mismatch in offloaded op <op_name>:"
                        |
                        v
        [Step 3: Apply Immediate Workaround]
   Disable offloading for the specific op family:
   e.g. --xla_tpu_enable_offloading_sort_to_sparsecore=false
   (Op moves to TensorCore -> unblocks production training)
                        |
                        v
        [Step 4: Collect Artifacts & Escalate]
   - Mismatch detector log output
   - HLO text dumps (--xla_dump_to)
   - File XLA compiler defect ticket
```

1.  **Isolate Problematic Operation**: Parse the mismatch log message to
    pinpoint the offending HLO operation name (e.g.,
    `all-gather.2.cloned.call-start`).

## Mitigations & Workarounds

1.  **Apply Workaround (Disable Op Offloading)**: Because SparseCore mismatch
    failures almost exclusively stem from **XLA compiler bugs** rather than user
    model code, immediately disable offloading for that operation family.
    Executing the operation on TensorCore eliminates the mismatch and unblocks
    training while the compiler team investigates.
2.  **Escalate to Compiler Team**: Capture HLO dumps via
    `--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text` and file a bug report
    attaching the HLO dump and the mismatch log.

--------------------------------------------------------------------------------

## Connections

-   **Workflows**:
    -   [[sdc-and-numerics-triage]]
-   **Playbooks**:
    -   [[pallas-mosaic-kernel-debugging]]
    -   [[data-non-determinism-triage]]
    -   [[numerics-error-triage]]
    -   [[on-device-debugging-triage]]
    -   [[debugging-mindset]]
-   **Actions**:
    -   [[analyze-sparsecore-offload]]
    -   [[triage-sdc-and-numerics]]
    -   [[inspect-hlo-dumps]]
    -   [[analyze-sflag-timeouts]]
    -   [[query-workload-logs]]
-   **Symptoms**:
    -   [[sdc-checksum-mismatch]]
    -   [[wait-timeout-on-sflag]]
    -   [[mxla-hardware-fault]]
-   **Concepts**:
    -   [[tpu-logging-flags]]
    -   [[multi-chip-hangs]]
-   **Entities**:
    -   [[sparsecore]]
    -   [[tensorcore]]
    -   [[cloud-tpu]]
    -   [[xla]]

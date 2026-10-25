---
title: "Numerics Error Triage Playbook"
description: "Standard operating procedure for isolating and resolving numerical divergence, intermediate tensor drift, and precision-related errors using HLO diff, comparison tools, and host-assisted logging."
type: playbook
internal_only: false
tags: [tpu, triage, methodology, analysis]
created: 2026-09-28
updated: 2026-10-09
---


# Numerics Error Triage Playbook

## Overview

Numerical errors on TPUs generally fall into two broad scenarios:

1.  **Comparing Runs**: A program's numerical output diverges from an
    established reference run (such as a previous software release, execution on
    CPU/GPU, or a run with different compiler flags).
2.  **Numerical Instability in New Programs**: A newly developed model
    experiences training loss explosion, unexpected NaNs/Infs, or loss of
    convergence without an external baseline.

Debugging numerical issues requires systematic localization down to the culprit
High-Level Optimizer (HLO) instruction, compiler pass, or hardware
floating-point format limitation.

This playbook establishes a 4-tier triage methodology leveraging:

-   **`hlo-diff`**: Semantic structural comparison between compiled HLO graphs.
-   **`hlo-isolation`** *(and 1P **`comparison-tool`***): Bisecting and
    comparing intermediate tensor outputs across target and reference
    environments.
-   **Logging Tools**: Fine-grained intermediate tensor logging
    (`log_hlo_output`), host-assisted CPU verification
    (`host_assisted_verification`), fusion debugging (`fusion_debugger`), and
    Python debug callbacks.
-   **XLA Numerics Flags & Floating-Point Precision Management**: Isolating
    optimization/MSA/fusion passes and navigating TPU hardware format behaviors
    (`f32`/`bf16` subnormal flush-to-zero, `e4m3b11` saturation, `e4m3b7`
    overflow-to-NaN, and multi-pass matrix multiplication emulation).

## Diagnostic Steps and Triage Workflow

```
[Numerical Divergence / Instability]
                |
     Are baseline runs available?
        /                  \
    [Yes]                  [No]
      |                      |
[Tier 1: hlo-diff]     [Tier 2: HLO Isolation / Comparison Tool]
      |                      |
   Diffs found?              |
   /        \                |
[Yes]       [No]             |
  |           |              |
Compiler    [Tier 2:         |
Pass /      HLO Isolation /  |
Flags       Comparison Tool] |
              \             /
          [Tier 3: Logging, Callbacks & Fusion Debugger]
                     |
          [Tier 4: XLA Numerics Flags & Precision Formats]
```

--------------------------------------------------------------------------------

### Tier 1: Semantic Graph Comparison with `hlo-diff`

When comparing two runs (e.g. baseline vs. target), start by comparing their
compiled HLO modules using the **HLO Diff Tool**. The tool evaluates the
computational graph structure while ignoring superficial differences like
instruction naming or parameter ordering.

#### Model Explorer Visualization and Color Coding

The HLO diff tool visualizes graph diffs via
[Model Explorer](https://ai.google.dev/edge/model-explorer):

-   **Unmatched Instructions**: Instructions removed in the target module appear
    in **Red** (`kLeftUnmatched`). Instructions newly added appear in **Green**
    (`kRightUnmatched`).
-   **Updated Instructions**: Instructions with modified shapes, layouts, or
    parameters appear in **Yellow** (`kUpdated`).
-   **Unchanged Instructions**: Identical instructions appear in **White**
    (`kUnchanged`).
-   **Hiding Unchanged Lineage**: For massive graphs, focus strictly on modified
    subgraphs by hiding nodes with unchanged ancestors:

    ```bash
    --hide_nodes_with_unchanged_ancestors=true
    ```
-   **Manual Node Mapping**: If heuristic graph alignment misaligns nodes, force
    explicit correspondences:

    ```bash
    --force_mapping="lhs_fusion_node:rhs_fusion_node"
    ```

*Triage Decision*:

-   If significant structural graph diffs exist, the divergence likely
    originates in an XLA compiler pass or differing compiler flags. Inspect the
    differing pass.
-   If HLO graphs are structurally identical or nearly identical, the divergence
    originates in lower-level numerical behavior or hardware execution. Proceed
    to Tier 2.

--------------------------------------------------------------------------------

### Tier 2: HLO Isolation and Intermediate Tensor Summaries

#### 1. OpenXLA HLO Isolation Tool (`hlo-isolation`)

To evaluate numerical differences between a target environment and a reference
environment—or to isolate compilation issues in a new program without a separate
baseline run—use the
[**HLO Isolation Tool**](https://openxla.org/xla/hlo_isolation):

-   **Comparing Runs**: Pass both target and `--reference_platform`
    configurations to bisect the HLO module and isolate the first instruction
    where outputs diverge.
-   **New Programs (Single-Run Verification)**: Run `hlo-isolation` without
    `--reference_platform` to find compilation bugs during execution on the test
    platform; similar to the host-assisted verifier, it uses the CPU interpreter
    as a reference baseline.

--------------------------------------------------------------------------------

### Tier 3: Logging HLO Outputs, Host Verification, and Fusion Debugger

Once suspect HLO operations are localized, use XLA's instrumentation logging
tools to inspect values without modifying the HLO graph structure. All logging
tools require `--xla_tpu_enable_log_recorder=true` (and optionally
`--xla_tpu_dump_logs_to_dir="path_to_dir"` to bypass HBM logging buffer size
limits, or `--xla_hbm_logging_buffer_size_bytes`).

#### 1. Host-Assisted Verification

Host-assisted verification evaluates the identical HLO subgraph on the Host CPU
using `HloEvaluator` (or `TpuEvaluator` via
`--xla_tpu_use_tpu_evaluator_in_host_verify=true`) and automatically calculates
absolute and relative differences against TPU output:

```bash
--xla_tpu_enable_log_recorder=true
--xla_tpu_enable_host_assisted_instrumentation=true
--xla_tpu_hlo_graph_host_verify=true
# Optional filters and thresholds:
# --xla_tpu_host_instrumentation_filter="foo.123*"
# --xla_tpu_host_verify_max_output_size=536870912
# --xla_tpu_host_verify_min_abs_diff_to_log=...
# --xla_tpu_host_verify_min_rel_diff_to_log=...
# --xla_tpu_host_verify_ignore_tuple_ops=true
```

In Python/JAX, verify or log specific code regions surgically using
`set_xla_metadata` (as a context manager, function decorator, or wrapper call)
with `_xla_log=True`, `_xla_log_if_name="foo.123*"`, `_xla_host_verify=True`,
and `_xla_tag="<string>"`:

```python
from jax.experimental.xla_metadata import set_xla_metadata

with set_xla_metadata(_xla_host_verify=True, _xla_log=True):
  output = suspect_layer(inputs)
```

The runtime logs maximum absolute and relative errors:

```
I0920 13:26:07.272415 instrumentation_handler.cc:293] Results of host-assisted verification of %fusion: Max abs diff: f32[] 0.45396 Max rel diff: f32[] 8339.31 tpu value: 23.4163 host value: 22.9623
```

#### 2. Logging Intermediate HLO Outputs and Instruction Filtering

To inspect statistical summaries (`{min, max, mean, stddev}`) or raw tensor
values across HLO instructions:

```bash
--xla_tpu_enable_log_recorder=true
--xla_jf_log_hlo_output=true
--xla_tpu_log_hlo_output_summaries=true
```

To filter full tensor logging without modifying model code, use:

-   `--xla_tpu_debug_emitter_hlo_filter="foo.123*"` (non-fused instructions)
-   `--xla_tpu_fusion_debug_hlo_filter="foo.123*"` (fused instructions)

#### 3. Fusion Debugger (`fusion_debugger`)

When the localized HLO instruction is a fusion operation, enable the **Fusion
Debugger** to inspect intermediate values inside fused computations:

```bash
--xla_tpu_enable_log_recorder=true
--xla_tpu_enable_fusion_debugger=true
--xla_tpu_hlo_graph_fusion_debug=true
```

If a `NaN` occurs inside a fusion, you can enable the fusion debugger while
disabling runtime full-tensor retrieval via
`--xla_tpu_disable_fusion_debug_handler=true` so the first `NaN` observed among
instrumented fused instructions is immediately detected.

#### 4. Python Debug Callback API (`_xla_log_callback`)

To obtain intermediate HLO outputs (including fused instructions inside a
fusion) directly as `np.array` objects in Python without modifying the HLO graph
(unlike `jax.debug.print`):

```python
from jax.experimental.xla_metadata import set_xla_metadata

log_output = {}
def callback(array, xla_tag: str, core_location: str):
  log_output.setdefault(xla_tag, {}).setdefault(core_location, []).append(array)

callback_id = handler.register_debug_callback(callback)

@jax.jit
def f(x):
  with set_xla_metadata(_xla_log_callback=True):
    y = jnp.add(x, x)
    return jnp.pow(y, 2)

f(2).block_until_ready()
handler.unregister_debug_callback(callback_id)
```

--------------------------------------------------------------------------------

### Tier 4: XLA Numerics Flags and Floating-Point Precision on TPU

#### 1. XLA Flags Related to Numerics Errors

When compiler optimizations, Memory Space Assignment (MSA), or fusions introduce
numerical drift or NaNs, test the following diagnostic and isolation flags:

-   **MSA, Copy & Collective Fusion Isolation**:
    -   `--xla_msa_enable_sync_copy_replacement=false`: Forces all copies to use
        synchronous copy.
    -   `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`:
        Disables fusing collective ops with compute ops.
    -   `--xla_tpu_autofdo=false`: Disables all AutoFDO optimizations.
    -   `--xla_msa_enable=false`: Disables Memory Space Assignment (MSA)
        specific optimizations.
    -   `--xla_tpu_scoped_vmem_limit_kib`: Limits VMEM available for scoped
        allocations (default is 16 MiB on v4, v5e, v5p; 32 MiB on
        v6e+).
-   **Fusion & BF16 Lowering Flags**:
    -   `--xla_tpu_dot_dot_fusion`: Controls `Dot(Dot)` fusions.
    -   `--xla_tpu_enable_multi_level_nested_loop_fusion`: Controls nested loop
        fusions.
    -   `--xla_tpu_vmem_scavenging_mode`: Allows Fusion+MSA to recuperate excess
        scoped memory for deeply nested fusions (`"AGGRESSIVE"` or `"SAFE"`).
    -   `--xla_tpu_impure_enable_packed_bf16_math_ops`: Controls whether
        `fusion_emitter` lowers loop fusions with `bf16` math ops using
        hardware-native BF16 operations.
    -   `--xla_allow_excess_precision`: When `true`, permits XLA to use higher
        precision for intermediate calculations than strictly required by the
        op's data type.
    -   Cost model tuning: `--xla_msa_cost_model_options`,
        `--xla_tpu_fusion_cost_model_options`,
        `--xla_tpu_latency_hiding_scheduler_cost_model_options`.
-   **HLO and LLO Special FP / NaN / Inf Detection**:
    -   `--xla_tpu_detect_special_fp=true`: Checks HLO outputs and raises an
        error on `NaN`, `inf`, or `-inf`.
    -   `--xla_tpu_detect_nan=true` / `--xla_sc_detect_nan=true`: Checks
        TensorCore or SparseCore HLO outputs and raises an error on `NaN`.
    -   `--xla_tpu_detect_inf=true`: Checks HLO outputs and raises an error on
        `inf` or `-inf`.
    -   `--xla_tpu_detect_only_on_fusion=true`: Restricts `xla_tpu_detect_*`
        checks to fusion node outputs.
    -   `--xla_tpu_detect_llo_nan` (and `--xla_tpu_detect_llo_nan_hlo_filter`):
        Runs an LLO pass checking every `sreg` and `vreg` for `NaN` at higher
        fidelity than HLO-level checkers.

#### 2. Hardware Supported Floating-Point Formats

Many numerical errors stem from fundamental differences between IEEE-754
standard CPU floating-point behavior and TPU hardware execution.

Format      | Exponent Bits | Mantissa Bits | Exponent Bias | Subnormal Support | Infinity Support          | Supported TPU Generations
:---------- | :------------ | :------------ | :------------ | :---------------- | :------------------------ | :------------------------
**f32**     | 8             | 23            | 127           | **Flush-to-Zero** | Yes                       | v2, v3, v4, v5, v6e, 7x
**bf16**    | 8             | 7             | 127           | **Flush-to-Zero** | Yes                       | v2, v3, v4, v5, v6e, 7x
**hf16**    | 5             | 10            | 15            | Supported         | Yes                       | v5, v6e, 7x
**e5m2**    | 5             | 2             | 15            | Supported         | Yes                       | v5, v6e, 7x
**e4m3b11** | 4             | 3             | 11            | Supported         | **No** (Saturates)        | v5, v6e
**e4m3b7**  | 4             | 3             | 7             | Supported         | **No** (Overflows to NaN) | 7x

#### 3. Critical Precision Caveats

1.  **Subnormal Flush-to-Zero (`f32` and `bf16`)**:
    -   Subnormals are supported on `hf16`, `e5m2`, `e4m3b11`, and `e4m3b7`, but
        for `f32` and `bf16`, TPUs **always flush subnormal results to zero**.
        Algorithms sensitive to underflow near zero may abruptly evaluate to
        zero or trigger division-by-zero `NaN`s.
2.  **8-Bit Formats: `e4m3b11` (v5/v6e) vs. `e4m3b7` (7x)**:
    -   Neither `e4m3b11` nor `e4m3b7` supports Infinities.
    -   **`e4m3b11` (TPU v5, v6e)**: Encodes `NaN` with sign bit `1` and all
        other bits `0` (`0x80`); negative zero is not supported. Values
        exceeding the representable range **saturate** to the largest
        representable finite number with the same sign.
    -   **`e4m3b7` (TPU 7x-only)**: Encodes `NaN` with sign bit `0` and all
        other bits `1` (`0x7F`). Unlike `e4m3b11`, any operation whose result
        exceeds the representable range **is converted to `NaN`** rather than
        saturating to a finite number.
3.  **Transcendental Accuracy in Extended Unary Pipeline (EUP)**:
    -   TPUs evaluate reciprocal, reciprocal square root, and transcendental
        functions (`sin`, `cos`, `erf`, `exp`, `log`) in the EUP.
    -   For `sin` and `cos` (TPU v5, v6e, 7x), hardware implementation covers
        $[-\pi/4, \pi/4]$ (returning `NaN` outside this domain), and the
        compiler automatically inserts Payne-Hanek range reduction instructions.
        For reciprocal and reciprocal square root on targets with $>1$ ULP
        error, the compiler inserts a Newton-Raphson iteration.
    -   *Mitigation*: In JAX, tune unary accuracy via `lax.AccuracyMode`
        (`accuracy=lax.AccuracyMode.HIGHEST` vs `DEFAULT`) or `lax.Tolerance`.
4.  **f32 Matrix Multiplication Emulation**:
    -   The TPU Matrix Multiplication Unit (MXU) natively operates on `bfloat16`
        inputs.
    -   Standard 1-Pass matmuls cast `f32` inputs to `bf16` (reducing mantissa
        from 23 to 7 bits via round-to-nearest-even), accumulating products in
        `f32`.
    -   To achieve higher `f32` fidelity, configure iterative multi-pass
        refinement:
        -   **`jax.lax.Precision.DEFAULT`**: 1-pass standard MXU execution
            (native `bf16` multiplication).
        -   **`jax.lax.Precision.HIGH`**: 3-pass MXU execution (2-part split
            into `bf16` high and `f32` residual terms).
        -   **`jax.lax.Precision.HIGHEST`**: 6-pass MXU execution (3-part split
            into high, mid, and low terms) achieving near-exact `f32` precision.

--------------------------------------------------------------------------------

## Mitigations and Resolution Strategies

1.  **Precision Escalation**: For numerically sensitive matrix operations or
    loss instability, escalate precision from `jax.lax.Precision.DEFAULT` to
    `Precision.HIGH` (3-pass) or `Precision.HIGHEST` (6-pass).
2.  **Transcendental High Accuracy**: Use `accuracy=lax.AccuracyMode.HIGHEST` or
    `lax.Tolerance` for unary transcendental functions (`exp`, `log`, `sin`)
    requiring tight precision.
3.  **Compiler Pass & Fusion Isolation**: Use
    `--xla_msa_enable_sync_copy_replacement=false`,
    `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`, or
    `jax.lax.optimization_barrier` to isolate compiler fusions or asynchronous
    copies causing numerical drift.

--------------------------------------------------------------------------------

## Compiler Flags Quick Reference

| Flag                                             | Default | Description     |
| :----------------------------------------------- | :------ | :-------------- |
| `--xla_tpu_enable_log_recorder` | `false` | Enables LLO log recording framework (required for logging tools). |
| `--xla_tpu_dump_logs_to_dir` | `""` | Directory to output runtime `LogData` protos without buffer size limits. |
| `--xla_jf_log_hlo_output` | `false` | Enables logging of intermediate HLO output buffers. |
| `--xla_tpu_log_hlo_output_summaries` | `false` | Restricts HLO output logging to statistical summaries (`min`, `max`, `mean`, `stddev`). |
| `--xla_tpu_enable_host_assisted_instrumentation` | `false` | Instruments HLO graph for CPU comparison. |
| `--xla_tpu_hlo_graph_host_verify` | `false` | Executes HLO graph on CPU host to compare absolute and relative numerical error. |
| `--xla_tpu_enable_fusion_debugger` | `false` | Enables compiler and runtime instrumentation for inspecting fused computations. |
| `--xla_tpu_hlo_graph_fusion_debug` | `false` | Enables fusion debug instrumentation across the entire HLO graph. |
| `--xla_tpu_detect_special_fp` | `false` | Checks HLO outputs and errors on `NaN`, `inf`, or `-inf`. |
| `--xla_tpu_detect_llo_nan` | `false` | LLO pass checking every `sreg` and `vreg` for `NaN`. |
| `--hide_nodes_with_unchanged_ancestors` | `false` | HLO Diff flag to collapse unchanged upstream dependency subgraphs. |

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
    -   [[sparsecore-debugging-triage]]
    -   [[compile-time-crash-reporting]]
    -   [[on-device-debugging-triage]]
    -   [[debugging-mindset]]
-   **Symptoms**:
    -   [[sdc-checksum-mismatch]]
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[tpu-logging-flags]]
    -   [[known-issues]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[tensorcore]]
    -   [[xla]]
    -   [[jax]]

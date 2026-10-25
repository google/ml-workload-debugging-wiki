---
title: MaxText
description: The JAX LLM training framework most workloads in this wiki run, and the
  application layer where configuration and resharding defects originate.
type: entity
internal_only: false
tags:
- training
- tpu
- pathways
created: 2026-09-15
updated: 2026-10-16
---


# MaxText

## Overview

**MaxText** is the [[jax]] large-model training framework that the majority of
workloads documented here run. It sits at the top of the stack — layer 1 in the
[[layered-debugging-framework]] — so its defects present as application
tracebacks and hangs rather than as infrastructure events, and are often misread
as hardware faults.

## Technical Specifications

-   Configuration is supplied as command-line key/value arguments parsed by a
    config library (`omegaconf` style); unknown or malformed keys fail fast at
    startup.
-   Checkpointing is delegated to [[orbax]] and runs on JAX Process 0, not
    necessarily on the GKE coordinator pod — see [[mcjax]].
-   Elastic recovery re-traces the step function at the new slice count after a
    slice drops, rebuilding `in_shardings` from the model spec.

## Common Failure Modes

-   **Startup configuration crash (exit code 1)**: a mistyped config key, or
    trailing whitespace in an XPK `--args` string producing
    `ValueError: '' not in [...]`. Deterministic across retries. See
    [[workload-startup-crash]] and [[workload-startup-failures]].
-   **Grain elastic iterator local device batch divisibility crash**: when
    `grain_use_elastic_iterator=True`, `packing=False`, and
    `grain_worker_count > 0`, multiprocessing Grain workers can yield a per-host
    batch slice (e.g., shape `(2, 8192)`) that is not evenly divisible by the
    local TPU device count (`len(devices)=4` on `v6e-8`), causing
    `np.split(array, len(devices), axis=0)` in
    `multihost_dataloading.py:form_global_array_colocated_python` to fail with
    `ValueError: array split does not result in an equal division` and
    `ValueError: Unable to put to devices shape (2, 8192) with local device
    count 4 at ['inputs']` (`While processing examples for
    unknown_output_specs_and_push_result...
    [type.googleapis.com/pathwaysutils.PyException='']`). Set
    `grain_worker_count=0` when `grain_use_elastic_iterator=True` and
    `packing=False` (or enable `packing=True`). See [[workload-startup-crash]]
    and [[pathways-sidecar]].
-   **Blocking checkpoint save**: a save that blocks on `jax.block_until_ready`
    trips the `step-timebomb` watchdog and dumps thread tracebacks, which can in
    turn paralyze the node proxy. See [[step-timebomb-watchdog]].
-   **Elastic reshard pytree mismatch**: on re-trace after a slice drop,
    `in_shardings` was rebuilt without the `rngs` subtree under
    `self_attention`, so recovery *detection* worked but recovery *execution*
    raised. Observed in a single incident and attributed to the application
    layer, not hardware. See [[elastic-reshard-pytree-mismatch]].
-   **Checkpoint restore failures**: partially written step directories are
    selected as latest and raise instead of falling back
    ([[incomplete-checkpoint-restore]]); corrupted checkpoints in GCS block the
    training loop ([[cloud-storage]]).
-   **Static `active_slices` telemetry caching during external slice
    suspension**: `metric_logger.py:221` (via `elastic_utils.py`) queries
    `elastic_manager.active_slice_indices` in
    `pathwaysutils/elastic/manager.py`, which is statically populated during
    `wait_for_slices()` at attempt startup and only updated on slice scale-up
    (`_monitor_new_slices()`). When a full slice is externally suspended in
    Kueue/JobSet (e.g., `example-elastic-4x1k`), step logs continue reporting
    `completed step: 115, seconds: 49.783, active_slices: 4...`
    (`insertId: no53pxjgznizk9yk`) through
    `completed step: 134, seconds: 49.889, active_slices: 4...`
    (`insertId: yhjqzzaui60a4l65`) until peer TCP disconnect timeouts
    (`lost connection to peer at http://machine/<node>/events`) force an attempt
    restart. Reconcile active slices dynamically via `live_slice_indices()`
    querying `live_devices()`. See [[elastic-vs-permanent-failure]] and
    [[pathways-fault-tolerance]].

## Connections

-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
-   **Symptoms**:
    -   [[elastic-reshard-pytree-mismatch]]
    -   [[step-timebomb-watchdog]]
    -   [[workload-startup-crash]]
-   **Concepts**:
    -   [[pathways-fault-tolerance]]
    -   [[pathways-sidecar]]
    -   [[workload-startup-failures]]
-   **Entities**:
    -   [[cloud-storage]]
    -   [[jax]]
    -   [[mcjax]]
-   **Tools**:
    -   [[orbax]]

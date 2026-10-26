---
title: Step Timebomb Watchdog
description: The step-timebomb watchdog indicates a distributed blocking hang, such
  as deep checkpointing hangs that trigger a watchdog dump, or an elastic monitor
  watchdog reset livelock where step-timebomb is repeatedly reset every ~50s while
  training is wedged.
type: symptom
internal_only: false
tags:
- tpu
- pathways
- timeout
- hangs
created: 2026-07-27
updated: 2026-10-20
---


# Step Timebomb Watchdog

## Overview

The `step-timebomb` watchdog on `pathways-head` (`jax-tpu`) and
`Error print traceback for thread:` log signatures are primary indicators of a
distributed blocking stall (such as MaxText checkpoint saves, JAX Colocated
Python / Grain `RemoteIterator` initialization and restore stalls, or
distributed MXLA collective stalls like reshape all-gather) or an
under-provisioned recovery timeout on multi-thousand-chip topologies. When the
`step-timebomb` watchdog triggers during a stall in collective operations (such
as reshape all-gather), the watchdog terminates the process with
SIGABRT/SIGTERM, which Kubernetes and GKE report as a **worker pod failure or
eviction**. Conversely, when the background elastic monitor thread repeatedly
deregisters and re-registers `step-timebomb` every ~50 seconds after a slice
disruption, it starves the 900-second timer and creates an indefinite **watchdog
reset livelock**.

## Log Signatures

### 1. Watchdog Expiry & Process Abort (`SIGTERM` / `SIGABRT`)

When a blocking stall (`jax.block_until_ready`, checkpoint save, Colocated
Python initialization/restore stall, or MXLA collective stall) exceeds the
`step-timebomb` budget (600s–900s), the watchdog dumps all Python thread stacks
and aborts `jax-tpu` (or triggers secondary worker cancellation watchdogs):

```text
CRITICAL - Timeout from watchdog 'step-timebomb' - aborting process with SIGTERM
Error print traceback for thread: ...
Fatal Python error: Aborted
```

-   `step-timebomb`
-   `step-stack-status`
-   `Error print traceback for thread:`
-   `mega_scale_Lazy.cc: Failed to poison execution launch_id=<N>
    sequencer_id=<N> when trying to cancel megascale execution on device cpu:0:
    NOT_FOUND`
-   `WatchDog Timebomb for MegaScale cancellation for launch_id=<N>
    sequencer_id=<N>`

### 2. Scale-Up / Large-Scale 4x1k Recovery Watchdog Expiry — `Aborted (core dumped)`

In large-scale 4x1k (4,096-chip) elastic snapshot runs (e.g.,
`example-elastic-4x1k`), intermittent `Aborted (core dumped)` crashes on the
`pathways-head` container occur when legitimate elastic `recover()` operations
exceed the hardcoded 900-second (15-minute) `step-timebomb` budget in MaxText
`train.py`:

```text
'step-timebomb' watchdog thread stack dump every 900 seconds. Count: 0 | File "/deps/src/maxtext/trainers/pre_train/train.py", line 972, in recover: restored_dict = jax.device_put(active_dict, sharding_dict)
Fatal Python error: Aborted | File "/deps/src/maxtext/trainers/pre_train/train.py", line 1461 in <module>
```

or while waiting for 1,024-chip slices to rejoin after an external user-space
`SIGILL (si_code=0)` kill on a slice rank-0 leader (`worker-job0-3-0-6hs64`):

```text
'step-timebomb' watchdog thread stack dump every 900 seconds. Count: 0 | File "/deps/src/maxtext/trainers/pre_train/train.py", line 861, in recover: all_active_slices = elastic.wait_for_slices(
bash: line 88:     9 Aborted                 (core dumped) ( export TPU_STDERR_LOG_LEVEL=0 ... python3 -m maxtext.trainers.pre_train.train ... )
```

A rapid shutdown timebomb may also follow on a subsequent attempt:

```text
WatchDog Timebomb for clean shutdown shutdown | Worker abnormally lost connection to the scheduler (reason: INTERNAL: Breaking pipe because shutdown doesn't happen in tests
```

-   **Root Cause:** Sharding and transferring a 405B model state across 4,096
    TPU chips via `jax.device_put(active_dict, sharding_dict)` (`train.py:972`)
    or waiting for a 1,024-chip slice to re-register via
    `elastic.wait_for_slices()` (`train.py:861`) after an external
    `SIGILL (si_code=0)` fault injection (`@0x631a = PID 25370`,
    `@0x3139 = PID 12601`, `@0x2e48 = PID 11848`; see [[sigill-worker-kill]])
    legitimately takes longer than 900 seconds (15 minutes). The `step-timebomb`
    watchdog thread mistakes this recovery progress for a hang and calls
    `abort()`, producing `Fatal Python error: Aborted` and
    `Aborted (core dumped)`.
-   **Mitigation / Fix:** Increase the `step-timebomb` watchdog timeout to
    `>=1800s` (30 minutes) for 4x1k+ topologies, or pause/reset the
    `step-timebomb` timer during `recover()` (`elastic.wait_for_slices()` and
    `jax.device_put()`). Do **not** isolate/replace the CPU head node or the
    role-confounded slice rank-0 leader hosts when host health and GCE
    maintenance checks confirm healthy machine state.

### 3. Elastic Monitor Watchdog Reset Livelock (Watchdog Starvation After Slice Loss)

After an elastic slice disruption (e.g., an external `SIGILL (si_code=0)` kill
and container restart on `worker-job0-0-0-pzf4j` in `example-elastic-4x1k`
Attempt 6), the training loop makes zero step progress, yet the 900s
`step-timebomb` never fires because the elastic monitor thread resets it every
~50 seconds:

```text
Deregistering 'step-timebomb' watchdog
Registering 'step-timebomb' watchdog with timeout 900 seconds and repeat=False
```

In this livelock state, the JobSet remains permanently wedged (e.g., for 25m32s
in `example-elastic-4x1k` Attempt 6) until an operator manually deletes the
JobSet (`io.x-k8s.jobset.v1alpha2.jobsets.delete` in
`cloudaudit.googleapis.com/activity`).

## Verification Steps

1.  Search coordinator (`pathways-head` / `jax-tpu` / `user-workload`) and
    worker logs for `step-timebomb` and `step-stack-status`.
2.  If `Timeout from watchdog 'step-timebomb'` or
    `'step-timebomb' watchdog thread stack dump every 900 seconds. Count: 0`
    fired:
    -   Inspect the dumped Python thread stacks
        (`Error print traceback for thread:`, `step-stack-status`, or
        `in recover:`) on `pathways-head` to determine whether the main thread
        was inside `train.py:recover()` (`jax.device_put` or
        `elastic.wait_for_slices`), blocked in collective operations,
        Orbax/Pathways Snapshotter checkpointing, or Colocated Python / Grain
        `RemoteIterator` deserialization (`_deserialize_specs` /
        `gcs_glob_pattern`).
    -   Check `pathways-worker` logs in the preceding 5–15 minutes for
        `Slow PjRt TPU operation detected`, `Megascale hang detected`
        ([[megascale-hang]]), or a worker crash ([[query-crash-signatures]]).
    -   If specific TPU workers crashed with `SIGABRT` (`si_code=-6`) around the
        same window, check for `WatchDog Timebomb for MegaScale cancellation`
        and verify host health with [[query-vm-termination-events]] and
        [[query-node-health-events]].
3.  If `Deregistering 'step-timebomb' watchdog` /
    `Registering 'step-timebomb' watchdog with timeout 900 seconds` repeats
    every ~50s without step completions:
    -   Identify the preceding worker disruption (such as `SIGILL (si_code=0)`
        on `worker-job0-*-0`, [[sigill-worker-kill]]) that wedged the session.
    -   Check Kubernetes/GCE audit logs (`cloudaudit.googleapis.com/activity`)
        for manual `jobsets.delete` termination.

## Playbooks & Root Causes

-   **Distributed MXLA collective hang or checkpointing stall:** Follow
    [[debugging-megascale-hangs]] and [[megascale-hang]]. When the
    `step-timebomb` watchdog triggers during a stall in collective operations
    (such as reshape all-gather), the watchdog terminates the process with
    SIGABRT/SIGTERM, which Kubernetes and GKE report as a **worker pod failure
    or eviction**.
-   **Colocated Python / Grain `RemoteIterator` (`_deserialize_specs` /
    `gcs_glob_pattern`) Step-0 / Restore Stall & Secondary MegaScale
    Cancellation Watchdog (`SIGABRT`)**:
    -   **Coordinator Stall Mechanism**: When `step-stack-status` shows the
        coordinator main thread blocked in `multihost_dataloading.py`
        (`_deserialize_specs` / `jax/experimental/colocated_python/func.py`) or
        `checkpointing.py` (`maybe_save_checkpoint` ->
        `pathways_sidecar/colocated_python_backend.py` `_result_fn`) during Step
        0 or Pathways Snapshotter restore, per-host redundant
        `gcs_utils.gcs_glob_pattern` calls or colocated Python state
        deserialization across 1,000+ hosts can exceed the 600s `step-timebomb`
        budget.
    -   **Secondary Worker `SIGABRT` Mechanism**: When Pathways attempts to
        cancel an in-flight colocated CPU execution (`mega_scale_Lazy.cc: Failed
        to poison execution launch_id=<N> sequencer_id=<N> when trying to cancel
        megascale execution on device cpu:0: NOT_FOUND`), the 3-minute `WatchDog
        Timebomb for MegaScale cancellation for launch_id=<N> sequencer_id=<N>`
        fires `SIGABRT` (`si_code=-6`) on specific TPU workers, causing hundreds
        of peers to report `lost connection to peer`
        ([[extract-culprit-hosts]]). These workers are software watchdog
        victims, **not** hardware faults, and should be verified with
        [[query-vm-termination-events]] and [[query-node-health-events]].
    -   **Remediation**:
        1.  Build initial state once on the single controller and pass it via
            function closure (rather than rebuilding per host).
        2.  Avoid broad `gcs_glob_pattern` scans on buckets with thousands of
            objects.
        3.  Raise the `step-timebomb` timeout (`>=900s`) for multi-slice `3x2k`
            colocated Python + Snapshotter workloads.
-   **Large-scale (4x1k) elastic recovery exceeding 900s `step-timebomb` budget
    (`Aborted (core dumped)`):** When `train.py:recover()` spends >900s in
    `jax.device_put(active_dict, sharding_dict)` or `elastic.wait_for_slices()`,
    increase the `step-timebomb` timeout to `>=1800s` or pause/reset the
    watchdog during `recover()`.
-   **Elastic monitor watchdog reset livelock:** The background elastic
    health/monitor loop unconditionally deregisters and re-registers
    `step-timebomb` every ~50 seconds even when no training steps have completed
    after a slice loss, preventing the 900-second watchdog from aborting a
    wedged attempt. Fix by gating `step-timebomb` registration resets on actual
    training-step progress or bounding the maximum duration of an elastic
    recovery transition.

## Connections

-   **Actions**:
    -   [[extract-culprit-hosts]]
    -   [[query-crash-signatures]]
    -   [[query-node-health-events]]
    -   [[query-vm-termination-events]]
-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
    -   [[pathways-debugging-principles]]
-   **Concepts**:
    -   [[pathways-sidecar]]
-   **Symptoms**:
    -   [[megascale-hang]]
    -   [[sigill-worker-kill]]

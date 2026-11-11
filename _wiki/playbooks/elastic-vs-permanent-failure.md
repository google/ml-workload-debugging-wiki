---
title: 'Elastic Training: Acceptable Slice Loss vs Permanent Hardware Failure'
description: 'In elastic Pathways runs a lost slice (`placed -> unplaced`, `1/2 Pathways
  Slices Now Ready`, DATA_LOSS) is expected and absorbed; it is a finding only if
  recovery fails or the same host recurs. Decide by: did the run resume (`Sufficient
  slices active`, `New slice available`, checkpoint restored)? Is there a hardware
  record on the host (HardwareFatal, hostError, node maintenance)? Is it the same
  host 2+ times?'
type: playbook
internal_only: false
tags:
- tpu
- pathways
- training
- hardware_fault
- triage
created: 2026-10-03
updated: 2026-10-21
---


# Elastic Training: Acceptable Slice Loss vs Permanent Hardware Failure

## Overview

With elastic mode a slice can drop and rejoin without a JobSet restart, so
`DATA_LOSS` / unplaced-slice errors appear in healthy runs. Do not report each
one as a root cause, and do not dismiss a host that keeps failing. Heuristic
from live-debugging sessions and past failure tables.

## Diagnostic Steps

Signatures of an elastic event (Pathways head / RM logs):

```
Transition slice ClientSliceId(...) placed -> unplaced
*** 1/2 Pathways Slices Now Ready ***
DATA_LOSS: Device device(N,...,slice=K) is unplaced
Elastic attempt ...    New slice available    Setting new slice event
Sufficient slices active    remaining assignable ...
```

Classify each event (evidence -> verdict):

-   Run resumes (`Sufficient slices active` / steps advance) and no hardware
    record on the lost slice's hosts -> acceptable elasticity: list as a
    recovered disruption, no action.
-   Hardware record on the host in the window (`error type: HardwareFatal`,
    `observed errors are: [HARDWARE]. Now taking a TPU core dump...`,
    `compute.instances.hostError`, node-problem-detector TPU error) -> hardware
    event; if elastic recovery absorbed it and the host only had a single
    isolated incident (e.g., `gke-tpu-node-xxxx` in `example-workload-2x1k`),
    record the hardware fault and monitor for recurrence (`>=2` primary
    incidents or persistent hardware repair state) before isolating/replacing
    the host per [[node-isolation-methodology]].
-   Same host in 2+ distinct primary incidents, or the same chip in 2+ Megascale
    `BAD_TPU_CHIP` digests -> candidate permanent hardware failure: **first rule
    out the Role-vs-Host Confound** ([[node-isolation-methodology]],
    [[sigill-worker-kill]]). If every incident on that host occurred on an
    index-0 slice leader pod (`-worker-job0-*-0-`) via external user-space
    `SIGILL (si_code=0, SI_USER)` and telemetry shows the same physical machine
    ran with 0 crashes when assigned a non-index-0 worker role
    (`-worker-job0-*-240-`) in a later attempt while `SIGILL` followed the
    rescheduled index-0 leader, **do NOT isolate/replace the host**.
-   Bad host differs on every restart -> not hardware; look at the software /
    recovery path.
-   Head pod logs `[PID: ...] Error while querying goodput for job <workload>.
    Skipping this cycle. Error: Productive training time is invalid. Please fix
    the logging entries.` -> **telemetry calculation edge case** in
    `ElasticGoodputCalculator` / `CheckpointBadputCalculator` after in-process
    elastic recovery without resetting `job_start_time` (fixed in the recorded
    incident; see [[goodput]]). Decouple this non-fatal Goodput query error from
    actual training stalls or worker hardware faults.
-   Head pod step logs (`metric_logger.py:221`) continue reporting
    `completed step: <N>, seconds: ..., active_slices: 4...` after a full slice
    is externally suspended via Kueue/JobSet -> **static `active_slices`
    telemetry caching** in `pathwaysutils/elastic/manager.py`
    (`self.active_slice_indices` initialized once during `wait_for_slices()` and
    only updated on slice scale-up, not external slice suspension) until peer
    TCP disconnect timeouts
    (`lost connection to peer at http://machine/<node>/events`) force an attempt
    restart.
-   Run does not recover (`Job cancelled due to`, `In-memory snapshot recovery
    failed (No active replicas found.). Falling back to persistent checkpoint.`
    followed by incomplete Orbax `.zarray` restore due to async checkpoint save
    starvation when `checkpoint_period` and `elastic_snapshot_interval` overlap,
    `RemoteIteratorWrapper.restore_state` `FileNotFoundError` (`The specified
    path gs://<bucket>/.../checkpoints/<step>/iter/process_<idx>-of-<N>.json was
    not found`) when recovering from an in-memory snapshot where
    `elastic_snapshot_interval < checkpoint_period`,
    `jax._src.sharding.IndivisibleError` when scaling to a non-power-of-2 slice
    count such as $3 \times 1\text{k} = 6,144$ devices dividing axis size
    $8,192$, Orbax `'rngs'` PyTree reshard mismatch, `step-timebomb` watchdog
    expiry/livelock during 4x1k scale-up `recover()` (`jax.device_put` or
    `elastic.wait_for_slices` exceeding 900s in the recorded incident), or
    post-recovery stall where `*** 3/3 Pathways Slices Now Ready ***` is
    restored after a GCE `compute.instances.hostError` replacement yet `jax-tpu`
    stalls at `elastic_utils.py:108] Checking latest checkpoint:` while
    replacement slice workers stall inside `UpdateAffectedComponent:
    jit_train_step.<id> load_slices: 1 reincarnate_slices: 2` /
    `Attempting to load component jit_train_step.<id> @ 1 at worker`) ->
    recovery-path software defect: [[elastic-reshard-pytree-mismatch]],
    [[incomplete-checkpoint-restore]], [[step-timebomb-watchdog]],
    [[pathways-fault-tolerance]].

## Two-Stage GCE `hostError` + Post-Recovery `UpdateAffectedComponent` (`load_slices: 1 reincarnate_slices: 2`) / `Checking latest checkpoint` Stall

In elastic replica-resize workloads (e.g., `example-workload-3x2k` on $3 \times
2,048$ TPU7x chips with `--num_elastic_instances=3`), a confirmed physical host
failure can trigger a **two-stage failure** where infrastructure recovery
succeeds (`*** 3/3 Pathways Slices Now Ready ***`) but the workload stalls
silently in the post-recovery software path:

1.  **Stage 1 — Confirmed GCE `hostError` & Kueue Replacement:** A worker node
    (e.g., `gke-tpu-node-xxxx`, hosting
    `example-workload-3x2k-worker-job0-1-144-6jd7h`) suffers a sudden
    host/daemon failure (`compute.instances.hostError`,
    (`DEADLINE_EXCEEDED: lost connection to peer at
    http://machine/gke-tpu-node-xxxx/events`), `pathways-rm` removes the dead
    slice job (`*** 2/3 Pathways Slices Now Ready ***`) and propagates
    `DATA_LOSS` to `jax-tpu`. Because the node died abruptly, the pod remains
    bound until `kueue-pod-termination-controller` forcefully deletes it
    pod `example-workload-3x2k-worker-job0-1-144-f4lnd` to spawn on replacement
    node `gke-tpu-node-xxxx` and restore `*** 3/3 Pathways Slices Now Ready ***`
2.  **Stage 2 — Dual Post-Recovery Software Stall
    (`Checking latest checkpoint` + `UpdateAffectedComponent`):** Despite `3/3`
    slices being ready:
    -   **Head (`jax-tpu`) Stall:** After catching `DATA_LOSS` in
        `pathwaysutils.elastic.manager`, `jax-tpu` on
        `example-workload-3x2k-pathways-head-0-0-9mvnk` logs
        `elastic_utils.py:108] Checking latest checkpoint:
        emits zero further logs for 54.5 minutes until `SIGTERM`
        (`EXIT_CODE=143`).
    -   **Replacement Slice Component Load Stall:** `pathways-proxy` issues
        `UpdateAffectedComponent: jit_train_step.<id> load_slices: 1
        replacement worker logs
        `Attempting to load component jit_train_step.<id> @ 1 at worker`
        (`PERSISTENT_FETCH`), followed by `Device set detected idle`
        replacement worker logs `Component is not loaded yet; deferring unload
        until load completes: jit_train_step.<id> @ 1`
        deadlocked.
3.  **Resolution:** Because GCE `compute.instances.hostError` already drained
    `gke-tpu-node-xxxx` and replacement node `gke-tpu-node-xxxx` is healthy, no
    manual node cordon/isolate is required; guard `elastic_utils.py` checkpoint
    checks and Pathways `UpdateAffectedComponent`
    (`load_slices: 1, reincarnate_slices: 2`) with a recovery watchdog timeout
    so stalled slice reloads fail fast.

## Static `active_slices` Telemetry Caching During External Kueue/JobSet Slice Suspension

In elastic Pathways runs (e.g., `example-workload-4x1k` on $4 \times 1,024$
TPU7x chips), operators testing fault tolerance by suspending a full slice
externally via Kueue or JobSet may observe that MaxText step logs on
`pathways-head` (`metric_logger.py:221`) continue reporting `active_slices: 4`
across multiple steps
(`completed step: 115, seconds: 49.783, active_slices: 4...`,
`completed step: 134, seconds: 49.889, active_slices: 4...`,

1.  **Static `self.active_slice_indices` Caching:** `metric_logger.py:221` calls
    `elastic_utils.py` (`len(pathwaysutils.elastic.get_active_slice_indices())`
    / `len(elastic_manager.active_slice_indices)`). In
    `pathwaysutils/elastic/manager.py`, `self.active_slice_indices` is
    statically populated during `wait_for_slices()` at attempt startup. The
    background monitor thread (`_monitor_new_slices()`) only polls for newly
    available slices scaling **up** and does not dynamically decrement
    `self.active_slice_indices` when a slice is suspended externally in
    Kubernetes.
2.  **Delayed Failure Until Peer Socket Timeout:** Step logs continue printing
    the stale `active_slices: 4` value until the suspended slice's worker VMs
    sever their peer TCP connections (`lost connection to peer at
    http://machine/gke-tpu-node-xxxx/events since 17.767s ago`,
    `example-workload-4x1k-pathways-worker-104-0-v5jcb` / `gke-tpu-node-xxxx`),
    forcing the attempt to abort and restart.
3.  **Resolution:** Do not mistake stale `active_slices: 4` step logs for a
    hardware fault; reconcile active slices dynamically via
    `live_slice_indices()` querying `live_devices()` in `elastic_utils.py` and
    extend `_monitor_new_slices()` in `pathwaysutils/elastic/manager.py` to
    detect slice suspension/termination events in real time.

## Non-Elastic Pathways Interruption Behavior — Slice Drop to 0 Without JobSet Restart

When a Pathways workload is configured **without elasticity**
(`elastic_enabled=False`, e.g., `example-nonelastic-4x1k` in the recorded
incident), interrupting slice coordinator pods (`worker-job0-*-0`) via external
signals (`SIGILL (si_code=0)`, [[sigill-worker-kill]]) can drop active slices
(`N/M Pathways Slices Now Ready` down to `0/4`) **without triggering a
Kubernetes JobSet restart**:

1.  **In-Place Pod Container Restart Below `backoffLimit`:** When a single
    `pathways-worker` container is killed by `SIGILL (si_code=0)` (e.g., on
    `example-nonelastic-4x1k-worker-job0-0-0-tgwvd` at `09:39:26Z`,
    `worker-job0-1-0-b8bwr` at `11:05:43Z`, and `worker-job0-2-0-srftg` at
    `12:23:45Z` and `13:38:19Z`), Kubelet restarts the container in-place inside
    the existing pod. Because the Pod does not permanently fail and the child
    Kubernetes Job does not exceed its `backoffLimit`, the JobSet controller's
    `failurePolicy: RestartJobSet` is never triggered.
2.  **`pathways-head` Remains Alive Without Exiting:** With
    `elastic_enabled=False`, the Pathways Resource Manager (`pathways-rm`) does
    not perform elastic slice resharding or in-process `recover()`, and if the
    head `jax-tpu` process stays blocked in an IFRT call or prints watchdog
    thread tracebacks (`Error print traceback for thread:`) without exiting with
    a non-zero status code, the head pod also remains `Running`.
3.  **Resolution:** This is expected orchestration behavior when combining
    in-place container restarts with `elastic_enabled=False`. To force a full
    JobSet restart on any worker interruption in non-elastic mode, either set
    the worker pod `restartPolicy: Never` with `backoffLimit: 0` in the JobSet
    spec, or ensure `step-timebomb` / IFRT fatal error propagation terminates
    the `pathways-head` container with a non-zero exit code (see
    [[pathways-fault-tolerance]] and [[jobset-yaml-structure]]).


## `TimeoutError: Timed out waiting for N slices. Only M active after ... seconds` (`MAXTEXT CONFIG ERROR`) & `PlaceSlice() failed: currently infeasible` During Rapid Coordinator Restarts

When `pathways-head` restarts rapidly (for example, after a
[[step-timebomb-watchdog]] abort) while TPU workers are still holding the prior
Pathways session lease, `SliceBuilder` logs:

```
UNAVAILABLE: Worker <pod>:8471 is in session <id>, waiting until expires
PlaceSlice() failed: currently infeasible
Virtual slice reservation goes over capacity for dimension 0:0 ... failure mode: EVERY_SLICE
```

Because the new coordinator cannot place the slice until the prior worker
session lease expires, the user container exits after 300s with:

```
TimeoutError: Timed out waiting for N slices. Only M active after ... seconds
```

(often surfaced under `MAXTEXT CONFIG ERROR`). Classify this as a transient
session-lease expiry overlap caused by a rapid coordinator restart rather than a
hardware fault or static configuration defect, and verify that the subsequent
coordinator restart places all `N/N` slices once the prior lease expires.

## Resolution

Find the host from the RM's own words ([[extract-culprit-hosts]]:
`lost connection to peer at http://machine/<node>`), never from the `slice=K`
coordinate. Count primary incidents, not reporting peers, and check
`role_confound_candidates` before recommending hardware replacement.

## Connections

-   **Actions**:

    -   [[extract-culprit-hosts]]
    -   [[query-crash-signatures]]

-   **Playbooks**:

    -   [[rca-invariants]]
    -   [[node-isolation-methodology]]

-   **Symptoms**:

    -   [[sigill-worker-kill]]
    -   [[elastic-reshard-pytree-mismatch]]
    -   [[incomplete-checkpoint-restore]]
    -   [[step-timebomb-watchdog]]
    -   [[worker-hardware-fatal-error]]
    -   [[megascale-hang]]

-   **Concepts**:

    -   [[goodput]]
    -   [[jobset-yaml-structure]]
    -   [[pathways-fault-tolerance]]
    -   [[failure-tables]]

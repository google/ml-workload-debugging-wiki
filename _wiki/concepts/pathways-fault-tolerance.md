---
title: Pathways Fault Tolerance and Error Propagation
type: concept
internal_only: false
tags:
- tpu
- pathways
- training
created: 2026-06-13
updated: 2026-10-16
---


## Overview: error poisoning

When a task fails on a worker (in XLA or in Colocated Python), Pathways marks
its result reference with the error ("poisoned"). Anything that depends on that
reference fails at once with the same error, so the user job fails fast instead
of hanging.

## DATA_LOSS "pipe closed" / "socket closed"

The gRPC connection between the User Job/Proxy and the Pathways backend was cut.
This is almost always downstream of a worker death (hardware error, OOM kill).
Check the pod's Kubernetes events for OOM or Eviction.

## Error Aggregator

The Resource Manager runs an error aggregator that collects error reports from
all workers and tries to deduce the primary failure. Look in the `pathways-rm`
logs.

## Recovery Modes and Resilient Training

Pathways supports three resilient training recovery strategies on GKE alongside
persistent checkpointing:

### 1. Suspend-Resume (Planned Maintenance)

*   **Mechanism**: Enabled by default on Cloud deployments. Pathways listens for
    GKE maintenance preemption notices, suspends the job, saves an HBM snapshot
    to persistent storage (`--gcs_scratch_location`), and restores it once
    maintenance finishes without restarting the training script or reloading
    from an Orbax checkpoint.
*   **Limitation**: Only stateless components (such as XLA computations) are
    restorable. Stateful components (such as [[pathways-sidecar]] / Colocated
    Python) cannot use transparent HBM Suspend-Resume and fall back to Elastic
    Training or full JobSet restart.

### 2. Elastic Training (`pathwaysutils.elastic.manager.Manager`)

*   **Mechanism**: Allows a multi-slice training loop to continue when a slice
    fails or is preempted (reshard down) and scale back up when replacement
    slices become healthy (reshard up), keeping both the JAX controller and
    Resource Manager alive so compilation caches remain warm.

*   **In-Memory Host Snapshots (`Snapshotter`)**: `manager.maybe_snapshot()`
    saves fast asynchronous host-memory snapshots at `snapshot_period` (or on a
    preemption signal). On `jax.errors.JaxRuntimeError`,
    `manager.maybe_reshard_down()` restores from the host-memory snapshot if a
    complete replica survives, or falls back to the persistent [[orbax]]
    checkpoint on GCS.

*   **Key Parameters**:

    *   `snapshot_period`: Steps between in-memory host snapshots.
    *   `reshard_check_period`: Steps between checks for newly healthy slices in
        `manager.maybe_reshard_up()`.
    *   `max_reshard_retry_count`: Retry limit for reshard operations.
    *   `max_elastic_down_event_count`: Maximum slice reductions before failing
        the job to trigger a full restart.

*   **Single-Replica Restore**: Pathways can restore one replica from GCS and
    broadcast it to the other replicas over DCN/ICI, cutting checkpoint restore
    time.

### 3. GKE JobSet Hot-Swap (Spare Capacity or Priority Preemption)

*   **Default Mode**: When a TPU slice fails, GKE remaps the failed worker job
    onto an idle spare slice in the cluster (`spare` node pool) without tearing
    down healthy slices.
*   **Priority Mode**: Uses Kubernetes `PriorityClass` (`priorityClassName` on
    the `pathways-worker` Pod spec) so a high-priority production JobSet
    immediately preempts a lower-priority workload on spare slices, recovering
    in under a minute.

## Non-Elastic Pathways Interruption Behavior — Slice Drop to 0 Without JobSet Restart

When a Pathways workload runs with elasticity disabled (`elastic_enabled=False`,
e.g., `example-nonelastic-4x1k` in the recorded incident) and slice coordinator
pods (`worker-job0-*-0`) are interrupted by external signals
(`SIGILL (si_code=0)`, [[sigill-worker-kill]]), active slices can drop
(`N/M Pathways Slices Now Ready` down to `0/4`) without triggering a Kubernetes
JobSet restart:

1.  **In-Place Container Restart Below `backoffLimit`:** Kubelet restarts the
    terminated `pathways-worker` container in-place within the existing pod, so
    the Pod never transitions to `Failed` and the child Kubernetes Job never
    exceeds its `backoffLimit` to trigger `failurePolicy: RestartJobSet`.
2.  **`pathways-head` Continues Running:** With `elastic_enabled=False`,
    `pathways-rm` does not execute elastic resharding or in-process `recover()`,
    and if `jax-tpu` on `pathways-head` remains alive
    (`Error print traceback for thread:` without exiting non-zero), the JobSet
    controller sees all pods as `Running`.
3.  **Configuration Guidance:** If whole-JobSet recreation is desired upon
    worker interruption in non-elastic mode, configure `restartPolicy: Never`
    and `backoffLimit: 0` on the worker Job template, or ensure `step-timebomb`
    aborts `pathways-head` with a non-zero exit code (see
    [[elastic-vs-permanent-failure]] and [[jobset-yaml-structure]]).

## Post-Recovery Elastic `UpdateAffectedComponent` (`load_slices: 1 reincarnate_slices: 2`) & `Checking latest checkpoint` Stall

In elastic replica-resize workloads (`--num_elastic_slices=3`, e.g.,
`example-elastic-3x2k` on $3 \times 2,048$ TPU7x chips in the recorded
incident), a slice lost to a physical host failure
is forcefully evicted by Kueue (`KueueForcefullyDeleted`,
(`example-elastic-3x2k-worker-job0-1-144-f4lnd` on `gke-tpu-node-xxxx`),
restoring `*** 3/3 Pathways Slices Now Ready ***`
due to a dual post-recovery stall:

1.  **Head `elastic_utils.py:108` Checkpoint Verification Stall:** After
    catching `DATA_LOSS` in `pathwaysutils.elastic.manager`, `jax-tpu` on
    `pathways-head` logs `elastic_utils.py:108] Checking latest checkpoint:
    without emitting any further log lines until `SIGTERM` (`EXIT_CODE=143`).
2.  **Replacement Slice `UpdateAffectedComponent` Load Stall:** When
    `pathways-proxy` triggers `UpdateAffectedComponent: jit_train_step.<id>
    replacement slice worker logs
    `Attempting to load component jit_train_step.<id> @ 1 at worker`
    (`PERSISTENT_FETCH`), followed by `Device set detected idle`
    logs `Component is not loaded yet; deferring unload until load completes:
    post-recovery component load never completed.

## Static `active_slices` Telemetry Caching During External Slice Suspension

In elastic Pathways workloads (`example-elastic-4x1k` on $4 \times 1,024$ TPU7x
chips), `pathwaysutils/elastic/manager.py` initializes
`self.active_slice_indices` statically during `wait_for_slices()` at attempt
startup and its background monitor (`_monitor_new_slices()`) only watches for
new slices scaling up. When a full slice is externally suspended via
Kueue/JobSet, MaxText's `metric_logger.py:221` (via `elastic_utils.py`)
continues logging `completed step: 115, seconds: 49.783, active_slices: 4...`
`completed step: 134, seconds: 49.889, active_slices: 4...`
(`lost connection to peer at http://machine/gke-tpu-node-xxxx/events`,
slices dynamically via `live_slice_indices()` / `live_devices()` and update
`_monitor_new_slices()` to detect slice suspension.

## Connections

-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
    -   [[workload-disruption-orchestration]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[elastic-reshard-pytree-mismatch]]
    -   [[false-preemption-sigterm]]
    -   [[incomplete-checkpoint-restore]]
    -   [[sigill-worker-kill]]
    -   [[step-timebomb-watchdog]]
-   **Concepts**:
    -   [[ici-errors]]
    -   [[jobset-yaml-structure]]
    -   [[pathways]]
    -   [[pathways-sidecar]]
-   **Entities**:
    -   [[maxtext]]
-   **Tools**:
    -   [[orbax]]

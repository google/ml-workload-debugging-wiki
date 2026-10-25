---
title: Pathways Debugging Principles
type: playbook
internal_only: false
tags:
- tpu
- pathways
- training
- triage
created: 2026-06-11
updated: 2026-10-16
---


## Overview

Triage order and common failure patterns for Pathways on GKE (`JobSet`,
`LeaderWorkerSet`, `xpk`, and Cluster Director) workloads.

## Container Triage Procedure

Read logs container by container in this order:

1.  **`user-workload` / `main` / `jax-tpu` (JAX & Colocated Python on
    `pathways-head`):** JAX runtime errors, Orbax dict key mismatches, IFRT
    proxy client errors, Colocated Python `pathwaysutils.PyException` dataset
    errors ([[pathways-sidecar]]), `step-timebomb` / `step-stack-status`
    watchdog timeouts (`Fatal Python error: Aborted` / `Aborted (core dumped)`
    when 4x1k scale-up `recover()` spends >900s in `jax.device_put` or
    `elastic.wait_for_slices`; see [[step-timebomb-watchdog]] and), and
    post-recovery stalls where `jax-tpu` stops emitting logs after
    `elastic_utils.py:108] Checking latest checkpoint:
    gs://<bucket>/.../checkpoints/<step>/`.

2.  **`pathways-rm`:** slice allocation and clustering (e.g.
    `"*** N/M Pathways Slices Now Ready ***"`), Error Aggregator primary failure
    reports, and GCS scratch bucket validation.

3.  **`pathways-worker` & `colocated-python-sidecar`:** start with leader
    workers (`completion-index = 0`, `worker-job0-*-0`) to find the first
    failing worker (such as TPU HBM OOM `RESOURCE_EXHAUSTED`, external
    user-space `SIGILL (si_code=0)` kills in [[sigill-worker-kill]], or
    Colocated Python sidecar crashes in [[pathways-sidecar]]) and avoid chasing
    downstream cascades. On replacement slices after an elastic recovery
    (`*** 3/3 Pathways Slices Now Ready ***`), also check for
    `Attempting to load component jit_train_step.<id> @ <slice> at worker`
    followed at teardown by `Component is not loaded yet; deferring unload until
    load completes: jit_train_step.<id> @ <slice>`
    ([[elastic-vs-permanent-failure]]).

4.  **Proxy server (`pathways-proxy`):** skip secondary proxy socket-closure
    logs when a worker died first ([[ifrt-proxy-terminated]]). Inspect
    `pathways-proxy` directly when diagnosing XLA compilation failures, IFRT
    protocol version mismatches, compiler flag issues, or post-recovery elastic
    component reload hangs where `pathways-proxy` logs `UpdateAffectedComponent:
    jit_train_step.<id> load_slices: 1 reincarnate_slices: 2` followed by
    `Device set detected idle`.

## Colocated Python / Grain `RemoteIterator` Stall & Secondary MegaScale Cancellation Watchdog (`SIGABRT`)

-   **Coordinator Stall (`step-timebomb`)**: When `step-stack-status` in the
    coordinator (`user-workload` / `jax-tpu`) shows the main thread blocked in
    `multihost_dataloading.py` (`_deserialize_specs` /
    `jax/experimental/colocated_python/func.py`) or `checkpointing.py`
    (`maybe_save_checkpoint` -> `pathways_sidecar/colocated_python_backend.py`
    `_result_fn`) during Step 0 or Pathways Snapshotter restore, per-host
    redundant `gcs_utils.gcs_glob_pattern` calls or colocated Python state
    deserialization across 1,000+ hosts can exceed the 600s `step-timebomb`
    budget ([[step-timebomb-watchdog]]).
-   **Secondary Worker `SIGABRT` (`si_code=-6`)**: When Pathways attempts to
    cancel an in-flight colocated CPU execution (`mega_scale_Lazy.cc: Failed to
    poison execution launch_id=<N> sequencer_id=<N> when trying to cancel
    megascale execution on device cpu:0: NOT_FOUND`), the 3-minute `WatchDog
    Timebomb for MegaScale cancellation for launch_id=<N> sequencer_id=<N>`
    fires `SIGABRT` (`si_code=-6`) on specific TPU workers, causing hundreds of
    peers to report `lost connection to peer` ([[extract-culprit-hosts]]). These
    workers are software watchdog victims, **not** hardware faults
    (`isolate_recommended: []`), and should be verified with
    [[query-vm-termination-events]] and [[query-node-health-events]].

-   **Remediation**: Build initial state once on the single controller and pass
    it via function closure (rather than rebuilding per host), avoid broad
    `gcs_glob_pattern` scans on buckets with thousands of objects, and raise the
    `step-timebomb` timeout (`>=900s`) for multi-slice `3x2k` colocated Python +
    Snapshotter workloads.

## Passing XLA Flags in Pathways

Because XLA compilation runs inside `pathways-proxy` rather than the user
container, `LIBTPU_INIT_ARGS` set on the user container does **not** affect the
Pathways compiler. Pass XLA compiler flags (including `--xla_dump_to`) to the
`pathways-proxy` container via its container `args:` in the JobSet manifest or
`--pathways-proxy-args` in `gcluster`.

## Common Configuration & Runtime Signatures

*   **Benign Warning (`GetTopologyForDevices`)**: `UNIMPLEMENTED:
    GetTopologyForDevices is not supported for the IFRT proxy client` is an
    expected informational warning on startup and can be safely ignored.

*   **GCS Scratch Bucket Lifecycle Limit (`50 prefixes per config`)**:
    `INVALID_ARGUMENT: Lifecycle matches_prefix cannot specify more than 50
    prefixes per config` occurs when too many workloads register distinct
    lifecycle prefixes on the same GCS bucket. Share a common
    `--gcs_scratch_location` prefix across workloads and delete stale `Delete`
    lifecycle rules on the bucket.

*   **IFRT Protocol Version Mismatch**: `IFRT Proxy client and server failed to
    agree on the protocol version; supported versions: client = [1, 1], server =
    [3, 14]` indicates the JAX client image is out of sync with the
    `pathways-proxy` image (`jax-<version>`). Align both container versions.

*   **Worker `SIGTERM` + `RESOURCE_EXHAUSTED`**: A worker receiving `SIGTERM`
    accompanied by `RESOURCE_EXHAUSTED` in worker logs indicates a TPU HBM
    Out-Of-Memory error (reduce per-device batch size or adjust sharding),
    whereas `SIGTERM` with `node_shutdown` and no host/HBM fault points to
    [[false-preemption-sigterm]] or GKE maintenance.

*   **JobSet Name Length Exceeds 22 Characters**: Kubernetes headless service
    DNS labels have a 63-character limit; Pathways JobSet names longer than 22
    characters fail pod creation when suffixed with replicated job and pod
    indices ([[forever-suspended-jobset]]).

*   **Missing Cloud Logging IAM Role**:
    `Permission 'logging.logEntries.create' denied` when the node pool service
    account lacks `roles/logging.logWriter`.

## Connections

-   **Actions**:
    -   [[extract-culprit-hosts]]
    -   [[query-node-health-events]]
    -   [[query-vm-termination-events]]
-   **Workflows**:
    -   [[pathways-cluster-readiness]]
    -   [[pathways-debugging-playbook]]
-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[false-preemption-sigterm]]
    -   [[forever-suspended-jobset]]
    -   [[ifrt-proxy-terminated]]
    -   [[proxy-connection-failed]]
    -   [[sigill-worker-kill]]
    -   [[step-timebomb-watchdog]]
-   **Concepts**:
    -   [[jobset-yaml-structure]]
    -   [[pathways]]
    -   [[pathways-cluster-readiness-checks]]
    -   [[pathways-fault-tolerance]]
    -   [[pathways-sidecar]]
    -   [[superslicing-subslicing]]

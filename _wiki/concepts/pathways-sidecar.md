---
title: Pathways Sidecars and Colocated Python
type: concept
internal_only: false
tags:
- tpu
- pathways
- training
created: 2026-06-13
updated: 2026-10-16
---


## Overview

The Pathways **Sidecar** runs Colocated Python
(`jax.experimental.colocated_python`) — user code for data loading,
preprocessing, and checkpointing — directly on the TPU worker hosts instead of
the central JAX controller, avoiding a DCN bottleneck at the single controller
CPU.

## Architecture and Communication Protocol

*   **Container Deployment**: In GKE deployments (including `xpk` and Cluster
    Director), the Colocated Python sidecar runs as a companion container inside
    each TPU worker Pod alongside `pathways-worker`, typically using the same
    user workload container image and starting a gRPC server (`server.py`).

*   **gRPC Transport**: Each `pathways-worker` communicates with its local
    Colocated Python sidecar over a custom versioned protocol on gRPC transport
    to dispatch host-local Python functions and exchange device/host buffers.

*   **Distributed Data Loading**: Frameworks like [[maxtext]] use Colocated
    Python (`RemoteIterator` in
    `MaxText/input_pipeline/multihost_dataloading.py`) so each TPU host fetches
    and preprocesses its own batch shard directly into local TPU HBM rather than
    streaming all input batches through the single controller.

## Execution Modes and Failure Behavior

1.  **In-process**: runs as a library inside the Pathways worker. Lowest
    latency, no isolation: a sidecar crash or OOM kills the worker at once and
    looks like a hardware fault.
2.  **Subprocess / Sidecar container**: runs as a separate process or container
    alongside the worker. The worker survives a sidecar crash and reports it, so
    Python-level bugs are faster to triage.

3.  **Suspend-Resume Limitation**: Because Colocated Python sidecars hold
    arbitrary Python host state, workloads using Colocated Python cannot use
    transparent HBM Suspend-Resume during GKE maintenance preemptions and must
    rely on Elastic Training or checkpoint recovery.

## Orbax `array_metadata_store` Scale-Up Barrier Deadlock

In Pathways workloads using Colocated Python (`colocated_python_sidecar`) and
[[orbax]](e.g., `example-elastic-2x64`), calling
`ocp_pathways.register_type_handlers()` **after**
`create_orbax_checkpoint_manager` (with
`array_metadata_store=array_metadata_store_lib.Store(primary_host=None)`) leaves
Orbax's `ArrayHandler` configured with `primary_host=0`. During dynamic elastic
scale-up (`1x64 -> 2x64`), non-zero sidecar processes on the newly added slice
block in `_maybe_create_base_dir()` waiting up to `600s` for worker 0 to create
`array_metadatas/`, preventing JAX from queuing jitted TPU programs and
triggering `Megascale detects a hang that is caused some worker not queuing
programs to TPU` (`StepGloballyInProgress: false`, `Already completed: 0`,
`copy_to_host_buffer: DEADLINE_EXCEEDED`; see [[megascale-hang]]). Always call
`ocp_pathways.register_type_handlers()` **before** creating the Orbax checkpoint
manager and pass `array_metadata_store=None` (MaxText commit
`71633707dec161897d7b342d31df286413f03a6d`).

## Colocated Python Dataset Preprocessing Exceptions (`pathwaysutils.PyException`)

Unhandled Python exceptions raised inside Colocated Python dataset loading or
batch formation (such as `np.split` raising
`ValueError: array split does not result in an equal division` / `ValueError:
Unable to put to devices shape (2,

8192\) with local device count 4 at
['inputs']`in`multihost_dataloading.py:form_global_array_colocated_python`when`grain_use_elastic_iterator=True`,`packing=False`, and`grain_worker_count >
0`in`example-elastic-workload`) abort the IFRT operation and surface on the Pathways coordinator (`pathways-head`)
as:

```text
While processing examples for unknown_output_specs_and_push_result.<id> [type.googleapis.com/ifrt.UserContext='<id>'] [type.googleapis.com/pathwaysutils.PyException='']
```

Inspect the preceding Python traceback on `pathways-head` or the sidecar logs to
identify the underlying application `ValueError` ([[workload-startup-crash]]).

## Scale-Up Recovery Watchdog Aborts (`step-timebomb` / `Aborted (core dumped)`)

In 4x1k (4,096-chip) elastic snapshot workloads using Colocated Python (e.g.,
`example-elastic-4x1k`), `train.py:recover()` can legitimately spend

> 900 seconds either sharding/distributing model weights across 4,096 chips in
> `jax.device_put(active_dict, sharding_dict)` (`train.py:972`) or waiting for a
> 1,024-chip slice to rejoin in `elastic.wait_for_slices()` (`train.py:861`)
> after an external user-space `SIGILL (si_code=0)` kill on a slice rank-0
> leader. When this exceeds the hardcoded 900-second `step-timebomb` watchdog
> budget on `pathways-head`, the watchdog thread dumps the stack
> (`'step-timebomb' watchdog thread stack dump every 900 seconds. Count: 0`) and
> aborts Python (`Fatal Python error: Aborted` /
> `bash: line 88: 9 Aborted (core dumped)`). Increase the `step-timebomb`
> timeout to `>=1800s` or pause/reset the watchdog during `recover()` (see
> [[step-timebomb-watchdog]]).

## Connections

-   **Concepts**:
    -   [[pathways]]
    -   [[pathways-fault-tolerance]]
-   **Playbooks**:
    -   [[pathways-debugging-principles]]
-   **Entities**:
    -   [[maxtext]]
-   **Tools**:
    -   [[orbax]]
-   **Symptoms**:
    -   [[megascale-hang]]
    -   [[sigill-worker-kill]]
    -   [[step-timebomb-watchdog]]
    -   [[workload-startup-crash]]

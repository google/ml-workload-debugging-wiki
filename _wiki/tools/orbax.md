---
title: Orbax Tips and Tricks
type: tool
internal_only: false
tags:
- tpu
- pathways
- training
created: 2026-05-13
updated: 2026-10-16
---


## Overview

Orbax is a library used for checkpointing in JAX workloads. This page documents
tips and tricks for debugging performance and recovery issues related to Orbax,
including Pathways distributed persistence (`ENABLE_PATHWAYS_PERSISTENCE=1`),
slow in-memory checkpoint times, and incomplete checkpoint errors after elastic
scale-down.

## Common Use Cases in Debugging

### Pathways Distributed Checkpointing (`ENABLE_PATHWAYS_PERSISTENCE=1`)

In Pathways single-controller workloads, reading or writing full model
checkpoints through the single CPU controller creates a severe DCN bottleneck.
Setting `ENABLE_PATHWAYS_PERSISTENCE=1` on the workload container instructs
Orbax (`orbax.checkpoint.CheckpointManager`) to use the Pathways Persistence API
and [[pathways-sidecar]](Colocated Python) so each TPU worker host reads and
writes its own array shards directly to and from Cloud Storage (`gs://`).

### Slow In-Memory Checkpoint Time

**Symptom**: Orbax in-memory checkpointer takes an unexpectedly long time (e.g.,
up to 300 seconds) to block. It should typically take 3 seconds or less.

**Example Log Message**:

```
I0528 00:33:52.689874 133463539525632 checkpointer_orbax_emergency.py:761] In-mem ckpt blocking time is 308.148077s.
```

**Mitigation**: This issue may be resolved by setting the
`TPU_PREMAPPED_BUFFER_SIZE` to be large enough to host the full checkpoints on
the host. Set the following environment variables and flags:

```bash
export TPU_PREMAPPED_BUFFER_SIZE=34359738368
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=34359738368
```

In addition, also set:

```bash
megascale_grpc_premap_memory_bytes=34359738368
```

### Checkpoint Corruption & Incomplete Checkpoints

**Symptom**: Restoring from a checkpoint fails with errors indicating that the
files are corrupted or incomplete (`ValueError: Found incomplete checkpoint`).
This often happens if the training job was interrupted while saving the
checkpoint.

**Mitigation/Debugging**:

*   **Cancelled Async Saves Remain Cached in `CheckpointManager._checkpoints`**:
    When an elastic disruption interrupts an in-flight async checkpoint save
    (`Started an asynchronous checkpoint save for step <N>`) and cancels
    background saves (`Immediately cancelling background checkpoint saving
    operations on CheckpointManager...`), Orbax leaves an unfinalized directory
    on GCS **and** leaves `<N>` in the live `CheckpointManager._checkpoints`
    in-memory list. If the same coordinator process later falls back to
    `checkpoint_manager.latest_step()`, it selects `<N>` from `_checkpoints` and
    raises `ValueError: Found incomplete checkpoint`. Reloading
    `CheckpointManager._checkpoints` from finalized GCS directories (or
    re-instantiating `CheckpointManager`, as fixed in
    `AI-Hypercomputer/maxtext#5260`) resolves this.

*   **Atomic Saves**: Orbax writes checkpoints to a temporary directory and
    finalizes them on completion (`commit_success.txt`), so a *fresh*
    `CheckpointManager` in a new process automatically ignores unfinalized
    directories.

*   **Verify Checkpoint Completeness**: Check if the directory structure is
    complete (`_CHECKPOINT_METADATA`, `commit_success.txt`, `items/`, and
    `iter/`).

*   **Fallback to Previous Checkpoint**: If an unfinalized directory blocks
    restore, reload `_checkpoints` or fall back to the previous finalized
    checkpoint directory.

### Shape & PyTree Mismatches during Restore

**Symptom**: When attempting to restore a checkpoint, Orbax throws an error
indicating that the shape or PyTree keys of the tensor in the checkpoint do not
match the expected model state.

**Mitigation/Debugging**:

*   **Missing `abstract_train_state` on Fallback**: In MaxText
    `train.py:recover()`, falling back from a failed in-memory snapshot restore
    (`No active replicas found.`) to `checkpointing.load_state_if_possible()`
    without passing `abstract_train_state` returns a raw dict lacking the `rngs`
    subtree (`ValueError: Mismatch details (1 found): ... symmetric difference
    on key sets is rngs`). Fixed in `AI-Hypercomputer/maxtext#5260`.

*   **Model Definition Changes**: Ensure that the model definition used for
    restore matches the one used for saving.

*   **Sharding Changes**: If the sharding strategy has changed (e.g., different
    number of devices or different mesh axes), pass the target
    `abstract_train_state` so Orbax remaps the arrays onto the current mesh.

*   **Inspect Shapes**: Use Orbax utilities to inspect the shapes stored in the
    checkpoint if possible, or compare the current model's parameter shapes with
    the expected shapes in the error message.

### Pathways Colocated Python Scale-Up Deadlock (`array_metadata_store`)

**Symptom**: During dynamic scale-up (`1x64 -> 2x64`) in a Pathways Colocated
Python ([[pathways-sidecar]]) workload (e.g., `example-elastic-2x64`),
`pathways-head` aborts with `copy_to_host_buffer: DEADLINE_EXCEEDED: Megascale
hang detected: Timed out waiting for 24 graphs to complete at launch_id 1920.
Already completed: 0. StepGloballyInProgress: false` and `Megascale detects a
hang that is caused some worker not queuing programs to TPU`
([[megascale-hang]]).

**Root Cause & Mitigation**: If `ocp_pathways.register_type_handlers()` is
called *after* `create_orbax_checkpoint_manager` with
`array_metadata_store=array_metadata_store_lib.Store(primary_host=None)`, Orbax
defaults `primary_host=0`. During scale-up, non-zero `colocated_python_sidecar`
workers on Slice 1 block in
`orbax/checkpoint/_src/metadata/array_metadata_store.py`
(`_maybe_create_base_dir()`) waiting up to `600s` for worker 0 to create
`array_metadatas/`. Call `ocp_pathways.register_type_handlers()` **before**
`create_orbax_checkpoint_manager` and pass `array_metadata_store=None` (MaxText
commit `71633707dec161897d7b342d31df286413f03a6d`).

### Dataset Iterator Checkpointing vs. In-Memory Snapshotter Intervals (`RemoteIteratorWrapper`)

**Symptom**: During elastic scale-down recovery when using the Pathways
in-memory Snapshotter (`elastic_backup_kind=snapshot`,
`elastic_snapshot_interval=10` < `checkpoint_period=50`), `train.py:recover()`
restores model weights from an in-memory snapshot step (e.g., step `31` or `81`
in `example-snapshot-workload`) and crashes with:

```
FileNotFoundError: The specified path gs://<bucket>/.../checkpoints/81/iter/process_0-of-8.json was not found.
```

**Root Cause & Mitigation**: Dataset iterator persistence
(`RemoteIteratorWrapper` in `multihost_dataloading.py` writing
`checkpoints/<step>/iter/process_<idx>-of-<N>.json` to GCS) is coupled to
persistent Orbax checkpoint intervals (`checkpoint_period`), **not** to
in-memory Snapshotter intervals (`elastic_snapshot_interval`). Calling
`new_data_iter.restore_state(restored_step)` during in-memory snapshot recovery
queries a GCS checkpoint directory that was never written
([[elastic-reshard-pytree-mismatch]]). Skip `new_data_iter.restore_state()` when
`is_snapshot_enabled` is True and transfer dataset iterator state in memory via
`get_state()` / `set_state()` (MaxText commit
`324eb962dc93d80eda89aa93689e4b352800405b`).

### Coordinator Process Bottlenecks (Worker 0 / JAX Process Zero)

In distributed JAX/Orbax checkpointing, GCS directory operations and background
deletion of old checkpoint folders are executed exclusively by **worker 0 (JAX
process zero)**. When GCS rate limits or slow deletion requests stall this
operation, worker 0 is the blocked coordinator, while workers 1..N wait and
eventually experience secondary barrier timeouts. When reporting root cause for
checkpoint hangs, always explicitly attribute the blocked coordinator to
**worker 0 (JAX process zero)** (pod `*-worker-0`).

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Concepts**:
    -   [[known-issues]]
    -   [[pathways]]
    -   [[pathways-sidecar]]
-   **Symptoms**:
    -   [[elastic-reshard-pytree-mismatch]]
    -   [[incomplete-checkpoint-restore]]
    -   [[megascale-hang]]
-   **Entities**:
    -   [[maxtext]]

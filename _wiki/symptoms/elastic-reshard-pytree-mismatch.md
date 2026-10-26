---
title: Elastic Reshard Pytree / Sharding Mismatch
type: symptom
internal_only: false
tags:
- tpu
- pathways
- training
- triage
- analysis
created: 2026-09-02
updated: 2026-10-15
---


# Elastic Reshard Pytree / Sharding Mismatch

## Overview

When a slice drops out or is still initializing during an elastic transition,
Pathways elastic recovery detects the active slice count and attempts to compile
a new device mesh or restore checkpoint state at the new topology — and fails in
the JAX/Orbax application layer. Recovery detection works; recovery *execution*
fails in software.

This is a **recurring family** with two primary sub-modes observed across
elastic runs:

| Error                                                                | Mechanism                            | Incident                    |
| :------------------------------------------------------------------- | :----------------------------------- | :-------------------------- |
| `jax._src.sharding.IndivisibleError: Sharding NamedSharding(mesh=Mesh(... 'data'\ implies that array axis 0 is partitioned 6144 times, but the dimension size is 8192` | `pathways-head` forms a 3, ... 'fsdp'\ 1\text{k} = 6,144$ TPU devices while 1 of 4 slices is unready/unplaced) whose combined `('data', 'fsdp')` tiling factor ($3 \times 2048 = 6,144$) does not evenly divide model tensor dimension $8,192$. | `example-elastic-4x1k` 2048 ...)) |
| `ValueError: Mismatch details (1 found): pytree structure error ... in_shardings[0]['model']['decoder']['layers']['self_attention'] ... symmetric difference on key sets is rngs` | During post-disruption elastic restore (e.g. after `In-memory snapshot recovery failed (No active replicas found.). Falling back to persistent checkpoint.` in `train.py\ state PyTree (`5 child keys\ attention_op key out query value`) disagrees with `pjit in_shardings` (`6 child keys\ | `example-elastic-2x64` ( ), `example-elastic-4x1k` Attempts 3 & 4 recover()`) the restored ... rngs`). |
| `ValueError: One of pjit outputs ...` | Output sharding mismatch on re-trace after slice drop. | `example-workload-2x7k` attempt 53 |
| `FileNotFoundError: The specified path: gs\ was not found.` | During in-memory snapshot scale-down //<bucket>/.../checkpoints/<step>/iter/process_<idx>-of-<N>.json (`elastic_backup_kind=snapshot`, `elastic_snapshot_interval=10` < `checkpoint_period=50`), `train.py\ weights from in-memory snapshot at step `31`/`81` but unconditionally calls `new_data_iter.restore_state(step)`, querying GCS for a persistent iterator checkpoint that only exists at multiples of `checkpoint_period`. | `example-snapshot-workload` recovery recover()` restores |

All members surface during elastic startup, scale-down, or restore, and leave
the underlying slice disruption survivable — the run dies on the software
recovery/sharding path, not on a hardware fault.

## Log Signatures

### 1. Non-Power-of-2 Elastic Mesh `IndivisibleError` (`3x1k = 6,144` devices vs. `8192` axis)

```
jax._src.sharding.IndivisibleError: Sharding NamedSharding(mesh=Mesh('diloco': 1, 'data': 3, 'stage': 1, 'fsdp': 2048, 'fsdp_transpose': 1, 'sequence': 1, 'context': 1, 'context_autoregressive': 1, 'tensor': 1, 'tensor_transpose': 1, 'tensor_sequence': 1, 'expert': 1, 'autoregressive': 1), spec=P(('data', 'fsdp'), None), memory_kind=device) implies that array axis 0 is partitioned 6144 times, but the dimension size is 8192 (full shape: (8192, 8192), per-dimension tiling factors: [6144, 1] should evenly divide the shape)
```

### 2. Orbax Checkpoint Restore PyTree Mismatch (`rngs` Symmetric Difference)

```
Slice 1 is not ready yet during health check: DATA_LOSS: Device device(272,TPU_DEVICE,
  coords=[0,0,0,0],logical_task=0,slice=1) is unplaced
  [type.googleapis.com/learning_pathways.PlacementDataLoss='']
In-memory snapshot recovery failed (No active replicas found.). Falling back to persistent checkpoint.
...
  File "/deps/src/maxtext/trainers/pre_train/train.py", line 763, in training_loop_iteration
    state, metrics = p_train_step(state, example_batch, *step_rng_args)
ValueError: Mismatch details (1 found):
pytree structure error: different numbers of pytree children at key path
    pjit in_shardings[0]['model']['decoder']['layers']['self_attention']
At that key path, the prefix pytree pjit in_shardings has a subtree of type
    <class 'dict'>
with 5 child keys
    attention_op key out query value
but at the same key path the full pytree has a subtree of the same type but with 6 child keys
    attention_op key out query rngs value
so the symmetric difference on key sets is
    rngs
```

Terminating with `XPK End ... EXIT_CODE=1`.

### 3. `RemoteIteratorWrapper.restore_state` `FileNotFoundError` During In-Memory Snapshot Scale-Down Recovery

```
*** 1/2 Pathways Slices Now Ready ***
...
  File "/deps/src/maxtext/trainers/pre_train/train.py", line ..., in recover
    new_data_iter.restore_state(restored_step)
  File "/deps/src/maxtext/input_pipeline/multihost_dataloading.py", line ..., in restore_state
    ...
FileNotFoundError: The specified path gs://<bucket>/pathways_2_slice_v6e-8_llama3-8b-8192/example-snapshot-workload/checkpoints/81/iter/process_0-of-8.json was not found.
```

(Observed at step `31` for `checkpoints/31/iter/process_4-of-8.json` and step
`81` for `checkpoints/81/iter/process_0-of-8.json` in
`example-snapshot-workload` after in-memory snapshot scale-down when
`elastic_snapshot_interval=10` and `checkpoint_period=50`.)

## Verification Steps

1.  **Check the active slice count and chronology:**
    -   For `IndivisibleError`, inspect `pathways-head` logs just prior to the
        exception to verify how many slices were ready (e.g., `3/4` slices =
        `data: 3, fsdp: 2048` = `6,144` devices while one slice was still
        initializing or unplaced).
    -   For the `rngs` PyTree `ValueError`, confirm an elastic slice loss
        preceded the error (`*** N/M Pathways Slices Now Ready ***`,
        `Transition slice ... placed -> unplaced`, `In-memory snapshot recovery
        failed (No active replicas found.). Falling back to persistent
        checkpoint.`) and note the timestamp: the `rngs` `ValueError` occurs
        *after* the worker disruption during subsequent elastic checkpoint
        restore (sometimes 1–30 minutes after an external `SIGILL` kill on
        `worker-job0-*-0`).
    -   For `RemoteIteratorWrapper.restore_state` `FileNotFoundError`, check the
        missing GCS step number
        (`checkpoints/<step>/iter/process_<idx>-of-<N>.json`) against
        `elastic_snapshot_interval` (e.g., `10`) and `checkpoint_period` (e.g.,
        `50`). When `<step>` (such as `31` or `81`) is an in-memory snapshot
        step rather than a persistent checkpoint step (`0`, `50`, ...), no GCS
        `checkpoints/<step>/iter/` directory exists.
2.  **Trace the upstream slice disruption:** The
    `DATA_LOSS ... device(N,...,slice=K) is unplaced` message identifies where
    the symptom surfaced; the slice index is frequently *not* the slice that
    died. Get the real culprit host from [[extract-culprit-hosts]], then find
    the first process death via [[query-crash-signatures]] — typically an
    external user-space [[sigill-worker-kill]](`si_code=0`).
3.  Check whether another elastic recovery in the same run (such as Attempt 6 in
    `example-elastic-4x1k` restoring step 120) **succeeded without raising the
    `rngs` error**.

## Playbooks & Root Causes

-   **Elastic mesh divisibility bug (`jax._src.sharding.IndivisibleError`) vs.
    Grain elastic iterator local batch split `ValueError`.** When an elastic
    JobSet (`4x1k` = 8,192 chips) starts or scales down with 3 active slices
    (`3x1k` = 6,144 chips), partitioning an `8192`-wide model tensor across
    `('data', 'fsdp') = (3, 2048)` requires $8192 \bmod 6144 = 0$, which raises
    `jax._src.sharding.IndivisibleError`. Fix by restricting valid elastic slice
    counts to divisors of the sharded tensor dimensions (e.g., powers of two: 1,
    2, or 4 slices) or waiting for all 4 slices before compiling the training
    mesh. By contrast, if `pathways-head` fails in
    `multihost_dataloading.py:form_global_array_colocated_python` with
    `ValueError: array split does not result in an equal division` and
    `ValueError: Unable to put to devices shape (2, 8192) with local device
    count 4 at ['inputs']`, the failure is **not** a mesh resharding error but a
    Grain multiprocessing batch-slicing mismatch when
    `grain_use_elastic_iterator=True`, `packing=False`, and
    `grain_worker_count > 0` (mitigate by setting `grain_worker_count=0` or
    enabling `packing=True`; see [[workload-startup-crash]]).
-   **`in_shardings` / Orbax restore PyTree mismatch (`rngs`) & async checkpoint
    save starvation.** When in-memory snapshot recovery fails (`In-memory
    snapshot recovery failed (No active replicas found.). Falling back to
    persistent checkpoint.`), `train.py:recover()` falls back to persistent
    Orbax checkpoint restore. If `checkpoint_period` and
    `elastic_snapshot_interval` are too close or an async checkpoint save is
    starved/interrupted mid-write, the fallback either hits an unfinalized step
    directory ([[incomplete-checkpoint-restore]]) or restores a state dictionary
    whose PyTree disagrees with `pjit in_shardings` on the `rngs` child key
    under `self_attention` (`AI-Hypercomputer/maxtext#5260`). Fix by passing a
    consistent `abstract_train_state` PyTree structure across checkpoint save
    and elastic restore, and spacing `checkpoint_period` vs.
    `elastic_snapshot_interval` so `AsyncCheckpointer` commits complete before
    fallback.
-   **`RemoteIteratorWrapper.restore_state` `FileNotFoundError` during in-memory
    snapshot scale-down.** In elastic training with in-memory snapshots
    (`elastic_backup_kind=snapshot`, `elastic_snapshot_interval=10` <
    `checkpoint_period=50`), `train.py:recover()` restored the model weights
    from the in-memory snapshot at step `31` / `81` after scale-down, but
    unconditionally called `new_data_iter.restore_state(step)`.
    `RemoteIteratorWrapper.restore_state()` looked on GCS for
    `gs://<bucket>/.../checkpoints/<step>/iter/process_<idx>-of-<N>.json`, which
    is only written at multiples of `checkpoint_period` (`0, 50, ...`), crashing
    the Colocated Python sidecar ([[pathways-sidecar]]) and head pod. Fixed in
    MaxText commit `324eb962dc93d80eda89aa93689e4b352800405b` by skipping
    `new_data_iter.restore_state()` when `is_snapshot_enabled` is True and using
    dataset iterator `get_state()` / `set_state()` for in-memory snapshot
    recovery.
-   **Do NOT isolate/replace hardware for `IndivisibleError`, `rngs`
    `ValueError`, or `RemoteIteratorWrapper.restore_state`
    `FileNotFoundError`.** All three are pure software/configuration bugs on the
    CPU head pod (`jax-tpu`) or Colocated Python sidecar, and upstream index-0
    `SIGILL (si_code=0)` kills must be checked for role-vs-host confound per
    [[sigill-worker-kill]] and [[node-isolation-methodology]].

## Connections

-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
    -   [[node-isolation-methodology]]
    -   [[workload-disruption-orchestration]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[incomplete-checkpoint-restore]]
    -   [[sigill-worker-kill]]
    -   [[workload-startup-crash]]
-   **Concepts**:
    -   [[pathways-fault-tolerance]]

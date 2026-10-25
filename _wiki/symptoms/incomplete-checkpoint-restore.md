---
title: Incomplete Checkpoint Blocks Restore
type: symptom
internal_only: false
tags: [tpu, training, triage]
created: 2026-09-02
updated: 2026-09-02
---

# Incomplete Checkpoint Blocks Restore

## Overview

A workload dies mid-flight during an asynchronous checkpoint save, leaving a
partially-written step directory. On restart, Orbax selects that directory as
the latest step and raises rather than falling back to the previous,
fully-committed step. A recoverable interruption becomes a hard failure.

The damage is entirely avoidable: in the observed case a complete checkpoint
from 6 seconds before the crash was sitting right next to the partial one.

## Log Signatures

```
ValueError: Found incomplete checkpoint at gs://<bucket>/<run>/checkpoints/1820.
```

Preceded, usually minutes earlier, by an async save that never finished:

```
[process=0] [async] Started save checkpoint @ gs://<bucket>/<run>/checkpoints/1820.
[process=0][thread=async_save] Background save thread started. Deadline for this save operation is ...
```

with **no** matching completion line for that step:

```
[process=0][thread=async_save] Finished saving checkpoint (finalized tmp dir) to ...
[process=0][thread=save_finalize] No errors found in background save thread=async_save.
```

## Verification Steps

1.  Read the step number out of the `ValueError` and inspect that directory in
    GCS:

    ```
    gcloud storage ls -r gs://<bucket>/<run>/checkpoints/<step>/
    ```

    A complete step has `_CHECKPOINT_METADATA`, `commit_success.txt`, and
    **both** the `items/` and `iter/` subtrees. A partial step is missing one of
    them or the parent commit marker.
2.  Compare against the preceding step to see what a good checkpoint looks like
    for this run. Do not assume a directory layout — check a known-good
    neighbour.
3.  **Check object timestamps before concluding the checkpoint is corrupt.**
    `gcloud storage ls -l --recursive gs://<bucket>/<run>/checkpoints/<step>/` A
    later run frequently rewrites the same step correctly. If the objects
    post-date the failed restore, the directory you are looking at is the
    *repaired* one and there is nothing wrong with it now.
4.  Identify the last fully-committed step — that is the correct restore point
    and the true cost of the incident (here: 20 steps, not the whole run).
5.  Do not read a truncated listing. `ls -r | head` will hide `iter/` and
    manufacture a false corruption finding.

## Playbooks & Root Causes

-   **Async saves are not atomic across items.** Orbax's
    `CompositeCheckpointHandler` writes `items` and `iter` separately; a process
    death between them leaves the step half-written.
-   **Restore selects the newest step directory, not the newest *valid* one.**
    The fix is to fall back to the last step with a complete commit marker
    instead of raising.
-   Ask *why the process died mid-save* — that is the real incident. See
    [[sigill-worker-kill]].
-   Interim mitigation: delete or quarantine the partial step directory so the
    next restart selects the last good step.
-   For genuine corruption (as opposed to truncation), see
    [[check-checkpoint-corruption]] and [[configure-orbax-checkpointing]].

## Connections

-   **Actions**:
    -   [[check-checkpoint-corruption]]
-   **Playbooks**:
    -   [[workload-disruption-orchestration]]
-   **Symptoms**:
    -   [[elastic-reshard-pytree-mismatch]]
    -   [[sigill-worker-kill]]

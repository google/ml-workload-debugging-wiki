---
title: SIGILL Worker Kill (si_code=0 / SI_USER)
type: symptom
internal_only: false
tags: [tpu, pathways, training, hardware_fault, triage]
created: 2026-09-02
updated: 2026-10-20
---


# SIGILL Worker Kill (si_code=0 / SI_USER)

## Overview

A healthy `pathways-worker` process (typically on an index-0 slice leader pod
`worker-job0-*-0`) is killed abruptly by `SIGILL` while actively executing
training programs (`Enqueueing program ...`). The signal lands on the idle main
thread parked in `absl::Notification::WaitForNotification()`.

Despite the name, **`si_code=0` is never a CPU illegal-instruction fault**. In
Linux `siginfo_t`, both `si_code` and the `@0x<hex>` field printed by the Abseil
`FailureSignalHandler` carry specific diagnostic meaning:

| Signal & `si_code` | Meaning of `@0x<hex>` | Interpretation |
| :-------------- | :----------------- | :-------------------------------------------------------- |
| `SIGILL`, `si_code=0` (`SI_USER`) | `siginfo->si_pid`: **Sender's PID in hex** (e.g., `@0x3dac` = PID 15788, `@0x2b00` = PID 11008, `@0x101e` = PID 4126, `@0x10ed` = PID 4333, `@0x11f4` = PID 4596, `@0x30f5` = PID 12533, `@0xc6e` = PID 3182, `@0x50bb` = PID 20667, `@0x2859` = PID 10329, `@0x9ce` = PID 2510, `@0xd90` = PID 3472, `@0xaa52` = PID 43602, `@0x631a` = PID 25370, `@0x3139` = PID 12601, `@0x2e48` = PID 11848, `@0x830c` = PID 33548, `@0x7103` = PID 28931, `@0x6e9d` = PID 28317) | **External user-space kill (`kill(2)` / `sigqueue(3)`).** Not a CPU/TPU hardware fault and not an illegal instruction. In snapshot/elastic resiliency tests, targets `worker-job0-*-0` to trigger elastic slice loss. (If the sender process is not named in container logs, label its exact identity `'unsupported - operator judgement'`.) |
| `SIGILL` / `SIGABRT`, `si_code=-6` (`SI_TKILL`) with `from PID <self>` | Self PID | Coordinated self-`abort()` (e.g. `WatchDog Timebomb for MegaScale cancellation`). **Not an independent hardware fault.** |
| `SIGILL`, `si_code > 0` (`ILL_ILLOPC`, etc.) | Faulting instruction address (`si_addr`) | Genuine CPU illegal-instruction trap (rare). |
| `SIGSEGV`, `si_code=1` (`SEGV_MAPERR`) | Faulting memory address (`si_addr`, e.g., `@0x5bb228cc8718`) | **Genuine C++ segmentation fault** (e.g., `proto2\ in `learning_pathways\ Software memory-corruption bug, not external `kill(2)`. |

## Log Signatures & Event Chronology

```
*** SIGILL (@0x3dac), si_code=0, received by PID 14 (TID 14) on cpu 111; stack trace: ***
PC: @  0x...  absl::synchronization_internal::FutexWaiter::Wait()
    @  0x...  FailureSignalHandler()
    @  0x...  absl::Notification::WaitForNotification()
    @  0x...  learning_pathways::CloudHTTPServer2::JoinServer()
    @  0x...  learning_pathways::StartServer()
RAW: Dumping core locally.
```

### Strict Chronology (`SIGILL` -> Pipe Break / Sidecar Restart -> Elastic Restore)

Do not invert cause and effect when correlating container logs around a
`SIGILL (si_code=0)` event:

1.  **`t = 0s` (`SIGILL` during active training):** `pathways-worker` (PID
    13/14) on `worker-job0-<slice>-0` is actively running
    (`Enqueueing program ...`) when killed by
    `*** SIGILL (@0x<hex>), si_code=0`.
2.  **`t = +5s to +25s` (Peer pipe break & slice unplacement):** Surviving
    workers and `pathways-head` log
    `lost connection to peer at http://machine/<node>/events`,
    `*** 3/4 Pathways Slices Now Ready ***`, and
    `Transition slice ... placed -> unplaced` ([[data-loss-pipe-closed]]).
3.  **`t = +38s to +45s` (`colocated-python-sidecar` restart):** As the killed
    worker container restarts, `colocated-python-sidecar` logs
    `Process exited, restarting...` and
    `Shm directory not found in flags or env var`. These sidecar logs are a
    **downstream consequence** of the container restart, not the trigger of
    `SIGILL`.
4.  **`t = +1m to +29m` (Elastic checkpoint restore & scale-up):** During
    subsequent elastic recovery or scale-up (`1x64 -> 2x64`), `pathways-head`
    may crash with an Orbax `'rngs'` PyTree `ValueError` or
    `RemoteIteratorWrapper.restore_state` `FileNotFoundError` (e.g., after
    `SIGILL (@0x9ce)` and `SIGILL (@0xd90)` on `worker-1-0-<pod>` at steps 31
    and 81, where `train.py:recover()` called
    `new_data_iter.restore_state(step)` for an in-memory snapshot step; see
    [[elastic-reshard-pytree-mismatch]]), enter an elastic watchdog reset loop
    ([[step-timebomb-watchdog]]), or stall at scale-up with
    `copy_to_host_buffer: DEADLINE_EXCEEDED: Megascale hang detected ...
    StepGloballyInProgress: false` (`Megascale detects a hang that is caused
    some worker not queuing programs to TPU`) when non-zero Colocated Python
    sidecars block in Orbax `array_metadata_store` `_maybe_create_base_dir()`
    waiting for worker 0 (e.g., after `SIGILL (@0x11f4), si_code=0` on
    `worker-job0-0-0-<pod>`; see [[megascale-hang]], [[pathways-sidecar]], and
    [[orbax]]).

## Verification Steps & Role-vs-Host Confound Check

1.  Sweep the **full run window** for signal deaths via
    [[query-crash-signatures]](do not scope only to JobSet restarts, as elastic slice losses are absorbed in-attempt).
2.  Read `si_code` and decode `@0x<hex>` per the table above. Distinguish
    `SIGILL (si_code=0)` on `worker-job0-*-0` from genuine C++
    `SIGSEGV (si_code=1)` crashes on non-index-0 workers (e.g.,
    `proto2::internal::ThreadSafeArena::CleanupList()`).
3.  Establish the **node** via
    [[extract-culprit-hosts]](`lost connection to peer at http://machine/<node>`).
4.  **Verify Role-vs-Host Confound before recommending host replacement
    ([[node-isolation-methodology]]):**

    -   When `extract-culprit-hosts` flags a node in `repeat_offenders` /
        `role_confound_candidates` (such as a node with 2x–3x `SIGILL` on
        `worker-1-0-<pod>` or `worker-job0-*-0-<pod>` across attempts), check
        whether every crash on that host occurred on an **index-0 slice leader
        (`worker-job0-*-0` / `worker-*-0-*`)** with `SIGILL (si_code=0)`.
    -   Check pod-to-node mappings (`Lookup Workload Mapping` or
        `[[resolve-workload-nodes]]`) across attempts for a **natural
        experiment**: check what happened when Kubernetes rescheduled
        `worker-job0-*-0` onto different nodes and assigned the prior culprit
        nodes to **non-index-0 worker roles** (e.g., `worker-job0-*-240`). When
        the prior nodes run as non-index-0 workers (`*-240`) with **0 crashes**
        while `SIGILL (si_code=0)` follows `worker-job0-0-0-<pod>` onto its new
        node, the failure is role-targeted rather than host-induced.

    -   Confirm GCE audit logs ([[query-vm-termination-events]]) and `k8s_node`
        health events ([[query-node-health-events]]) are clean. **Do NOT
        isolate/replace any node whose repeat incidents are index-0
        `SIGILL (si_code=0)` kills.**

## Playbooks & Root Causes

-   **External user-space `SIGILL (si_code=0)` targeting index-0 slice leaders
    (`worker-job0-*-0` / `worker-*-0-*`).** In elastic and non-elastic
    multi-slice workloads, an external user-space process (with higher PIDs
    encoded in `@0x<hex>`) sends `SIGILL` (`si_code=0`) to `pathways-worker` PID
    8/13/14 on `worker-*-0`. Because the crash tracks the logical index-0 role
    rather than the physical machine, hardware replacement is strictly
    contraindicated.
-   **Survivable disruption, downstream software failure.** Pathways elastic
    recovery absorbs the worker kill; when the run fails afterward, fix the
    downstream recovery or scale-up defect ([[megascale-hang]],
    [[elastic-reshard-pytree-mismatch]], [[incomplete-checkpoint-restore]], or
    [[step-timebomb-watchdog]]).

## Connections

-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
    -   [[node-isolation-methodology]]
    -   [[workload-disruption-orchestration]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[elastic-reshard-pytree-mismatch]]
    -   [[false-preemption-sigterm]]
    -   [[incomplete-checkpoint-restore]]
    -   [[megascale-hang]]
    -   [[slice-failure]]
    -   [[step-timebomb-watchdog]]

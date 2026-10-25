---
title: Distributed Training Gotchas
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-06-13
updated: 2026-10-03
---

## Distributed Training & ML Framework Gotchas

1.  **Barrier timeouts are symptoms.** `Barrier sync with id ... timed out`
    means another worker stalled; read the logs before the timeout.
2.  **Restart attempts.** With non-zero restart attempts, active logs may show
    only the latest crash; search earlier attempt suffixes.
3.  **Debug flags cost performance.** Turn verbose flags on for the next run if
    logs lack detail, but not "always on" in production
    ([[enable-tpu-verbose-logging]]).
4.  **Timestamp alignment.** For multi-restart jobs, group logs by time range
    per attempt (e.g. waves of ~4096 logs for 2k jobs).
5.  **Check slice scope via labels.** Extract `job-global-index` or
    `replicatedjob-name` from worker logs; workers sharing an error are not
    necessarily in the same slice.
6.  **Two "worker 0s".** GKE worker 0 of slice 0 (`...-worker-job0-0-...`) is
    the coordinator/aggregator; JAX process 0 (`[process=0]` in logs) does
    checkpointing. They are not always the same pod; check both. If worker 0 has
    no logs (startup crash), search other workers in the slice.
7.  **Look for the outlier error.** A severe error on every worker is often a
    benign warning or secondary symptom.
8.  **Exit 143 is normal shutdown.** In JAX/MaxText, when the coordinator exits
    0, GKE SIGTERMs the other workers (143). See [[exit-codes]].
9.  **Query each run suffix.** For multi-run failure tables, query each suffix
    (e.g. `my-training-job-4-`, `my-training-job-5-`), not a broad prefix, or data is
    missed.
10. **`failed call to cuInit: INTERNAL: CUDA error` on TPU is a red herring.**
    JAX platform auto-detection logs it before falling back to TPU.
11. **Partial `StopIteration` = disk full.** Only some workers log
    `StopIteration` and the rest stall: suspect local VM disk exhaustion, not
    end of data ([[stopiteration-disk-full]]).
12. **TCP window size.** At 32k+ chips over DCN, default OS network config can
    lower MFU; increase `tcp_rmem` on hosts if step time is slow.
13. **Host skew from eval.** Heavy eval steps inside the training loop stall the
    whole cluster at barriers.
14. **Misleading "bad chip".** `Potential cause: BAD_TPU_CHIP` or suspect-host
    lists in coordinator logs often mask software hangs. Example: JAX process 0
    (alone on one host for checkpointing) hangs and Megascale reports a "bad
    chip". Don't isolate right away; check recurrence across restarts and rule
    out software (e.g. slow GCS checkpoint folder deletion).
15. **Global to slice/worker ID.** Slice ID = `Global Index // VMs per slice`
    (`1025 // 512 = 2`); local worker ID = `Global Index % VMs per slice`
    (`1025 % 512 = 1`) -> pod `slice-2-worker-1`.
16. **Chronological ordering trap.** C++ `RET_CHECK` or `FATAL` compilation
    exceptions outrank watchdog stall signatures (`step-timebomb`,
    `jax.block_until_ready`). The stall may appear earlier in the logs, but the
    fatal exception is the root cause, not a secondary casualty.
17. **Shardy reshape `All-Gather` OOM looks like a hardware scale fault.**
    Signature: `llo::CouldLeS32(0, bound)` with
    `Fatal Python error: Segmentation fault`. Rarely triggers below 256 chips
    but fails consistently on 6x2k topologies due to axis mis-distribution. It
    is software, not hardware.

## Connections

-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[general-ml-debugging]]
    -   [[mcjax-debugging-principles]]

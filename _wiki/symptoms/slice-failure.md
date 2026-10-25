---
title: SLICE_FAILURE (Chip Driver Error / SW Inject Error)
type: symptom
internal_only: false
tags: [tpu, hardware_fault]
created: 2026-05-13
updated: 2026-10-03
---

## Symptom signature

SliceBuilder ICI slice failure:

```
Detected failure SLICE_FAILURE_CHIP_DRIVER_ERROR in session 57ca08a286d30683; Start failure handling
```

Usually followed by `Session master detected TPU session error from worker...`.
Like barrier timeouts, it usually reports an earlier failure. Failure types:

-   `SLICE_FAILURE_CHIP_DRIVER_ERROR`: chip driver error (bad TPU, ICI link
    down) after the slice was built.
-   `SLICE_FAILURE_SW_INJECT_ERROR`: injected by software (e.g. the TPU runtime)
    to mimic a slice hardware error, for control actions or testing.
-   `SLICE_FAILURE_INIT_ERROR`: failed during slice init (ICI network config).
-   `SLICE_FAILURE_WORKER_UNAVAILABLE`: worker disconnected after the heartbeat
    threshold (network partition, or check K8s events for eviction/OOM).
-   `SLICE_FAILURE_FLAPPING_TASK_ERROR`: flapping task, possibly a fast task
    restart misconfiguring a previously built slice; amenable to job quick_exit.

## Actionable Steps

1.  Read earlier logs on the same worker. If app logs are silent, check K8s
    events for evictions or OOMs.
2.  Use the following Session Leader message to find the right worker.
3.  No root cause in logs: investigate by unique exit codes
    ([[exit-code-investigation]]).
4.  **Retry**: Retry the workload to see if it's a reproducible hardware issue
    or an intermittent failure.

## Connections

-   **Workflows**:
    -   [[exit-code-investigation]]
-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[hardware-failure-modes]]
    -   [[mcjax-debugging-principles]]
-   **Symptoms**:
    -   [[session-failing]]

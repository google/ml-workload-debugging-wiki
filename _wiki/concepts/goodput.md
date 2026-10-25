---
title: Goodput and Recovery Metrics
type: concept
internal_only: false
tags:
- tpu
- pathways
- training
created: 2026-06-13
updated: 2026-10-16
---


## Overview: step-time deviation

In some environments (e.g. dedicated customer clouds) raw step time is
proprietary and is not exposed to the infrastructure team. You only get **Step
Time Deviation** in the Goodput dashboard (see [[slow-step-time]]).

-   A rise in deviation means one or more workers slowed down and the
    synchronous job stalls at barriers.
-   **Coarseness**: if step time is reported every 30-40 steps (or 30-40
    minutes), short slowdowns are masked. To see them, parse raw worker logs and
    diff the step start and end timestamps.

## Details: time to recovery (TTR)

TTR runs from the crash to the first resumed training step: infra repair time
(repair nodes, replace node pools, provision) plus job init time (restart, XLA
compile, checkpoint load).

## Time to Detect & Exclude (TDE)

TDE is the part of the cleanup phase between the fault and the bad node being
removed from the slice.

-   Long eviction/detection latency shows up as a silent stall on `sflag`
    collective timeouts, not as an error.
-   **Double interruption**: if the node is excluded too slowly, the restart
    lands on the same bad node and fails again. This inflates the interruption
    count and the recovery time.

## Goodput Calculation Edge Cases & In-Process Elastic Recovery

In elastic Pathways workloads using in-memory snapshot recovery (e.g.,
`example-elastic-2x1k`), the Goodput monitoring thread on `pathways-head`
(`example-elastic-2x1k-pathways-head-0-0-smbf6`) can emit a non-fatal telemetry
calculation error during post-recovery query cycles:

```text
[PID: 4297] Error while querying goodput for job example-elastic-2x1k. Skipping this cycle. Error: Productive training time is invalid. Please fix the logging entries.
```

-   **Root Cause (`ElasticGoodputCalculator` / `CheckpointBadputCalculator`):**
    When an elastic snapshot job recovers in-process from a slice disruption
    without a full JobSet restart (`job_start_time` is not reset), unproductive
    checkpoint/snapshot restore intervals can overlap across elastic
    reinitializations or leave `None` checkpoint duration fields, causing the
    derived productive training time to become negative or invalid.
-   **Resolution:** Fixed in the recorded incident
    (`Fix elastic and checkpoint goodput/badput calculation edge cases`) by
    discounting overlapping unproductive restore windows during in-process
    elastic reinitialization and safely handling `None` checkpoint durations.
-   **Triage Guidance (Decoupling Telemetry Bugs from Hardware/Training
    Faults):**
    1.  `Error: Productive training time is invalid. Please fix the logging
        entries.` (`Skipping this cycle`) is a **head-side telemetry calculation
        bug**, not a training stall or worker crash. Verify whether training
        steps continue advancing in `jax-tpu` logs.
    2.  In large-scale elastic runs (e.g., 2x1k `example-elastic-2x1k`),
        orthogonal slice disruptions may occur in the same window—such as
        external user-space `SIGILL (@0xaa52, si_code=0)` fault injections
        ([[sigill-worker-kill]]) or an isolated single-host TPU `[HARDWARE]`
        fatal (`[/dev/vfio/4 tpu1907:pe0:0] vf_id:0 !!!! FATAL ERROR !!!!
        observed errors are: [HARDWARE]. Now taking a TPU core dump...` on
        `example-elastic-2x1k-worker-job0-1-231-l5cgv`, node
        `gke-tpu-node-xxxx`). Attribute the Goodput query error to
        `ElasticGoodputCalculator` on `pathways-head`, and evaluate any worker
        `[HARDWARE]` event independently under [[elastic-vs-permanent-failure]]
        and [[node-isolation-methodology]].

## Connections

-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
    -   [[general-ml-debugging]]
    -   [[node-isolation-methodology]]
-   **Symptoms**:
    -   [[sigill-worker-kill]]
    -   [[slow-step-time]]
    -   [[worker-hardware-fatal-error]]

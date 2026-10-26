---
title: Trace Program Execution Times
description: Measure per-worker device execution time from the enqueue log and the real_program_continuator.cc completion log.
type: playbook
internal_only: false
tags: [metrics, analysis]
created: 2026-06-15
updated: 2026-10-03
---

## Overview

Dashboard step times can be delayed or batched (e.g. every 500 steps). For a
[[slow-step-time]] or a worker doing extra local work (data loading,
recompilation, evaluation) before enqueuing, measure device execution time per
worker from two logs:

-   Enqueue (host prep done, program submitted): `Enqueueing program ...`
-   Completion (device done; emitted by `real_program_continuator.cc`):
    `last_finished_run_id ...`

## Calculation and Analysis

1.  On one worker, device execution time = timestamp of `last_finished_run_id`
    minus timestamp of the matching `Enqueueing program`.
2.  Compare across all workers in the slice/pod:
    -   Similar everywhere but step time is slow: the computation itself or a
        universal network barrier.
    -   **Unique-worker pattern:** one worker has a *smaller* delta. It spends
        extra time before enqueuing (fetching data, evaluation, checkpointing)
        while the others wait at the barrier; their total step time grows but
        their device delta stays small.

This separates slow device execution (hardware/network) from host-side
bottlenecks before enqueue.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Symptoms**:
    -   [[slow-step-time]]

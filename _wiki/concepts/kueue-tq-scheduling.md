---
title: Kueue and Throughput Quota Scheduling
type: concept
internal_only: false
tags:
- tpu
- training
- scheduling
created: 2026-06-16
updated: 2026-10-04
---

## Overview

In GKE clusters configured with **Kueue** (such as XPK-managed or multi-tenant
TPU clusters), workloads are queued and admitted by Kueue rather than scheduled
directly onto nodes at submission time. Kueue ensures that workloads are
admitted only when there is sufficient quota and physical capacity, preventing
resource starvation and thrashing. (Kueue is optional on GKE; clusters that
schedule Jobs or Pods directly do not create Kueue `Workload`, `LocalQueue`, or
`ClusterQueue` resources.)

-   Inspect queues: `kubectl describe lq <queue-name>`,
    `kubectl describe cq <cluster-queue-name>`.
-   ResourceFlavors map to TPU accelerator topologies (such as full 4x4x4 cubes
    or smaller sub-cube topologies).
-   Admission order: priority class first, then creation timestamp (requeued
    workloads keep their place).

## Team Priority Policy

-   **Medium** (default): standard training and eval runs.
-   **High**: only when you must preempt other workloads.
-   **Very-High**: **do not use** without explicit permission. It disrupts other
    team members.
-   **Low**: long-running, interruption-tolerant, or background workloads.

## Common Scheduling Issues

### Head-of-Line Blocking

A workload at the front of the queue that cannot be admitted (slice too large,
or an invalid YAML such as a bad topology, wrong machine type or mismatched
partition ID) can block smaller workloads behind it that would fit.

### Stuck After Preemption

The bug itself is fixed.

When a workload is preempted, Kueue should requeue it. Historically, bugs in the
requeuing logic caused these readmitted workloads to get stuck in the queue
indefinitely, requiring manual intervention (deactivating and reactivating the
workload) to kick them back into the scheduling loop. Similar state-sync issues
can occur in the future.

## Connections

-   **Playbooks**:
    -   [[debugging-submission-failures]]
-   **Symptoms**:
    -   [[stuck-in-queue]]
-   **Concepts**:
    -   [[ai-hypercomputer]]
    -   [[jobset-yaml-structure]]

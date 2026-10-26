---
title: MCJax Debugging Principles
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-06-11
updated: 2026-10-03
---

## Definition

Debugging tips for multi-controller JAX (MCJax) workloads on GKE TPUs.

## Core Principles

1.  **Worker 0-0 is not necessarily Process 0:** check which worker logs
    `[process=0]` (it does checkpointing).
2.  **Verify every worker initialized:** search setup milestones such as
    `"Setting up checkpoint logger..."`.
3.  **Post-mortem:** use [[cloud-logging]] with time filters, not local files on
    deleted pods.
4.  **Look a few minutes before `DEADLINE_EXCEEDED`** for the triggering event.

Pod name:
`{workload_name}-slice-job-{slice_index}-{worker_index}-{random_suffix}`.

## Exit Code Issues

-   `143` (SIGTERM): graceful termination. `134` (SIGABRT): active failure. `1`:
    application error (e.g. Python `ValueError`, config); look for tracebacks.
-   No exit code: strong suspect for a stall or eviction; check Kubernetes
    events for eviction or OOM ([[missing-exit-codes-eviction]]).

## Connections

-   **Workflows**:
    -   [[exit-code-investigation]]
    -   [[mcjax-debugging-playbook]]
-   **Playbooks**:
    -   [[debugging-gotchas]]
-   **Concepts**:
    -   [[megascale]]
-   **Entities**:
    -   [[mcjax]]
-   **Tools**:
    -   [[cloud-logging]]

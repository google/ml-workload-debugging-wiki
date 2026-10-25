---
title: Debugging Gotchas
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-05-13
updated: 2026-10-03
---

## Overview

Index of debugging gotchas by domain.

## Gotchas Categories

-   [[gotchas-cloud-logging]]: traceback ordering and reverse sorting, inexact
    matching and search limits, red herrings, severity filters.
-   [[gotchas-kubernetes-gke]]: post-mortem on deleted pods, events for
    evictions, `kubectl` connectivity limits, GKE Metadata Server crashloops,
    Kueue admission freezes, software crashes preceding evictions (core dumps).
-   [[gotchas-distributed-training]]: JAX Process 0 vs GKE Worker 0, barrier
    timeouts vs root cause, red-herring CUDA errors on TPU, premature data
    exhaustion, DCN TCP tuning, host skew from expensive evaluations.

## Connections

-   **Workflows**:
    -   [[debugging-failed-provisioning-ai-hypercomputer]]
-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[mcjax-debugging-principles]]
-   **Concepts**:
    -   [[gke-identifier-taxonomy]]
-   **Tools**:
    -   [[cloud-logging]]

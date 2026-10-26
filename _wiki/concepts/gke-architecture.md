---
title: GKE Details
type: concept
internal_only: false
tags: [tpu, training, scheduling]
created: 2026-05-13
updated: 2026-10-03
---

## Overview and details

TPU workloads on GKE run as a JobSet (see [[jobset-yaml-structure]]). A "worker"
usually means one task or VM running part of the workload. Large jobs are
gang-scheduled: they wait until all requested resources are available at once.

## Connections

-   **Workflows**:
    -   [[k8s-workload-verification]]
-   **Concepts**:
    -   [[project-quotas]]

---
title: Provisioning Failure Analysis
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-06-11
updated: 2026-10-03
---

## Overview

Finding where a TPU provisioning request fails in traditional (non-AI
Hypercomputer) environments. Covers the provisioning and GCE/hardware layers of
[[layered-debugging-framework]].

## Diagnosis Procedure by Stack Layer

The request flows `Project Quota` -> `Capacity Pools` -> `Cluster Scheduler` ->
`GKE / GCE Cluster`.

1.  **Project quota:** rejected immediately at the API.
2.  **Capacity pools:** accepted (`WAITING_FOR_RESOURCES` or still queued) but
    not scheduled because the capacity pools have no free chips.
3.  **GKE / GCE:** VMs fail to create or become ready. A partial slice (e.g.
    63/64 nodes ready) points to a node-specific hardware fault.

## Connections

-   **Workflows**:
    -   [[debugging-failed-provisioning]]
-   **Playbooks**:
    -   [[layered-debugging-framework]]
-   **Concepts**:
    -   [[ai-hypercomputer]]
    -   [[project-quotas]]
    -   [[queued-resources]]

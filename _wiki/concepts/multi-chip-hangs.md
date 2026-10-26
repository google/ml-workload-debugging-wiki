---
title: Multi-Chip Hangs
type: concept
internal_only: false
tags: [tpu, hardware_fault]
created: 2026-06-11
updated: 2026-10-03
---

## Overview and debugging

All chips look stuck while they wait on one straggler, so the symptom is global
but the cause is one chip or a few. Check Kubernetes events for evictions or
OOMs that may have killed the culprit silently.

Causes seen: slow or degraded chips, chips that halt on an error without
stopping the job cleanly, SparseCore discrepancies in ops that use both
TensorCore and SparseCore (e.g. All-Reduce), and ICI congestion.

## Connections

-   **Playbooks**:
    -   [[multi-chip-hang-investigation]]
    -   [[multi-workload-triage]]
    -   [[tpu-state-analysis-methodology]]
-   **Symptoms**:
    -   [[wait-timeout-on-sflag]]
-   **Entities**:
    -   [[sparsecore]]

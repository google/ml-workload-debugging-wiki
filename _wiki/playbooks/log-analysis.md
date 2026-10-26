---
title: Tail-End Log Analysis Workflow
type: playbook
internal_only: false
tags: [training, analysis, triage]
created: 2026-06-16
updated: 2026-10-20
---

## Overview

Tail-end log techniques for unknown crashes or hangs in large TPU workloads.
First run [[ai-triage-classify-and-mask-logs]]: ingestion
([[log-ingestion-pipeline]]), masking identifiers to dedupe lines into templates
([[log-processing-pipeline]]), then matching against the symptom catalog
([[ai-match-symptoms]]).

## Manual Analysis Steps

1.  **Start at the tail:** the 5–10 minutes before the failure or preemption.
    Ignore transient warnings from hours earlier.
2.  **Key workers first:** JAX Process 0 and GKE coordinator slice 0. Search
    `"process ID"` (or similar) to see which worker printed a line.
3.  **Global worker ID -> slice/worker:** with $N$ VMs per slice (typically 512
    at large scale), slice = $\lfloor \text{Global ID} / N \rfloor$, local
    worker = Global ID mod $N$. Example: `1025` with $N = 512$ is worker 1 on
    slice 2.
4.  **Fingerprints:** the JAX/XLA fingerprint identifies the compiled program
    running. Matching fingerprints across workers shows whether all nodes ran
    the same program when the stall started.
5.  **Manual step time:** completion timestamp (`"run completion"`) minus
    enqueue timestamp (`"incued"` or similar start event).

## Connections

-   **Workflows**:
    -   [[ai-triage-classify-and-mask-logs]]
    -   [[exit-code-multi-pod-analysis]]
    -   [[log-ingestion-pipeline]]
    -   [[log-processing-pipeline]]
-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Concepts**:
    -   [[application-log-analysis]]
    -   [[exit-codes]]
    -   [[goodput]]
    -   [[multi-pod-exit-code-analysis]]
-   **Tools**:
    -   [[cloud-logging]]
    -   [[mask-segment-tool]]

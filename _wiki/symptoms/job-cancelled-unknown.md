---
title: Job Cancelled Due To
type: symptom
internal_only: false
tags: [pathways, hangs, timeout]
created: 2026-07-20
updated: 2026-10-03
---

# Job Cancelled Due To

**Symptom**: The Pathways Head Node logs "Job cancelled due to" `<reason>`.

## Context

Pathways operates as the central orchestrator and scheduler for the slice. When
a critical failure occurs—either in software (e.g. a Megascale hang, an
unhandled exception) or hardware/infrastructure (e.g. preemption on a
worker)—Pathways will explicitly abort the distributed execution. When this
happens, it logs a "Job cancelled due to" message on the Head Node indicating
why it aborted the job.

## Diagnostic Value

-   **Single Instance**: If you observe only *one* "Job cancelled due to" log
    for a given restart, it is highly significant. It explicitly tells you what
    Pathways suspects is the root cause for the job cancellation. You should
    highlight this reason in your diagnostic report.
-   **Multiple Instances**: If there are several of these "Job cancelled due to"
    logs spanning multiple TPU VMs or GKE instances during the same narrow
    timeframe, it is likely an infrastructure occurrence (e.g., a systemic
    cluster preemption, or network partition affecting multiple instances). You
    should point this out clearly as an infrastructure-level event rather than a
    single node fault.

## Query Filter

```
resource.type="k8s_container"
resource.labels.pod_name:"-pathways-head-0"
"Job cancelled due to"
```

## Connections

-   **Symptoms**:
    -   [[megascale-hang]]
-   **Concepts**:
    -   [[pathways]]

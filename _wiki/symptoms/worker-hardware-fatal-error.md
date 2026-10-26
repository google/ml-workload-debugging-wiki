---
title: Worker Hardware Fatal Error
type: symptom
internal_only: false
tags:
- tpu
- hardware_fault
created: 2026-07-20
updated: 2026-10-03
---

# Worker Hardware Fatal Error

**Symptom**: A TPU worker node logs an error beginning with
`!!!! FATAL ERROR !!!! observed errors are:` followed by an array of error
labels (e.g., `[USER, HARDWARE]`).

## Context

When a TPU worker encounters an unrecoverable fault during training execution,
it intercepts the exception and attempts to categorize it. It emits a definitive
string `!!!! FATAL ERROR !!!! observed errors are: [...]`. The array explicitly
lists what the system suspects went wrong based on internal probing.

## Diagnostic Value

-   If the `observed errors` list includes the string **`HARDWARE`** (e.g.,
    `[HARDWARE]`, `[USER, HARDWARE]`), you must point this out explicitly.
-   The presence of the `HARDWARE` tag points to physical hardware faults on
    that specific worker node, rather than software code issues or
    infrastructure preemption.
-   **Exception: many hosts in the same second.** If dozens of hosts log
    `[HARDWARE]` within ~100 ms, suspect one fabric-wide event (for example a
    large slice being programmed), not dozens of bad chips.

-   **CRITICAL**: To identify the specific GKE TPU instance name associated with
    the log, you MUST extract it from the
    `labels."compute.googleapis.com/resource_name"` field (or
    `resource.labels.node_name` if the former is unavailable) in the JSON log
    payload. This indicates the underlying GKE node (e.g., `gke-tpu-...`). Do
    NOT use the pod name for this error. List these underlying physical compute
    nodes as the culprit GKE instances, along with a count of their fatal
    hardware logs.

## Query Filter

When querying, explicitly specify the 20-minute window (10 minutes before and 10
minutes after the downtime event) and filter using the job name as a prefix to
efficiently target all TPU workers for that specific run.

```
resource.type="k8s_container"
resource.labels.pod_name:"<JOB_NAME>"
resource.labels.pod_name:"-worker-"
"!!!! FATAL ERROR !!!! observed errors are"
timestamp>="T-10m" timestamp<="T+10m"
```

## Connections

-   **Playbooks**:
    -   [[hardware-failure-modes]]

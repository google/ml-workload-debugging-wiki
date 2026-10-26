---
title: Job Disruption Detection Methodology
type: concept
internal_only: false
tags: [training, pathways, analysis, triage]
created: 2026-07-17
updated: 2026-10-15
---

# Job Disruption Detection Methodology

## Overview

This page serves as the source of truth for how the **Job Disruption Detector**
agent analyzes ML workloads to find interruptions, restarts, and reshards.

The core objective of this analysis is to identify a clear chronological
timeline of events mapping to jobset changes, without doing brute-force log
scraping on Application logs, so that downstream analyzer agents know *exactly*
when disruptions occurred.

## 0. Pre-requisites: Workload Validation

Before analyzing disruptions (restarts/reshards) on a target workload, verify
that the workload actually initiated.

1.  **Check for Initialization Logs:** Query for the `InitGoogle` anchor log for
    the target workload.
2.  **If No Logs Found:**
    -   Query GKE Audit Logs to check if there were any launch attempts (head
        pod creations):
        *   Query:
            `logName:"projects/<project-id>/logs/cloudaudit.googleapis.com%2Factivity"
            AND protoPayload.methodName:"io.k8s.core.v1.pods.create" AND
            protoPayload.resourceName:"pathways-head" AND "<workload-name>"`
    -   **If Audit Logs show creations:** This indicates **Failed Startup
        Attempts**. Do not proceed with disruption analysis. Execute the **Pod
        Analyzer** ([[pod-analyzer]]) scoped to the creation time window to
        diagnose why the pods failed to schedule or start.
    -   **If no Audit Logs found:** The workload was never submitted. Report
        `WORKLOAD_NOT_FOUND`.

## 1. Disruption Detection Strategies

Always dynamically query logs from GCP Cloud Logging (e.g., using
`gcloud logging read` or `google-cloud-logging` SDK). **Do NOT rely on static
datasets or CSVs.**

### Unified Strategy: Application Initialization Anchors

Instead of scraping millions of generic k8s container logs or trying to run
multiple isolated queries, use the application's footprint as the anchor for
**all** restart tracking.

Every time a TPU worker spins up its binaries—whether it's an automated
kubernetes pod replacement, a manual job recreation, or an in-place internal
crash reboot—it emits a standardized anchor:
`"Logging before InitGoogle() is written to"`.

-   **Global Search Query Basis:** Perform a single `gcloud logging read` query
    filtered by `resource.type="k8s_container"` AND
    `textPayload:"Logging before InitGoogle() is written to"`, scoped to the
    target `jobset-name`.
-   **Classification Logic:** Once the timeline of initialization logs is
    extracted, chronologically compare each log's `resource.labels.pod_name` and
    `labels."k8s-pod/job-name"` against the previous known state of that worker:
    1.  **Manual Restarts:** If the overarching `labels."k8s-pod/job_name"` or
        topology suffix sequence changes (e.g., `cr4` to `cr5`), the workload
        was manually deleted and recreated.
    2.  **Automated Restarts:** If the overarching jobset name remains identical
        but a specific worker or head `pod_name` hash changes (e.g.,
        `worker-0-abc` to `worker-0-xyz`), Kubernetes caught a crash and
        self-healed the pod.
    3.  **In-Place Restarts:** If the overarching jobset name and the specific
        `pod_name` both remain completely identical to the previous known state,
        but the `InitGoogle` anchor appears again chronologically, the
        application binaries have rebooted themselves internally.
-   **Node Identity Extraction:** When assembling the JSON output, you must
    explicitly extract the active pod names for the overarching job.
    -   For `cpu_pathways_head_podname`, locate the active pod whose name
        contains `pathways-head`.
    -   For `tpu_worker_headname`, you MUST locate the worker pod whose name
        contains the exact substring `0-0` (e.g., `worker-job0-0-123`,
        `worker-job-0-0-123`, `worker-job-slice0-0-123`, etc.). The naming
        prefix before `0-0` can vary unpredictably, so zero in strictly on the
        `0-0` pattern to identify the primary TPU worker node.

## 2. Output Analysis Schema

After parsing the telemetry from Cloud Logging, the timeline of collected
metadata must be provided to the analyzer agents matching the following JSON
Array layout:

```json
[
  {
    "attempt": 1,
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
    "type": "Automated",
    "trigger_reason": "Pod Name Change",
    "job_name": "my-jobset-v1",
    "cpu_pathways_head_podname": "my-jobset-v1-pathways-head-0-0-a1b2c",
    "tpu_worker_headname": "my-jobset-v1-worker-0-0-x9y8z"
  },
  {
    "attempt": 2,
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
    "type": "Manual",
    "trigger_reason": "Job Name Change",
    "job_name": "my-jobset-v2",
    "cpu_pathways_head_podname": "my-jobset-v2-pathways-head-0-0-d3e4f",
    "tpu_worker_headname": "my-jobset-v2-worker-0-0-l5m6n"
  }
]
```

## Execution Rules

-   Always process events sequentially based on timestamp.
-   Use explicit 10-minute timeouts per attempt if making complex log queries.
-   Limit query scope tightly to the precise environment, utilizing known
    standard project IDs like `[GCP_PROJECT_ID]`.

## Connections

-   **Playbooks**:
    -   [[workload-disruption-orchestration]]
-   **Concepts**:
    -   [[jobset-yaml-structure]]
    -   [[workload-monitoring]]
-   **Tools**:
    -   [[cloud-logging]]

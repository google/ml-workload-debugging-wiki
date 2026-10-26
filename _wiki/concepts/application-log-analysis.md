---
title: Application Log Analysis Methodology
type: concept
internal_only: false
tags: [training, pathways, analysis, triage, metrics]
created: 2026-07-17
updated: 2026-10-20
---

# Application Log Analysis Methodology

This page serves as the source of truth for how the **Application Logs
Analyzer** agent queries and analyzes application-level runtime logs to isolate
the root cause of ML workload disruptions.

## 1. Input Context

The agent receives input from the orchestration layer (discovered via the Job
Disruption Detector):

*   `attempt`: The restart iteration index.
*   `timestamp`: The time the disruption or restart was detected.
*   `type`: The restart type (`Automated`, `Manual`, `In-Place`).
*   `job_name`: The name of the GKE JobSet/workload.
*   `cpu_pathways_head_podname`: The identifier of the Pathways CPU head
    coordinator pod.
*   `tpu_worker_headname`: The identifier of the primary TPU worker head node
    (the `0-0` worker).

--------------------------------------------------------------------------------

## 2. Log Querying Strategy

Do NOT scrape all application logs blindly. Focus queries around the disruption
event using a precise sliding lookback window.

### Sliding Lookback Window

*   **Start Time ($T_{\text{start}}$)**: `timestamp - 10 minutes`
*   **End Time ($T_{\text{end}}$)**: `timestamp + 2 minutes`

### Target Logs & Pod Scopes

1.  **Coordinator and Process 0 Logs (Primary Target)**:
    *   Query container logs where `resource.labels.pod_name` matches
        `tpu_worker_headname` OR `cpu_pathways_head_podname`, AND separately
        search all pods for `[process=0]`.
    *   *Rationale*: the coordinator (`0-0`) and the Pathways head log
        coordination barriers, init and step progress. JAX Process 0 is
        frequently a *different* pod and carries checkpoint/GCS exceptions
        ([[mcjax]]).
2.  **Cluster-wide Application logs (Fallback / Secondary Target)**:
    *   If coordinator logs do not show a clear crash, query all container logs
        in the `job_name` (e.g., `resource.labels.pod_name:"{job_name}"` or
        `labels."k8s-pod/jobset-name"="{job_name}"`).
    *   Filter with a union of the critical keywords to keep query execution
        light.

--------------------------------------------------------------------------------

## 3. Failure Heuristic Patterns

Use the following regular expression patterns to classify the root failure
cause:

### A. Elastic Reshard / Upshard / Downshard

*   **Classification**: `Elastic Reshard`
*   **Signatures**:
    *   `RESTART_INTERRUPT_AFTER_CHECKPOINT`:
        `Checkpoint save completed\. Interrupting`
    *   `RESTART_ELASTIC_ATTEMPT`: `Elastic attempt`
    *   `RESTART_UNASSIGN_SLICES`: `Unassigning slices from job`
*   **Description**: Expected framework scale transitions where the job is
    intentionally interrupted right after a checkpoint to scale up or down.

### B. Fatal Hardware Failure

*   **Classification**: `Hardware Failure`
*   **Signatures**:
    *   `CRASH_FATAL_HARDWARE`:
        `!!!! FATAL ERROR !!!! observed errors are: \[([^\]]*HARDWARE[^\]]*)\]`
    *   `HALT_DEVICE`: `Accelerator device halted prematurely|Node \d+ halted
        unexpectedly at tag:pc|halted at tag:pc`
    *   `CRASH_CORE_HALT`: `RuntimeUnexpectedCoreHalt`
    *   `CRASH_BAD_TPUS`: `The host that have bad TPUs are|bad TPUs are:`
*   **Description**: Low-level hardware failures, sequencer halts, or
    compiler/driver-reported TPU failures.

### C. Heartbeat / Connectivity Timeout

*   **Classification**: `Heartbeat Timeout`
*   **Signatures**:
    *   `STALL_TPU_SFLAG_TIMEOUT`: `Wait timeout on sflag`
    *   `CRASH_MEGASCALE_HANG`: `Megascale detects a hang`
    *   `CRASH_DEADLINE_EXCEEDED`: `DEADLINE_EXCEEDED|lost connection to peer`
    *   `TIMEOUT_BARRIER`: `Barrier sync with id .* timed out`
*   **Description**: Network hangs, worker-to-worker synchronization timeouts,
    or pathways sidecar lost connections.

### D. Software / User Program Crash

*   **Classification**: `Software User Crash`
*   **Signatures**:
    *   `CRASH_PYTHON_FATAL`: `Fatal Python error`
    *   `CRASH_TRACEBACK`: `Traceback \(most recent call first\):`
    *   `CRASH_SEGFAULT`: `Segmentation fault`
    *   `CRASH_SIGABRT`: `SIGABRT` (without coordination barrier context)
    *   `STOP_ITERATION`: `StopIteration` (often indicates local dataset/disk
        issues)
*   **Description**: Standard Python exceptions, runtime syntax errors, memory
    leaks, segment faults, or training loops running out of data.

--------------------------------------------------------------------------------

## 4. Root Cause Determination Rules

When analyzing the collected log timeline, follow these rules:

1.  **Temporal Ordering**: Find the **earliest** timestamped issue in the
    lookback window.
2.  **Cascade Identification**:
    *   If the coordinator aborts with `SIGABRT` or `Barrier sync timed out`,
        look for preceding errors (`CRASH_*`, `HALT_DEVICE`) on individual
        workers. The earliest worker error is the root cause.
    *   If no application errors are found, classify as `Unknown` to allow the
        GKE/VM termination layers to override or merge with their infrastructure
        findings.

--------------------------------------------------------------------------------

## 5. Output Schema

Provide a single JSON block representing the findings:

```json
{
  "attempt": 1,
  "root_cause_category": "Elastic Reshard",
  "root_cause_summary": "Workload completed saving checkpoint and initiated scale transition.",
  "matched_patterns": [
    "RESTART_INTERRUPT_AFTER_CHECKPOINT"
  ],
  "culprit_node": "N/A",
  "earliest_error_timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "logs_snippet": "Checkpoint save completed. Interrupting..."
}
```

## Connections

-   **Playbooks**:
    -   [[log-analysis]]
-   **Symptoms**:
    -   [[job-cancelled-unknown]]
    -   [[worker-hardware-fatal-error]]
-   **Concepts**:
    -   [[exit-codes]]
    -   [[failure-tables]]
    -   [[tpu-logging-flags]]
-   **Tools**:
    -   [[mask-segment-tool]]

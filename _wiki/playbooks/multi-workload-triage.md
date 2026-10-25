---
title: Multi-Workload Triage Playbook
description: A systematic approach to debugging a series of related workload runs
  that fail consecutively by compiling a Failure Table.
type: playbook
internal_only: false
tags: [tpu, training, triage]
created: 2026-06-13
updated: 2026-10-06
---

## Overview

When debugging a series of related workload runs (e.g., `workload-name-1`, `-2`,
`-3`) that fail consecutively, you should use a systematic approach to compile a
**Failure Table** to identify repeating patterns vs one-off issues.

## Triage Automation Tool

To automate Steps 2 through 4, you should use the [[triage-script]] helper:

```bash
python3 tools/scripts/triage_workloads_api.py <workload_prefix> --project=<project_id>
```

Run this script first to collect exit code distributions, eviction warnings,
quota errors, TPU OOMs, and Python tracebacks in a single sweep.

## Step-by-Step Triage Workflow

### Step 1: Initialize the Failure Table

Create or locate the failure table for the workload family (e.g.,
`cases/failure-table-workload-name.md`). Use the standard template.

### Step 2: Compile Exit Codes and Timestamps

For each run in the series, query the exit codes of all workers to identify:

1.  The distribution of exit codes (e.g., mostly 143, some 134 or 1).
2.  The earliest worker to exit (the "culprit" or "first reporter").
    *   Use chronological ascending sorting to find the absolute first timestamp
        of exit.

### Step 3: Investigate Infrastructure Events (Evictions)

For any run with exit code 143 (SIGTERM) or missing exit codes, check GKE events
for evictions:

*   **Query Gcloud Logging**: Use `resource.labels.pod_name` prefix filter.
*   **CRITICAL FILTER**: When filtering GKE events in Cloud Logging, use
    `logName:events` (contains) rather than `logName="events"` (exact match).
    GKE event log names are fully qualified (e.g.,
    `projects/PROJECT_ID/logs/events`), so exact match with `=` will return 0
    results.
*   If you find `reason: Evicted` due to `ephemeral-storage` pressure, document
    this as `Infrastructure / Eviction` category.

### Step 4: Investigate Application Logs for Quotas and Configs

If no evictions are found, or if workers aborted (134) or crashed (1) before
eviction:

1.  **Check for Quota Errors**: Search all worker logs for `Quota exceeded` or
    `429` (HTTP Too Many Requests) errors.
    *   *Example*: Cloud Storage egress bandwidth limits
        (`MultiRegionGoogleEgressBandwidth`).
    *   If found, classify as `SW` or `Data` or `Quota` and identify GCS as the
        trigger.
2.  **Check for Config Errors**: Search logs of the earliest failing pod for
    startup errors, tracebacks, or configuration validation failures (e.g.
    `ValueError`, `KeyError`).
    *   If found, classify as `Configuration`.

### Step 5: Differentiate Hangs from Active Failures

If you see coordination timeouts (e.g. `Wait timeout on sflag`, `SetError` in
`CoordinationServiceAgent`), it is likely a **Distributed Hang**.

*   Verify if the hang was triggered by a silent death of a worker (OOM,
    eviction) or a software deadlock.
*   Only classify as `Distributed Hang` if there is no evidence of prior
    eviction, quota, or config error on any worker.

### Step 6: Populate the Table

For each run, add a row with:

*   **Suspected Category**: Use standardized categories
    (`Infrastructure / Eviction`, `Configuration`, `Distributed Hang`, `SW`).
*   **Reason**: The primary root cause message (e.g. `Quota exceeded`,
    `Evicted (ephemeral-storage pressure)`).
*   **Analysis**: Explain the chain of events (e.g., "GCS Quota exceeded during
    dataset load, causing worker to abort and triggering cascade shutdown").
*   **Culprit**: The specific pod that failed first (e.g. `slice-job-0-72`).

## Cross-run host analysis (good vs bad runs)

Use this when restarts land on new node pools or replacement nodes and a later
attempt fails or slows down.

1.  **Machine delta**: pick a good attempt (normal step time) and the first bad
    attempt. List the GKE nodes and physical hosts for both (map them with Find
    Gce Instance Mapping). Suspects = hosts in the bad attempt that were not in
    the good one.
2.  **Repeat offender**: across several bad attempts with mixed slices,
    intersect the bad host sets. A host present in several independent failures
    is a repeat offender.
3.  **Silent outliers**: a worker with consistently higher step time before the
    stall marks its host as a suspect even without an explicit error.
4.  **Exclusion test**: see the failure twice with the same suspects before you
    blame them. Then rerun with the suspect node pools excluded, after all old
    pods are gone ([[exclusion-test-workflow]]). Under
    `spec.replicatedJobs[0].template.spec`:

    ```yaml
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
          - matchExpressions:
            - key: cloud.google.com/gke-nodepool
              operator: NotIn
              values: [{bad_nodepool_1}, {bad_nodepool_2}]
    ```

    Kueue's quota math does not see this exclusion: a large job that excludes
    healthy or suspect pools can wait on topology ("fit only N out of M") while
    the queue looks free ([[kueue-tas-fit-only-n-of-m]]). Count what the
    exclusion costs before you rerun, and keep one list per run in the failure
    table: lists tend to drift between runs. If the same pools keep failing a
    class of workload, record them as [[known-bad-cubes]] instead of leaving
    each user to exclude them.

    If the run progresses without them, isolate via [[node-isolation-methodology]].
    If the error appears on all nodes and slices, it is config or software, not
    a host.

## Connections

-   **Playbooks**:
    -   [[gotchas-cloud-logging]]
    -   [[gotchas-kubernetes-gke]]
    -   [[node-isolation-methodology]]
-   **Workflows**:
    -   [[exclusion-test-workflow]]
-   **Concepts**:
    -   [[failure-tables]]
-   **Entities**:
    -   [[cloud-storage]]

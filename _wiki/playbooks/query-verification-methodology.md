---
title: Log Query Verification Methodology
type: playbook
internal_only: false
tags: [training]
created: 2026-06-12
updated: 2026-06-12
---

# Log Query Verification Methodology

## Overview

When querying logs during an investigation, an empty result (0 entries found)
must NEVER be interpreted as "no logs exist" or "logs have expired" without
testing the query itself.

## The Verification Loop

If your query `resource.labels.pod_name:"workload-name" AND "KEYWORD"` returns 0
results, you MUST execute the following verification steps before making any
claims:

### Step 1: Query for Presence (Broaden Filter)

Remove the specific keyword and search for *any* log entry containing the
workload name:

*   **Query**: `resource.labels.pod_name:"workload-name"` (or a substring match
    like `"workload-name"` in textPayload).
*   **Goal**: Confirm if GKE/GCP Logging has captured *any* logs for this run.
    If this returns entries, **the logs exist**, and your keyword filter
    (`KEYWORD`) was simply incorrect or did not occur.

### Step 2: Query for Naming Variations

If Step 1 still returns 0, search for naming variations in the cluster. Pod
prefixes might have suffixes like `-worker-job0-` or different controller
wrappers.

*   **Query**: Search for parts of the workload name in
    `resource.labels.pod_name` (e.g., search for the unique hash prefix).

### Step 3: Query Logging Health (Cluster Control Plane)

If Step 2 returns 0, verify if logging was active for that cluster at that
timestamp.

*   **Query**: Query system components or daemonset logs (like `fluentbit` or
    `tpu-device-plugin`) in the same cluster and time window.
*   **Goal**: If system logs exist, the logging service was healthy, meaning the
    workload itself did not start or was named completely differently.

## Critical Rule & Gotchas

You must document these verification steps in your reasoning. Assertions that
logs are "expired" or "missing" without showing a broader query proving the
entire workload's logs are absent will be marked as a validation failure.

## Connections

-   **Playbooks**:
    -   [[debugging-mindset]]
    -   [[general-ml-debugging]]
-   **Tools**:
    -   [[cloud-logging]]

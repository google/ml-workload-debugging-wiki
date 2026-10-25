---
title: Triage Workloads Script
type: tool
internal_only: false
tags: [tpu, training, triage]
created: 2026-06-13
updated: 2026-10-03
---

## Overview

The `triage_workloads_api.py` script is a diagnostic helper utility designed to
scan Google Cloud Logging for a series of related workloads. It aggregates exit
codes, detects Kubelet evictions, scans for API quota limits (such as GCS
Egress), flags TPU OOMs, and extracts Python tracebacks from the earliest
failing pods.

Using this script avoids common logging traps (like temporal delays or
case-sensitive exact phrase match errors).

## Location

The script is executable and located at: `tools/scripts/triage_workloads_api.py`
(action: [[triage-workload-signals]])

## Usage

Run the script from your wiki directory, passing the workload prefix and the
project ID:

```bash
python3 tools/scripts/triage_workloads_api.py <workload_prefix> --project=<project_id>
```

### Example

To triage the series of workloads `my-training-job-4` through `-7` in project
`[GCP_PROJECT_ID]`:

```bash
python3 tools/scripts/triage_workloads_api.py my-training-job --project=[GCP_PROJECT_ID]
```

## Output Analysis

The script outputs a summary block for each run identified under the prefix:

1.  **Exit Code Distribution**: Lists counts of all recorded worker exit codes.
2.  **Earliest Exit Code**: Pinpoints the timestamp and pod name of the very
    first worker that terminated.
3.  **Eviction Warning**: Displays GKE node resource pressure events (e.g. low
    ephemeral storage) matching the pod.
4.  **Quota/API limit error**: Displays HTTP 429 or Multi-Region quota errors
    during data load.
5.  **TPU OOM**: Detects if the TPU HBM memory was exhausted (`TpuOOM`), which
    often precedes evictions due to writing core dumps.
6.  **Traceback Signature**: Extracts the Python traceback (e.g., config
    `ValueError`) if the pod exited with code 1 or 134.

## Connections

-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[multi-workload-triage]]
-   **Symptoms**:
    -   [[missing-exit-codes-eviction]]

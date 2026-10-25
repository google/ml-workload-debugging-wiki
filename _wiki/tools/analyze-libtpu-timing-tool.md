---
title: libtpu Timing Analysis Script
description: Diagnostic utility for parsing libtpu program enqueue and completion
  timestamps, computing device execution and host dispatch metrics, and isolating
  outlier workers.
type: tool
internal_only: false
tags: [tpu, training, analysis, logging, observability, metrics, methodology]
created: 2026-09-01
updated: 2026-10-03
---

## Overview

`tools/scripts/analyze_libtpu_timing.py` reads `libtpu` logs (stdin or a file),
matches program enqueue and device completion events across TensorCores and
workers, and reports:

1.  **Device Execution Duration ($T_{exec}$)**: Time from program submission to
    device finish.
2.  **Host Dispatch Gap ($T_{dispatch\_gap}$)**: Inter-step delay between
    finishing execution and enqueueing the next program.
3.  **Intra-Host TensorCore Skew**: Time difference between the first and last
    TensorCore enqueue on the same worker host.
4.  **HLO Fingerprint Consistency**: Verifies that all workers in an execution
    cycle run identical compiled program hashes.
5.  **Statistical Outlier Detection**: Flags straggler workers using Median
    Absolute Deviation (MAD), IQR, and ratio tests.

## Usage & Commands

### 1. Basic Invocation via Log File

```bash
python3 tools/scripts/analyze_libtpu_timing.py /path/to/workload_libtpu.log
```

### 2. Piped Invocation from Cloud Logging

```bash
gcloud logging read 'logName="projects/MY_PROJECT/logs/libtpu"' --format="value(textPayload)" | \
  python3 tools/scripts/analyze_libtpu_timing.py
```

### 3. Structured JSON Output

```bash
python3 tools/scripts/analyze_libtpu_timing.py /path/to/workload_libtpu.log --json > /tmp/timing_report.json
```

### 4. Downloading Logs in Batch across Slices (Helper Script)

Use the batch downloader helper `tools/scripts/download_libtpu_logs.py` to
retrieve multi-slice logs without query timeouts:

```bash
python3 tools/scripts/download_libtpu_logs.py \
  --project="<PROJECT_ID>" \
  --cluster="<CLUSTER_NAME>" \
  --start-time="2026-08-12T00:00:00Z" \
  --end-time="2026-08-12T00:01:00Z" \
  --num-slices=4 \
  --output="/tmp/workload_libtpu_logs.txt"
```

### 5. BigQuery / Log Analytics

The SQL query is not reproduced here; use the full query in
[[libtpu-enqueue-timing-analysis]].

## Troubleshooting Scenarios

-   [[libtpu-timing-outlier-analysis]]: step performance across multi-slice
    clusters.
-   [[libtpu-enqueue-timing-analysis]]: host-side bottleneck (data loading,
    Python serialization) vs accelerator degradation.

## Connections

-   **Workflows**:
    -   [[libtpu-timing-outlier-analysis]]
-   **Actions**:
    -   [[analyze-libtpu-timing]]
-   **Playbooks**:
    -   [[libtpu-enqueue-timing-analysis]]
-   **Symptoms**:
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[libtpu-program-lifecycle]]
-   **Entities**:
    -   [[libtpu]]
    -   [[tensorcore]]

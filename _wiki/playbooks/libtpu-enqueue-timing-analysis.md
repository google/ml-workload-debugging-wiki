---
title: libtpu Enqueue and Execution Timing Analysis Playbook
description: Heuristic playbook for computing per-worker libtpu exec time and host dispatch gaps from program lifecycle logs to find stragglers.
type: playbook
internal_only: false
tags: [tpu, training, analysis, logging, observability, metrics, methodology, triage]
created: 2026-09-01
updated: 2026-10-03
---

# libtpu Enqueue and Execution Timing Analysis Playbook

## Overview

Use for step-time regressions, latency spikes, or uneven worker progress. Per
TensorCore enqueue/finish timestamps separate host stalls, device slowdowns,
compile divergence, and intra-host thread contention.

## Log signatures and formulas

```text
INFO <timestamp> [<worker_id>] <coord>_TC<id>: Enqueueing program <prog_id>: <fingerprint> to continuation queue (launch_id=<launch_id>)
INFO <timestamp> [<worker_id>] <coord>_TC<id>: last_finished_run_id updated to <prog_id>, remaining programs=<rem>
```

-   $T_{exec}$ = finish time − enqueue time of the same `prog_id` on the same
    TC.
-   $T_{dispatch\_gap}$ = next enqueue − previous finish on the same TC.
-   TC skew = spread of enqueue times across the TCs of one worker for one
    `launch_id`.
-   Ironwood has 2 TensorCores (TC0, TC1) per chip, so a 4-chip VM produces 8
    log streams.
-   Application logs (loss, metrics, checkpoint saves) printed between
    `last_finished_run_id updated` and the next `Enqueueing program` show what
    the host was doing during a long dispatch gap.

## Procedure

1.  Extract the lifecycle lines:

    ```bash
    gcloud logging read 'logName="projects/<PROJECT>/logs/libtpu" AND ("Enqueueing program" OR "last_finished_run_id updated")' \
      --format="value(textPayload)" \
      --freshness=1h > /tmp/workload_libtpu.log
    ```

    Multi-slice:

    ```bash
    python3 tools/scripts/download_libtpu_logs.py \
      --project="<PROJECT_ID>" \
      --cluster="<CLUSTER_NAME>" \
      --start-time="2026-08-12T00:00:00Z" \
      --end-time="2026-08-12T00:01:00Z" \
      --num-slices=4 \
      --output="/tmp/workload_libtpu.log"
    ```

2.  Analyze:

    ```bash
    python3 tools/scripts/analyze_libtpu_timing.py /tmp/workload_libtpu.log
    python3 tools/scripts/analyze_libtpu_timing.py /tmp/workload_libtpu.log --json > /tmp/timing_analysis.json
    ```

3.  Classify with the matrix below.

## Outlier matrix (Analysis)

-   **A. Host straggler**: one worker has $T_{dispatch\_gap}$ > 2.5× median
    (e.g. 1200 ms vs 45 ms) while its $T_{exec}$ is normal or short; peers show
    normal gap but long $T_{exec}$ (they wait at collectives for it). Causes:
    local disk/GCS read stall, thread deadlock, memory copy, GC pause. Action:
    read that worker's app logs in the pre-enqueue window for I/O errors, data
    loader timeouts, GC pauses.
-   **B. Global host stall**: all workers have $T_{dispatch\_gap}$ > 2.5× median
    at once. Causes: GCS 429/503 rate limits, synchronous checkpoint dump, heavy
    host-side eval, Python-level global sync. Action: check GCS metrics for
    429/503 ([[cloud-storage]]).
-   **C. Device hardware straggler**: all workers enqueue in sync
    ($T_{dispatch\_gap}$ ≈ 50 ms) but one worker has $T_{exec}$ > 1.5× median.
    Causes: hardware fault, HBM ECC retries, degraded optical ICI link, PCIe
    throttling. Action: run [[ai-hw-analyzer]]; check dmesg/driver logs for ECC
    errors or PCIe link downshifts.
-   **D. Fingerprint mismatch**: `analyze_libtpu_timing.py` reports more than
    one HLO fingerprint in the same cycle on a subset of workers. Causes: shape
    mismatch, dynamic batching, process role divergence. Action: read XLA
    compile logs.
-   **E. Intra-host TC skew**: TC0-to-TC7 enqueue spread on one worker > 100 ms.
    Causes: host CPU starvation, cgroup CPU throttling, lock contention in the
    host TPU driver. Action: check CPU use, cgroup throttling, thread pinning.

## Log Analytics SQL (thousand-worker clusters)

Enable Log Analytics and link the bucket:

```bash
gcloud logging buckets update _Default --location=global --enable-analytics
gcloud logging links create tpu_logs_link --bucket=_Default --location=global
```

Run in BigQuery or **Logging > Log Analytics**. The `CASE` holds the outlier
thresholds:

```sql
WITH parsed_events AS (
  SELECT
    timestamp,
    COALESCE(
      resource.labels.container_name,
      resource.labels.pod_name,
      REGEXP_EXTRACT(text_payload, r'\[resource\.labels\.containerName:\s*([^\]]+)\]'),
      'worker-unknown'
    ) AS worker_id,
    REGEXP_EXTRACT(text_payload, r'([0-9a-zA-Z_x]+_TC[0-9]+):') AS tc_id,
    CASE
      WHEN REGEXP_CONTAINS(text_payload, r'Enqueueing program') THEN 'ENQUEUE'
      WHEN REGEXP_CONTAINS(text_payload, r'last_finished_run_id updated') THEN 'FINISH'
      ELSE 'OTHER'
    END AS event_type,
    CAST(COALESCE(
      REGEXP_EXTRACT(text_payload, r'Enqueueing program (\d+):'),
      REGEXP_EXTRACT(text_payload, r'last_finished_run_id updated to (\d+)')
    ) AS INT64) AS program_id,
    REGEXP_EXTRACT(text_payload, r'Enqueueing program \d+:\s*([0-9a-fA-F]+)') AS fingerprint,
    CAST(REGEXP_EXTRACT(text_payload, r'launch_id=(\d+)') AS INT64) AS launch_id
  FROM `<PROJECT_ID>.global._Default.tpu_logs_link._AllLogs`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR)
    AND (REGEXP_CONTAINS(text_payload, r'Enqueueing program')
         OR REGEXP_CONTAINS(text_payload, r'last_finished_run_id updated'))
),
matched_runs AS (
  SELECT enq.worker_id, enq.tc_id, enq.program_id, enq.fingerprint, enq.launch_id,
    enq.timestamp AS t_enqueue, fin.timestamp AS t_finish,
    TIMESTAMP_DIFF(fin.timestamp, enq.timestamp, MICROSECOND) / 1000000.0 AS t_exec_sec
  FROM (SELECT * FROM parsed_events WHERE event_type = 'ENQUEUE') enq
  JOIN (SELECT * FROM parsed_events WHERE event_type = 'FINISH') fin
    ON enq.worker_id = fin.worker_id AND enq.tc_id = fin.tc_id
    AND enq.program_id = fin.program_id
),
ordered_steps AS (
  SELECT *, LEAD(t_enqueue) OVER (PARTITION BY worker_id, tc_id ORDER BY t_enqueue) AS t_next_enqueue
  FROM matched_runs
),
timing_intervals AS (
  SELECT *, TIMESTAMP_DIFF(t_next_enqueue, t_finish, MICROSECOND) / 1000000.0 AS t_dispatch_gap_sec
  FROM ordered_steps
),
worker_step_aggregates AS (
  SELECT worker_id, launch_id, fingerprint,
    MIN(t_enqueue) AS worker_step_start, MAX(t_finish) AS worker_step_finish,
    AVG(t_exec_sec) AS avg_t_exec, AVG(t_dispatch_gap_sec) AS avg_dispatch_gap,
    TIMESTAMP_DIFF(MAX(t_enqueue), MIN(t_enqueue), MICROSECOND) / 1000.0 AS tc_enqueue_skew_ms
  FROM timing_intervals
  GROUP BY worker_id, launch_id, fingerprint
),
step_baselines AS (
  SELECT *,
    PERCENTILE_CONT(avg_t_exec, 0.5) OVER (PARTITION BY launch_id) AS median_t_exec,
    PERCENTILE_CONT(avg_dispatch_gap, 0.5) OVER (PARTITION BY launch_id) AS median_dispatch_gap,
    STDDEV(avg_dispatch_gap) OVER (PARTITION BY launch_id) AS stddev_dispatch_gap,
    COUNT(DISTINCT fingerprint) OVER (PARTITION BY launch_id) AS unique_fingerprints_count
  FROM worker_step_aggregates
)
SELECT worker_id, launch_id, fingerprint,
  ROUND(avg_t_exec, 3) AS t_exec_sec,
  ROUND(avg_dispatch_gap * 1000, 1) AS dispatch_gap_ms,
  ROUND(tc_enqueue_skew_ms, 2) AS tc_skew_ms,
  ROUND(median_t_exec, 3) AS cluster_median_t_exec_sec,
  ROUND(median_dispatch_gap * 1000, 1) AS cluster_median_gap_ms,
  CASE
    WHEN avg_dispatch_gap > GREATEST(0.2, median_dispatch_gap * 2.5, median_dispatch_gap + 2.5 * COALESCE(stddev_dispatch_gap, 0))
      THEN 'OUTLIER: Host Data/CPU Straggler (Late Enqueue)'
    WHEN avg_t_exec > GREATEST(0.5, median_t_exec * 1.5) AND avg_dispatch_gap <= median_dispatch_gap * 1.5
      THEN 'OUTLIER: Device Hardware Straggler (Slow Compute)'
    WHEN tc_enqueue_skew_ms > 100.0
      THEN 'OUTLIER: Intra-Host Thread Contention'
    WHEN unique_fingerprints_count > 1
      THEN 'OUTLIER: Fingerprint Mismatch'
    ELSE 'HEALTHY'
  END AS status
FROM step_baselines
ORDER BY launch_id DESC, dispatch_gap_ms DESC;
```

## Connections

-   **Workflows**:
    -   [[libtpu-timing-outlier-analysis]]
-   **Actions**:
    -   [[analyze-libtpu-timing]]
-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[trace-program-execution-times]]
-   **Symptoms**:
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[libtpu-program-lifecycle]]
-   **Entities**:
    -   [[cloud-storage]]
    -   [[libtpu]]
    -   [[tensorcore]]
-   **Tools**:
    -   [[cloud-logging]]

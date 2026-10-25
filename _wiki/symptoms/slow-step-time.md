---
title: Slow Step Time (Performance Degradation)
description: "Workload executes training steps at an abnormally slow pace without crashing, often caused by stragglers, data loader bottlenecks, or degraded interconnect links."
type: symptom
internal_only: false
tags: [tpu, training, hangs, metrics, monitoring, gke, mxla]
created: 2026-06-13
updated: 2026-10-09
---


# Slow Step Time (Performance Degradation)

## Definition / Explanation

A **Slow Step Time** symptom occurs when the training workload continues to
progress and does not crash or hang, but the time required to complete each
training step (forward/backward pass) increases significantly compared to the
baseline. This results in reduced training efficiency and lower Model FLOPs
Utilization (MFU), increasing the cost and duration of the job.

On GKE clusters running version `1.36.0-gke.4447000` and later, **ML Diagnostics
Workload Monitoring** ([[workload-monitoring]]) automatically tracks aggregated
TPU duty cycle across the workload and triggers a **`PERFORMANCE_DEGRADATION`**
monitored event when it detects a **15% drop in aggregated TPU duty cycle**
(default threshold, evaluated after a **15-minute `duty cycle threshold`
warmup gate** from the last notification time on each `RUNNING` transition,
comparing backward-looking `[t-5m, t]` vs. `[t-15m, t]` windows).

## How to Identify

1.  **ML Diagnostics `PERFORMANCE_DEGRADATION` Events & AutoCaptured XProf
    Sessions (GKE `>= 1.36.0-gke.4447000`)**: Run
    [[query-mldiagnostics-monitored-events]] via [[ml-diagnostics-cli]] to
    inspect automated `analyzerReports` across all 11 `PERFORMANCE_DEGRADATION`
    subsystems, negative hardware rule-out (`not_detected_analyzers`),
    pinpointed `affected_instances` / `affected_gce_instance_ids`, and
    AutoCaptured XProf `.xplane.pb` traces (`profiler_sessions`).

2.  **Step Time Deviation**: Check goodput/performance dashboards. An increase
    in step time deviation indicates a slowdown.

3.  **Metrics Nuance**: At massive scales, actual step times might be hidden
    from cloud-side telemetry due to privacy/data egress rules. In such cases,
    monitor the *relative step time deviation* rather than raw step times.

4.  **Step Count Graph**: Look at the step count graph over time. A flattening
    slope indicates a slowdown. Gaps in the graph indicate unsuccessful restart
    attempts.

5.  **Data Granularity Challenge**: Step times are often reported to dashboards
    in batches (e.g., every 'n' steps, taking 30-40 minutes). This can mask
    short-lived slowdowns. To calculate granular step times:

    -   Search worker logs for "step started" and "step finished" events.

    -   Calculate the precise timestamp difference between the "incued" and "run
        completion" timestamps.

    -   Specifically, trace the `Enqueueing program` log and match it to its
        corresponding `last_finished_run_id` completion log emitted by
        `real_program_continuator.cc`. The delta between `Enqueueing program`
        and `last_finished_run_id` can be compared across all the workers. In
        one example, you could see that on a unique worker, this delta is
        smaller compared to the rest of the workers, and that is because it is
        spending additional time before enqueuing the program to run unique
        work. For more details on this process, see
        [[trace-program-execution-times]], the
        [[libtpu-timing-outlier-analysis]] workflow, and
        [[analyze-libtpu-timing]].

6.  **XProf Trace MapDMA Buffer Registrations**: In XProf Trace Viewer, inspect
    the worker execution timeline during steady-state steps. Host memory
    registrations (`MapDmaBuffer` calls) are expected only once early during
    process startup. If `MapDmaBuffer` calls occur during steady-state
    execution, it indicates a buffer pool shortage, causing on-the-fly dynamic
    memory mapping and stalls. In addition, inspect Megascale "Communication
    Transport" slices for fallback host-to-host memory copies.

### System Metrics (1-Minute Cloud Monitoring & 10-Second SDK)

Monitor the following Cloud Monitoring metrics (collected every 1 minute by
default, or every 10 seconds when `log_system_metrics=true` is enabled in the
`google-cloud-mldiagnostics` SDK):

-   **TPU Device**: `node/accelerator/duty_cycle`,
    `node/accelerator/tensorcore_utilization`
-   **HBM**: `node/accelerator/memory_bandwidth_utilization`,
    `node/accelerator/memory_used`, `node/accelerator/memory_total`
-   **Host CPU & Memory**: `node/cpu/allocatable_utilization`,
    `node/memory/allocatable_utilization`
-   **MXLA Multi-Slice Latencies (`libtpu >= 0.40.0`)**:
    -   `container/multislice/network/dcn_transfer_latencies`
    -   `container/multislice/network/dcn_inbound_transfer_latencies`
    -   `container/multislice/accelerator/compute_latencies`
    -   `container/multislice/network/collective_end_to_end_latencies`
    -   `container/multislice/accelerator/device_to_host_transfer_latencies`
    -   `container/multislice/accelerator/host_to_device_transfer_latencies`

## Common Root Causes & ML Diagnostics `PERFORMANCE_DEGRADATION` Analyzers

1.  **Single Degraded Machine (Bad Actor)**: In synchronous distributed
    training, the entire cluster must wait at barrier synchronization points. If
    a single TPU machine is performing poorly (due to thermal throttling, PCIe
    bandwidth limitations, or memory degradation), it will drag down the speed
    of the entire workload.
2.  **ML Diagnostics `PERFORMANCE_DEGRADATION` Bottlenecks (11 Automated
    Analyzers)**: When `PERFORMANCE_DEGRADATION` triggers after the 15-minute
    warmup gate, ML Diagnostics evaluates 11 infrastructure and framework
    analyzers to isolate silent metric-only bottlenecks:

    -   **Thermal Analyzer**: Detects TPU nodes overheating in [[tensorcore]]s
        or HBM.
    -   **TPU Throttling Analyzer**: Detects TPU chip throttling due to power,
        thermal, or hardware constraints.
    -   **HBM Capacity Analyzer**: Detects HBM utilization approaching ~90%
        capacity or causing OOMs (recommends XProf traces or batch/model size
        reduction).
    -   **HBM Bandwidth Analyzer**: Detects throttled HBM read/write bandwidth
        due to high chip temperature or bandwidth limits.
    -   **Host Memory Utilization Analyzer**: Detects non-evictable host RAM
        approaching limits from data pipeline prefetching/buffering, checkpoint
        RAM spikes, or custom loader leaks.
    -   **CPU Utilization Analyzer**: Detects host CPU utilization approaching
        allocated pod limits (recommends Cloud Profiling or raising pod CPU
        limits).
    -   **ICI Link Analyzer**: Detects intra-slice Inter-Chip Interconnect
        ([[ici-errors]]) degradation between TPU processors.
    -   **ToR NIC Link Analyzer**: Detects link errors between the Top-of-Rack
        (ToR) switch and host NICs.
    -   **ToR Uplink Congestion**: Detects ToR switch uplink port high
        utilization and fabric congestion.
    -   **PyGrain Data Analyzer**: Detects port-`9431` Prometheus input pipeline
        stalls or PyGrain data loader bottlenecks.
    -   **Async Checkpointing Analyzer**: Detects port-`9431` Prometheus
        asynchronous checkpointing contention or stalls.

3.  **Host Skew (Expensive Evaluations)**: Running expensive evaluation,
    validation, or metric-logging steps synchronously within the main training
    loop (often on JAX Process 0 or a subset of workers) causes other workers to
    wait at the barrier, stalling overall progress.

4.  **Network Performance Issues**:

    -   **TCP Window Size**: At massive scales (e.g. 32k chips) over DCN,
        incorrect TCP configurations (like `tcp_rmem` window sizes) can limit
        bandwidth and slow down communication.

    -   **ICI / OCS Link Degradation**: Packet drops or retransmits on the
        optical switch (OCS) or copper links (see [[ici-straggler-triage]]).

5.  **Checkpointing Bottlenecks**: High GCS latency during checkpoint saves, or
    slow deletions of older checkpoint folders in GCS blocking the next step.

6.  **MapDMA Buffer Shortage and Registration Stalls**: In Megascale workloads,
    host memory must be registered before being used for DMA transfers to/from
    TPUs. If the pre-mapped memory pool (`--megascale_grpc_premap_memory_bytes`)
    is exhausted, the Megascale XLA runtime dynamically invokes `MapDmaBuffer`
    calls during steady-state execution. This introduces synchronous
    registration delays and memory copies into what should be a zero-copy
    transport path, causing step-time degradation.

## Diagnosis Workflow

1.  **Check ML Diagnostics `MonitoredEvent` Reports & AutoCaptured XProf
    Sessions**: Run [[query-mldiagnostics-monitored-events]] via
    [[ml-diagnostics-cli]] to see if any of the 11 `PERFORMANCE_DEGRADATION`
    analyzers flagged specific nodes, inspect `not_detected_analyzers` to rule
    out healthy subsystems, and check `profiler_sessions` for AutoCaptured
    `.xplane.pb` XProf traces.

2.  **Locate the Window**: Determine when the slowdown started and whether it is
    associated with a specific restart attempt (e.g., beginning at attempt 92).

3.  **Analyze Machine Delta**: Compare the list of active physical hostnames
    from a "good" run (fast step time) against the "bad" run (slow step time).
    The delta (machines present only in the bad run) contains the suspect
    degraded machine.

4.  **Check for Host Skew**: Run [[libtpu-timing-outlier-analysis]] and
    [[analyze-libtpu-timing]] to inspect the distribution of step durations and
    host dispatch gaps across all workers.

5.  **Review Evaluations & Checkpoint Logs**: Verify if the slow steps correlate
    with checkpoint writes or evaluation intervals. Refer to
    [[gotchas-distributed-training]] for evaluation gotchas.

6.  **Capture and Inspect XProf Traces**: Capture an XProf trace for the
    degraded run using `jax.profiler.trace`. Open the Trace Viewer in XProf
    (`http://localhost:6006`) and search for `MapDmaBuffer` calls on worker
    timelines during steady-state. Inspect the Megascale "Communication
    Transport" traces for unintended memory copies, verify collective slack
    time, and tune `--megascale_grpc_premap_memory_bytes` following
    [[megascale-performance-tuning]].

## Connections

-   **Workflows**:
    -   [[monitoring-progress]]
    -   [[libtpu-timing-outlier-analysis]]
-   **Actions**:
    -   [[query-mldiagnostics-monitored-events]]
    -   [[analyze-libtpu-timing]]
-   **Playbooks**:

    -   [[trace-program-execution-times]]
    -   [[gotchas-distributed-training]]
    -   [[megascale-performance-tuning]]

-   **Concepts**:

    -   [[workload-monitoring]]
    -   [[goodput]]
    -   [[ici-errors]]

-   **Tools**:

    -   [[ml-diagnostics-cli]]
    -   [[xprof]]

-   **Entities**:

    -   [[tensorcore]]

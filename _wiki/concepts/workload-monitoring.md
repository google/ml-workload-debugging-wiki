---
title: Workload Progress Monitoring
description: Monitoring Cloud TPU GKE workload execution phases via coordinator logs and automated ML Diagnostics workload monitoring (states, 12 analyzers, and system metrics).
type: concept
internal_only: false
tags: [tpu, training, metrics, monitoring, observability, gke, mxla, hangs]
created: 2026-06-11
updated: 2026-10-07
---


# Workload Progress Monitoring

Workload progress monitoring tracks the execution lifecycle, health states, and
performance metrics of distributed AI/ML training jobs on Cloud TPUs via
coordinator logs and the built-in **ML Diagnostics Workload Monitoring**
platform (`hypercomputecluster.googleapis.com`).

## Overview: Progress Phases in Coordinator Logs

An MCJax or Pathways GKE workload progresses through five canonical stages in
the coordinator logs:

1.  **Initialization**: Environment setup, distributed coordinator handshake,
    and runtime configuration.
2.  **Topology Discovery**: `MegaScale Topology Discovery in progress...`
3.  **Compilation**: `XLA::TPU running hlo passes...`
4.  **Program Enqueueing**: `Enqueueing program {N}...`
5.  **Training Steps**: `completed step {N}...`

## Architecture & Design: ML Diagnostics Workload Monitoring

**Workload Monitoring** is a built-in capability within **ML Diagnostics**
(backed by the **Cluster Director API**, `hypercomputecluster.googleapis.com`)
that automatically detects and diagnoses issues on Cloud TPUs on GKE without
manual code instrumentation.

### Prerequisites & Automatic Tracking

-   **GKE Version**: Automatically enabled on GKE clusters running
    **`1.36.0-gke.4447000` and later**.
-   **libtpu Version**: **`libtpu >= 0.40.0`** is required for the **Megascale
    XLA Hang Analyzer** and **Megascale XLA multi-slice metrics**.
-   **Automatic `machineLearningRun` Creation**: Creates
    `<job-name>-<timestamp>` per GKE job and continuously monitors aggregated
    TPU duty cycle.

### Explicit Scope & Division of Labor in Debug Wiki

In Debug Wiki, [[query-mldiagnostics-monitored-events]] is scoped **specifically
to cover silent metric-only `PERFORMANCE_DEGRADATION` bottlenecks and
AutoCaptured XProf traces** that do not emit error signatures in Cloud Logging:

1.  **Host cgroup resource saturation**: `CPU Utilization Analyzer`,
    `Host Memory Utilization Analyzer`
2.  **Accelerator memory saturation**: `HBM Capacity Analyzer`,
    `HBM Bandwidth Analyzer`
3.  **Silent hardware/fabric throttling**: `Thermal Analyzer`,
    `TPU Throttling Analyzer`, `ICI Link Analyzer`, `ToR NIC Link Analyzer`,
    `ToR Uplink Congestion`
4.  **Port-`9431` Prometheus framework stalls**: `PyGrain Data Analyzer`,
    `Async Checkpointing Analyzer`
5.  **Negative hardware rule-out**: `not_detected_analyzers` across all 11
    subsystems
6.  **AutoCaptured XProf `.xplane.pb` traces**: `profiler_sessions`
    (`autoCaptured: true`, `kind`, `state`, `profilerTargets`, and GCS artifact
    paths)

#### 15-Minute `duty cycle threshold` Warmup Gate & When Not to Use ML Diagnostics

-   **Warmup Gate (`15 minutes`)**: `PERFORMANCE_DEGRADATION` detection requires
    a **15-minute `duty cycle threshold` warmup gate** starting from
    `last notification time` on each `RUNNING` transition or restart,
    comparing backward-looking `[t-5m, t]` short windows against `[t-15m, t]`
    baseline windows (`>= 15%` aggregated TPU duty cycle drop).
-   **When to Use Core Debug Wiki Workflows Instead**: For crashes, hangs,
    restarts, preemptions, or `<15m` startup/checkpoint-restore stalls, do
    **not** rely on ML Diagnostics; use Debug Wiki's core Cloud Logging and
    host/node triage workflows ([[debugging-megascale-hangs]],
    [[workload-disruption-orchestration]], [[mxla-hardware-triage]])
    instead.

### Tracked Workload States

1.  **`running`**: Actively executing on the cluster (not displayed as an
    interruption state in the Cloud Console).
2.  **`termination`**: Workload terminated due to an orchestrator
    interruption/restart, successful completion, or failure.
3.  **`hang`**: Prolonged period of minimal to no TPU activity without
    completion.
4.  **`performance degradation`**: Triggered by a significant metric drop (**15%
    drop in aggregated TPU duty cycle** by default after the 15-minute
    `duty cycle threshold` warmup gate from `last notification time`).
5.  **`Orchestrator interruption`**:
    -   **Slice interruptions** on **TPU7x (Ironwood)**: Identifies the affected
        slice and physical sub-block.
    -   **Nodepool interruptions** on **TPU v6e (Trillium) and earlier**:
        Registers individual VMs as nodepools and applies
        `node.kubernetes.io/unreachable` taints and `NotReady` conditions.

### Automated Diagnostic Analyzers & Deterministic Wiki Mapping

When [[query-mldiagnostics-monitored-events]] inspects `analyzerReports` where
`detectionState == "DETECTED"`, it extracts `affected_instances` and
`affected_gce_instance_ids` (plus `not_detected_analyzers` for negative proof of
healthy subsystems and `profiler_sessions` when `PERFORMANCE_DEGRADATION` is
present) and maps each analyzer (supporting both space-separated and CamelCase
API identifiers) to wiki symptoms, workflows, and actions:

-   **Silent `PERFORMANCE_DEGRADATION` Subsystems (11 Analyzers)**:
    -   **CPU Utilization Analyzer** (host cgroup CPU utilization approaching
        pod limits; emits `Instance Id <id>: [Pods: <pod>]`) ->
        [[slow-step-time]] -> [[libtpu-timing-outlier-analysis]]
    -   **Host Memory Utilization Analyzer** (non-evictable host cgroup RAM near
        limits from prefetching, checkpoints, or leaks) ->
        [[tpu-coordinator-oom]], [[slow-step-time]] ->
        [[workload-disruption-orchestration]]
    -   **HBM Capacity Analyzer** (HBM utilization near ~90% capacity) ->
        [[slow-step-time]] -> [[libtpu-timing-outlier-analysis]]
    -   **HBM Bandwidth Analyzer** (HBM bandwidth throttled by temperature or
        limits) -> [[slow-step-time]] -> [[repair-in-place]],
        [[node-isolation-and-rr-api]]
    -   **Thermal Analyzer** (overheating in [[tensorcore]]s or HBM) ->
        [[slow-step-time]] -> [[repair-in-place]], [[node-isolation-and-rr-api]]
    -   **TPU Throttling Analyzer** (power, thermal, or hardware chip
        throttling) -> [[slow-step-time]] -> [[repair-in-place]],
        [[node-isolation-and-rr-api]]
    -   **ICI Link Analyzer** (intra-slice ICI networking degradation) ->
        [[slow-step-time]], [[grpc-error-start-session-failed]] ->
        [[ici-straggler-triage]]
    -   **ToR NIC Link Analyzer** (ToR switch to host NIC link errors) ->
        [[slow-step-time]] -> [[libtpu-timing-outlier-analysis]]
    -   **ToR Uplink Congestion** / **ToR Uplink Congestion Analyzer** (high
        utilization or congestion on ToR uplink ports) -> [[megascale-hang]],
        [[slow-step-time]] -> [[debugging-megascale-hangs]],
        [[query-node-health-events]]
    -   **PyGrain Data Analyzer** (port-`9431` Prometheus input pipeline
        bottlenecks or PyGrain data loading stalls during
        `PERFORMANCE_DEGRADATION`) -> [[slow-step-time]],
        [[step-timebomb-watchdog]] -> [[libtpu-timing-outlier-analysis]],
        [[analyze-libtpu-timing]], [[query-workload-logs]]
    -   **Async Checkpointing Analyzer** (port-`9431` Prometheus asynchronous
        checkpointing stalls or contention during `PERFORMANCE_DEGRADATION`) ->
        [[incomplete-checkpoint-restore]], [[step-timebomb-watchdog]] ->
        [[workload-disruption-orchestration]], [[query-workload-logs]]
-   **Reference Mapping for Orchestrator & Framework Events (Handled Primarily
    by Core Cloud Logging & Host/Node Triage Workflows)**:
    -   **GKE Orchestrator Slice Interruption Analyzer** /
        **`SliceInterruptionAnalyzer`** -> [[tpu-provisioner-slice-recreation]],
        [[slice-failure]] -> [[mxla-hardware-triage]],
        [[workload-disruption-orchestration]]
    -   **GKE Nodepool Interruption Analyzer** /
        **`NodepoolInterruptionAnalyzer`** ->
        [[tpu-provisioner-slice-recreation]], [[false-preemption-sigterm]],
        [[data-loss-pipe-closed]] -> [[workload-disruption-orchestration]]
    -   **Megascale XLA Hang Analyzer (`libtpu >= 0.40.0`, 11 Causes)**: Mapped
        if present (`BAD_TPU_CHIP`, `UNRECOVERABLE_ERROR`, `BAD_SC_CHIP`,
        `ICI_ERROR`, `DATA_INPUT_STALL`, `DIFFERENT_MODULE`,
        `FINGERPRINT_MISMATCH`, `LAUNCHES_ORDER_INCONSISTENT`,
        `PROGRAM_NOT_QUEUED`, `NETWORKING_ISSUE`, `UNKNOWN`), though active hang
        triage uses [[debugging-megascale-hangs]] directly.

## Key Metrics (1-Minute Cloud Monitoring & 10-Second SDK)

Exported at **1-minute granularity** in Cloud Monitoring by default, or
**10-second granularity** with `log_system_metrics=true` in
`google-cloud-mldiagnostics`:

-   **TPU Device**: `node/accelerator/duty_cycle`,
    `node/accelerator/tensorcore_utilization`
-   **Host Device**: `node/cpu/allocatable_utilization`,
    `node/memory/allocatable_utilization`
-   **HBM**: `node/accelerator/memory_bandwidth_utilization`,
    `node/accelerator/memory_total`, `node/accelerator/memory_used`
-   **Megascale XLA (`libtpu >= 0.40.0`)**:
    `container/multislice/network/dcn_transfer_latencies`,
    `container/multislice/network/dcn_inbound_transfer_latencies`,
    `container/multislice/accelerator/compute_latencies`,
    `container/multislice/network/collective_end_to_end_latencies`,
    `container/multislice/accelerator/device_to_host_transfer_latencies`,
    `container/multislice/accelerator/host_to_device_transfer_latencies`

## Operational Impact & Troubleshooting Stalls

-   **15% Duty Cycle Drop (`PERFORMANCE_DEGRADATION` after 15m Warmup Gate)**:
    Run [[query-mldiagnostics-monitored-events]] to inspect the 11
    `PERFORMANCE_DEGRADATION` analyzers, `not_detected_analyzers` negative
    hardware rule-out, and AutoCaptured XProf `.xplane.pb` `profiler_sessions`
    (see [[slow-step-time]] and [[ml-diagnostics-cli]]).
-   **Crashes, Hangs, Restarts, Preemptions, or `<15m` Startup/Restore Stalls**:
    Use Debug Wiki's core Cloud Logging and host/node triage workflows
    ([[debugging-megascale-hangs]], [[workload-disruption-orchestration]],
    [[hang-diagnosis-methodology]], [[tpu-state-analysis-methodology]]).

## Connections

-   **Workflows**:
    -   [[monitoring-progress]]
    -   [[debugging-megascale-hangs]]
-   **Actions**:
    -   [[query-mldiagnostics-monitored-events]]
    -   [[check-pod-status]]
    -   [[tail-coordinator-logs]]
-   **Playbooks**:
    -   [[hang-diagnosis-methodology]]
    -   [[mcjax-debugging-principles]]
    -   [[tpu-state-analysis-methodology]]
-   **Symptoms**:
    -   [[megascale-hang]]
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[goodput]]
    -   [[megascale]]
-   **Tools**:
    -   [[ml-diagnostics-cli]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[libtpu]]
    -   [[tensorcore]]
    -   [[sparsecore]]

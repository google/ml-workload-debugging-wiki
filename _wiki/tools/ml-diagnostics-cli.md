---
title: ML Diagnostics CLI and Cluster Director API (gcloud alpha mldiagnostics)
description: Public gcloud alpha mldiagnostics CLI and hypercomputecluster.googleapis.com/v1alpha REST API for querying Cloud TPU machineLearningRun records, MonitoredEvents, analyzerReports, and on-demand XProf profiler sessions.
type: tool
internal_only: false
tags: [tpu, gke, gcloud, gcp, monitoring, observability, triage, mxla]
created: 2026-10-07
updated: 2026-10-07
---


# ML Diagnostics CLI and Cluster Director API (gcloud alpha mldiagnostics)

`gcloud alpha mldiagnostics` and the Cluster Director REST API
(`hypercomputecluster.googleapis.com/v1alpha`) provide command-line and
programmatic access to **ML Diagnostics Workload Monitoring**
([[workload-monitoring]]) for Cloud TPU workloads on GKE
(`>= 1.36.0-gke.4447000`).

## Overview

For every GKE job on a supported cluster (`GKE >= 1.36.0-gke.4447000`), ML
Diagnostics automatically creates a `machineLearningRun` resource named
`<job-name>-<timestamp>`. Operators and diagnostic agents use
`gcloud alpha mldiagnostics` or the `hypercomputecluster.googleapis.com/v1alpha`
REST API to:

-   Manage `machineLearningRun` executions (`create`, `describe`, `list`,
    `update`, `delete`) and inspect their state (`running`, `termination`,
    `hang`, `performance degradation`, `Orchestrator interruption`).
-   Query `monitoredEvents` (`list`, `describe`) and structured
    `analyzerReports` across all 12 Infrastructure, Orchestrator, and Framework
    analyzers (`detectionState`, `details`, affected instances, and
    `recommendedActions`).
-   Capture and inspect programmatic (`gs://<bucket>/<run>/plugins/profile/...`
    via `injection-webhook` + ML Diagnostics SDK) and on-demand
    (`profiler.start_server(port=9999)` via `connection-operator`) XProf
    profiling sessions (`profiler-target list`, `profiler-session list`,
    `profiler-session capture`).

### Prerequisites

-   **Project**: Set `gcloud config set project <project_id>` and enable
    `hypercomputecluster.googleapis.com`.
-   **IAM Roles**: `roles/hypercomputecluster.viewer` (read-only inspection) or
    `roles/hypercomputecluster.editor` (creating/updating runs or capturing
    profiler sessions).
-   **CLI Component**: Install the `gcloud` alpha component
    (`gcloud components install alpha`).

## Usage & Commands

### 1. Manage Machine Learning Runs (`machine-learning-run`)

```bash
# Register existing profiles in GCS (state "Completed"):
gcloud alpha mldiagnostics machine-learning-run create <run_name> \
    --location=<location> --run-group=<run_group> \
    --gcs-path=gs://<bucket>/<run_dir> --display-name=<display_name> \
    --labels="list_existing_sessions_only"="true"

# Create an active GKE JobSet run for on-demand profiling:
gcloud alpha mldiagnostics machine-learning-run create <run_name> \
    --location=<location> --run-group=<run_group> \
    --gcs-path=gs://<bucket>/<run_name> --display-name=<run_name> \
    --orchestrator=gke \
    --gke-cluster-name=/projects/<project_id>/locations/<location>/clusters/<cluster_name> \
    --gke-namespace=default --gke-workload-name=<jobset_name> \
    --gke-kind=JobSet --gke-workload-create-time=<iso_timestamp> --run-phase=ACTIVE

# List, describe, update (--etag), or delete ML runs (delete preserves GCS/GKE data):
gcloud alpha mldiagnostics machine-learning-run list --location=<location> --project=<project_id> --format=json
gcloud alpha mldiagnostics machine-learning-run describe <ml_run_id> --location=<location> --project=<project_id> --format=json
gcloud alpha mldiagnostics machine-learning-run update <ml_run_id> \
    --location=<location> --etag=<etag> --run-group=<run_group> \
    --gcs-path=gs://<bucket>/<run_name> --display-name=<display_name> --run-phase=COMPLETED
gcloud alpha mldiagnostics machine-learning-run delete <ml_run_id> --location=<location> --project=<project_id>
```

### 2. Inspect Monitored Events & Analyzer Reports (`monitored-events`)

```bash
gcloud alpha mldiagnostics monitored-events list \
    --machine-learning-run=<ml_run_id> --location=<location> --project=<project_id> --format=json
gcloud alpha mldiagnostics monitored-events describe <event_id> \
    --machine-learning-run=<ml_run_id> --location=<location> --project=<project_id> --format=json
```

### 3. On-Demand XProf Profiling (`profiler-target` & `profiler-session`)

```bash
gcloud alpha mldiagnostics profiler-target list --machine-learning-run=<ml_run_id> --location=<location>
gcloud alpha mldiagnostics profiler-session list --machine-learning-run=<ml_run_id> --location=<location>
gcloud alpha mldiagnostics profiler-session capture <session_name> \
    --machine-learning-run=<ml_run_id> --location=<location> \
    --targets=<target_1>,<target_2> --duration=10s \
    --device-tracer-level=device-tracer-level-enabled \
    --host-tracer-level=host-tracer-level-info \
    --python-tracer-level=python-tracer-level-disabled
```

### 4. Cluster Director REST API (`hypercomputecluster.googleapis.com/v1alpha`)

```bash
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://hypercomputecluster.googleapis.com/v1alpha/projects/<project_id>/locations/<location>/machineLearningRuns/<ml_run_id>/monitoredEvents"
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://hypercomputecluster.googleapis.com/v1alpha/projects/<project_id>/locations/<location>/machineLearningRuns/<ml_run_id>/monitoredEvents/<event_id>"
```

### Cloud Console Entrypoints

-   **Cluster Director**:
    `https://console.cloud.google.com/cluster-director/diagnostics` (**Run
    Diagnostics** tab)
-   **GKE AI/ML**:
    `https://console.cloud.google.com/kubernetes/aiml/diagnostics` (**AI/ML >
    Run Diagnostics** tab)

## Explicit Scope & Division of Labor in Debug Wiki

In Debug Wiki, [[query-mldiagnostics-monitored-events]] is used **specifically
to cover silent metric-only `PERFORMANCE_DEGRADATION` bottlenecks** (and
AutoCaptured XProf traces) that do not emit error signatures in Cloud Logging:

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

### 15-Minute `duty cycle threshold` Warmup Gate

-   `PERFORMANCE_DEGRADATION` events require a **15-minute
    `duty cycle threshold` warmup gate** starting from
    `lastNotificationTime` on each `RUNNING` transition or restart,
    evaluating backward-looking `[t-5m, t]` vs. `[t-15m, t]` aggregated TPU duty
    cycle windows.
-   For crashes, hangs, restarts, preemptions, or `<15m` startup/restore stalls,
    agents should use Debug Wiki's core Cloud Logging and host/node triage
    workflows ([[debugging-megascale-hangs]],
    [[workload-disruption-orchestration]], [[mxla-hardware-triage]])
    instead.

## Troubleshooting Scenarios & Deterministic Symptom Mapping

[[query-mldiagnostics-monitored-events]] queries `gcloud alpha mldiagnostics`
(with REST fallback), filters `analyzerReports` where
`detectionState == "DETECTED"`, records `not_detected_analyzers`, extracts
`affected_instances` and `affected_gce_instance_ids`, and (when
`PERFORMANCE_DEGRADATION` is present) queries `profiler-session list` to extract
AutoCaptured `.xplane.pb` `profiler_sessions`:

1.  **Triaging Silent `PERFORMANCE_DEGRADATION` Slowdowns
    ([[slow-step-time]], >= 15% Duty Cycle Drop After 15m Warmup Gate)**:
    -   **CPU Utilization Analyzer** -> [[slow-step-time]] ->
        [[libtpu-timing-outlier-analysis]]
    -   **Host Memory Utilization Analyzer** -> [[tpu-coordinator-oom]],
        [[slow-step-time]] -> [[workload-disruption-orchestration]]
    -   **HBM Capacity / ToR NIC Link Analyzers** -> [[slow-step-time]] ->
        [[libtpu-timing-outlier-analysis]]
    -   **Thermal / TPU Throttling / HBM Bandwidth Analyzers** ->
        [[slow-step-time]] -> [[repair-in-place]], [[node-isolation-and-rr-api]]
    -   **ICI Link Analyzer** -> [[slow-step-time]],
        [[grpc-error-start-session-failed]] -> [[ici-straggler-triage]]
    -   **ToR Uplink Congestion** -> [[megascale-hang]], [[slow-step-time]] ->
        [[debugging-megascale-hangs]], [[query-node-health-events]]
    -   **PyGrain Data Analyzer** (port-`9431` Prometheus) ->
        [[slow-step-time]], [[step-timebomb-watchdog]] ->
        [[libtpu-timing-outlier-analysis]], [[analyze-libtpu-timing]],
        [[query-workload-logs]]
    -   **Async Checkpointing Analyzer** (port-`9431` Prometheus) ->
        [[incomplete-checkpoint-restore]], [[step-timebomb-watchdog]] ->
        [[workload-disruption-orchestration]], [[query-workload-logs]]
    -   **AutoCaptured XProf Traces (`profiler_sessions`)**: Automatically
        retrieves `profilerSessions` (`autoCaptured: true`, `profilerTargets`,
        and `.xplane.pb` GCS paths) for immediate inspection in XProf.
2.  **Reference Mapping for Hang & Orchestrator Events**:
    -   While `query_mldiagnostics_monitored_events.py` retains deterministic
        mappings for `Megascale XLA Hang Analyzer` (11 causes) and Orchestrator
        interruption analyzers (`SliceInterruptionAnalyzer`,
        `NodepoolInterruptionAnalyzer`), active crash/hang/disruption triage
        uses [[debugging-megascale-hangs]] and
        [[workload-disruption-orchestration]] directly.

## Connections

-   **Workflows**:
    -   [[debugging-megascale-hangs]]
    -   [[monitoring-progress]]
    -   [[workload-disruption-orchestration]]
-   **Actions**:
    -   [[query-mldiagnostics-monitored-events]]
    -   [[query-crash-signatures]]
    -   [[query-workload-logs]]
-   **Playbooks**:
    -   [[hang-diagnosis-methodology]]
    -   [[workload-disruption-orchestration]]
-   **Symptoms**:
    -   [[megascale-hang]]
    -   [[slow-step-time]]
    -   [[tpu-provisioner-slice-recreation]]
    -   [[mxla-hardware-fault]]
    -   [[wait-timeout-on-sflag]]
    -   [[tpu-coordinator-oom]]
-   **Concepts**:
    -   [[workload-monitoring]]
    -   [[goodput]]
    -   [[megascale]]
-   **Tools**:
    -   [[cloud-logging]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[libtpu]]

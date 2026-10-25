---
title: General ML Debugging Steps for TPUs
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-05-13
updated: 2026-10-20
---

## Overview

Router for triaging large TPU workload failures (up to 50k+ chips), from the
application down to hardware. Stack model: [[layered-debugging-framework]].

## Triage Steps

1.  **Verbose logs:** set `TPU_STDERR_LOG_LEVEL`, `TPU_MIN_LOG_LEVEL`,
    `TF_CPP_MIN_LOG_LEVEL` ([[enable-tpu-verbose-logging]]).
2.  **Post-mortem: use [[cloud-logging]], not `kubectl`.** Pods of past runs are
    usually deleted. Use `kubectl` only on a live workload, e.g.
    `kubectl get pods -w | grep your-job-prefix` to see which pod goes `Running`
    -> `Error` first.
3.  **Coordinator and Process 0:** check worker 0, slice 0 (GKE coordinator) and
    JAX Process 0 (checkpointing), which may be a different worker; search
    `[process=0]`. If a query returns 0 results, follow
    [[query-verification-methodology]] before assuming logs expired.
4.  **Search crash signatures across the whole workload prefix** (exit codes go
    missing on silent crashes): `"Fatal Python error"`, `"Traceback"`,
    `"Segmentation fault"`, `"SIGABRT"`, `"Out of memory"`.
5.  **Don't filter by `severity>=ERROR`:** JAX collective timeouts are often
    logged at `INFO` ([[gotchas-cloud-logging]] Gotcha 9).
6.  **Route by layer:**
    -   Application: [[workload-startup-failures]], [[inspect-hlo-dumps]],
        [[deadline-exceeded]], [[megascale-hang]].
    -   Orchestration (evictions, memory pressure): [[stopiteration-disk-full]],
        [[missing-exit-codes-eviction]].
    -   Provisioning (quota, scheduler): [[project-quotas]].
    -   GCE / hardware (dynamic slicing, links): [[slice-failure]].
7.  **Several runs failing differently** (`workload-name-1`, `-2`, `-3`): build
    a [[failure-tables]] table to separate repeating patterns from one-offs.
8.  **Hardware vs software** ([[debugging-mindset]] outlier and success-baseline
    tests): map `Workload Logs -> GKE Instance -> Host Name` with Find Gce
    Instance Mapping. Hardware if errors repeat on the same hosts or stop when
    they are excluded ([[multi-workload-triage]], Mxla Startup Hang 16 Slices);
    software if random, cross-machine, or code-deterministic.

## Mitigation

For hardware: [[mitigate-tpu-hardware-fault]] or [[mitigate-and-escalate]]. Find
the node pool from the GKE instance name, taint per [[node-isolation-methodology]],
and file a support case/bug with logs, GKE instance name, error, host name, and
repeated failures on the same hardware.

## Connections

-   **Workflows**:
    -   [[exit-code-investigation]]
-   **Playbooks**:
    -   [[investigation-report-template]]
    -   [[layered-debugging-framework]]
    -   [[tpu-sdk-debugging-overview]]
-   **Symptoms**:
    -   [[deadline-exceeded]]
    -   [[session-failing]]
    -   [[slice-failure]]
-   **Concepts**:
    -   [[workload-monitoring]]
-   **Tools**:
    -   [[cloud-logging]]
    -   [[orbax]]
    -   [[python-glue-runner]]

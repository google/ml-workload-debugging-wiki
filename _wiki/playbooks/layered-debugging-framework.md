---
title: Layered Debugging Framework (Triage across the Stack)
type: playbook
internal_only: false
tags: [tpu, training, methodology]
created: 2026-06-13
updated: 2026-10-20
---


## Overview

A symptom in a higher layer is often caused by a lower layer. Before treating a
Python traceback as a code bug or a pod restart as random, check the layers
below it.

## Per-Layer Log Indicators (Triage)

-   **Application (JAX / MaxText / Orbax):** `Wait timeout on sflag`;
    `Training stopped: You may have run out of training data` (`StopIteration`);
    Python tracebacks; slow steps or checkpoint hangs. Read container logs
    (`resource.type="k8s_container"`) and Orbax metadata.
-   **Workload orchestration (JobSet / Kueue / kubelet):** pod `Evicted` or
    `Failed`; `exit_codes.txt` has far fewer entries than pods; exit `134`
    (abort) or `143` (SIGTERM). Query
    `logName:"/logs/events" AND resource.type="k8s_pod"` for `reason: "Evicted"`
    or `reason: "OOMKilling"`.
-   **Provisioning (node pools, Kueue):** workload stuck in `QuotaReserved` or
    `Requeued`; `DiskPressure` / `MemoryPressure` taints. Check
    `kubectl get workload`, `kubectl describe node`, Kueue events.
-   **GCE / hardware:** TPU driver init failure (`failed call to cuInit` is a
    red herring; look for real link faults or driver crashes); OCS link down or
    dynamic slicing failures; host maintenance preemption. Check the `telemetry`
    VM-to-host mapping, `gcloud compute operations list`, and the GCE serial
    console.

## Failure Analysis: Example Scenarios

### Scenario A: False "data exhaustion" hang

Worker 101 logs `StopIteration` and JAX shutdown; the other 1022 workers hang at
step 40 in a collective. Pod events show Worker 101 was **Evicted** for
`ephemeral-storage` pressure: the full local disk made the data loader fail to
write cache files, which surfaced as a false `StopIteration`, and the
coordinated shutdown deadlocked the rest. Classification: **Resource / SW**
(disk eviction), not training success.

### Scenario B: Missing exit codes

`exit_codes.txt` shows 835 pods exited `143`; 189 pods have no exit code. Pod
events show kubelet **Evicted** them for memory/disk pressure, bypassing the
exit wrapper. Classification: **Resource / SW** (eviction), not a software hang.

### Scenario C: Startup hang

McJax hangs on init (`GRPC_ERROR START_SESSION failed`). `telemetry` on the
outlier VM shows a TPU driver crash on the physical host (`host-xxxx`).
Classification: **Hardware** (TPU link/driver error).

## Connections

-   **Actions**:
    -   [[run-cross-layer-analysis]]
-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[debugging-mindset]]
-   **Concepts**:
    -   [[exit-codes]]
    -   [[goodput]]
-   **Entities**:
    -   [[maxtext]]
-   **Tools**:
    -   [[cloud-logging]]

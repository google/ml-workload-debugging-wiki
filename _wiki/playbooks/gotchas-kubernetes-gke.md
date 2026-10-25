---
title: Kubernetes & GKE Gotchas
type: playbook
internal_only: false
tags:
- tpu
- training
created: 2026-06-13
updated: 2026-10-03
---


## Kubernetes & GKE Gotchas & Tips

These gotchas focus on issues related to Kubernetes scheduling, GKE
infrastructure, node state, and `kubectl` usage.

### 1. Deleted Pods

If pods are deleted after failure, `kubectl` commands will fail. Always use
Cloud Logging for post-mortem analysis.

### 2. Permission / Scope Errors

Messages like `"Provided scope(s) are not authorized"` indicate IAM issues
accessing resources like GCS buckets, causing silent failures.

### 3. Check K8s Events for Failing Pods

Application logs might only show a generic `Aborted` or `SIGTERM` message.
Always check Kubernetes events for the failing pod (e.g., look for
`reason: Evicted` or `reason: Killing`) to find system-level causes like
resource exhaustion (ephemeral storage, OOM).

### 4. Connection Timeout with `kubectl` and `--dns-endpoint`

If `kubectl` commands fail with a connection timeout after running
`gcloud container clusters get-credentials`, the API server might be private or
not directly reachable via IP. Try adding the `--dns-endpoint` flag to the
`get-credentials` command to use the DNS-based endpoint instead.

### 5. GKE/GCE Identifier Mismatches

When invoking actions that interact with GCP VM-level resources (like
`report-host-as-faulty` or `taint-node`), you must pass the GCE VM instance name
(format: `gke-tpu-...`). Do not pass the pod hostname (format:
`workload-worker-...`). Always run node mapping lookups first to resolve the
correct GCE instance name. See [[gke-identifier-taxonomy]].

### 6. Kueue Jobs Stuck After Preemption

Kueue jobs that are readmitted after preemption can get stuck in the queue and
require manual recreation/intervention to be admitted again (marked FIXED
2026-04-14; check whether your Kueue version has the fix before using the kick
below).

*   **Symptom**: Workloads remain in `QuotaReserved` or `Requeued` state without
    being admitted, even if capacity is available.
*   **Mitigation**: Deactivate and reactivate the workload to force Kueue to
    re-process it:

    1.  Run `kubectl edit workload/<workload-name>`.
    2.  Set `spec.active=false`.
    3.  Wait a few seconds, then set `spec.active=true`.

*   **Caution on TPU slices**: the kick evicts and immediately re-admits the
    workload. Confirm the old pods are gone first.

### 7. Missing Exit Codes due to Aggressive Eviction

If your exit code analysis reveals that a significant number of pods did not log
exit status (e.g. the number of exit codes in exit_codes.txt is much smaller
than the total pods in the slice), do not default to 'Unknown' failure. When
Kubelet evicts a pod due to extreme ephemeral disk pressure, it terminates the
container instantly, preventing the wrapper script from printing the exit
status. Treat missing exit codes as a high-probability signature of eviction or
node failure. Check Kubernetes events for 'Evicted' for the specific pod IDs
that did not log exit codes. See [[missing-exit-codes-eviction]].

### 8. GKE Metadata Server Crashloop on Cordoned Nodes

When nodes are cordoned (tainted) and later reused, they can occasionally suffer
from image corruption. This corruption can cause the `gke-metadata-server`
container to crashloop, blocking the VM's access to GCS credentials and causing
subsequent job restarts to fail with permission errors (e.g.,
`storage.objects.list` denied). If you see permission errors on restart, check
the status of system pods. See [[gke-metadata-server-crashloop]].

### 9. Temporal Priority of Software Crashes over Evictions (Core Dumps)

If logs show a software crash (e.g., a compiler assertion, C++ check failure, or
Python `AttributeError`/`ValueError`) followed a few minutes later by a
Kubernetes `Evicted` event (due to `ephemeral-storage` limits), **do not default
to classifying the failure as an Eviction**.

*   **The Core Dump Trap**: Software crashes often trigger the creation of
    massive core dumps (especially at scale). If the container's ephemeral
    storage is close to its limit, writing these core dumps can fill the disk
    and trigger a post-crash GKE eviction.
*   **Temporal Test**: Compare the timestamps. If the software abort/traceback
    timestamp (T1) is earlier than the Kubernetes eviction event timestamp (T2),
    the software crash is the root cause. Classify the failure as `SW` or
    `Config` rather than `Resource` (Eviction).

### 11. Force-Deleting a TPU Pod on a Live Node Does Not Free the TPU

`kubectl delete pod --grace-period=0 --force` removes only the Pod object. On a
Ready node the container can keep running and hold `/dev/vfio`; the scheduler
sees the node as free. Use it only when the node is really gone (NotReady /
unreachable, VM recreated). For TPU pods stuck Terminating on Ready nodes
(`FailedKillPod`, KillContainer `DeadlineExceeded`), escalate with
`--grace-period=1`, then reset the VM.

## Connections

-   **Playbooks**:

    -   [[debugging-gotchas]]

-   **Concepts**:

    -   [[gke-architecture]]
    -   [[gke-identifier-taxonomy]]

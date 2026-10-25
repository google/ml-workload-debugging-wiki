---
title: Workload Stuck in Queue
type: symptom
internal_only: false
tags:
- tpu
- training
- scheduling
created: 2026-06-16
updated: 2026-06-16
---


## Description

This symptom occurs when a TPU workload is successfully submitted to the cluster
but remains in a non-running state (e.g., `Pending`, `Requeued`, `Suspended`, or
stuck in `QuotaReserved`) for an extended period, and no application logs are
generated.

## Verification Steps & Log Signatures

### 1. Workload Status is "Requeued" or "Suspended"

When checking the workload status, the `Status` column shows `Requeued` or
`Suspended`, and the job does not transition to `Admitted` or `Running`.

**Command:**

```bash
kubectl get workloads --ignore-not-found -o=custom-columns="Jobset Name:.metadata.ownerReferences[0].name,Created Time:.metadata.creationTimestamp,PRIORITY:.spec.podSets[0].template.spec.priorityClassName,TPU VMs Needed:.spec.podSets[0].count,TPU VMs Running/Ran:.status.admission.podSetAssignments[-1].count,TPU VMs Done:.status.reclaimablePods[0].count,Status:.status.conditions[-1].type,Status Message:.status.conditions[-1].message,Status Time:.status.conditions[-1].lastTransitionTime"
```

**Output Example:** `Jobset Name Created Time PRIORITY TPU VMs Needed TPU VMs
Running/Ran TPU VMs Done Status Status Message Status Time my-stuck-job-0
2026-03-19T17:14:07Z medium 16 16 <none> Requeued The workload backoff was
admitted 2026-03-19T18:47:55Z` *Note: Even though "TPU VMs Running/Ran" shows
16, the Status is "Requeued", meaning it was kicked back to the queue.*

### 2. No Slice Object Created

For Superslicing workloads, Kueue must create a `Slice` object to configure ICI
links. If the workload is admitted but no `Slice` is created, the pods will
never start.

**Check command:**

```bash
kubectl get slice
```

If you do not see a slice corresponding to your `JobSet` name, the slice manager
has failed to provision the interconnects.

### 3. Flavor Assignment Failure (Topology Constraint)

The workload remains in `QuotaReserved` status but cannot be admitted because
Kueue cannot satisfy the topology requirements.

**Log/Status Message Signature:** `couldn't assign flavors to pod set slice-job:
Flavor "<flavor-name>" supports only TopologyAwareScheduling` This indicates
that the requested topology does not match the available physical node layout or
that the flavor configuration is restricted.

### 4. Head-of-Line Blocking in ClusterQueue

Your workload is healthy and has quota, but it is blocked behind a stuck
workload at the front of the queue.

**Diagnosis:** To inspect queue state, Head-of-Line blocking, and admission
check failures, always dispatch the formal wiki actions
`[[check-localqueue-status]]` and `[[get-cluster-queues]]` via the glue runner
(`python3 _wiki/tools/scripts/python_glue_runner.py ...`). Do not rely on raw
unformatted commands. When inspecting `admissionChecks`, check for Slice
Controller errors where the generated slice name exceeds the 49-character limit
(`must be no more than 49 characters`), causing the stuck slice to hold quota
and block subsequent workloads in the queue.

Compare the total capacity used in the ClusterQueue vs the requested capacity.
If there is free capacity but your job is not admitted, check the workloads
created before yours.

### 5. GKE Warden Rejection (Expected for Non-Admitted Jobs)

You see events indicating that GKE Warden rejected Job creation due to a missing
topology label.

**Log/Event Message Signature:** `job "..." creation failed with error:
admission webhook "warden-validating.common-webhooks.networking.gke.io" denied
the request: GKE Warden rejected the request because it violates one or more
constraints. Violations details: {"[denied by
tpu-accelerator-topology-constraints]": ["Missing nodeSelector/nodeAffinity
label cloud.google.com/gke-tpu-topology..."]}`

**Interpretation:**

> [!IMPORTANT] This error is **expected** for dynamic slicing workloads that
> have not yet been admitted by Kueue.

In dynamic slicing, the user does not specify the `gke-tpu-topology` label in
the JobSet nodeSelector. The GKE Slice Controller dynamically injects this label
*after* Kueue admits the workload. If GKE Warden rejects the Job creation with
this error, it indicates that the JobSet is attempting to create pods before
Kueue has admitted it (often due to startup race conditions or when the workload
is suspended/requeued).

**Do not attempt to fix this by manually adding the topology label.** Instead,
investigate why Kueue has not admitted the workload (e.g. check queue capacity,
blockage, or priority).

## Root Causes

1.  **Kueue Stuck after Preemption**: Jobs readmitted after preemption get stuck
    in a suspended/requeued state.
2.  **Incompatible XPK Manifests**: Older XPK versions can generate manifests
    that fail Kueue validation.
3.  **Lack of Physical Capacity (Stockout)**: The cluster does not have enough
    healthy nodes/cubes to satisfy the request shape.

4.  **Incorrect Priority Class**: Workload is queued behind many higher-priority
    jobs.

5.  **Flavor Mismatch**: Requesting a topology that is not supported by the
    active ClusterQueue flavors.

6.  **Workload Not Admitted**: The GKE Warden topology error is a symptom of the
    job waiting for Kueue admission, not a configuration error.

## Connections

-   **Actions**:
    -   [[check-localqueue-status]]
    -   [[get-cluster-queues]]
-   **Playbooks**:
    -   [[debugging-submission-failures]]
    -   [[launch-workload-dashboard]]
-   **Symptoms**:
    -   [[forever-suspended-jobset]]
-   **Concepts**:
    -   [[kueue-tq-scheduling]]

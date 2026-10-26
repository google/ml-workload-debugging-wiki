---
title: Debugging Workload Submission Failures
type: playbook
internal_only: false
tags:
- tpu
- training
- scheduling
- triage
created: 2026-06-16
updated: 2026-10-20
---


## Overview

Use this playbook when a job submitted to a GKE TPU cluster fails to start,
emits no logs, or stays stuck. (Note: Kueue and JobSet are optional on GKE
clusters; if your cluster schedules Pods or Jobs directly without Kueue, skip
the Kueue admission steps in Steps 1 and 3 and go directly to Step 2a.)

## Step-by-Step Triage

### Step 1: Verify Workload Status & Slicing Mode

1.  Check workload admission (`Suspended` / `Requeued` vs `Admitted`):

    ```bash
    kubectl get workloads --ignore-not-found -o=custom-columns="Jobset Name:.metadata.ownerReferences[0].name,Created Time:.metadata.creationTimestamp,PRIORITY:.spec.podSets[0].template.spec.priorityClassName,TPU VMs Needed:.spec.podSets[0].count,TPU VMs Running/Ran:.status.admission.podSetAssignments[-1].count,TPU VMs Done:.status.reclaimablePods[0].count,Status:.status.conditions[-1].type,Status Message:.status.conditions[-1].message,Status Time:.status.conditions[-1].lastTransitionTime"
    ```

2.  Check **Dynamic Slicing** (`kubectl get jobset <workload-name> -o
    jsonpath='{.metadata.annotations.cloud\.google\.com/gke-tpu-slice-topology}'`):
    **present** (`4x4x4`, `4x8x8`) -> **Step 2**; **absent** -> **Step 2a**.

3.  **GKE Warden**: `GKE Warden rejected the request because it violates one or
    more constraints` (missing `cloud.google.com/gke-tpu-topology`) is
    **expected before Kueue admission** on dynamic slicing; go to **Step 3**.

### Step 2: Check Slice Creation (Dynamic Slicing Only)

If `Admitted` with no pods running, check `kubectl get slice` and
`kubectl describe slice <slice-name>` for `ACTIVE`. If no slice exists, inspect
admission checks (`kubectl get workloads
-o=custom-columns="NAME:.metadata.name,CHECKS:.status.admissionChecks[*].state,MESSAGE:.status.admissionChecks[*].message"`)
for `metadata.name: Invalid value: ... must be no more than 49 characters`.

### Step 2a: Check Pod Status (Static Slicing or Active Dynamic Slices)

Run `kubectl get pods -l jobset.sigs.k8s.io/jobset-name=<workload-name>` and
`kubectl describe pod <pod-name>`.

-   **Case A — 8k (`8x32x32` / `8x1024`, 2,048 pods) `tpu-device-plugin`
    `Allocate` rejections (`UnexpectedAdmissionError`; see Rec D)**:
    -   **Sig A (`tpu-device-plugin` OOMKilled mid-`Allocate`)**:
        `UnexpectedAdmissionError: Pod was rejected: Allocate failed due to rpc
        error: code = Unavailable desc = error reading from server: EOF, which
        is unexpected` or `... read unix @->/device-plugin/tpu-<id>.sock: read:
        connection reset by peer, which is unexpected`
    -   **Sig B (`Allocate` fallback `Pods("").List` timeout)**:
        `UnexpectedAdmissionError: Pod was rejected: Allocate failed due to rpc
        error: code = Unknown desc = the server was unable to return a response
        in the time allotted, but may still be processing the request (get
        pods), which is unexpected` or `... Get
        "https://.../api/v1/pods?fieldSelector=spec.nodeName%3D...": http2:
        client connection lost, which is unexpected`
    -   **Sig C (`vbar-control-agent` OOMKilled)**: `vbar-control-agent` in
        `kube-system/tpu-device-plugin-*` `OOMKilled` (`exitCode: 137`) at
        `150Mi`, breaking sessions or `unix:///var/run/tpu-plugin/vbar.sock`.

### Step 3: Check Queue Capacity and Blockage

If stuck in `Requeued` or `Suspended`, check if the queue is full or blocked by
dispatching `[[check-localqueue-status]]` and `[[get-cluster-queues]]`:

1.  **LocalQueue & ClusterQueue Usage**: `kubectl describe lq multislice-queue`
    (`[[check-localqueue-status]]`) and `kubectl describe cq cluster-queue`
    (`[[get-cluster-queues]]`)

2.  **Physical Node Pool Readiness**:
    `kubectl get nodes -l cloud.google.com/gke-tpu-accelerator`

3.  **Head-of-Line Blocking**:

    ```bash
    kubectl get workloads --ignore-not-found -o=custom-columns="NAME:.metadata.ownerReferences[0].name,TIME:.metadata.creationTimestamp,PRIORITY:.spec.podSets[0].template.spec.priorityClassName,STATUS:.status.conditions[-1].type,MSG:.status.conditions[-1].message" | grep -v Finished | awk '
    NR==1 {print "SORT_KEY\t" $0; next}
    {p=2; if ($3=="Very-High") p=4; else if ($3=="high") p=3; else if ($3=="medium") p=2; else if ($3=="low") p=1; print p "_" $2 "\t" $0}' | sort -t'_' -k1,1r -k2,2 | cut -f2-
    ```

--------------------------------------------------------------------------------

## Mitigations and Recommendations

> [!IMPORTANT] Do not perform automated mutations in production without explicit
> confirmation.

### Recommendation A: Reactivate Stuck Preempted Workloads (Kick, last resort)

Only after confirming 0 pods remain (none `Terminating`, no `FailedKillPod`),
running `[[fixing-stuck-kueue-workload]]`, and checking Kueue version
([[gotchas-kubernetes-gke]] gotcha 6, [[kueue-stuck-termination-recovery]]):

```bash
kubectl patch workload <workload-name> --type='merge' -p '{"spec":{"active":false}}' && sleep 5 && kubectl patch workload <workload-name> --type='merge' -p '{"spec":{"active":true}}'
```

### Recommendation B: Verify Priority Class Configuration

Never `kubectl replace --force` a JobSet (recreates while old pods hold TPUs).
Use [[change-workload-priority]]: in-place WorkloadPriorityClass change, or
foreground delete, wait for 0 pods, then resubmit.

### Recommendation C: Check for Configuration Validation Errors

If Kueue events show `Flavor ... supports only TopologyAwareScheduling`,
re-verify the YAML against [[jobset-yaml-structure]].

### Recommendation D: Mitigate 8k-Scale `UnexpectedAdmissionError` (Per-Index Retry & `tpu-device-plugin` `Allocate` OOM / Pod List Timeout)

-   **Root Cause**: On GKE 1.36 (`1.36.10-gke.0`), `ds/tpu-device-plugin` has
    zero-headroom limits (`tpu-device-plugin` `80Mi` vs p50 `68.1`, p95 `76.1`,
    p99 `78.4 MiB`; `vbar-control-agent` `150Mi` vs p99 `153-155 MiB`;
    `sidecar-log-collector` `20Mi`). At 8k (2,048 pods): (a) `:8431` Megascale
    scraping (`megascale.dcn_transfer_latencies`, `mxla_compute_latencies`,
    `collective_end_to_end_latencies` every 10-60s) pushes `tpu-device-plugin`
    p99 to `80.5 MiB` (OOMKill exit 137); (b) when `podInformer` lags during a
    2,048-pod burst, `Allocate` falls back to unpaginated `Pods("").List`
    (`GET /api/v1/pods?fieldSelector=spec.nodeName=<node>`, no
    `ResourceVersion: "0"` or retry)—when `kube-apiserver` watch cache lags >3s,
    it scans `pods watch cache` in etcd, timing out `(get pods)` or OOM-killing
    `tpu-device-plugin` (`EOF` / `connection reset by peer`).
-   **Workload Mitigation (`backoffLimitPerIndex: 3` + `maxFailedIndexes: 0` or
    `backoffLimit: 16` + `podFailurePolicy`)**: `UnexpectedAdmissionError` means
    kubelet rejected the pod before any container ran (no exit code), bypassing
    `podFailurePolicy` `onExitCodes` rules into the Job's backoff. With
    `backoffLimit: 0`, one rejected pod fails the whole Job and burns
    `maxRestarts: 1`: on a 2,048-pod slice all pods are deleted and recreated,
    and every new pod repeats the lookup, which can re-overload the apiserver
    and cascade. In the slice Job template, remove `backoffLimit: 0` and retry
    per index (needs `completionMode: Indexed`, the JobSet default, and pod
    `restartPolicy: Never`; with `backoffLimitPerIndex` set, an unset
    `backoffLimit` defaults to `2147483647`, or set `backoffLimit: 16` when not
    using per-index backoff). Keep the existing `podFailurePolicy` (`FailJob` on
    non-zero exit codes) and JobSet `failurePolicy`:

```yaml
replicatedJobs:
- name: slice
  template:
    spec:
      completionMode: Indexed   # JobSet default
      backoffLimitPerIndex: 3   # retry only the failed index
      maxFailedIndexes: 0       # an index out of retries fails the Job -> JobSet failurePolicy
      # no backoffLimit: 0 here
      podFailurePolicy: ...     # unchanged
      template:
        spec:
          restartPolicy: Never
```

-   **Effect**: only the rejected index is recreated, on the same
    Kueue/TAS-pinned host, after the Job pod-failure backoff (10s, 20s, 40s,
    ...). The other 2,047 pods keep waiting in `jax.distributed.initialize`
    (default timeout 300s; MaxText
    `jax_distributed_initialization_timeout: 300`). An index that runs out of
    retries (e.g. a host with a bad device) fails the Job and the JobSet
    `failurePolicy` restarts it as before.
-   **Why 3**: retries must finish inside the 300s JAX init window. Under
    apiserver overload each attempt can take ~60s plus backoff (10+20+40 = 70s).
    If peers time out first they exit 1, which matches a `FailJob`
    `podFailurePolicy` rule and, with a JobSet rule
    `onJobFailureReasons: [PodFailurePolicy]` -> `FailJobSet`, fails the JobSet
    permanently. For more retries, also raise
    `jax_distributed_initialization_timeout` (e.g. 600); then 4-5 retries are
    fine.
-   **Mid-run caveat**: a single pod killed mid-run with a signal exit (e.g.
    137, ignored by `podFailurePolicy`) is now replaced on its own instead of
    failing the Job at once; its peers then fail and their exit codes decide
    (exit 1 -> `FailJob` -> `FailJobSet`). The main benefit is at startup.
-   **Cluster Mitigation (K8s 1.36 `pods/resize` + `MutatingAdmissionPolicy`,
    admin)**: Never patch `ds/tpu-device-plugin` (`kube-addon-manager` reverts
    in ~60s; rolling kills sessions). Patch `kube-system/tpu-device-plugin-*`
    via `pods/resize` (`Burstable`, `NotRequired`, updates `memory.max` in
    `<2s`, 0 restarts) + `MutatingAdmissionPolicy`
    `device-plugin resize policy`. On GKE 1.36.9-gke.0+ (where `acs-agent` was
    relocated to `ai-telemetry-collector`), omit `acs-agent` from the patch
    because Kubernetes in-place resize rejects patches referencing absent
    containers (include `acs-agent` only on pre-1.36.9 pods):

```bash
kubectl -n kube-system patch pod <tpu-device-plugin-pod> --subresource=resize -p '{"spec":{"containers":[{"name":"tpu-device-plugin","resources":{"requests":{"cpu":"80m","memory":"256Mi"},"limits":{"memory":"512Mi"}}},{"name":"sidecar-log-collector","resources":{"requests":{"cpu":"20m","memory":"64Mi"},"limits":{"memory":"128Mi"}}},{"name":"vbar-control-agent","resources":{"requests":{"cpu":"10m","memory":"512Mi"},"limits":{"memory":"1Gi"}}}]}}'
```

## Connections

-   **Workflows**: [[fixing-stuck-kueue-workload]]
-   **Actions**: [[check-localqueue-status]],
    [[get-cluster-queues]]
-   **Symptoms**: [[argument-list-too-long]], [[forever-suspended-jobset]],
    [[stuck-in-queue]]
-   **Concepts**: [[kueue-tq-scheduling]]

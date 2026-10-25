---
title: Kueue Stuck Termination Recovery
type: playbook
internal_only: false
tags:
- triage
created: 2026-07-27
updated: 2026-10-03
---


# Kueue Stuck Termination Recovery

## Overview

This playbook addresses a scenario in GKE TPU nodes (such as v7x generations)
where pods can get stuck in a `Terminating` state. This often occurs when a TPU
node enters an "RIP repair" state without a proper termination signal being sent
to the pod, blocking JobSets from recovering.

## Trigger Conditions

-   Pods are observed stuck in a `Terminating` state for an extended period.
-   The pods belong to a Kueue-managed workload (e.g., JobSet on GKE TPU nodes).
-   System alerts or logs indicate node non-responsiveness (e.g., "RIP repair").

## Diagnostic Steps

1.  Execute `[[check-pod-status]]` during troubleshooting to inspect pod phases
    and identify pods stuck in `Terminating` states.
2.  Get the Kueue version (`kubectl -n kueue-system get deploy
    kueue-controller-manager -o jsonpath='{..image}'`). The fix below needs
    `>= 0.15.0`; older Kueue: one-time force delete under the warning below.
3.  Check that the `FailureRecoveryPolicy` feature gate (alpha, off by default)
    is enabled on the Kueue controller manager.
4.  Check the JobSet's pod templates for the opt-in annotation. Its name changed
    in Kueue `0.15.6`, `0.16.3` and `0.17.0` (KEP-6757):
    -   `>= 0.15.6` (0.15.x), `>= 0.16.3` (0.16.x), `>= 0.17.0`:
        `kueue.x-k8s.io/safe-to-forcefully-delete: "true"`
    -   `0.15.0`-`0.15.5`, `0.16.0`-`0.16.2`:
        `kueue.x-k8s.io/safe-to-forcefully-terminate: "true"`
5.  The controller only acts on annotated pods that are Terminating on a node
    tainted `node.kubernetes.io/unreachable`, 1 min after their grace period; it
    marks them `Failed`, deletes them without grace and records event reason
    `KueueForcefullyTerminated`. Pods on Ready or merely NotReady nodes are
    ignored by design.
6.  Attribution: the culprit is the node that went into repair without
    terminating its pods (`culprit_host` = that node, `culprit_pod` = the stuck
    pod). Kueue is not the culprit; missing FailureRecoveryPolicy / annotation
    is the reason recovery stalled.

## Resolution Workflow

Real fix in a McJax large-scale test (pods stuck Terminating 68 min on a node in
RIP, scheduler `0/2717 nodes are available ... Preemption is not helpful`):
FailureRecoveryPolicy plus the annotation, after which Kueue terminated the pod
and the JobSet recovered.

1.  Cluster admin: execute `[[enable-kueue-failure-recovery]]` if the feature
    gate is missing.
2.  Workload owner: execute `[[annotate-pod-failure-recovery]]` on the affected
    workload, passing the cluster's detected Kueue version so the correct string
    is applied.
3.  Observe if Kueue automatically reaps the stuck pods and recovers the JobSet
    (`kubectl get events -A --field-selector reason=KueueForcefullyTerminated`).
    Do not clear finalizers or deactivate / reactivate the Workload as a first
    step.

> [!WARNING] A one-time forceful deletion
> (`kubectl delete pod <name> --grace-period=0 --force`) is acceptable **only
> when the pod's node is really gone** (NotReady / unreachable, RIP repair, VM
> being recreated or reset). On a **Ready** node it removes only the Pod object
> while the container can keep the TPU (`/dev/vfio`) open; the scheduler then
> sees the node as free and a new slice can be programmed under the live
> process. For pods stuck Terminating on Ready nodes, escalate with
> `--grace-period=1`, cordon the node, and reset the VM if the container is
> still stuck.

## Connections

-   **Workflows**:
    -   [[fixing-stuck-kueue-workload]]
-   **Playbooks**:
    -   [[general-ml-debugging]]

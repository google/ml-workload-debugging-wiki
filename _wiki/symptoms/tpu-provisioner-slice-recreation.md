---
title: "TPU Provisioner Slice Recreation (JobSet Suspended at readyCount N-1/N)"
description: "Training stalls cluster-wide, then the JobSet is suspended and resumed minutes later: `tpu-provisioner-controller-manager` logs `recreation condition matched: FAILED` for one slice, suspends the JobSet (`readyCount: 7, expectedCount: 8`), deletes the slice's pods, provisions a replacement and unsuspends on `All Slices are Ready`. The cause is the failed slice's nodes, not the workload."
type: symptom
internal_only: false
tags: [tpu, gke, scheduling, hardware_fault, triage]
created: 2026-10-03
updated: 2026-10-03
---

# TPU Provisioner Slice Recreation

## Overview

The TPU provisioner (namespace `tpu-provisioner-system`, service account
`tpu-provisioner-controller-manager`) recreates a slice it considers failed and
suspends the whole JobSet while it does so. Every worker stops at the same step
(often inside a barrier); the stall is a victim of the slice failure. Single
observation (one incident review card, 2026-10-03); strings below are as quoted by the
RCA agent from the controller logs, not verified against the controller code.

## Log Signatures

```
tpu-provisioner-controller-manager ... slice <jobset>-job-<k> ... recreation condition matched: FAILED
JobSet suspended ... readyCount: 7, expectedCount: 8
... All Slices are Ready      (JobSet unsuspended)
```

Sequence seen: stall at step S on all slices -> ~3 min later
`recreation condition matched: FAILED` -> suspend + rolling delete of that
slice's pods -> replacement slice -> unsuspend ~3 min later -> containers
restore the last checkpoint and resume. Training-side errors in between (barrier
/ collective timeouts, SIGTERM) are consequences.

## Verification Steps

1.  Controller logs for the workload: `resource.type="k8s_container"
    resource.labels.namespace_name="tpu-provisioner-system" "<jobset_name>"`
    over the stall window ([[query-workload-logs]] style filter, scoped to the
    JobSet name).
2.  JobSet state changes: [[get-jobset-events]], [[verify-jobset-status]]: look
    for suspend / resume and pod deletions of one replicated job only.
3.  Which slice: the `job-<k>` in the provisioner line; map its nodes with
    [[get-slices]] and the pod -> node binding before deletion.
4.  Why it failed (the real root cause): node events of that slice only,
    [[query-vm-termination-events]] with a scoped node filter,
    [[query-node-health-events]], [[list-unhealthy-gce-instances]].

## Playbooks & Root Causes

-   Report the culprit as the recreated slice's failing node(s) with the quoted
    node event, or `INCONCLUSIVE` if none is visible; never the workload code or
    worker 0 ([[rca-invariants]] rules 3, 4, 6).
-   Clean resume from checkpoint = recovered disruption: no workload action.
    Repeated recreations on the same nodes = hardware candidate for node
    replacement ([[node-isolation-methodology]]).

## Connections

-   **Actions**:
    -   [[get-jobset-events]]
    -   [[verify-jobset-status]]
    -   [[get-slices]]
    -   [[query-vm-termination-events]]
    -   [[query-node-health-events]]
    -   [[list-unhealthy-gce-instances]]
-   **Playbooks**:
    -   [[rca-invariants]]
    -   [[node-isolation-methodology]]
-   **Symptoms**:
    -   [[slice-failure]]

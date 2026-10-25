---
title: "Kueue TAS: \"allows to fit only N out of M slice(s)\" / Job Waits While Cubes Look Free"
description: "A TPU job stays pending with the Topology-Aware Scheduling message \"allows to fit only N out of M slice(s) ... excluded: affinity A, resource R, taint T\" (or \"insufficient unused quota ... more needed\") although the cluster reports enough usable cubes. How to decode the excluded buckets into cubes, compute the capacity for this job, and spot self-excluded pools and cubes held by other workloads."
type: symptom
internal_only: false
tags: [tpu, gke, kueue, scheduling, capacity, triage]
created: 2026-10-06
updated: 2026-10-06
---

# Kueue TAS: "allows to fit only N out of M slice(s)" / Job Waits While Cubes Look Free

## Overview

A large TPU job (for example 8192 chips = 128 cubes of 4x4x4) waits in Kueue
while the cluster's health count says there is enough capacity. The cluster
count is the wrong number: what matters is the capacity **for this job**, after
its own node affinity, other workloads and taints. Investigations that start
from the cluster count conclude "not enough healthy cubes" and are wrong.

## Log Signatures

Workload condition (`kubectl get workload -o yaml`, or Cloud Logging `events`)
with Topology-Aware Scheduling:

```text
couldn't assign flavors to pod set ...: topology "<topology>" allows to fit
only 126 out of 128 slice(s) ... Total nodes 2293; excluded: affinity 64,
resource google.com/tpu 32, taint <custom-node-taint> 181
```

Earlier, while lower-priority borrowers still hold quota, the same workload can
show only the quota message, which hides the topology result:

```text
couldn't assign flavors to pod set ...: insufficient unused quota in cohort for
google.com/tpu in flavor <flavor>, 4352 more needed
```

## Verification Steps

All steps are read-only.

1.  **Get the message history, not just the latest condition.** `kubectl` events
    expire after about 1 h; read `events` from Cloud Logging for the workload in
    ascending order ([[gotchas-cloud-logging]]).
2.  **Decode each `excluded:` bucket into cubes** (TPU7x: 16 nodes per cube):

    | Bucket | Meaning | Check |
    |---|---|---|
    | `affinity A` | Nodes rejected by **this job's own** nodeSelector / nodeAffinity (NotIn node pools, partition-state selector) | Read the job's affinity; A/16 should equal the pools it excludes |
    | `resource google.com/tpu R` | Nodes whose TPUs are already used by **other pods** | Find which workloads hold them in the same time window |
    | `taint ... T` | Tainted nodes (broken or partial cubes, maintenance) | Check node taints |
3.  **Read the exact run's spec.** Use the Workload's `podSets` affinity. If the
    JobSet was already deleted, recover it from GKE audit logs: the
    jobset-controller's `io.k8s.batch.v1.jobs.create` request bodies carry the
    pod template (`jobsets.create` may be absent). Never infer one run's
    affinity from a sibling run: lists often change between runs.
4.  **Compute the capacity for this job:** `free = usable cubes allowed by its
    state selector − pools it excludes − cubes held by other workloads`, and
    compare with the cubes it needs. A job that pins `4x4x4-state=HEALTHY` also
    loses `DEGRADED` cubes.
5.  **Name the holders.** For cubes in the `resource` bucket, list the pods on
    those nodes at that time (audit `pods.delete` / binding entries by node, or
    pod listings if still live). Do not explain a past state with a live
    snapshot.

## Playbooks & Root Causes

-   **The job excluded capacity itself.** A user NotIn list on healthy or
    suspect node pools removes those cubes from placement, but Kueue's quota
    math still counts their chips. The job then waits on topology although the
    queue looks free. Tell the user how many cubes each excluded pool costs.
-   **Other workloads hold whole cubes** (another queue, a custom-tainted pool, a
    job still terminating). Name them and their queue; preempting them helps
    only if the job then fits.
-   **Known-bad cubes avoided by each user.** If users keep excluding the same
    pools because they fail a class of workload (for example DCN), make the
    exclusion visible to the scheduler instead ([[known-bad-cubes]]).
-   **Quota message masks topology.** While borrowers run, Kueue may report only
    `insufficient unused quota`; preemption is simulated and skipped if the job
    would still not fit. Check the topology result before concluding that
    preemption is broken.
-   **Genuine shortage.** If free < needed with no self-exclusion and no
    holders, follow [[stuck-in-queue]].

## Connections

-   **Symptoms**:
    -   [[stuck-in-queue]]
-   **Concepts**:
    -   [[kueue-tq-scheduling]]
    -   [[known-bad-cubes]]
-   **Playbooks**:
    -   [[gotchas-cloud-logging]]
    -   [[multi-workload-triage]]
    -   [[run-forensics-report-template]]
-   **Workflows**:
    -   [[workload-run-forensics]]

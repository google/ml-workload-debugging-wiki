---
title: "Known-Bad Cubes (Healthy in GKE, Failing a Class of Workload)"
description: "Cubes whose GKE partition state is HEALTHY but that repeatedly fail one class of workload (for example multislice/DCN jobs) while still working for others (single-slice ICI). How to recognize, record and expose them to the scheduler instead of relying on per-user NotIn node-pool lists."
type: concept
internal_only: false
tags: [tpu, gke, kueue, scheduling, capacity, hardware_fault]
created: 2026-10-06
updated: 2026-10-06
---

# Known-Bad Cubes (Healthy in GKE, Failing a Class of Workload)

## Overview

A **known-bad cube** reports `HEALTHY` in its partition labels, but jobs of one
class keep failing on it. In the 2026-10-05 case, two node pools tracked in a
DCN bug crashed every 8192-chip multislice run placed on them, while the same
cubes stayed usable for single-slice ICI jobs.

Users then work around it with `cloud.google.com/gke-nodepool NotIn [...]` lists
in their JobSets. That has three problems:

1.  **The scheduler cannot see it.** Kueue's quota math still counts the
    excluded chips, so the job can wait on topology ("fit only N out of M")
    while the queue looks free ([[kueue-tas-fit-only-n-of-m]]).
2.  **Lists drift.** In that case the list changed between runs: 0 → 4 → 0 → 1 →
    4 → 2 pools. Runs without the list crashed; runs with the full list could
    not fit.
3.  **Each user rediscovers it.** The knowledge lives in one team's YAML.

## Recording a known-bad cube

Keep one record per cube or pool:

-   failure class (for example DCN, ICI, HBM, host);
-   evidence: workloads and runs that failed there, the signature, timestamps;
-   tracking bug;
-   which workloads can still use it (for example single-slice ICI jobs);
-   status: suspected (one incident or placement-only evidence), confirmed
    (repeat offender across runs, or an SRE/vendor diagnosis), cleared.

Use the repeat-offender test before marking a cube: the same pool in two or more
failed runs and absent from runs that succeeded ([[multi-workload-triage]]). A
link to a known bug is circumstantial unless the bug's signature appears in this
workload's logs.

## Exposing it to the scheduler

Prefer making the constraint visible to scheduling over per-user lists:

-   **Taint** only when no workload can use the cube (cluster admin action;
    requires approval).
-   For cubes that still serve some workloads, a node **label per failure
    class**, with the affected workload class requiring the label to be absent,
    lets Topology-Aware Scheduling count the same constraint the pods will see,
    while other workloads keep using the cubes. (Design direction, not a GKE
    feature.)
-   Until then, keep the record visible in capacity reports and subtract the
    cubes when telling users how much capacity their job class has.

## Operational Impact

-   Crashes: every run of the affected class that lands on the cubes fails,
    often minutes into the job, wasting the whole allocation each time.
-   Hidden capacity: per-user exclusions make jobs wait while the queue looks
    free, and capacity numbers quoted by different people stop agreeing.
-   Repair cost: sending the cubes to repair at once lowers capacity for the
    workloads that can still use them; recording the failure class lets repair
    be scheduled when capacity allows.

## Connections

-   **Symptoms**:
    -   [[kueue-tas-fit-only-n-of-m]]
-   **Playbooks**:
    -   [[multi-workload-triage]]
    -   [[node-isolation-methodology]]
    -   [[run-forensics-report-template]]
-   **Workflows**:
    -   [[workload-run-forensics]]

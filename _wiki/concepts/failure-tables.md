---
title: Failure Tables as a Debugging Tool
type: concept
internal_only: false
tags: [training, pathways, triage]
created: 2026-05-14
updated: 2026-10-15
---


## Overview

A failure table has one row per interruption or restart of a workload, so that
repeat offenders, cascades and the reported-vs-actual culprit become visible
across incidents. The machine that logs the fatal error is often not the one
that died (in Pathways a cube peer can log `FATAL ERROR USER` while another
worker died silently). Recording VM and workload recovery time per row also
exposes slow detection (one table showed a dead machine taking 10 min to be
terminated, doubling recovery time).

## Attribution Rules (mandatory)

Node Replacement decisions are made from the culprit column. Each rule below
comes from a misattribution that actually shipped.

### 1. Quote the culprit host; never infer it

Pathways names the failing host literally in its own logs:

```
lost connection to peer at http://machine/<node>/events
... first level distributor <id> <node> exiting [reported by worker pipe closing]
```

Use [[extract-culprit-hosts]], which parses exactly these lines.

**Never derive a culprit from `DATA_LOSS` device coordinates.** A message such
as `DATA_LOSS: Device
device(272,TPU_DEVICE,coords=[0,0,0,0],logical_task=0,slice=1) is unplaced`
identifies **where the symptom surfaced**, and the slice index is frequently
*not* the slice that died. In My Tpu Workload Sigill 20260827 this inference
named a node in a node group that was not even part of the failing slice, while
the RM had printed the correct host in plain text seconds earlier.

### 2. Count offences by incident, not by log line

When a host dies, **every surviving peer reports it within the same second**.
Raw event counts therefore overstate failures by an order of magnitude. Cluster
same-host events into incidents, and classify by **reporting breadth**:

| Reporters                         | Meaning                               |
| :-------------------------------- | :------------------------------------ |
| Most of the fleet (tens of pods)  | **Primary** — a genuine host death    |
| 1–2 pods, seconds after a primary | **Cascade** — teardown artefact, does not count |

Measured on one incident: 3 genuine failures reported by 17–33 pods each,
alongside 26 cascade artefacts reported by 1–2 pods. Counting raw events yielded
29 "failures" and nominated **7 healthy nodes** for node replacement.

`Recommend Node Replacement` requires **two or more distinct primary incidents**
on the same instance id.

### 3. State the scope of every negative verdict

`No VM Termination` read alone means "no hardware problem", and has caused real
faults to be dismissed. Always name what was checked —
`No VM Termination (all 65 nodes across 4 node groups, 07:45-10:05)` — and never
assert a fleet-wide negative over a node set whose completeness is unconfirmed
(see [[resolve-workload-nodes]]).

Where the layers disagree, say so: *`No VM Termination ..., but 3 workers killed
by SIGILL si_code=0; fault is invisible at the GCE layer`*.

### 4. Preemption requires audit corroboration

`EXIT_TYPE_QUICKLY_DURING_PREEMPTION` and
`"Resource manager is being preempted"` are emitted for **any** SIGTERM — see
[[false-preemption-sigterm]]. A preemption row needs a matching audit event on
one of *this workload's own* nodes.

### 5. Begin each chain at the first process death

Client-side `DATA_LOSS`, `PlacementDataLoss`, `ValueError` and `JaxRuntimeError`
are almost always downstream. Start at the crash signal, pipe break, or host
loss — and read `si_code` (`0` = `SI_USER` = external `kill()`;
`-6 from PID <self>` = self-`abort()` teardown, not an independent fault).

### 6. Give recovered disruptions a row

A crash the workload survived without restarting is absent from restart-derived
timelines, yet it is often the most informative event in the incident: it proves
the fault is survivable and reclassifies every failed recovery as a
recovery-path defect.

## Columns

### Automated analyzer schema

`Jobset Name`, `Restart Attempt`, `Timestamp`, `Case`, `Overall analysis`,
`App log analysis of crash`, `GKE Logs analysis`, `VM Termination analysis`.

*   **Case**: The categorization based on heuristic pattern analysis. Must
    strictly be one of:
    *   `Monitor for Potential HW Issue`: No HW interruption seen, but App logs
        or GKE pods note a suspicious machine (first known case for this
        instance).
    *   `Recommend Node Replacement`: Two or more distinct **primary incidents**
        on the same bad GKE instance id. Count incidents, not raw events: one
        host death is reported by every surviving peer, so raw counts inflate a
        single failure into a dozen. Cube-level node replacement is a real
        operator decision but is **not automated** -- [[extract-culprit-hosts]]
        aggregates per instance id only, so `isolate_recommended` will never
        name a cube.
    *   `Expected HW Interruption - No Action`: HW interruption on a machine
        with no previous issues.
    *   `Expected HW Interruption - Detected repeated bad machine`: A bad
        machine was identified in a previous case (via any previous status) and
        is later involved in another HW interruption.
    *   `Elastic Upshard`: No HW interruption, but App logs indicate the
        workload, specifically Pathways, restarted to elastic upshard to utilize
        more slices after checkpoint.
    *   `Manual Intervention`: JobSet sequence change (e.g., `cr1` -> `cr2`) or
        manual deletion/re-creation by operator or Kueue eviction.
    *   `Unknown Case`: Automatically classify as Unknown when unable to decide
        on any of the above patterns. Must provide the specific reason for the
        uncertainty in the analysis.
*   **App log analysis of crash** must carry a logs deep link:
    *   **Logs Link**: Include a deep link to the Cloud Logging query targeting
        the relevant workload/pods. Template:
        `[Logs](https://console.cloud.google.com/logs/query;query=resource.labels.pod_name:%22{workload_prefix}-%22;project={project_id})`
        (construct this by replacing `{workload_prefix}` with the workload name
        prefix and `{project_id}` with the active project ID).

### Manual schema

`Interruption Number`, `Workload`, `Suspected Category` (HW / SW / Provisioning
/ Unknown), `Reason / Log Message`, `Analysis`, `Culprit Pod / Slice`, `Status`,
`Duration`.

*   **Culprit for global errors**: For global errors (such as software bugs,
    configuration typos, or project-wide quota limits) that are not tied to a
    single faulty machine, **identify the specific pod or slice where the
    error/log was first detected** (e.g., `Worker 0` or `Slice 0` or a specific
    pod name) rather than using `N/A`. This helps developers locate the log
    context. If it truly affected the entire slice equally without a clear
    single detector, use `all`. Only use `N/A` if no logs or location can be
    determined.

## Writing rules

-   Record the steps that found each fact (e.g. "sorted exit timestamps
    ascending"), not only the conclusion.
-   Check whether a suspect message appears on one worker or on all, and in
    earlier interruptions.
-   Look at K8s events (`Evicted`, `OOMKilling`) when application logs are
    silent.
-   Keep deep-link filters loose enough to survive small message variations.

## Automated analyzer workflow

1.  Get every restart/crash with timestamps for the jobsets and time window
    ([[job-disruption-timeline]]).
2.  For each restart, run the GKE, hardware ([[ai-hw-analyzer]]) and application
    ([[ai-app-analyzer]]) analyzers in parallel, each limited to that restart
    attempt, with a 10 min timeout per analyzer (keep partial output on
    timeout).
3.  Join the outputs on restart attempt, timestamp and machine identity into the
    automated schema; check every row maps to the right attempt and carries the
    deep link.
4.  Assign one of the 7 `Case` values per row.
5.  Output the table and pattern analysis only. Never run a mitigation.

### Example

| Jobset | Attempt | Case | Overall analysis | App logs | GKE logs | VM Termination |
| ------------- | ------- | ------------ | ------------ | -------- | --------- | ----------- |
| MyFavoriteJob | 1 | `Monitor for Potential HW Issue` | First failure on machine X, no VM termination. | `!!!! FATAL ERROR !!!!` HARDWARE on X | Pods started | No VM Termination (all nodes, window) |
| MyFavoriteJob | 2 | `Recommend Node Replacement` | Second primary incident on X; still schedulable. | `!!!! FATAL ERROR !!!!` HARDWARE on X | Pods started | No VM Termination (all nodes, window) |
| MyFavoriteJob | 3 | `Expected HW Interruption - No Action` | App logs and VM termination agree on Y. | MXLA hang, culprit Y | Pods started | VM Termination on Y |
| MyFavoriteJob | 4 | `Expected HW Interruption - Detected repeated bad machine` | Y came back from attempt 3 and failed again; ask for testing of Y and its cube. | MXLA hang, culprit Y | Pods started | VM Termination on Y |
| MyFavoriteJob | 5 | `Elastic Upshard` | Pathways scaled up after checkpoint. | Elastic upshard event | Scheduled | No VM Termination |

## Connections

-   **Playbooks**:
    -   [[multi-chip-hang-investigation]]
    -   [[workload-disruption-orchestration]]

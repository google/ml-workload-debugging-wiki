---
title: "Run Forensics Report Template (Multi-Run Scheduling and Crash Triage)"
description: "Report format for a GKE TPU JobSet series where some runs stay pending ('allows to fit only N out of M slice(s)') and others start and crash: bottom line, per-run exclusion and placement table, story, time-aligned evidence with run scope, system and observability gaps, and ranked fixes with owners."
type: playbook
internal_only: false
tags: [tpu, gke, kueue, scheduling, capacity, triage]
created: 2026-10-06
updated: 2026-10-06
---


# Run Forensics Report Template (Multi-Run Scheduling and Crash Triage)

## Overview

Use this template for any multi-run investigation driven by
[[workload-run-forensics]](for example, when a JobSet series alternates between waiting in Kueue and crashing after placement).
Run [[ai-verify-failure-attribution]] on the draft before publishing.

## 1. Bottom Line

Two sentences:

-   **Root cause**: why the unscheduled runs could not fit (in 4x4x4 cubes for
    that specific run) and why the started runs crashed (naming the
    repeat-offender node pools and whether a known bug link is direct or
    circumstantial).
-   **One immediate action**: the exact change that lets the next run both fit
    and avoid the bad hardware (plus any logging flag required to make future
    failure digests visible).

## 2. Diagnostic Steps & Per-Run Table

Build one row per run (including recreated runs that reused a JobSet name) from
[[recover-jobset-spec-from-audit]], [[scheduled-events-by-pool]],
[[decode-tas-pending-message]], and [[cube-ledger]]:

| Run | Created (PDT / UTC) | Pools excluded (audit) | Placed on (events) | Free cubes for this run = formula | Outcome | Why |
| :----------- | :------ | :------- | :------- | :-------- | :-------- | :----------- |
| `<prefix>-1` | `PDT (UTC)` | `none` | `np-a (16), np-b (16), ...` | `132 usable - 0 self-excl - 2 held = 130 >= 128` | Crashed (~3m) | Placed on DCN-suspect pools `np-a`, `np-b` |
| `<prefix>-5` | `PDT (UTC)` | `np-a, np-b, np-c, np-d (4)` | `—` | `132 usable - 4 self-excl - 2 held = 126 < 128` | Never scheduled | `fit only 126 out of 128`\ excluded 2 healthy pools (`np-c`, `np-d`) in addition to `np-a`, `np-b` while `rl-train-0` held 2 cubes |

Rules for the table:

-   Never infer one run's `NotIn` list from a sibling run; recover each run's
    spec from `io.k8s.batch.v1.jobs.create` audit entries.
-   Express all capacity math in cubes (`nodes / 16` on TPU7x) and name the
    holders of any busy cubes.

## 3. Story

A short narrative explaining how the columns connect across runs:

-   How the user's `NotIn` exclusion list changed across runs
    (`0 -> 4 -> 0 -> 1 -> 4 -> 2` pools).
-   Why runs without the full exclusion list scheduled and crashed (landing on
    the suspect pools), while runs that excluded extra healthy pools fell below
    the needed cube count.
-   Why an earlier `insufficient unused quota` message masked the topology
    shortfall while borrowers were still running
    ([[kueue-tas-fit-only-n-of-m]]).

## 4. Time-Aligned Evidence

State the verdict (`CONFIRMED`, `INCONCLUSIVE`, or `NO_DATA` per
[[rca-invariants]]) and the **explicit run scope** for every claim:

-   **Per-run exclusion lists (`CONFIRMED`, runs `<prefix>-1` ..
    `<prefix>-8`)**: quoted from GKE audit `io.k8s.batch.v1.jobs.create` request
    bodies at each run's creation timestamp.
-   **TAS capacity shortfall (`CONFIRMED`, run `<prefix>-5` at `<timestamp>`)**:
    decoded from `excluded: affinity A, resource R, taint T` and matched to the
    cube ledger at that timestamp (never from a live snapshot hours later).
-   **Crash placement (`CONFIRMED` / `INCONCLUSIVE`, runs `<started_runs>`)**:
    grouped `Scheduled` events per node pool + coordinator/worker stack trace.
    If linking to a known bug whose log signature has 0 hits in these runs'
    logs, label the link **circumstantial** (`INCONCLUSIVE`).

## 5. System & Observability Gaps

Answer two questions explicitly:

1.  **Are bad machines visible to the system?** Check
    [[cube-ledger]](`known_bad_cube_visibility`): do the known-bad pools report
    `HEALTHY` in GKE without a cluster taint, forcing users to maintain drifting
    `NotIn` lists that Kueue quota math cannot see ([[known-bad-cubes]])?
2.  **Can the user see the failure evidence?** Check
    [[check-failure-observability]]: what is the effective
    `TPU_STDERR_LOG_LEVEL`? At the default `3` (`FATAL`), `LOG(ERROR)` Megascale
    digests (`Megascale detects a hang`, `Full error digest`) are suppressed
    from Cloud Logging and local `/tmp/tpu_logs` disappears when pods exit
    ([[tpu-logging-flags]]).

## 6. Ranked Fixes with Owners

List concrete actions ordered by impact, naming the owner and approval
requirement for each:

1.  **Cluster admin (requires approval — ask the user before running)**: taint
    or cordon confirmed known-bad pools (`[[taint-node]]`) so the scheduler and
    Kueue see the constraint without per-user `NotIn` lists.
2.  **Workload owner**: trim the JobSet `NotIn` list to only confirmed bad pools
    (verified with the cube ledger math check so `free_cubes >= needed_cubes`),
    and set `TPU_STDERR_LOG_LEVEL=2` and `TPU_MIN_LOG_LEVEL=0` so future
    Megascale error digests reach Cloud Logging.
3.  **Capacity / queue operator**: if `free_cubes` is still short after trimming
    self-exclusions, reclaim cubes held by lower-priority queues only after
    verifying that reclaiming them makes `free_cubes >= needed_cubes`.

## Connections

-   **Symptoms**:
    -   [[kueue-tas-fit-only-n-of-m]]
    -   [[megascale-hang]]
-   **Concepts**:
    -   [[known-bad-cubes]]
        -   [[tpu-logging-flags]]
-   **Playbooks**:
    -   [[rca-invariants]]
    -   [[multi-workload-triage]]
-   **Workflows**:
    -   [[workload-run-forensics]]
-   **Actions**:
    -   [[ai-verify-failure-attribution]]

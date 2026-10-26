---
title: "RCA Invariants (Read First)"
description: "Mandatory reasoning rules for every RCA, read before any other page: preflight log access (PERMISSION_DENIED / logging.logViews.access -> stop, NO_DATA), no evidence no verdict, earliest error across all pods not worker 0, barrier / DEADLINE_EXCEEDED timeouts are victims, scope every query to the workload, infra before user code for simultaneous stalls, internal host = cluster machine, verdicts CONFIRMED / REFUTED / INCONCLUSIVE / NO_DATA."
type: playbook
internal_only: false
tags: [methodology, triage]
created: 2026-10-03
updated: 2026-10-03
---


# RCA Invariants (Read First)

## Overview

Apply all rules on every run; they override any workflow step. Each comes from a
real wrong RCA.

## Diagnostic Steps

1.  **Preflight access.** Before analysis, read one log entry of the workload in
    its window
    (`gcloud logging read '<workload filter>' --project=<project> --limit=1`).
    On `PERMISSION_DENIED: Permission denied for all log views`, missing
    `logging.logViews.access`, or authentication failure:
    stop, verdict `NO_DATA` with the exact error. Never "RCA ready". Why: 6 of
    30 incident reviews read zero logs and still said RCA ready.
2.  **No evidence, no verdict.** Every causal claim quotes a timestamped line,
    event or metric. "Not provable from customer telemetry" is `INCONCLUSIVE`,
    not refuted and not confirmed. Why: scenario 9.C read no logs, said "not enough
    slices"; it was preemption by a higher-priority workload.
3.  **Earliest error across all pods.** Sort errors, exits and signals of every
    pod by timestamp ([[find-earliest-failure]], [[query-crash-signatures]] over
    the full window). Worker 0 / pathways-head logs the most and reports others'
    failures; the loudest error is usually downstream. Why: scenario 10.A blamed an
    Orbax error on pod 0-0.
4.  **Timeouts are victims.** `DEADLINE_EXCEEDED: Barrier timed out`,
    `Shutdown barrier has failed`, Megascale hang, sflag waits: find who did not
    arrive (`# of tasks that reached the barrier: 1023/1024 ... Some timed out
    task names: 106`) and debug that task before the timeout.
5.  **Scope to the workload.** Filter by namespace + JobSet / pod prefix and the
    workload's nodes ([[resolve-workload-nodes]]); drop other workloads' pods.
    Why: a incident alert quoted an unrelated MCP-server pod.
6.  **Infra before user code** when all workers stop together or many hosts fail
    in the same second: check node, slice and DNS events first. Why: scenario 2
    chased a YAML mismatch; the fix was kube-dns -> CoreDNS
    ([[kubedns-resolution-failure]]).
7.  **Verdicts** (any confirm / refute step, incl. internal verification):
    -   `CONFIRMED`: a quoted record in the incident window on the named
        node/machine supports culprit and mechanism.
    -   `REFUTED`: a quoted record contradicts it (earlier other culprit, or the
        machine healthy with complete coverage).
    -   `INCONCLUSIVE`: data readable but neither; list checks and gaps.
    -   `NO_DATA`: required data unreadable, or no culprit to check; give the
        error. A `NO_DATA` RCA is never `CONFIRMED` or PASS.

## Resolution

Report: verdict, culprit (node / machine / task), mechanism, and the quoted
timestamped evidence for each claim; for `NO_DATA`, the exact access error.

## Connections

-   **Playbooks**:
    -   [[debugging-mindset]]
    -   [[elastic-vs-permanent-failure]]
-   **Symptoms**:
    -   [[tpu-provisioner-slice-recreation]]
    -   [[kubedns-resolution-failure]]

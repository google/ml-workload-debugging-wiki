---
title: Investigation Report Template
type: playbook
internal_only: false
tags: [training]
created: 2026-05-14
updated: 2026-10-03
---

Report skeleton for a complex TPU workload failure: group workers, then build a
timeline.

## Summary

Workload name; cluster; failure timestamp; one-line summary.

## Symptom

Primary error, traceback, or where the job stopped responding.

## Worker Groupings & Diagnostic Breakdown

Group workers by behavior, exit code, or program-tracker state at failure:

-   **Primary suspects:** stopped logging with no exit code, or stuck on an
    earlier HLO. Pods: `...`
-   **Active failures:** specific exit code such as 134. Pods: `...`
-   **Bystanders:** graceful exit 143. Pods: `...`

## Timeline of Events (UTC)

-   `YYYY-MM-DD HH:MM:SS`: event (e.g. workers initialize, coordinator receives
    error report, primary failure).

## Root Cause

## Resolution / Recommended Actions

E.g. node replacement a node, adding debug flags.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[tpu-state-analysis-methodology]]

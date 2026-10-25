---
title: Workload Debugging Flowchart
type: playbook
internal_only: false
tags: [training]
created: 2026-05-14
updated: 2026-10-03
---

## Overview

Decision order for an unknown workload failure. Commands are in
[[exit-code-investigation]].

## Decision Flow

1.  Check worker 0 of slice 0 (coordinator) **and** JAX Process 0 (see
    [[debugging-gotchas]]). Clear error: confirm it is unique to failing
    workers, record it in the failure table, fix.
2.  No clear error: run exit-code analysis across all pods.
3.  Not all `143`: take the outlier codes (e.g. `134`, `137`), sort by
    timestamp, and read the first failing worker's logs.
4.  All `143`: compare the number of exit-code logs with the expected pod count.
    -   Missing: check Kubernetes events (`Evicted`, `Killing`) for the silent
        pods. OOM or disk full explains it; otherwise suspect a silent stall or
        data loading issue.
    -   None missing: check for cluster maintenance or a global interruption.
5.  Always finish by recording findings in the failure table.

## Connections

-   **Workflows**:
    -   [[exit-code-investigation]]
-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[debugging-mindset]]

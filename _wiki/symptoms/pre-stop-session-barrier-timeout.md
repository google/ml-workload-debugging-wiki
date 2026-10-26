---
title: PRE_STOP_SESSION_BARRIER Timeout
type: symptom
internal_only: false
tags: [tpu, timeout]
created: 2026-05-13
updated: 2026-10-03
---

## Symptom signature

```
Barrier sync with id PRE_STOP_SESSION_BARRIER timed out. Invoking stop session without sync.
```

In `session_manager.cc` this barrier is best-effort: it stops ICI traffic before
the network goes down, and teardown proceeds without it if it fails. Almost
always secondary, not the root cause.

## Triage

1.  Read earlier logs on the same worker for the originating error. If app logs
    are silent, check K8s events for evictions or OOMs.
2.  If the log names another worker (e.g. slice 0-10), read that worker's logs.
3.  Many workers timing out at once suggests a cluster-wide issue such as DNS.
4.  **Fallback**: If no root cause is visible in logs, perform an
    [[exit-code-investigation]] or retry the workload to gather more data
    points.

## Routing

-   Session errors: [[session-failing]]; slice errors: [[slice-failure]].

## Connections

-   **Workflows**:
    -   [[exit-code-investigation]]
-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[mcjax-debugging-principles]]
-   **Symptoms**:
    -   [[session-failing]]
    -   [[slice-failure]]

---
title: Session Failing with Status
type: symptom
internal_only: false
tags: [tpu, training]
created: 2026-06-11
updated: 2026-10-03
---

## Symptom signature and causes

Logged by the Session Master on McJax GKE worker 0 of slice 0 (coordinator) or
JAX process 0 (search `[process=0]`). Strings and the `session_master.cc`
failure type behind each:

```
Session is failing with the following status: Session master detected TPU session error from worker <worker_address>. The reporting TPU chip locations are: <chips>
```

-> chip driver errors (`SLICE_FAILURE_CHIP_DRIVER_ERROR`).

```
Session master detected error injected from worker with address: <worker_address>
```

-> software-injected error (`SLICE_FAILURE_SW_INJECT_ERROR`).

```
Session manager lost connection from worker with address <worker_address>. It has missed X health checks.
```

-> missed heartbeats (`SLICE_FAILURE_WORKER_UNAVAILABLE`); check K8s events for
OOM or eviction.

```
Session is failing due to the following chips having zero as chip id
```

-> init error (`SLICE_FAILURE_INIT_ERROR`).

If several workers fail, the named worker may only be the first one noticed.

## Triage Steps

1.  Open the named worker's logs and look for errors before this message.
2.  Nothing there: check GKE worker 0 or JAX process 0 of each slice or node
    pool.
3.  Read raw logs around the failure time; keyword filters miss messages. Filter
    one worker (e.g. worker 10 in slice 0):

    ```
    resource.labels.pod_name=~"llama2-70b-4096-16-111920-6dt-slice-job-0-10-"
    ```

## Connections

-   **Workflows**:
    -   [[exit-code-investigation]]
-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[mcjax-debugging-principles]]
-   **Symptoms**:
    -   [[pre-stop-session-barrier-timeout]]
    -   [[slice-failure]]

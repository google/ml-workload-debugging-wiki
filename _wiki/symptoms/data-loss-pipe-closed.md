---
title: Data Loss Pipe Closed
type: symptom
internal_only: false
tags: [tpu, pathways, training, triage]
created: 2026-05-13
updated: 2026-10-15
---


## Symptom signature

```
DATA_LOSS: Pipe from scheduler to aggregator <jobset-name>-worker-0-0.:29001 closed unexpectedly
```

And on surviving peer `pathways-worker` pods when another worker in the slice is
killed (for example by an external `SIGILL` on `worker-job0-*-0`):

```
Server pipe /worker_to_worker id=7647480592439503971: Break idle pipe DEADLINE_EXCEEDED: lost connection to peer at http://machine/<culprit_node>/events ... (Quick-restart requested but unsupported in Cloud/OSS.) Binary exited with code: 1
```

A shard of a JAX array or a worker-to-worker RPC pipe became unavailable because
a peer host died or exited. `<jobset-name>-worker-0-0` is the GKE pod name; JAX
process 0 may be a different worker ([[debugging-gotchas]]).

## Verification

-   **Distinguish primary culprit from cascade victim:** When a worker logs
    `Server pipe /worker_to_worker ... lost connection to peer at
    http://machine/<culprit_node>/events ... (Quick-restart requested but
    unsupported in Cloud/OSS.) Binary exited with code: 1`, the pod that exited
    with code 1 (e.g., `worker-job0-1-193-fxkvb` on `gke-tpu-node-xxxx` in
    `example-elastic-4x1k` Attempt 4) is a **cascade victim** of
    `<culprit_node>` (e.g., `gke-tpu-node-xxxx`, where `worker-job0-1-0-zf9cx`
    was killed by `SIGILL` 22 seconds earlier). Never attribute the root cause
    to the cascade victim.
-   Use [[extract-culprit-hosts]](`min_reporting_pods >= 3`) to separate broadly
    reported primary node deaths from narrow 1–2 pod teardown cascades.
-   Check K8s events for `Evicted` or `OOMKilling` on the primary culprit pod
    and the rest of the slice.

## Routing

-   Worker logs: [[pathways-cluster-readiness]].
-   Culprit node extraction: [[extract-culprit-hosts]].

## Connections

-   **Concepts**:
    -   [[pathways]]
-   **Symptoms**:
    -   [[sigill-worker-kill]]
    -   [[elastic-reshard-pytree-mismatch]]

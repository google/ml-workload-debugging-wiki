---
title: IFRT Proxy Terminated
type: symptom
internal_only: false
tags: [tpu, pathways, training]
created: 2026-06-11
updated: 2026-10-15
---

## Symptom signature

```text
IFRT proxy server was terminated: UNAVAILABLE: Socket closed
```

Or on the JAX client during array execution/deletion:

```text
Internal error from proxy server during Array::IsDeleted(): UNAVAILABLE: Connection to IFRT proxy server was terminated: FAILED_PRECONDITION: GrpcClientSession: writes no longer allowed.
```

Downstream only: the proxy dies or closes its gRPC session because a Pathways
worker (`pathways-worker`) or the Resource Manager (`pathways-rm`) container
crashed or was preempted. The proxy log rarely holds the root cause.

## Triage and Resolution

1.  Skip the proxy logs; read `pathways-worker` (starting with leader index 0)
    and `pathways-rm` logs (`gcluster job logs <WORKLOAD> --main-only=false` or
    Cloud Logging) for the first crash or Error Aggregator summary.
2.  Check K8s events on the proxy and worker pods for eviction, node shutdown
    ([[false-preemption-sigterm]]), or OOM kill (`RESOURCE_EXHAUSTED`).
3.  If running in interactive mode over an Internal Load Balancer (ILB), verify
    the backend service `timeoutSec` was not exceeded during an idle period.

## Connections

-   **Workflows**:
    -   [[pathways-debugging-playbook]]
-   **Playbooks**:
    -   [[pathways-debugging-principles]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[false-preemption-sigterm]]
-   **Concepts**:
    -   [[pathways]]
    -   [[pathways-fault-tolerance]]

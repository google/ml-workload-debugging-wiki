---
title: Proxy Connection Failed
type: symptom
internal_only: false
tags: [tpu, pathways, training]
created: 2026-06-11
updated: 2026-10-15
---

## Symptom signature

The JAX client hangs, times out, or immediately rejects connection to the
Pathways IFRT proxy (`grpc://<proxy>:29000` or `:38681`):

```text
RuntimeError: Unable to initialize backend 'proxy': UNAVAILABLE: Unable to establish connection to ifrt_proxy server, please check provided address 'grpc://...'; detailed error: DNS resolution failed for ...: C-ares status is not ARES_SUCCESS qtype=A ...
```

Or a session conflict / protocol mismatch during `ifrt_proxy.get_client`:

```text
RuntimeError: Unable to initialize backend 'proxy': CANCELLED: Connection to IFRT proxy server was terminated: CANCELLED: Cancelled
```

```text
RuntimeError: Unable to initialize backend 'proxy': INVALID_ARGUMENT: IFRT Proxy client and server failed to agree on the protocol version; supported versions: client = [1, 1], server = [3, 14]
```

## Triage and Causes

1.  **Proxy server not started or crashed**: Check `pathways-proxy` logs for
    `IFRT proxy server started with status OK` and verify
    `--gcs_scratch_location` bucket exists and is accessible.
2.  **`JAX_BACKEND_TARGET` / DNS misconfiguration**: Verify the proxy service
    hostname or port-forwarding tunnel (`kubectl port-forward`) is active and
    that `JAX_PLATFORMS=proxy` is set.
3.  **VPC / firewall or L4 ILB timeout**: In interactive mode using an internal
    load balancer, verify L4 ILB subsetting is enabled and `timeoutSec` is
    increased to prevent idle gRPC disconnects.
4.  **Single-client session conflict (`CANCELLED`)**: Pathways supports only one
    active interactive client per cluster at a time. Disconnect any other open
    notebook/script, or recreate the Pathways cluster if a previous client
    disconnected uncleanly.
5.  **Protocol version mismatch (`INVALID_ARGUMENT`)**: Align the JAX/MaxText
    client container version with the `pathways-proxy` server image tag
    (`jax-<version>`).

## Routing

-   [[pathways-cluster-readiness-checks]]
-   [[pathways-cluster-readiness]]

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[pathways-debugging-principles]]
-   **Concepts**:
    -   [[pathways]]
    -   [[pathways-cluster-readiness-checks]]

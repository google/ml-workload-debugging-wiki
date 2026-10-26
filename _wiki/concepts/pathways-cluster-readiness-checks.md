---
title: Pathways Cluster Readiness
type: concept
internal_only: false
tags: [tpu, pathways, training, observability, logging, monitoring]
created: 2026-06-11
updated: 2026-10-15
---

## Definition

**Pathways Cluster Readiness** details the expected log signatures, component
dependencies, and verification queries needed to ensure a Pathways on GKE
cluster (`pathways-rm`, `pathways-proxy`, and `pathways-worker`) is healthy and
ready to execute JAX computations.

## Readiness Log Signatures & Architecture Details

-   **Resource Manager (`pathways-rm`)**:
    -   Server listening:
        `server_lib.cc:1473] Pathways Server serving on [::]:29001`
    -   Slice registration:
        `multi_job_allocator.cc:1063] *** <N>/<M> Pathways Slices Now Ready ***`
    -   **Virtual Slices**: Pathways uses *virtual slices* to initialize the
        cluster before all physical TPU slices have finished booting. A JAX
        client can connect to `pathways-proxy`, query `jax.devices()`, and begin
        XLA compilation while `N < M` slices are ready, blocking only when
        device buffer execution requires the remaining slices.
-   **IFRT Proxy Server (`pathways-proxy`)**:
    -   Backend ready:
        `proxy_server.cc:125] IFRT proxy server started with status OK`
    -   Client connected:
        `proxy_server.cc:125] Connected to backend proxy_server`
-   **Workers (`pathways-worker`)**:
    -   Worker listening:
        `server_lib.cc:1473] Pathways Server serving on [::]:29001`
    -   Focus on leader workers (`completion-index = 0`, e.g.
        `<jobset>-worker-0-0`) to verify startup sequences.

## Cloud Logging & CLI Verification

To view logs across all containers for a given JobSet in Cloud Logging:

```text
resource.type="k8s_container"
resource.labels.project_id="<project-id>"
resource.labels.location="<cluster-region>"
resource.labels.cluster_name="<cluster-name>"
resource.labels.pod_name:"<jobset-name>-main-0" OR "<jobset-name>-rm-0" OR "<jobset-name>-proxy-0" OR "<jobset-name>-worker-0-0"
severity>=DEFAULT
```

For Cluster Director (`gcluster`) deployments, inspect logs across all pods
(rather than only the main container) with:

```bash
gcluster job logs <WORKLOAD_NAME> --main-only=false
gcluster job list --status Failed
```

## Cloud Monitoring MXLA Latency Metrics

When diagnosing post-startup performance bottlenecks or collective stalls, check
the following Cloud Monitoring metrics under `Kubernetes Container`:

-   `kubernetes.io/container/multislice/network/dcn_transfer_latencies`: DCN
    host-to-host transfer latency distribution across slices.
-   `kubernetes.io/container/multislice/collective/collective_e2e_latency`:
    End-to-end Megascale collective operation latency.
-   `kubernetes.io/container/multislice/accelerator/host_to_device_transfer_latencies`:
    Host-to-device (H2D) buffer transfer latency.
-   `kubernetes.io/container/multislice/accelerator/device_to_host_transfer_latencies`:
    Device-to-host (D2H) buffer transfer latency.

## Connections

-   **Workflows**:
    -   [[pathways-cluster-readiness]]
-   **Playbooks**:
    -   [[pathways-debugging-principles]]
-   **Symptoms**:
    -   [[proxy-connection-failed]]
-   **Concepts**:
    -   [[pathways]]

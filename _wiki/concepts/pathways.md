---
title: Pathways Architecture and Internals
type: concept
internal_only: false
tags:
- tpu
- pathways
- training
- inference
- gke
created: 2026-06-11
updated: 2026-10-16
---



## Overview

Pathways is a single-controller [[jax]] backend alternative to traditional
multi-controller JAX ([[mcjax]]), designed to orchestrate large-scale,
multi-task, and sparsely activated ML systems across thousands of [[cloud-tpu]]
accelerators.

*   **Multi-Controller ([[mcjax]])**: Python runs on every TPU VM
    (`jax.process_index()` varies per host; `jax.local_devices()` returns only
    host-local chips). All workers execute the same SPMD program in lockstep.
*   **Single-Controller (Pathways)**: Python runs on a single CPU VM (the User
    Job). `jax.process_index()` is **always `0`**, and both `jax.devices()` and
    `jax.local_devices()` return **all TPU devices across the entire job**. The
    controller sends StableHLO to the Pathways control plane for compilation,
    scheduling, and execution across multiple TPU slices, supporting both SPMD
    and asymmetrical pipeline parallelism or disaggregated serving.

--------------------------------------------------------------------------------

## Architecture & Design

```mermaid
graph TD
    subgraph Client / User Job (CPU VM)
        UserCode[JAX User Script + pathwaysutils] --> ProxyClient[IFRT Proxy Client]
    end
    subgraph Pathways Control Plane (CPU Node Pool: pathways-head)
        ProxyClient -->|1. gRPC port 29000 / 38681| ProxyServer[IFRT Proxy Server + Pathways Client]
        ProxyServer -->|2. Forward HLO port 29001 / 38677| RM[Pathways Resource Manager]
    end
    subgraph Accelerator Data Plane (TPU Slice Node Pools)
        RM -->|3. Gang-Schedule & Monitor| Workers[Pathways Workers on TPU VMs]
        ProxyServer -->|4. Compiled XLA Executables & Results| Workers
        Workers <-->|5. Custom gRPC Protocol| Sidecar[Sidecar Server: Colocated Python]
        Workers <-->|6. Multislice Collectives| Megascale[Megascale / DCN & ICI]
    end
```

### Core Components

*   **User Job & `pathwaysutils`**: Runs the user's JAX script on a CPU node
    (recommended minimum `n2-standard-64` or `n4-standard-64` with 64 vCPUs and
    256 GB RAM in the same GCP zone). Calling `pathwaysutils.initialize()` with
    `JAX_PLATFORMS=proxy` registers the `"proxy"` backend, disables the local
    JAX compilation cache in favor of the Pathways persistent cache, and
    configures distributed [[orbax]] checkpoint handlers and elastic training
    primitives.
*   **IFRT Proxy Client**: OpenXLA **Interim Framework Runtime (IFRT)** client
    in the user process that communicates with the IFRT Proxy Server over gRPC
    (`JAX_BACKEND_TARGET=grpc://<proxy-host>:29000`).
*   **IFRT Proxy Server (`pathways-proxy`) & Pathways Client**: CPU-only gRPC
    container
    (`us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/proxy_server:jax-<version>`)
    that receives IFRT requests and forwards HLOs to the **Pathways Client** for
    placement with the Resource Manager. **Important**: XLA compiler flags
    (including `--xla_dump_to=gs://<bucket>/<prefix>/` for HLO dumps) must be
    passed to **`pathways-proxy`** (e.g., via `--pathways-proxy-args`).
*   **Pathways Resource Manager (`pathways-rm`)**: Central CPU control plane
    container
    (`us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:jax-<version>`
    with `--node_type=resource_manager`). Manages accelerator slices
    (`--instance_count`, `--instance_type=tpu<type>:<topology>`), monitors
    worker health, gang-schedules computations, coordinates suspend-resume and
    elastic slice transitions, and runs the centralized error aggregator.
*   **Pathways Worker (`pathways-worker` / `worker`)**: Runs on each TPU VM
    (`us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:jax-<version>`),
    executes compiled XLA executables from the proxy server, and coordinates
    multislice collectives via
    [[megascale]](`MEGASCALE_NUM_SLICES`, `MEGASCALE_SLICE_ID`, `MEGASCALE_COORDINATOR_ADDRESS`).
*   **Sidecar Server (`colocated_python`)**: Co-located gRPC server on each TPU
    VM that runs user Python code (`RemoteIterator` data loading, checkpointing)
    directly on accelerator hosts over a custom versioned gRPC protocol,
    avoiding central CPU controller bottlenecks. See [[pathways-sidecar]].

### GKE Deployment Modes & Orchestration

Pathways workloads on GKE use three deployment patterns:

1.  **Batch Workloads (`JobSet` + Kueue / Cluster Toolkit)**:
    *   Submitted via `gcluster job submit --pathways` or `JobSet` manifests
        (`v0.8.0+`). Due to Kubernetes DNS label limits, the JobSet name must be
        **22 characters or fewer** ([[forever-suspended-jobset]]).
    *   The JAX user container runs alongside `pathways-rm` and
        `pathways-proxy`; completion of the JAX job terminates all Pathways
        components.
2.  **Interactive Mode (Headless `JobSet`)**:
    *   Submitted with `--pathways-headless`; client script completion does
        **not** shut down the Pathways cluster.
    *   Clients connect to port `29000` on `<jobset>-pathways-head-0-0` via
        `kubectl port-forward`, **Cloud DNS for GKE**
        (`<jobset>-pathways-head-0-0.<jobset>.default.svc.<domain>:29000`), or
        an **Internal Load Balancer (ILB)** Service (for clusters >250 nodes,
        enable `--enable-l4-ilb-subsetting`).
    *   Only **one active client session** is supported at a time; a second
        client receives `UNAVAILABLE: ... CANCELLED: Cancelled`
        ([[proxy-connection-failed]]).
3.  **Multihost Inference (`LeaderWorkerSet` + JetStream)**:
    *   Deployed via `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`). The
        leader pod (`<lws-name>-0`) runs `pathways-proxy` (port `38681`),
        `pathways-rm` (port `38677`), `jax-tpu`
        (`enable_single_controller=true`), and `jetstream-http` (port `8000`);
        worker pods run `worker` (port `38679`).
    *   Supports **disaggregated serving**
        (`leaderworkerset.sigs.k8s.io/subgroup-exclusive-topology:
        cloud.google.com/gke-nodepool`, `subGroupPolicy.subGroupSize`) to
        isolate *prefill* and *decode* onto separate TPU slices.

### Persistent Compilation Cache & Checkpointing

*   **Persistent Compilation Cache**: Enabled when
    `--gcs_scratch_location=gs://<bucket>` is passed to `pathways-rm`,
    `pathways-proxy`, and `pathways-worker` (omitting it falls back to in-memory
    cache). Pathways attaches a GCS lifecycle policy for cache TTL. Because GCS
    limits buckets to **50 lifecycle `matches_prefix` rules**, share a common
    `--gcs_scratch_location` across workloads; exceeding 50 prefixes fails
    startup with `INVALID_ARGUMENT: Permanent error, with a last message of
    Lifecycle matches_prefix cannot specify more than 50 prefixes per config.
*   **Distributed Checkpointing with [[orbax]]**: Setting
    `ENABLE_PATHWAYS_PERSISTENCE=1` and calling `pathwaysutils.initialize()`
    registers a custom `ArrayHandler` through the IFRT proxy so TPU workers read
    and write shards directly to [[cloud-storage]].

--------------------------------------------------------------------------------

## Operational Impact & Debugging Significance

*   **Container-Scoped Log Triage**: Logs are split across the user job,
    `pathways-rm`, `pathways-proxy`, and `pathways-worker`. Follow the triage
    order in [[pathways-debugging-principles]] and verify readiness signatures
    in [[pathways-cluster-readiness-checks]].
*   **Fault Tolerance & Elasticity**: Pathways uses **error poisoning** and an
    RM **error aggregator** to fail fast on worker crashes, and supports
    **Suspend-Resume** (automatic HBM snapshots to GCS on preemption notices for
    stateless XLA jobs), **Elastic Training**
    (`pathwaysutils.elastic.manager.Manager` with `maybe_snapshot`,
    `maybe_reshard_down`, `maybe_reshard_up`), and **JobSet Hot-Swap**
    (remapping failed slices onto spare or lower-`PriorityClass` slices in <1
    minute). See [[pathways-fault-tolerance]] and
    [[elastic-vs-permanent-failure]].
*   **Cloud Monitoring (MXLA Metrics)**: Pathways exports Megascale XLA latency
    distributions to Cloud Monitoring (filterable by `pod_name`):
    `dcn_transfer_latencies`, `collective_e2e_latency`,
    `host_to_device_transfer_latencies`, and
    `device_to_host_transfer_latencies`.
*   **Multi-Component Profiling (`xplane.pb`)**: JAX profiling
    (`jax.profiler.start_trace` / `start_server`) writes separate GCS traces for
    the Python client (`<client-vm>.xplane.pb`), IFRT proxy
    (`client.<head-node>.xplane.pb`), Resource Manager
    (`server.*<head-node>.xplane.pb`), and TPU workers
    (`server.*<tpu-node>.xplane.pb`). IFRT Proxy `Compile` and `Execute` RPC
    timings appear on thread `GrpcClientSessionUserFuturesWorkQueue`.
*   **Post-Recovery Component Load Hangs**: After a slice recovers
    (`*** 3/3 Pathways Slices Now Ready ***`), `pathways-proxy` may log
    `UpdateAffectedComponent` (`load_slices: 1 reincarnate_slices: 2`) while
    `jax-tpu` stalls after `elastic_utils.py:108] Checking latest checkpoint`
    because a replacement worker hangs inside
    `Attempting to load component jit_train_step.<id> @ <slice> at worker` (see
    [[elastic-vs-permanent-failure]] and [[pathways-fault-tolerance]]).
*   **Static `active_slices` Telemetry Caching During External Slice
    Suspension**: `pathwaysutils/elastic/manager.py` initializes
    `self.active_slice_indices` statically during `wait_for_slices()` and
    `_monitor_new_slices()` only checks for new slices scaling up, so MaxText's
    `metric_logger.py:221` continues printing `active_slices: 4` after external
    Kueue/JobSet slice suspension until peer TCP disconnect timeouts
    (`lost connection to peer at http://machine/<node>/events`) force an attempt
    restart.

## Connections

-   **Workflows**:
    -   [[pathways-cluster-readiness]]
    -   [[pathways-debugging-playbook]]
    -   [[pathways-topology-mismatch-triage]]
-   **Playbooks**:
    -   [[elastic-vs-permanent-failure]]
    -   [[pathways-debugging-principles]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[forever-suspended-jobset]]
    -   [[ifrt-proxy-terminated]]
    -   [[job-cancelled-unknown]]
    -   [[pathways-expected-topology-mismatch]]
    -   [[proxy-connection-failed]]
-   **Concepts**:
    -   [[ici-errors]]
    -   [[megascale]]
    -   [[pathways-cluster-readiness-checks]]
    -   [[pathways-fault-tolerance]]
    -   [[pathways-sidecar]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[jax]]
    -   [[mcjax]]
-   **Tools**:
    -   [[orbax]]

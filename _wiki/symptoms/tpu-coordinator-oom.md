---
title: TPU Coordinator OOM
type: symptom
internal_only: false
tags:
- tpu
- pathways
- training
- out_of_memory
created: 2026-06-13
updated: 2026-10-15
---


# TPU Coordinator OOM

## Symptom signature

-   The coordinator pod (usually `pod-0-0` / worker 0 in McJax, or
    `<workload>-pathways-head-0-0-*` on a CPU node pool in Pathways) exits with
    status `137` (`OOMKilled`).
-   Before the crash: MFU drops to 0-15% and host RAM grows steadily across
    steps or across repeated elastic slice replacements / checkpoint restores.
-   In `k8s_node` (`containerd` / kernel `Event`) logs on the CPU head node
    (e.g., `gke-...-cpu-np-...` via [[query-node-health-events]]):

```text
level=info msg="TaskOOM event container_id:\"0ecb63dcfe4e26d625b63ada75e8b356a57b97d5f5824879eb74e07cbb906cba\""
Memory cgroup out of memory: Killed process 1501424 (train.py) total-vm:287966520kB, anon-rss:96321092kB, file-rss:522008kB, shmem-rss:0kB, UID:0 pgtables:237676kB oom_score_adj:-997
```

## Triage

-   **Check `k8s_node` kernel OOM events on the CPU head node:** In Pathways
    runs, `train.py` runs inside the `jax-tpu` container on `pathways-head-0-0`
    (scheduled on a CPU node pool such as `cpu-np`). Always include the CPU head
    GKE node when running [[query-node-health-events]].
-   Plot host RAM over the job's life. Resets and stays flat after each
    restart = configuration/over-allocation; grows across steps or elastic
    transitions until crash = leak or uncollected host-side resharding/Orbax
    buffers.
-   Same few machines every time = suspect hardware or noisy neighbor;
    consistent across restarts and machines = software / container sizing.
-   Check whether `--megascale_grpc_enable_xor_tracer=true` is set.

## Root Cause and Mitigation

-   **Pathways CPU head (`pathways-head-0-0`) cgroup memory exhaustion after
    repeated elastic transitions (e.g., `example-elastic-4x1k` Attempt 5):**
    When an attempt undergoes multiple back-to-back slice losses and
    replacements (such as GCE `terminateOnHostMaintenance` preemptions and
    worker crashes), host-side Python/Orbax checkpoint and mesh-resharding
    buffers accumulate in `train.py` (`96.3 GB` `anon-rss`), breaching the
    `100Gi` `jax-tpu` container memory limit. Increase the Kubernetes memory
    request/limit on `pathways-head` (`jax-tpu`) above `100Gi` and release
    host-side checkpoint/resharding buffers across elastic recoveries.
-   Known leak (one historical example, not the general rule):
    `--megascale_grpc_enable_xor_tracer=true` leaked host RAM on the coordinator
    at 32k chips through a metrics-recording bug. Set
    `--megascale_grpc_enable_xor_tracer=false`.
-   Otherwise: excessive logging, large model-state aggregation, or data
    pipelines on the coordinator. Check [[known-issues]] for known leaks.

## Related Pages

-   [[known-issues]]
-   [[elastic-vs-permanent-failure]]
-   [[query-node-health-events]]

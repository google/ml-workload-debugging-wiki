---
title: MCJax (Multi-Controller JAX)
description: The traditional multi-controller JAX backend where every TPU VM runs
  the Python program in lockstep.
type: entity
internal_only: false
tags: [tpu, pathways, training, gke]
created: 2026-09-15
updated: 2026-10-15
---

# MCJax (Multi-Controller JAX)

## Overview

**MCJax** (also written McJAX) is the traditional multi-controller execution
model for [[jax]] on TPUs: the Python program runs on *every* TPU VM, and all
workers must execute the same program in lockstep. It is the alternative to the
single-controller [[pathways]] backend, and the distinction determines where
logs live — which is the first thing to establish when triaging.

## Technical Specifications

-   **Pod naming**:
    `{workload_name}-slice-job-{slice_index}-{worker_index}-{random_suffix}`;
    the coordinator pod is GKE worker 0 of slice 0 (`...-worker-job0-0-...`).
-   **Two "worker zeros"**: the McJax coordinator and JAX Process 0 (the process
    logging `[process=0]`) are frequently *different* pods. Application-level
    work (checkpoint writes, GCS I/O, global metrics) runs on JAX Process 0:
    always query `[process=0]` alongside the coordinator, because those
    exceptions do not appear on other workers.
-   **Topology checks**: MCJax does not enforce the strict expected-vs-physical
    topology check that Pathways does; it will compile and run on a mismatched
    mesh, potentially slower or incorrect, where Pathways crashes at startup
    ([[pathways-expected-topology-mismatch]]).
-   Workloads are launched on GKE as JobSets — see [[jobset-yaml-structure]].

## Common Failure Modes

-   **Looking at the wrong worker 0**: a checkpoint-restore hang was logged only
    under `[process=0]` while the coordinator showed a generic 5-minute
    collective timeout and a misleading "potential bad chip" error; the
    investigation stalled until the search was broadened across the cluster.
-   **Session failures and collective hangs** surface on the coordinator:
    [[session-failing]], [[megascale-hang]].
-   **High-concurrency startup**: simultaneous head-node DNS resolution across
    slices can bottleneck cluster DNS, see [[kubedns-resolution-failure]].

## Connections

-   **Workflows**:
    -   [[mcjax-debugging-playbook]]
-   **Playbooks**:
    -   [[mcjax-debugging-principles]]
-   **Symptoms**:
    -   [[session-failing]]
-   **Concepts**:
    -   [[pathways]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[jax]]

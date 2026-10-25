---
title: KubeDNS
description: The default DNS engine in GKE which can become bottlenecked during high-concurrency
  multi-slice TPU workload startups.
type: concept
internal_only: false
tags: [tpu, pathways, mxla, datacenter, hardware_fault, scheduling]
created: 2026-07-27
updated: 2026-10-15
---

## Overview and details: multi-slice startup bottleneck

At multi-slice startup (Pathways or McJAX), hundreds of worker pods resolve head
node IPs, slice services and control plane endpoints at the same time. This
burst can bottleneck the centralized KubeDNS (or CoreDNS) pods. Workers then
fail to acquire slices even though the slices are active and provisioned, and
the job hangs.

Signatures:

-   Kueue: `"The workload was unable to acquire any slices."` or `"Failed to
    acquire slices despite active slices and running workers in multi-slice
    run."`
-   Worker pods: `169.254.169.254:53: i/o timeout` and
    `dial UDP: operation was canceled`.

## Connections

-   **Symptoms**:
    -   [[pathways-expected-topology-mismatch]]
    -   [[step-timebomb-watchdog]]

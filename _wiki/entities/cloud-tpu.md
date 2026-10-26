---
title: Google Cloud TPU
description: Hub page for TPU hardware generations, topologies, VM architecture, backends, and failure modes.
type: entity
internal_only: false
tags: [tpu, pathways, training, inference]
created: 2026-06-04
updated: 2026-10-15
---


# Google Cloud TPU

## Overview

Hub page for Google Cloud Tensor Processing Unit (TPU) hardware generations
(`TPU7x` Ironwood, `v6e` Trillium, `v5p`, `v5e`, `v4`), interconnect topologies,
TPU VM host architecture, execution backends ([[mcjax]] vs [[pathways]]), and
their associated debugging pages.

## Cross-Generation Hardware Specifications

| Specification   | TPU v4     | TPU v5e                    | TPU v5p           | TPU v6e (Trillium)         | TPU7x (Ironwood)    |
| :-------------- | :--------- | :------------------------- | :---------------- | :------------------------- | :------------------ |
| **Chips per Pod** | 4,096 | 256 | 8,960 (max slice 6,144) | 256 | 9,216 |
| **Interconnect Topology** | 3D mesh / 3D torus | 2D torus | 3D torus (`>=4x4x4`) | 2D torus | 3D torus |
| **TensorCores per Chip** | 2 (MegaCore) | 1 | 2 (MegaCore) | 1 | 2 (Dual-chiplet) |
| **MXU Array Size** | 4 x `128x128` / TC | 4 x `128x128` / TC | 4 x `128x128` / TC | 2 x `256x256` / TC | `256x256` |
| **SparseCores per Chip** | Embedding HW | 0 | 4 | 2 | 4 (2 per chiplet) |
| **Peak Compute / Chip (BF16)** | 275 TFLOPs | 197 TFLOPs | 459 TFLOPs | 918 TFLOPs | 2,307 TFLOPs |
| **Peak Compute / Chip (FP8 / Int8)** | 275 TFLOPs (Int8) | 393 TOPs (Int8) | 459 TFLOPs (FP8) | 1,836 TOPs (Int8) | 4,614 TFLOPs (FP8) |
| **HBM Capacity / Chip** | 32 GiB (unified) | 16 GB | 95 GiB (unified) | 32 GB | 192 GiB (96 GB/chiplet) |
| **HBM Bandwidth / Chip** | 1,200 GBps | 800 GiBps | 2,765 GBps | 1,638 GBps | 7,380 GBps |
| **Bidirectional ICI BW / Chip** | 6 ICI links | 400 GBps (4 ports) | 1,200 GBps | 800 GBps (4 ports) | 1,200 GBps (200 GBps/axis) |
| **DCN Bandwidth / Chip** | Host NIC | 25 Gbps (2x100G/host) | 50 Gbps (200G/host) | 100 Gbps (4x200G/host) | 100 Gbps |
| **Chips per Physical Host** | 4 | 8 | 4 | 8 | 4 |
| **Machine Types** | v4 TPU VM | `ct5lp-hightpu-{1t,4t,8t}` | `ct5p-hightpu-4t` | `ct6e-standard-{1t,4t,8t}` | `tpu7x-standard-4t` |
| **4-Chip VM vCPUs / RAM / NUMA** | 2 NUMA nodes | 112 vCPUs / 192 GB / 1 NUMA | 208 vCPUs / 448 GB / 2 NUMA | 180 vCPUs / 720 GB / 1 NUMA | 224 vCPUs / 960 GB / 2 NUMA |

## Architecture & Topology Details

-   **On-Chip Compute & Memory Hierarchy**:

    -   Dense matrix operations run on [[tensorcore]] Matrix Multiply Units
        (MXUs: `128x128` systolic arrays prior to v6e; `256x256` on `v6e` and
        `TPU7x`), while embeddings and sparse DMA transfers run on
        [[sparsecore]], driven on the host by [[libtpu]].

    -   **TPU7x (Ironwood) Dual-Chiplet Architecture**: Unlike `v4` and `v5p`
        (which fuse two TensorCores into a single logical MegaCore with unified
        HBM), each `TPU7x` chip comprises **two separate chiplets** connected by
        a die-to-die (D2D) link **6x faster than a 1D ICI link**. Each chiplet
        has 1 TensorCore, 2 SparseCores, and 96 GB dedicated HBM (192 GiB/chip
        total). JAX and PyTorch expose each `TPU7x` chip as **two devices** and
        add a 4th topology dimension to select chiplets. TensorFlow is not
        supported on `TPU7x`.

    -   **VMEM, Cmem, and Host PCIe / NUMA**: On-chip VMEM SRAM acts as a
        scratchpad for custom Pallas kernels (`v4` also provides Load-Store
        Common Memory `Cmem`). On 2-NUMA VMs (`v4`, `v5p`, `tpu7x`, and 8-chip
        `ct5lp-hightpu-8t` / `ct6e-standard-8t`), cross-NUMA CPU-TPU traffic
        incurs latency penalties; binding CPU-heavy scripts via
        `numactl --cpunodebind=0` avoids cross-NUMA bottlenecks.

-   **Physical Grouping, Cubes, and Topologies**:

    -   **3D Torus Generations (`v4`, `v5p`, `TPU7x`)**: Use 4-chip hosts and
        group 16 hosts (64 chips) into a `4x4x4` **cube** — the unit of shared
        power, cooling, copper intra-cube ICI, and the minimum fault-isolation
        unit on 3D-torus TPU pods (see Cluster Infrastructure,
        [[node-isolation-methodology]], [[superslicing-subslicing]]). Inter-cube ICI
        links traverse Optical Circuit Switches (OCS) with **ICI Resiliency**
        enabled by default on slices `>= 4x4x4`. Shapes where `2A=B=C` or
        `2A=2B=C` (`4x4x8`, `4x8x8`, `8x8x16`, `8x16x16`, `12x12x24`) support
        **Twisted Torus** links on `v4` and `v5p` for ~40%–70% higher bisection
        bandwidth. Note that `v4-<N>` slice names count TensorCores (2x chips),
        whereas newer generations specify chip topologies or chip counts.

    -   **2D Torus Generations (`v5e`, `v6e`)**: Scale up to `16x16` (256 chips)
        per Pod using 8-chip physical hosts. Multi-host training slices use
        half-host 4-chip VMs (`ct5lp-hightpu-4t`, `ct6e-standard-4t`, 1 NUMA
        node), whereas single-host inference can use sub-host 1-chip VMs (`1t`),
        4-chip VMs (`4t`), or full-host 8-chip VMs (`ct5lp-hightpu-8t`,
        `ct6e-standard-8t`). Multi-host inference (`> 8` chips) uses
        [[pathways]](`LeaderWorkerSet` + vLLM/JetStream) or Sax.

-   **Execution Backends**:

    -   Multi-controller [[mcjax]] runs a Python process on every TPU VM;
        single-controller [[pathways]] runs a single user controller (`main`),
        `pathways-proxy`, `pathways-rm`, `pathways-worker`, and a
        [[pathways-sidecar]] container. The backend determines which container
        holds compiler vs worker logs.

## Cloud TPU Regions and Zones (Including AI Zones)

| TPU Generation | North America Zones | Europe Zones | Asia Pacific & South America Zones |
| :----------- | :----------------- | :--------------- | :--------------------- |
| **TPU7x (Ironwood)** | `us-central1-ai1a` (AI zone), `us-central1-c` | — | — |
| **TPU v6e (Trillium)** | `us-central1-b`, `us-east1-d`, `us-east5-a`, `us-east5-b`, `us-south1-ai1b` (AI zone) | `europe-west4-a` | `asia-northeast1-b`, `southamerica-west1-a` |
| **TPU v5p** | `us-central1-a`, `us-east5-a` | `europe-west4-b` | — |
| **TPU v5e** | `us-central1-a`, `us-south1-a`, `us-west1-c`, `us-west4-a` | `europe-west4-b` | — |
| **TPU v4** | `us-central2-b` (manual quota approval required) | — | — |

## Common Failure Modes

-   **Pathways connection or proxy failures**: Usually stem from proxy
    readiness, DNS/ILB configuration, or downstream worker deaths
    (`[[proxy-connection-failed]]`, `[[ifrt-proxy-terminated]]`).

-   **Scheduling stalls**: JobSet stuck waiting for slice topology alignment or
    spare slices (`[[forever-suspended-jobset]]`,
    `[[pathways-expected-topology-mismatch]]`).

-   **`DATA_LOSS` pipe closed**: Downstream symptom of a worker pod crash,
    preemption, or OOM (`[[data-loss-pipe-closed]]`).

-   **Intra-slice ICI vs Cross-slice DCN hangs**: ICI link faults or sflag wait
    timeouts (`[[ici-errors]]`, `[[wait-timeout-on-sflag]]`) vs multi-slice DCN
    collective hangs (`[[megascale-hang]]`).

## Connections

-   **Playbooks**:
    -   [[mcjax-debugging-principles]]
    -   [[pathways-debugging-principles]]
    -   [[tpu-state-analysis-methodology]]
-   **Symptoms**:
    -   [[data-loss-pipe-closed]]
    -   [[forever-suspended-jobset]]
    -   [[megascale-hang]]
    -   [[proxy-connection-failed]]
    -   [[slow-step-time]]
    -   [[tpu-coordinator-oom]]
    -   [[wait-timeout-on-sflag]]
-   **Concepts**:
    -   [[ai-hypercomputer]]
    -   [[ici-errors]]
    -   [[megascale]]
    -   [[pathways]]
    -   [[superslicing-subslicing]]
-   **Entities**:
    -   [[libtpu]]
    -   [[mcjax]]
    -   [[sparsecore]]
    -   [[tensorcore]]
    -   [[xla]]

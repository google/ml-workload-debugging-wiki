---
title: ICI Errors
description: Inter-Chip Interconnect (ICI) failures, degradations, and stragglers
  affecting collective communication in TPU pods.
type: concept
internal_only: false
tags:
- tpu
- pathways
- hardware_fault
created: 2026-05-13
updated: 2026-10-15
---


## Definition

Inter-Chip Interconnect (ICI) errors refer to failures or degradations in the
high-speed, low-latency copper or optical links that connect TPU chips directly
to each other within a slice (`2D torus` on `v5e`/`v6e`; `3D mesh` or `3D torus`
across `4x4x4` cubes on `v4`/`v5p`/`TPU7x`).

## Architecture, Bandwidth & Resiliency

-   **Per-Generation ICI Specifications**:
    -   **TPU7x (Ironwood)**: 3D torus scaling up to 9,216 chips; **1,200 GBps**
        bidirectional ICI bandwidth per chip (`200 GBps` per axis), plus an
        on-package die-to-die (D2D) link between the two chiplets that is **6x
        faster than a 1D ICI link**.
    -   **TPU v6e (Trillium)**: 2D torus up to 256 chips (`16x16`); **800 GBps**
        bidirectional ICI bandwidth per chip across 4 ICI ports (`102.4 TB/s`
        Pod all-reduce, `3.2 TB/s` bisection bandwidth).
    -   **TPU v5p**: 3D torus (`>= 4x4x4` cube) up to 6,144 chips per slice
        (8,960 per Pod); **1,200 GBps** bidirectional ICI bandwidth per chip.
        Sub-cube slices (`< 4x4x4`) are 3D-connected without wrap-around torus
        links.
    -   **TPU v5e**: 2D torus up to 256 chips (`16x16`); **400 GBps**
        bidirectional ICI bandwidth per chip across 4 ICI ports (`51.2 TB/s` Pod
        all-reduce, `1.6 TB/s` bisection bandwidth).
    -   **TPU v4**: 3D mesh / 3D torus up to 4,096 chips with 6 ICI links per
        chip (`1.1 PB/s` Pod all-reduce, `24 TB/s` bisection bandwidth).
-   **Copper Intra-Cube vs Optical Inter-Cube (OCS) Links**:
    -   Within a `4x4x4` cube (64 chips, 16 hosts on `v4`, `v5p`, `TPU7x`), ICI
        connections use passive **copper links**.
    -   Between cubes, ICI connections traverse **optical links and Optical
        Circuit Switches (OCS)** (`[[superslicing-subslicing]]`).
-   **ICI Resiliency**:
    -   Enabled by default on `v4`, `v5p`, and `TPU7x` slices of one cube
        (`4x4x4`) or larger.
    -   Routes inter-cube ICI connections around OCS and optical link faults to
        improve slice scheduling availability, with the trade-off of temporary
        degradation in ICI performance when fault routing is active.
-   **Twisted Torus Topologies**:
    -   Supported on `v4` and `v5p` 3D slices where `2A = B = C` (`4x4x8`,
        `8x8x16`, `12x12x24` -> **~70%** higher bisection bandwidth) or
        `2A = 2B = C` (`4x8x8`, `8x16x16` -> **~40%** higher bisection
        bandwidth), converting an asymmetric torus into a symmetric network with
        shorter packet routes.

## Why it matters for debugging

ICI is critical for the collective communication operations (like `all-reduce`,
`all-gather`, and `reduce-scatter`) required by large-scale ML models. A failure
in an ICI link can cause the entire workload to stop responding or fail during
initialization, as all participating chips must synchronize.

## Common issues related to it

-   **Initialization Failures**: Manifests as `GRPC_ERROR START_SESSION failed`
    appearing across all VMs associated with the cube containing the bad link.
-   **Hangs**: Can lead to `Wait timeout on sflag` messages in [[tensorcore]]
    logs when a chip is waiting for data from a neighbor that cannot be reached
    or is sending data too slowly (`[[wait-timeout-on-sflag]]`).
-   **Performance Degradation & Stragglers**: Abnormally slow or degraded links
    (or active ICI resiliency rerouting) bottleneck distributed collective
    communications, causing severe straggler effects where all chips synchronize
    down to the slowest link rate (`[[slow-step-time]]`). Slow links can also
    trigger `ici_slow_signal` metrics.
-   **CRC Errors**: Cyclic Redundancy Check errors indicate data corruption on
    the link. Excessive CRC errors can trigger link retraining or host
    isolation/replacement.
-   **Link Retraining**: The system may attempt to retrain links to recover from
    transient errors, which can cause temporary pauses or performance drops.

## Identification and Isolation

-   Check the **subblock health** via the API or Google Cloud Console dashboard
    to identify links marked as bad (`[[ai-hypercomputer]]`).
-   Look for patterns of `GRPC_ERROR START_SESSION failed` affecting a full
    cube's worth of VMs.

## Connections

-   **Playbooks**:

    -   [[hardware-failure-modes]]

-   **Symptoms**:

    -   [[slow-step-time]]
    -   [[wait-timeout-on-sflag]]

-   **Concepts**:

    -   [[ai-hypercomputer]]
    -   [[multi-chip-hangs]]
    -   [[pathways-fault-tolerance]]
    -   [[superslicing-subslicing]]

-   **Entities**:

    -   [[cloud-tpu]]
    -   [[tensorcore]]

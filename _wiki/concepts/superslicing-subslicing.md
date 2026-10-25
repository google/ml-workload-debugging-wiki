---
title: "Superslicing and Subslicing"
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-05-13
updated: 2026-10-03
---

## Definition

**Superslicing** and **Subslicing** are advanced resource allocation and
scheduling strategies enabled by AI Hypercomputer to optimize the utilization,
goodput, and resilience of TPU clusters, especially in the presence of faulty
hardware.

## Superslicing details

A superslice spans N nodepools (cubes). Slice creation is decoupled from
hardware provisioning: nodepools are provisioned per cube, and the slice is
modeled separately so the OCS links to a cube can be reprogrammed (OCS mirrors
move).

*   Superslice creation or re-creation takes 70-100 s per slice, with no
    manual intervention. This replaces idle "Hot Spare" capacity.
*   Any slice size of the form $4i \times 4j \times 4k$ with $i \le j \le k$.
*   Stack mapping in AI Hypercomputer: `One nodepool = one cube` (4x4x4 = 16 VMs
    / 64 chips).
*   Failures isolate to one cube (intra-cube ICI, which can disable specific
    topologies) or to the OCS links between cubes (cross-cube, which degrades or
    kills the whole superslice).

## Subslicing

Runs a workload on less than one cube.

*   Uses the healthy machines in cubes that have broken machines, without
    waiting for repair. A scheduler can send small workloads to cubes with known
    bad machines and keep fully healthy cubes for superslices.
*   Supports small dev environments (e.g. 8 chips).
*   Lets elastic training scale down to fewer cubes while other workloads
    finish.

## Connections

-   **Concepts**:
    -   [[ai-hypercomputer]]

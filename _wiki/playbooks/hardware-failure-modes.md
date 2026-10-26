---
title: Hardware Failure Modes
type: playbook
internal_only: false
tags: [tpu, hardware_fault]
created: 2026-05-13
updated: 2026-10-03
---

## Overview

TPU hardware faults usually surface as software symptoms (hangs, timeouts, slice
failures). Only field-observed signatures are listed. Confirm one bad host
before a host isolation/replacement request.

## Failure Modes (field-observed signatures)

-   **HBM ECC uncorrectable:** "HBM ECC uncorrectable" in host logs; fatal abort
    or MXLA stall. See [[ecc-errors]], [[mxla-hardware-fault]].
-   **PCIe overload (Trillium / TPU v6e):** fails consistently after a few steps
    with `DEADLINE_EXCEEDED` and `SLICE_FAILURE_CHIP_DRIVER_ERROR`. Mitigation:
    `LIBTPU_INIT_ARGS='--xla_tpu_iova_dma_chunk_size_bytes=104857'` (smaller
    IOVA DMA chunks).
-   **OCS optical link:** `DEADLINE_EXCEEDED` on cross-cube traffic, degraded
    Superpod. See [[ocs-errors]], [[ici-errors]].
-   **SparseCore stall:** one straggler causes an MXLA stall; RapidEye reports
    SparseCore stragglers. See [[sparsecore]].
-   **Transient slow chip:** step time degrades and recovers after a machine
    restart.

Thermal throttling and voltage droops have no log signature here; check
temperature metrics on the dashboards if step time drifts.

## Connections

-   **Symptoms**:

    -   [[deadline-exceeded]]

-   **Concepts**:

    -   [[ecc-errors]]
    -   [[ici-errors]]
    -   [[known-issues]]
    -   [[ocs-errors]]

-   **Entities**:

    -   [[sparsecore]]

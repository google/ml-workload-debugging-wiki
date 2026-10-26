---
title: MXLA Hardware Fault
description: Multi-slice MXLA hang caused by an HBM ECC or PCIe fault on one worker deep in the topology.
type: symptom
internal_only: false
tags: [tpu, hardware_fault, hangs, training]
created: 2026-07-23
updated: 2026-10-03
---

# MXLA Hardware Fault

## Symptom signature

-   Multi-slice MXLA run (4, 16, 48 slices or more) hangs with no error code on
    worker 0; collective timeouts or missing barrier responses across slices.
-   Kernel/driver logs on one host: `HBM ECC uncorrectable error` or
    `Fatal TPU PCIe bus error`.
-   The faulty worker stops without a clean teardown signal to the coordinator.
-   Not this: `llo::CouldLeS32(0, bound)` + segfault is a software Shardy OOM,
    and a `RET_CHECK`/`FATAL` outranks earlier watchdog stalls; see
    [[gotchas-distributed-training]].

## Verification Steps

1.  Find the slice that never reached the collective barrier, then the worker
    pod that stopped heartbeating.
2.  Query Cloud Logging via [[trace-hbm-ecc-errors]]for that host.
3.  Check whether HBM ECC or PCIe error counters incremented.

## Root Cause and Playbook

Degraded HBM on one chip gives silent uncorrectable ECC reads. Isolate the host
for repair or node replacement with [[hardware-failure-modes]] and
[[node-isolation-methodology]].

## Connections

-   **Workflows**:
    -   [[mxla-hardware-triage]]
-   **Actions**:
    -   [[trace-hbm-ecc-errors]]
-   **Playbooks**:
    -   [[hardware-failure-modes]]
    -   [[node-isolation-methodology]]
-   **Symptoms**:
    -   [[megascale-hang]]
-   **Concepts**:
    -   [[ecc-errors]]

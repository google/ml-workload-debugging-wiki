---
title: ECC Errors
type: concept
internal_only: false
tags: [tpu, hardware_fault]
created: 2026-05-13
updated: 2026-10-03
---

## Overview and details: signatures and handling

-   Uncorrectable HBM ECC errors abort `libtpu` or panic the OS. Look for
    `"HBM ECC uncorrectable"` in host logs.
-   Don't filter only on "ECC": the same errors can appear as "parity error" or
    "uncorrectable error". Read the raw logs around the failure time.
-   A high rate of correctable errors can precede an uncorrectable failure.
-   Post-Package Repair (PPR) can map out bad memory after an uncorrectable
    error on some platforms.
-   If the error recurs on the same machine or chip, isolate/replace the machine
    (see [[node-isolation-methodology]]).

## Connections

-   **Playbooks**:
    -   [[hardware-failure-modes]]
    -   [[node-isolation-methodology]]

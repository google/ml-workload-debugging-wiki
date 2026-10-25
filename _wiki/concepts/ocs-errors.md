---
title: OCS Errors
type: concept
internal_only: false
tags: [tpu, hardware_fault]
created: 2026-05-13
updated: 2026-10-03
---

## Overview and failure details

-   Cross-cube collectives fail with `DEADLINE_EXCEEDED` (see
    [[deadline-exceeded]]).
-   Degraded Superpod: OCS links down or degraded means reduced bandwidth or
    unreachable cubes.
-   Mirror control errors: the mirrors that steer light fail from high voltage
    or mechanical faults.
-   High insertion loss causes link failures. Optical link diagnostics
    monitor it.

## Connections

-   **Playbooks**:
    -   [[hardware-failure-modes]]
-   **Symptoms**:
    -   [[deadline-exceeded]]
-   **Concepts**:
    -   [[ici-errors]]

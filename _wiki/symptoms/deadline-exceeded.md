---
title: Deadline Exceeded (Timeout)
type: symptom
internal_only: false
tags: [tpu, timeout]
created: 2026-06-11
updated: 2026-10-03
---

## Overview

`DEADLINE_EXCEEDED` alone does not identify the failing component; it is almost
always a downstream effect of a stall, slow node, or link issue elsewhere.

## Routing

-   Megascale / TPU workloads: [[megascale-hang]].

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[multi-chip-hang-investigation]]

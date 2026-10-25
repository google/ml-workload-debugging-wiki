---
title: Multi-Pod Exit Code Analysis
description: Methodology for analyzing exit codes across large-scale pod deployments
  to isolate outlier hardware and process failures.
type: concept
internal_only: false
tags: [tpu, training, triage, analysis]
created: 2026-07-23
updated: 2026-10-03
---

## Overview and details

In a large job (e.g. 1000+ pods), one worker crashes and the orchestrator
SIGTERMs the rest (exit 143), hiding the primary failure. Group all pods by exit
code and look for the minority code (e.g. 1 of 1024 pods with exit code 1). Map
that pod to its GKE node and GCE instance. If that host has recurring hardware
errors, cordon or replace the node via Repair & Replace
([[node-isolation-and-rr-api]]).

## Connections

-   **Playbooks**:
    -   [[multi-workload-triage]]
-   **Workflows**:
    -   [[node-isolation-and-rr-api]]
-   **Concepts**:
    -   [[exit-codes]]

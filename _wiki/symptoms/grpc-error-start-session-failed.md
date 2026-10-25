---
title: GRPC_ERROR START_SESSION failed
type: symptom
internal_only: false
tags: [tpu, hardware_fault]
created: 2026-06-11
updated: 2026-10-03
---


## Symptom signature

Ironwood, at session start:

```
MaybeInitializeSliceBuilder failed: INTERNAL: GRPC_ERROR error message: START_SESSION failed. Contact GCE with UUID: <support-id> if this is recurring. Max # of tries(=3) exhausted.
```

Discriminator: on **all VMs of one cube** = bad ICI link on the nodepool (Case
1); on **one VM of the same machine repeatedly across runs** = bad machine (Case
2).

## Potential Causes

In AI Hypercomputer (`[[ai-hypercomputer]]`) multi-slice GKE TPU environments,
if a workload attempts to schedule on a nodepool that has a bad ICI (Inter-Chip
Interconnect) link, this is a common resulting error message. Verify the
nodepool health and look for general ICI failure patterns.

## Triage Steps

-   **Case 1 (ICI)**: check AI Hypercomputer subblock health for the nodepool
    (bad links show as degraded infrastructure in the Google Cloud Console; see
    `[[ai-hypercomputer-infrastructure-health]]`). Use superslicing to avoid
    cubes with degraded links.
-   **Case 2 (bad machine)**: confirm the same GKE instance name fails
    repeatedly, then request an isolate for that machine.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Concepts**:
    -   [[ai-hypercomputer]]
    -   [[ai-hypercomputer-infrastructure-health]]
    -   [[superslicing-subslicing]]

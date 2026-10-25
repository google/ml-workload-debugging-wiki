---
title: Forever Suspended JobSet
type: symptom
internal_only: false
tags: [tpu, pathways, training, gke]
created: 2026-05-13
updated: 2026-10-15
---

## Symptom signature

JobSet stays `Suspended` and never creates pods, so no container logs exist.
Look in GKE events and the JobSet controller logs (`jobset-system` namespace)
for truncation, length-limit, or name-generation errors.

## Diagnosis

-   **Usual cause (JobSet name > 22 characters)**: JobSet appends replicated-job
    names (such as `pathways-worker`, `pathways-proxy`, `pathways-rm`) and slice
    / pod indices to child resource names, which must be valid DNS labels
    (`<= 63` characters). For Pathways workloads (including `gcluster` JobSets),
    a JobSet name longer than **22 characters** causes generated pod or headless
    service names to fail Kubernetes validation.
-   **Not this**: quota exhaustion or scheduling constraints; check Kueue
    (`ClusterQueue` / `Workload` conditions) and cluster events.

## Resolution

1.  Recreate the Pathways JobSet / workload with a name of **22 characters or
    fewer** (only lowercase letters, numbers, and hyphens).
2.  Still suspended: read `jobset-controller-manager` logs for other webhook or
    validation errors.

## Connections

-   **Playbooks**:
    -   [[pathways-debugging-principles]]
-   **Concepts**:
    -   [[jobset-yaml-structure]]
    -   [[pathways]]

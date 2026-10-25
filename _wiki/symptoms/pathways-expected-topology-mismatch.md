---
title: Pathways Expected Topology Mismatch
description: Occurs during Pathways worker initialization when the expected topology
  derived by the Pathways Resource Manager does not match the actual physical topology
  reported by the workers.
type: symptom
internal_only: false
tags: [tpu, pathways]
created: 2026-07-27
updated: 2026-07-27
---

# Pathways Expected Topology Mismatch

## Overview

This symptom occurs during Pathways worker initialization on GKE/TPU clusters
when the expected topology derived by the Pathways Resource Manager does not
match the actual physical topology reported by the workers. While McJAX bypasses
this strict topology check and compiles successfully (though potentially with
lower performance or a corrupted mesh), Pathways has a strict
expected-vs-physical topology check and will crash during startup.

## Log Signatures

The primary log signature is a precondition failure emitted by the Pathways
Resource Manager during worker initialization:

```
ResourceManagerDone: crashing worker due to failed precondition: FAILED_PRECONDITION: The newly added instance does not match with the expected instances; this is currently not allowed.
```

Additionally, if you check the environment variables of the worker pods, you
will notice that the following critical metadata variables are missing:

-   `TPU_ACCELERATOR_TYPE`
-   `TPU_CHIPS_PER_HOST_BOUNDS`
-   `TPU_HOST_BOUNDS`

## Verification Steps

1.  **Search Worker Logs**: Look for the signature
    `crashing worker due to failed precondition` or
    `The newly added instance does not match with the expected instances`.
2.  **Inspect Pod Environment**: Verify if the `TPU_ACCELERATOR_TYPE`
    environment variable is defined in the worker pod spec. If it is missing,
    the TPU runtime fails to verify the accelerator generation and falls back to
    an unwrapped topology (`wrap=false`), while the expected topology
    configuration is `wrap=true`.
3.  **Verify GKE Version**: Check if the cluster is running on a version older
    than `1.35.0-gke.2479000`.

## Playbooks & Root Causes

The root cause is that under Incremental Provisioning or SuperSlicing, GKE node
pools are provisioned dynamically without the `gke-tpu-topology` VM metadata
label. As a result, the `tpu-device-plugin` does not inject the required
environment variables.

### Recommended Fix: Upgrade GKE

Upgrade the GKE cluster to version **1.35.0-gke.2479000 or above**. This version
contains the fix to automatically inject `TPU_ACCELERATOR_TYPE` and
`TPU_TOPOLOGY_WRAP=true,true,true` (or equivalent flags for wrap status
configuration) to the worker pods.

### Workaround (Before GKE Upgrade)

If the cluster cannot be upgraded immediately, manually inject the missing
environment variables into the worker pod specs in the workload configuration:

```yaml
            env:
            - name: TPU_CHIPS_PER_HOST_BOUNDS
              value: "2,2,1" # Adjust for the specific node type (e.g. 2,2,1 for v5p, etc.)
            - name: TPU_ACCELERATOR_TYPE
              value: "tpu7x-512" # E.g., tpu7x-512 for TPU7x (Ironwood)
            - name: TPU_HOST_BOUNDS
              value: "2,4,8" # Adjust for the specific topology bounds
```

## Related Workflows

-   [[pathways-topology-mismatch-triage]]

## Connections

-   **Playbooks**:
    -   [[pathways-debugging-principles]]
-   **Concepts**:
    -   [[pathways]]
    -   [[pathways-cluster-readiness-checks]]

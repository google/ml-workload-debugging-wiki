---
title: JobSet YAML Structure
type: concept
internal_only: false
tags: [tpu, training, scheduling]
created: 2026-05-13
updated: 2026-10-03
---

## Overview: key fields

-   Set `restartPolicy: Never` in the pod template so failed pods keep their
    logs. The replicated job is usually named `slice-job` with
    `replicas: {num_slices}`.
-   Debug flags go in the container `env` or in `LIBTPU_INIT_ARGS`.

## Super-Slicing (Dynamic Slicing)

Required annotations under
`spec.replicatedJobs[0].template.metadata.annotations`:

-   `cloud.google.com/gke-tpu-slice-topology`: {topology} (e.g., 4x8x8)
-   `kueue.x-k8s.io/podset-required-topology`:
    `cloud.google.com/gce-topology-block`
-   `kueue.x-k8s.io/podset-slice-required-topology`:
    `cloud.google.com/gke-tpu-partition-{subblock_size}-id`

Recommended node affinity (skips excluded partitions):

```yaml
            affinity:
              nodeAffinity:
                requiredDuringSchedulingIgnoredDuringExecution:
                  nodeSelectorTerms:
                  - matchExpressions:
                    - key: cloud.google.com/gke-tpu-partition-{subblock_size}-state
                      operator: In
                      values: [HEALTHY, DEGRADED]
```

GKE Warden webhook and Kueue: do **not** add the
`cloud.google.com/gke-tpu-topology` label to `nodeSelector` yourself. Kueue adds
it when it admits the workload, and adding it yourself can conflict.

## Connections

-   **Playbooks**:
    -   [[multi-workload-triage]]

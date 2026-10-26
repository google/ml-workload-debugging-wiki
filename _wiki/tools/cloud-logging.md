---
title: Cloud Logging
type: tool
internal_only: false
tags: [tpu, training]
created: 2026-06-13
updated: 2026-10-03
---

## Overview

**Cloud Logging** (accessed via `gcloud logging` CLI) is the primary tool used
to retrieve and analyze application and system logs for TPU workloads running on
AI Hypercomputer.

## Usage

```bash
gcloud logging read 'resource.type="k8s_container" \
  AND resource.labels.project_id="<PROJECT_ID>" \
  AND resource.labels.cluster_name="<CLUSTER_NAME>" \
  AND resource.labels.namespace_name="default" \
  AND resource.labels.pod_name:"<WORKLOAD_NAME>-"' \
  --limit <LIMIT> \
  --project <PROJECT_ID>
```

## Troubleshooting Tips

-   **Don't filter by severity (e.g. `severity>=ERROR`) early.** JAX sflag
    timeouts and VM preemption events are logged at `INFO` or `WARNING`. Find
    the failure time and read the unfiltered stream around it (`AND timestamp >=
    "YYYY-MM-DDTHH:MM:SSZ" AND timestamp <= "YYYY-MM-DDTHH:MM:SSZ"`).
-   **Coordinator vs workers:** pod naming is covered in [[mcjax]] and
    [[gke-identifier-taxonomy]]; adjust the `pod_name` filter accordingly.
-   sflag timeouts: `textPayload:"Wait timeout on sflag"`.
-   GKE VM preemptions: `resource.type="gce_instance" AND
    jsonPayload.event_subtype="compute.instances.preempted"`.
-   **Pod events (evictions, kills, OOMs)** survive pod deletion in the events
    log:

    ```bash
    gcloud logging read 'logName:"/logs/events" \
      AND resource.type="k8s_pod" \
      AND resource.labels.pod_name:"<WORKLOAD_NAME>-"' \
      --project <PROJECT_ID> \
      --limit 500

    gcloud logging read 'logName:"/logs/events" \
      AND resource.type="k8s_pod" \
      AND resource.labels.pod_name:"<WORKLOAD_NAME>-" \
      AND (jsonPayload.reason="Evicted" OR jsonPayload.reason="Killing" OR jsonPayload.reason="OOMKilling")' \
      --project <PROJECT_ID>
    ```

    `jsonPayload.message` says why, e.g. `The node was low on resource:
    ephemeral-storage. ... jax-tpu-1 was using 38GB`.

## Connections

-   **Actions**:
    -   [[analyze-sflag-timeouts]]
    -   [[query-workload-logs]]
-   **Playbooks**:
    -   [[debugging-gotchas]]
    -   [[debugging-mindset]]

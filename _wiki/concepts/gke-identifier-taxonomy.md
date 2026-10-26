---
title: GKE/GCE TPU Naming Taxonomy
type: concept
internal_only: false
tags: [tpu, training, scheduling]
created: 2026-06-12
updated: 2026-10-20
---


## Overview

When debugging TPU workloads in GKE / AI Hypercomputer environments, multiple
layers of abstraction introduce different names/identifiers for the same
physical resource. Using the incorrect identifier in actions (especially
mutating ones) will cause API failures.

## Taxonomy Layers

| Identifier Type | Example Value | Context / Scope | Resolved Via | Used By (Actions) |
| :--------- | :---------------------------------------- | :---------- | :--------------------------- | :-------------------------- |
| **GCP Project ID / Project Number** | `my-gcp-project` / `123456789012` | GCP Project scope and numeric resource identifier. | `[[lookup-project-number]]` (`gcloud projects describe`) | Cloud Logging, GCE, and reservation queries |
| **JAX Task ID** | `slice3-task255` | JAX process runtime logic. | Application Logs / Stack traces | Software stack debugging |
| **GKE Pod Name** | `example-workload-worker-job0-1-12-gr64r` | Kubernetes Pod Resource. | Cloud Logging (startup/error logs) or `kubectl get pods` (if live) | GKE log queries, Pod inspection |
| **GKE Hostname** | `example-workload-worker-job0-1-12` | Pod hostname (host label). | Pod metadata / State logs | Log filtering |
| **GKE Node Name / GCE VM Name** | `gke-tpu-node-xxxx` | The underlying GCE VM instance. | `[[get-node-from-pod-logs]]` | `[[report-host-as-faulty]]` |

**Stable Machine Identifier**: gives the VM + physical host a stable ID that
survives VM-ID and GKE-instance-name changes, allowing tracking of recurring bad
machines across rotations. Without it, rotating VM names hide repeat failures on
the same host.

## Key Debugging Rule

Before calling any Gcloud compute VM-level API (such as `report-host-as-faulty`
or `taint-node`), you **must** map the GKE Pod/Hostname to the GKE Node Name
(`gke-tpu-...`). Do not pass the pod hostname to Gcloud compute commands.

## Connections

-   **Actions**:
    -   [[lookup-project-number]]
    -   [[get-node-from-pod-logs]]
    -   [[report-host-as-faulty]]

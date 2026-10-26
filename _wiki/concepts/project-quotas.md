---
title: Project Quotas
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-06-11
updated: 2026-09-15
---

## Definition

Project Quotas are the first layer of gatekeeping in the TPU provisioning stack.
They define the maximum amount of resources a specific Cloud Project is allowed
to consume. For TPU workloads, this includes not just the TPU chips themselves
(Primary Quota), but also essential supporting resources (Secondary Quotas).

### Primary Quota

-   **TPU Chips**: The number of chips of a specific type (e.g., v5e, v5p) that
    the project is allowed to use.

### Secondary Quotas

-   **Persistent Disk**: Required for storing data, model checkpoints, etc.
-   **IP Addresses**: Needed for the VMs to communicate.
-   **GCS Egress / Ingress**: Network bandwidth for reading from and writing to
    Google Cloud Storage.

## Why it matters for debugging

Understanding quotas is crucial because a failure at this layer prevents the
request from ever reaching the underlying job scheduler (cluster). If a request
fails immediately without appearing in cluster or GKE logs as a scheduling
failure, quota exhaustion is a primary suspect.

### The Spare Quota Misconception

A common misunderstanding among users is how to interpret "available" quota. For
example, if a project has a quota of 100 v5e chips and is currently using 10,
the Cloud Console shows 90 chips "available."

-   **What it means**: The project is *allowed* to request up to 90 more chips.
-   **What it does NOT mean**: It does not guarantee that 90 chips are
    physically available in the data center or in the specific Cloud TPU capacity pool.
    Stockouts can still occur even if quota is available.

### Runtime vs. Provisioning Quotas

It is critical to distinguish when a quota error occurs:

-   **Provisioning-Time Quota Errors**: Occur before the job starts. The job
    fails to schedule or provision resources (e.g. not enough TPU chips quota).
    These should be categorized as **Provisioning** failures.
-   **Runtime Quota Errors**: Occur during job execution (e.g., GCS egress
    bandwidth quota exceeded during TFDS dataset loading or check-pointing).
    These should be categorized as **SW** or **SW / Data** failures, NOT
    provisioning failures, as the resources were successfully provisioned and
    the job was already running.

## Operational Inspection

To check the current quota usage and limits for a project, use the `telemetry`
tool:

-   **Fetch Quota details**:

    ```bash
    gcloud compute project-info describe --project=<project>
    ```

## Common issues related to it

-   **Quota Exhaustion**: Attempting to provision more chips or secondary
    resources than allowed by the quota.
-   **Stockouts despite available quota**: Failing to provision because the
    physical capacity is not available in the target zone or pool, despite the
    project having sufficient quota.

## Connections

-   **Concepts**:
    -   [[gke-architecture]]
-   **Entities**:
    -   [[cloud-storage]]

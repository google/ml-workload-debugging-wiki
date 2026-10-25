---
title: Google Cloud Storage (GCS)
description: The object store backing training data input and checkpoints, and a recurring
  source of stalls that present as job hangs.
type: entity
internal_only: false
tags: [storage, training, timeout]
created: 2026-09-15
updated: 2026-10-03
---

## Overview

Every worker VM reads and writes GCS at once, so it fails on request rate and
latency, not per node. The result usually shows up as a stalled collective or a
step-time regression, not as a storage error on the worker that caused it.

## Technical Specifications

-   At 32k chips, thousands of VMs read the same bucket at once and can exceed
    default request-rate and egress limits. **Anywhere Cache** and
    **Hierarchical Namespace (HNS)** are the recommended mitigations at that
    scale.
-   GCS egress/ingress bandwidth is a project-level secondary quota, separate
    from TPU chip quota (see [[project-quotas]]).
-   [[orbax]] writes larger consolidated files (200MB default) instead of many
    small objects, which reduces API call pressure
    ([[configure-orbax-checkpointing]]).
-   Workers get GCS credentials from the GKE metadata server, so every read and
    write depends on it.

## Common Failure Modes

-   **HTTP 429 / 503 during data loading**: the canonical large-scale symptom,
    see [[gcs-429-503-data-loading]].
-   **Slow object deletion blocking a step**: deleting old checkpoint folders
    took over 10 minutes and stalled the step. Because the checkpoint barrier
    was a TPU collective, not a JAX distributed barrier, the stall became an
    MXLA timeout. See [[verify-training-barrier-api-usage]].
-   **Corrupt or partial checkpoints**: an interrupted write leaves a directory
    that blocks restore; the traceback appears only on JAX Process
    0.  See [[incomplete-checkpoint-restore]].
-   **Credential failures**: a crashlooping `gke-metadata-server` denies
    `storage.objects.list` and blocks restarts with permission errors (see
    [[gke-metadata-server-crashloop]]).
-   **Global host dispatch gap**: when *all* workers show an elevated host
    dispatch gap, not just one, suspect shared storage throughput first
    ([[libtpu-enqueue-timing-analysis]]).

## Connections

-   **Symptoms**:
    -   [[gcs-429-503-data-loading]]
    -   [[gke-metadata-server-crashloop]]
    -   [[incomplete-checkpoint-restore]]
    -   [[slow-step-time]]
-   **Concepts**:
    -   [[project-quotas]]
-   **Entities**:
    -   [[maxtext]]
-   **Tools**:
    -   [[orbax]]

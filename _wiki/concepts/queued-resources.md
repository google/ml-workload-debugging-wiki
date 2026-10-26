---
title: Queued Resources (QR)
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-05-13
updated: 2026-10-03
---

## Overview

> [!NOTE] QR does not exist on AI Hypercomputer (see [[ai-hypercomputer]]).

## State details

-   **`WAITING_FOR_RESOURCES`**: usually means the request is waiting in a
    capacity pool queue (stockout in the pool or zone).
-   **`PROVISIONING`**: a failure here points to a lower infrastructure
    provisioning layer.

-   **`FAILED`**:
    `gcloud compute tpus queued-resources describe <qr_name> --zone=<zone>`
    shows the error.

-   Requests can also expire if not filled by their end time.

## Connections

-   **Playbooks**:
    -   [[provisioning-failure-analysis]]
-   **Concepts**:
    -   [[ai-hypercomputer]]

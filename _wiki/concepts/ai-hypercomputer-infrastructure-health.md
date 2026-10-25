---
title: AI Hypercomputer Infrastructure Health
type: concept
internal_only: false
tags:
- tpu
- training
created: 2026-06-16
updated: 2026-09-29
---


## Overview

AI Hypercomputer Infrastructure Health concerns the status of the underlying
physical hardware and provisioning structures (reservations, subblocks) in AI
Hypercomputer environments. It defines how operators verify node and
interconnect health using labels and compute APIs.

## AI Hypercomputer vs. Legacy Hardware Failure Handling

In legacy (non-AI Hypercomputer) environments, a hardware error or communication
failure typically resulted in the teardown and full recreation of the entire
nodepool. AI Hypercomputer changes this paradigm by introducing Restart in Place
/ Repair & Replace (R&R) and granular fault localization:

-   **Auto Repair**: Instead of recreating the entire nodepool, AI Hypercomputer
    uses **Restart in Place** and **Repair & Replace (R&R)** to target and
    repair the specific machine or link, marking them bad at the subblock level.
-   **Bad Machine**: Instead of recreating the nodepool, AI Hypercomputer marks
    the corresponding GKE Node as `NotReady` (or `Unschedulable`) while keeping
    the rest of the nodepool and slice alive.
-   **Bad ICI (Inter-Chip Interconnect)**: AI Hypercomputer marks the subblock
    as bad infrastructure. The nodepool remains alive, but attempting to run
    workloads on it will cause communication timeouts or GRPC errors.
-   **Bad OCS (Optical Circuit Switch)**: AI Hypercomputer marks the subblock as
    bad infrastructure and the cube as `DEGRADED`, but the nodepool itself
    remains usable.

## Key Infrastructure Labels

-   `cloud.google.com/reservation-name`: The compute reservation hosting the
    node.
-   `cloud.google.com/reservation-subblocks`: The specific subblocks associated
    with the reservation.
-   `cloud.google.com/gke-tpu-partition-4x4x4-state`: The state of the GKE TPU
    partition.

## Verifying Subblock and Reservation States

To diagnose underlying infrastructure degradation:

### 1. Reservation Health

Check the reservation API. If `healthStatus` is `DEGRADED`, at least one
constituent node or block has failed.

### 2. Subblock Health

More granular than reservation level. Check `resource.healthInfo.healthStatus`
and look for maintenance activities under `upcomingGroupMaintenance` (such as
`ONGOING` or `PENDING` states).

### 3. Node Repair States

Verify node GKE states (`NotReady`, `Unschedulable`, or `google.com/tpu: 0`
allocatable) and check if the API shows `degradedHostCount > 0`.

### 4. Sub-block Health Counters (TPU7x)

On TPU7x GKE on AI Hypercomputer, one 4x4x4 cube is one reservation sub-block
and one GKE TPU node pool of 16 VMs. Listing
the sub-blocks is read-only and cheap (about 2.5 s). The
[[list-reservation-sub-blocks-health]] action wraps the list, and
[[describe-reservation-sub-blocks]] covers the per-sub-block describe. `<block>`
is the full block name, e.g. `<reservation>-block-0001`:

```bash
gcloud beta compute reservations sub-blocks list <reservation> --block-name=<block> \
  --zone=<zone> --project=<project> --format=json
```

-   **Join key**: the sub-block's `physicalTopology.subBlock` equals the node
    label `cloud.google.com/gce-topology-subblock`, which also equals
    `gke-tpu-partition-4x4x4-id`.

-   **Fields**:
    `healthInfo{healthStatus,degradedHostCount,degradedInfraCount,healthyHostCount}`,
    `inUseCount`, and
    `reservationSubBlockMaintenance{maintenanceOngoingCount,instanceMaintenanceOngoingCount,upcomingGroupMaintenance.maintenanceStatus}`.

-   **`degradedInfraCount` > 0 means an infrastructure fault**: when the nodes
    are Ready but the partition label is UNHEALTHY and the sub-block reports
    `degradedInfraCount` > 0, the fault is in the ICI/OCS infrastructure, not
    the machines. Leave the machines alone and open an infrastructure support
    case. On the studied cluster, one cube had all 16 machines RUNNING/SERVING,
    the partition UNHEALTHY and `degradedInfraCount=1` (single-incident
    observation).

-   **`healthyHostCount` is not live placement**: it reflects capacity
    assurance, not live VM placement. In a sub-block claimed 16/16 healthy while
    a host could not start a VM; a power cycle fixed it.

-   **4-hourly DEGRADED flaps**: a sub-block can flip HEALTHY to DEGRADED and
    back for about 1 min, roughly every 4 h. Take no action; debounce by
    requiring DEGRADED for more than 5 min (heuristic). The flaps were
    attributed to TPU7x (Ironwood) RoT attestation timeouts ; that is a
    Support analysis, not confirmed by Eng.

-   **Cadence (heuristic)**: the source suggests every 60 s together with the
    cached node list.

## Connections

-   **Concepts**:
    -   [[ai-hypercomputer]]
    -   [[repair-in-place]]
    -   [[superslicing-subslicing]]
-   **Actions**:
    -   [[describe-reservation-sub-blocks]]
    -   [[get-tpu-cube-partition-state]]
    -   [[list-reservation-sub-blocks-health]]

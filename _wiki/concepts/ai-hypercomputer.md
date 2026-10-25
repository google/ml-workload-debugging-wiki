---
title: "AI Hypercomputer"
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-05-13
updated: 2026-10-03
---

## Definition

AI Hypercomputer represents a paradigm shift in how TPU resources are managed
and provisioned in GKE clusters, moving away from traditional methods like
Queued Resources (QR) and Cluster Lifecycle Handler. It provides a more direct and persistent
mapping between GKE instances and the underlying physical hardware.

## Debugging Implications

-   **GKE instance == persistent machine.** A GKE instance stays linked to the
    same physical machine and is never reused for a different one. You can debug
    with GKE instance names directly, without tracing down to the physical
    machine name, and you can track per-instance health history (client-side
    "bad node detectors" or machine schedulers).
-   **Restart in Place / Repair & Replace (R&R)** replaces nodepool recreation.
    Only the bad machine or bad infra (an OCS switch, or an ICI link in the
    cube) goes out of service and is repaired (human fix, reboot, cable work);
    the nodepool stays linked to the same machines. Schedulers must avoid cubes
    or subcubes marked with bad machines or bad infra (OCS/ICI).
-   **Provisioning**: nodepools are provisioned per cube. Superslicing maps the
    OCS to connect cubes into a slice; this is a fixed-cost operation.

## What Doesn't Translate From Legacy Provisioning

-   Cluster Lifecycle Handler and Queued Resources (QR) are gone.
-   For small dev environments (e.g. 8 chips) that used CLH, use PW Serverless
    Mode with Subslicing (VM-like or Jupyter experience) or Quick Subslicing
    single-command workloads.

## Infrastructure Health

See [[ai-hypercomputer-infrastructure-health]] for labels, reservation/subblock
health and node repair states.

## Provisioning Failures

Provisioning skips QR and maps directly to Compute reservations through Queue
Manager Service (Quota Management Service) and Instance Manager. Symptoms: nodepool stuck in
Creating/Error, nodes permanently `NotReady`, or `ZONE_RESOURCE_POOL_EXHAUSTED`
(ZRPE).

-   **Quota Management Service (scheduler) INFEASIBLE**: no capacity matches the topology constraints
    in the reservation pool.
-   **Instance Manager INFEASIBLE**: physical allocation failed at the low-level
    host management layer.

## Connections

-   **Workflows**:
    -   [[ai-hypercomputer-investigation-playbook]]
    -   [[shared-cluster-usage]]
-   **Playbooks**:
    -   [[provisioning-failure-analysis]]
-   **Symptoms**:
    -   [[grpc-error-start-session-failed]]
-   **Concepts**:
    -   [[queued-resources]]
    -   [[repair-in-place]]
    -   [[superslicing-subslicing]]

---
title: Node Isolation and Repair & Replace (R&R) Methodology
description: Techniques, safety rules, and validation steps for isolating and decommissioning
  faulty TPU/GPU nodes across GKE and GCE.
type: playbook
internal_only: false
tags:
- tpu
- hardware_fault
created: 2026-06-11
updated: 2026-10-21
contradictions:
- tpu7x-cube-machine-health.md
---


## Definition

**Node Isolation and Repair & Replace (R&R) Methodology** describes the
techniques, safety rules, and validation steps used to isolate and decommission
faulty TPU or GPU nodes from the active serving pool using Kubernetes taints and
the Repair & Replace (R&R) API.

Node isolation operates at the **GCE / Hardware Layer** (Layer 4) of the
[[layered-debugging-framework]]. It should only be triggered after
systematically ruling out root causes at higher layers of the stack.

## Tainting vs. Host Replacement (R&R)

-   **Tainting (GKE)**: A temporary/investigative measure. Prevents new
    workloads from scheduling on a node (using `NoSchedule` effect) while
    existing pods continue or are drained manually. Taints can be persistent
    (applied on the node pool) or temporary (using `kubectl taint`).
-   **Host Replacement (R&R)**: A permanent measure. Triggers hardware eviction,
    repair, and replacement workflows to decommission faulty physical hosts.

## Isolation Mechanisms by Platform

Choosing the correct isolation mechanism depends on the platform and environment
where the ML workload is running:

### 1. GKE-Managed TPUs (Standard Cloud Deployment)

*   **Method**: **R&R API for GKE** (Declarative).
*   **Action**: Apply the `cloud.google.com/fault-behavior` node label to
    trigger GKE auto-repair.
*   **Command**: Use the [[isolate-node-via-rr-label]] action:

    ```bash
    kubectl label nodes ${NODE_NAME} cloud.google.com/fault-behavior=SDC
    ```
*   **Important**: Do **not** use GCE instance APIs (like
    `report-host-as-faulty`) for GKE-managed TPUs. GKE must remain the source of
    truth for node lifecycle state.

### 2. GCE GPU VMs (Direct Compute Engine)

*   **Method**: **GCE Instance API** (Imperative).
*   **Action**: Call the `report-host-as-faulty` API on the VM instance.
*   **Command**: Use the [[report-host-as-faulty]] action:

    ```bash
    gcloud compute instances report-host-as-faulty ${VM_NAME} --zone=${ZONE} --fault-reasons behavior=SDC,description="GPU fault" --disruption-schedule=IMMEDIATE
    ```

## Pre-Isolation Verification Checklist

To avoid unnecessary capacity loss and availability impact, apply the
[[layered-debugging-framework]] to verify:

1.  **Clear Hardware Logs (Hardware Layer)**: Seek clear error codes
    (`UNRECOVERABLE_ERROR`, `cSSD media error`, etc.) in [[cloud-logging]].
2.  **Systemic vs. Isolated (Layer Disjointness)**: Confirm the failure points
    exclusively to the host/hardware and are not a cascading consequence of
    network issues or Kubernetes evictions at the Workload/Provisioning layers.
3.  **Software Debugging First (Application Layer)**: Rule out Application Layer
    issues such as async checkpointing, JAX/driver version mismatches, or user
    configuration bugs.
4.  **Pattern of Failure**: Wait for a repeat offender pattern across multiple
    run attempts.
5.  **Autopilot checks**: Verify if the system's automated checkers have already
    picked up the fault.

## Connections

-   **Workflows**:
    -   [[node-isolation-and-rr-api]]
-   **Actions**:
    -   [[isolate-node-via-rr-label]]
-   **Playbooks**:

    -   [[layered-debugging-framework]]
    
-   **Concepts**:

    -   [[repair-in-place]]

-   **Tools**:

    -   [[cloud-logging]]

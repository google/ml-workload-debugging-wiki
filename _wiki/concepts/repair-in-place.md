---
title: Restart in Place and Repair & Replace (R&R)
type: concept
internal_only: false
tags:
- tpu
- hardware_fault
created: 2026-06-11
updated: 2026-10-03
---


## Definition

**Restart in Place** and **Repair & Replace (R&R)** are capacity management
workflows in AI Hypercomputer that replace the legacy workflow of recreating
entire nodepools to resolve hardware issues. Instead of replacing the entire
slice, these workflows focus on restarting or repairing specific components
(e.g., rebooting a host, replacing a single faulty host, or adjusting a cable)
while keeping the rest of the slice intact.

## Legacy vs. AI Hypercomputer Comparison

| Step | Legacy Workflow | AI Hypercomputer Workflow (Restart in Place / R&R) |
| :------------ | :------------------------- | :------------------------------ |
| **Detection** | Nodepool auto-repair kicks in OR User drains slice. | Node becomes `NotReady` OR User notices issue. |
| **Action** | Full nodepool recreation (heavy-handed, slow). | Affiliated bad machine or specific infra (OCS/ICI) unavailable. |
| **Repair** | New machines provisioned. | Attempt to restart or repair specific component (reboot, cable adjustment, etc.). |
| **Outcome** | Whole slice replaced. | Slice repaired without replacing healthy components. |

## Labeling Bad Nodes (Repair & Replace)

If a bad node is identified and needs to be marked for repair or excluded from
scheduler selection, label it with a fault behavior to trigger **Repair &
Replace (R&R)**:

-   **Command**: `kubectl label nodes ${node_name}
    cloud.google.com/fault-behavior=${fault_behavior}`
-   **Fault Behaviors**: `performance`, `SDC`, `unspecified`.
-   **Annotation requirement**: Always provide a detailed reason description
    (e.g., referencing an issue ID) to prevent recidivist machines and aid
    hardware triage.

## Restart in Place VM System Events

-   **Restart in Place (`RESTART_IN_PLACE`)**: After a host failure, the VM goes
    to `REPAIRING` and GCE logs `compute.instances.hostError` and
    `restartInPlaceAttempting`, then `successfulRestartInPlace` when the VM is
    back. The instance ID survives a restart in place, whereas a MIG recreate
    issues a new ID. Read these events with [[query-gce-system-events-by-zone]].
    The workload-scoped [[query-vm-termination-events]] action does not match
    the `restartInPlace*` methods.

-   **Silent flappers (single-cluster observation)**: Successful restarts in
    place can hide a recidivist. On the one cluster studied, a host had 3
    `hostError` events in 3 days; each restart in place succeeded without
    permanently taking the host out of service. It became a candidate for R&R.
    Use [[detect-flapping-machines]].

## Connections

-   **Playbooks**:
    -   [[node-isolation-methodology]]
-   **Concepts**:
    -   [[ai-hypercomputer]]
-   **Actions**:
    -   [[detect-flapping-machines]]
    -   [[query-disruption-events]]
    -   [[query-gce-system-events-by-zone]]

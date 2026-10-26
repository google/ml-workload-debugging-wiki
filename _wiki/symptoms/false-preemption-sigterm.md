---
title: False Preemption Signal (SIGTERM reported as preemption)
type: symptom
internal_only: false
tags: [tpu, pathways, training, triage, scheduling]
created: 2026-09-02
updated: 2026-10-15
---

# False Preemption Signal (SIGTERM reported as preemption)

## Symptom signature

The Pathways resource manager (RM) reports *any* `SIGTERM` as a preemption. All
of these appear for an ordinary SIGTERM and are **not** evidence of preemption:

```
Got SIGTERM signal, forwarding to Pathways as a preemption notification
Resource manager is being preempted. Shutting down at ...
exit_type: EXIT_TYPE_QUICKLY_DURING_PREEMPTION
reason: "Received SIGTERM in cloud environment, forwarding as a preemption notification
         with death time constructed from command-line argument."
Removing job ... because: Received SIGTERM in cloud environment, forwarding as a preemption notification...
```

Read the wording as "something sent SIGTERM", nothing more. The project audit
log usually holds dozens of unrelated preemptions, which reinforces the wrong
reading.

## Triage

Both checks must pass before recording a preemption.

1.  **Ordering.** Real preemption: infrastructure signal first, then the app
    shuts down. Artifact: the app exits on its own, kubelet stops the sidecars,
    then the RM prints preemption wording. Example (artifact):

    ```
    08:50:45  jax-tpu   XPK End ... EXIT_CODE=1        <- application already gone
    08:50:46  kubelet   Killing: Stopping container pathways-rm / pathways-proxy
    08:51:16  pathways-rm  "Resource manager is being preempted"
    ```

    Kubelet stopping only the *sidecars* means the main container had already
    exited.

2.  **Node-scoped audit.** Require a `compute.instances.preempted` event on one
    of **this workload's own nodes**: [[resolve-workload-nodes]], then
    [[query-vm-termination-events]] with its `node_filter_clause`. One 2-hour
    window: node-scoped **0** events; unscoped **111** (76 `preempted`, 32
    `repair.recreateInstance`, 3 `terminateOnHostMaintenance`), all from other
    workloads.

3.  **Eviction check.** Kubelet eviction also sends SIGTERM and nothing in the
    signal tells it apart. Before writing `VM termination`, `SIGTERM` or
    `preempted` as a cause:

    ```bash
    kubectl get events --field-selector reason=Evicted
    # or, in Cloud Logging
    "Evicted" OR "ephemeral-storage" OR "DiskPressure" OR "The node was low on resource"
    ```

    Eviction is node resource pressure (often ephemeral storage). An evicted pod
    dies too fast to print an exit code, so a missing exit code plus SIGTERM
    wording points at eviction ([[missing-exit-codes-eviction]]).

## Routing

-   App died first: the wording is noise; find what killed the app. Common real
    senders: kubelet stopping sidecars after the main container exits
    ([[sigill-worker-kill]]), JobSet failure-policy teardown, manual deletion.
-   Eviction found: [[missing-exit-codes-eviction]].
-   Incident: an eval run reported `vm termination (sigterm)` for a workload
    whose cause was `Evicted due to low ephemeral storage`.

## Connections

-   **Actions**:
    -   [[query-vm-termination-events]]
    -   [[resolve-workload-nodes]]
-   **Playbooks**:
    -   [[workload-disruption-orchestration]]
-   **Symptoms**:
    -   [[missing-exit-codes-eviction]]
    -   [[sigill-worker-kill]]

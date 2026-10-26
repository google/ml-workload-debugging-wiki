---
title: Missing Exit Codes due to Eviction
type: symptom
internal_only: false
tags: [tpu, training]
created: 2026-06-13
updated: 2026-10-03
---

## Symptom signature

-   Far fewer exit codes in `exit_codes.txt` than pods in the slice.
-   Pod termination events exist but no exit code output.

Kubelet eviction under heavy ephemeral-disk pressure kills the container at
once, before the wrapper prints its exit status. Missing exit codes point to
eviction or node failure, not an "Unknown" failure.

## Playbook

-   Check K8s events for `Evicted` on the pod IDs with no exit code.
-   Get the exit code from GKE:
    -   `kubectl get pods <pod-name>
        -o=jsonpath='{.status.containerStatuses[*].state.terminated.exitCode}'`
    -   Cloud Logging GKE events
        (`resource.type=k8s_container AND logName=.../events`) or the pod's
        `lastState.terminated.exitCode`.
-   See Gotcha 7 in [[gotchas-kubernetes-gke]].

## Connections

-   **Playbooks**:
    -   [[gotchas-kubernetes-gke]]
-   **Concepts**:
    -   [[exit-codes]]

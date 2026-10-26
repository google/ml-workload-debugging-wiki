---
title: GKE Metadata Server Crashloop
type: symptom
internal_only: false
tags:
- tpu
- training
created: 2026-06-13
updated: 2026-10-03
---

## Symptom signature

Workload fails to start/restart with auth or permission errors on GCS. Workload
container logs:

```
google.auth.exceptions.TransportError: Failed to retrieve credentials from the Google Metadata Server.
...
MaybeInitializeSliceBuilder failed: INTERNAL: storage.objects.list access denied for bucket <bucket-name>.
```

The `gke-metadata-server` pod (in `kube-system` or the workload namespace) is in
`CrashLoopBackOff`:

```
kubectl get pods -n kube-system | grep metadata
gke-metadata-server-xxxxx   0/1     CrashLoopBackOff   15         57m
```

## Causes

-   Reused nodes (cordoned/tainted, then readmitted) with corrupt software state
    or images stop the metadata server agent from starting.
-   Underlying GKE VM configuration issues.

## Triage and Mitigation

1.  Confirm with `kubectl get pods -n kube-system` (or a Cloud Logging query).
2.  Find the nodes hosting the crashlooping pods.
3.  Isolate those nodes so they are replaced with clean images:
    [[isolate-node-via-rr-label]].

## Connections

-   **Concepts**:
    -   [[gke-architecture]]

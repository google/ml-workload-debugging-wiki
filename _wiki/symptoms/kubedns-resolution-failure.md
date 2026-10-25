---
title: KubeDNS Resolution Failure and Timeouts
description: Symptom signature, log messages, and diagnostic steps for KubeDNS resolution
  failures and timeouts on GKE.
type: symptom
internal_only: false
tags:
- timeout
- triage
- tpu
- pathways
created: 2026-07-23
updated: 2026-10-15
---

# KubeDNS Resolution Failure and Timeouts

## Summary

**When it bites:** multi-slice GKE TPU startup (Pathways or McJAX). Hundreds of
workers resolve head-node and slice services at the same time, the central
KubeDNS pods saturate, and the Kueue slice controller fails slice acquisition
even though the TPU VMs and slices are up.

## Log Signatures & Error Messages

1.  **Slice Controller Acquisition Failures**:

    ```text
    "The workload was unable to acquire any slices."
    "Failed to acquire slices despite active slices and running workers in multi-slice run."
    ```

2.  **Metadata Server & DNS Port Timeouts**:

    ```text
    169.254.169.254:53: i/o timeout
    ```

    *Query filter in Cloud Logging*:

    ```text
    jsonPayload.message=~"169.254.169.254:53: i/o timeout"
    resource.labels.container_name=("kubedns" OR "node-cache")
    resource.labels.namespace_name="kube-system"
    ```

3.  **UDP Dial Cancellation**:

    ```text
    dial tcp: lookup <service> on <dns-ip>:53: dial UDP <dns-ip>:53: operation was canceled
    ```

4.  **Node Resolv.conf Lock Conflict**:

    ```text
    ln: failed to create symbolic link '/etc/resolv.conf': Device or resource busy
    ```

5.  **Application Host Resolution Errors**:

    ```text
    name resolution failure
    host not found
    connection timed out
    ```

## Discriminator

-   `.cluster.local` lookups fail while the TPU VMs are healthy and the slices
    are active = DNS bottleneck, not capacity.
-   Pods with `hostNetwork: true` bypass KubeDNS and use the metadata server
    (`169.254.169.254`): debug those as GCE DNS issues.
-   On C4/C3D node pools, bursts to `169.254.169.254:53` time out on their own
    (mitigation in the playbook).

## Routing

Triage tree, mitigations and limits: [[kubedns-troubleshooting]]. Automated SOP:
[[remediate-kubedns-resolution-failure]].

## Connections

-   **Workflows**:
    -   [[kubedns-troubleshooting-workflow]]
    -   [[remediate-kubedns-resolution-failure]]
-   **Actions**:
    -   [[check-node-dns-resolv]]
    -   [[enable-coredns-nodelocaldns]]
    -   [[query-kubedns-logs]]
    -   [[scale-dns-pod-replicas]]
-   **Playbooks**:
    -   [[gotchas-kubernetes-gke]]
    -   [[kubedns-troubleshooting]]
-   **Symptoms**:
    -   [[deadline-exceeded]]
-   **Concepts**:
    -   [[gke-architecture]]
    -   [[kubedns]]
    -   [[kueue-tq-scheduling]]

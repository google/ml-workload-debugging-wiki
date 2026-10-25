---
title: KubeDNS and Cluster DNS Troubleshooting Playbook
description: Comprehensive guide for diagnosing KubeDNS, CoreDNS, NodeLocalDNS, and
  DNS resolution issues on GKE clusters.
type: playbook
internal_only: false
tags: [timeout, triage]
created: 2026-07-23
updated: 2026-10-03
---

# KubeDNS and Cluster DNS Troubleshooting Playbook

## Overview

In GKE clusters, DNS service discovery is provided by `kube-dns` (or Cloud DNS).
Pods resolve cluster-internal services (`*.cluster.local`), custom stub domains,
and external endpoints via the `kube-dns` Service ClusterIP. Starting in GKE
1.36+, Enhanced KubeDNS (CoreDNS engine) replaces legacy `dnsmasq`-based
implementations, while NodeLocalDNS (enabled by default for GKE 1.34.1+ Standard
clusters and all Autopilot clusters) caches DNS queries on each node.

Because the `kube-dns-autoscaler` scales replica count based on node/core counts
rather than QPS, high-traffic clusters may experience DNS query timeouts or
latency spikes if KubeDNS or local conntrack tables become overloaded.

## Diagnostic Steps & Triage Flow

### 1. Engine & Feature Identification

-   **DNS Provider**: Confirm whether the cluster uses KubeDNS, Cloud DNS, or a
    custom CoreDNS deployment via GKE cluster details or checking for
    `kube-system/kube-dns`.
-   **KubeDNS Engine**: In GKE 1.36+, `kube-dns` runs a single `coredns`
    container per pod. In GKE 1.35 and earlier, legacy `kube-dns` uses a
    4-container pod (including `dnsmasq` and `kube-dns`).
-   **NodeLocalDNS**: Check if NodeLocal DNSCache is enabled (`gcloud container
    clusters describe <cluster>
    --format="value(addonsConfig.nodeLocalDnsConfig.enabled)"`).

### 2. Verify Pod Configuration

-   **hostNetwork Pods**: Pods with `hostNetwork: true` bypass KubeDNS and
    inherit the host `/etc/resolv.conf` (pointing to GCE Metadata Server
    `169.254.169.254`). Debug hostNetwork DNS issues as GCE/GCP DNS issues.
-   **Pod `/etc/resolv.conf`**: Exec into the impacted pod
    (`kubectl exec -it <pod> -- cat /etc/resolv.conf`) and confirm `nameserver`
    is the NodeLocalDNS cache `169.254.20.10` when NodeLocalDNS is on
    ([[check-node-dns-resolv]]), else the `kube-dns` Service ClusterIP
    (`kubectl get svc -n kube-system kube-dns`).

### 3. Step-by-Step Query Isolation

From a test pod, execute `dig` or `nslookup` with a trailing dot (`.`) to
isolate the failure domain:

1.  **Cluster Local (`dig kubernetes.default.svc.cluster.local.`)**:
    -   Tests Pod -> KubeDNS engine. If this fails, KubeDNS pods, service
        endpoints, or node-to-pod network policies are failing.
2.  **Stub Domains / Custom Resolvers (`dig test.mycompany.local.`)**:
    -   Tests KubeDNS -> Custom Upstream DNS. If this fails, verify the
        `kube-dns` ConfigMap (`stubDomains` or `upstreamNameservers`) and
        TCP/UDP port 53 reachability to upstream DNS servers.
3.  **External Domains (`dig google.com.`)**:
    -   Tests KubeDNS -> Metadata Server (`169.254.169.254:53`) -> Cloud DNS ->
        Public Internet.
4.  **Cloud DNS Private Zones (`dig app.myzone.internal.`)**:
    -   Tests KubeDNS -> Metadata Server -> Cloud DNS Private Zone binding on
        the VPC network.

### 4. Advanced Triage & Metrics

-   **Individual KubeDNS Pod Probes**: Fetch endpoints
    (`kubectl get ep -n kube-system kube-dns`) and probe each pod IP directly
    (`dig @<pod-ip> kubernetes.default.svc.cluster.local.`). If only specific
    pods fail, isolate candidate nodes for network issues.
-   **Conntrack Errors**: Inspect CNI metrics for `insert_failed` or `dropped`
    conntrack errors caused by race conditions or hairpinning (common when
    Intranode Visibility is enabled).
-   **Query Logging**: Enable query logging by configuring the `log` plugin in
    CoreDNS (GKE 1.36+) or `-q` flag in `dnsmasq` container (legacy GKE <=
    1.35).

## Mitigations & Known Issues

### Mitigations

-   **Enable NodeLocal DNSCache**: Caches queries locally on each node,
    bypassing conntrack race conditions and reducing QPS load on `kube-dns`
    pods.
-   **Append Trailing Dot**: Use FQDNs ending with a dot (e.g., `google.com.`)
    in application code or lower `ndots` (e.g. `ndots:2`) in pod `dnsConfig` to
    avoid costly search domain expansions.
-   **Scale KubeDNS Replicas**: Edit `kube-dns-autoscaler` (or
    `coredns-autoscaler`) ConfigMap to lower `nodesPerReplica` or increase
    minimum replica count. For multi-slice TPU startup, the recommended minimum
    is 1 DNS pod per 16 TPU worker nodes ([[scale-dns-pod-replicas]]).
-   **C4/C3D Node Mitigation**: For DNS timeouts on C4, C3D, C4A, or C4D VM node
    pools during burst queries to `169.254.169.254`, create a Cloud DNS Inbound
    Server Policy and configure `upstreamNameservers` in `kube-dns` ConfigMap
    with the inbound policy IP.

### Limits & Constraints

-   **Headless Service Endpoints**: Legacy KubeDNS supports up to 1000 endpoints
    per headless service. Enhanced KubeDNS (CoreDNS) supports up to 4090 IPv4 /
    ~2340 IPv6 addresses.
-   **Upstream Connection Limits**: CoreDNS instances enforce a limit of 1500
    concurrent upstream connections to Cloud DNS.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[gotchas-kubernetes-gke]]
-   **Symptoms**:
    -   [[kubedns-resolution-failure]]
-   **Concepts**:
    -   [[gke-architecture]]

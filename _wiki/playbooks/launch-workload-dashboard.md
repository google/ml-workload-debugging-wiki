---
title: Launch Workload Dashboard
description: Configure and launch the local GKE Capacity & Triage Dashboard scoped to a workload, cluster, or project.
type: playbook
internal_only: false
tags: [monitoring, triage, capacity]
created: 2026-09-16
updated: 2026-10-03
---

# Launch Workload Dashboard

## Overview

`/dashboard` runs the `[[launch-workload-dashboard]]` workflow: it finds the
cluster, adds it to `_dashboard/clusters_config.json`, checks the local daemon,
starts a scan, and returns a deep-linked URL.

### State and scan memory

-   **Data dir**: state (`clusters_config.json`, cluster cache, trackers, scan
    memory) lives in `~/.cache/debug_ml_workloads_wiki/dashboard` and logs in
    `~/.cache/debug_ml_workloads_wiki/logs` (override with `DASHBOARD_DATA_DIR`
    and `PORTABLE_LOG_DIR`).
-   **No-access clusters**: 401/403 from the API server shows `no access` and
    the cluster is skipped on periodic scans. **Scan Now** re-checks all
    remembered clusters. If your own gcloud login expired, nothing is
    remembered: clusters show `Inaccessible` with a `gcloud auth login` hint.
-   **Missing reservations**: not re-queried for 24h; **Scan Now** clears this.
-   **AI triage backend**: the daemon triages with the harness that launched it,
    even if Antigravity is also installed. The launcher reports the running
    server's backend as `agent_backend` and sets `agent_backend_warning` when a
    reused server uses a different one.
-   **Quota warnings**: "authenticated using end user credentials" in the log:
    run `gcloud auth application-default set-quota-project <project_id>`.

## Procedure

1.  `[[gcloud-login]]`: checks gcloud auth and ADC; refreshes with
    `gcloud auth login --update-adc` if expired or blocked by CAA.
2.  `[[validate-workload]]`: reads Cloud Audit and `InitGoogle()` logs; extracts
    `cluster_name`, `project_id`, `location`.
3.  `[[resolve-cluster-metadata]]`: finds `generation` (e.g. `tpu7x`, `v6e`,
    `v5p`) and the `reservation` from node pool reservation affinities.
4.  `[[configure-and-launch-dashboard]]`:
    -   Adds clusters to `_dashboard/clusters_config.json` (idempotent).
    -   Checks `http://localhost:8080/api/health`; if down, starts
        `_dashboard/server.py --port 8080 --no-browser`.
    -   Sends `POST /api/scan`.
    -   URLs: workload
        `http://localhost:8080/?cluster=<cluster>&workload=<workload_id>&gen=<gen>`;
        cluster `http://localhost:8080/?cluster=<cluster>&gen=<gen>`; project
        `http://localhost:8080/?project=<project_id>`.
    -   Reports `server_status: UNRESPONSIVE` if an unresponsive server holds
        the port, instead of starting a second server that cannot bind.
5.  `[[verify-dashboard-url]]`:
    -   `socket.getfqdn()` can return a name that does not resolve (e.g.
        `*.local` on a laptop); the launcher falls back to `localhost`.
    -   Probes the page and `/api/health` on the URL's host while the scan runs;
        reports `url_status` (`VERIFIED`, `DEGRADED`, `UNRESOLVABLE`, `DOWN`,
        `UNRESPONSIVE`) and `url_probes` latency stats.
    -   Give the URL to the user only when `VERIFIED`, or `DEGRADED` with a
        latency caveat.

## Failure Modes

-   **`UNRESOLVABLE`**: FQDN not in DNS. Rerun with `--hostname localhost` (the
    launcher does this automatically).
-   **`DEGRADED`**: final probe round passed but some probes were slow or timed
    out, usually host load. Report the URL with a latency caveat; check
    `uptime`.
-   **`DOWN`**: connection refused (never started, exited, or host
    rebooted/slept). Rerun; if it keeps exiting, read `dashboard.log` and
    `server_stderr.log` in `log_dir`.

-   **`UNRESPONSIVE` (health passes once, then times out)**: gRPC fork handlers
    block in `fork()` with the GIL held on every `gcloud`/`kubectl` spawn.
    `_dashboard/server.py` sets `GRPC_ENABLE_FORK_SUPPORT=0` before any grpc
    import; restart servers started before that fix.

-   **`UNRESPONSIVE` on projects with hundreds of clusters**: one scan thread
    per cluster starves the HTTP loop of the GIL. Concurrency is capped by
    `SCAN_CLUSTER_WORKERS` (default 16); lower it if stalls persist.

-   **`no access`**: 401/403 for your credentials; skipped until the next manual
    scan. Fix access (`[[gcloud-login]]`, IAM), then **Scan Now**.

-   **`server_status: UNRESPONSIVE`**: an unresponsive server holds the port.
    `kill $(lsof -ti tcp:8080)`, rerun, then `[[verify-dashboard-url]]`.

## Invocation

-   `/dashboard my-gcp-project`: all active clusters in the project.
-   `/dashboard my-tpu-cluster`: resolves location and generation, adds the
    cluster to `clusters_config.json`, opens focused on it.
-   `/dashboard my-tpu-workload`: resolves the cluster via
    `[[validate-workload]]` and deep-links with `#workload-search` filtered.
-   `/dashboard`: uses the active gcloud project or the clusters already in
    `clusters_config.json`.

Deep links: with `?workload=` or `?search=`, `_dashboard/public/dashboard.js`
fills `#workload-search` / `#workload-search-input` and dispatches an `input`
event to filter the Workload Registry table (Running, Pending, Preempted, or
CrashLoop).

## Connections

-   **Actions**:
    -   [[configure-and-launch-dashboard]]
    -   [[gcloud-login]]
    -   [[resolve-cluster-metadata]]
    -   [[validate-workload]]
    -   [[verify-dashboard-url]]
-   **Playbooks**:
    -   [[general-ml-debugging]]
    -   [[launch-workload-dashboard]]
    -   [[workload-disruption-orchestration]]

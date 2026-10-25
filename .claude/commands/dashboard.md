# `/dashboard` Command

Claude Code command for configuring and launching the GKE Capacity & Triage
Dashboard.

## Description

Configures and launches the GKE Capacity & Triage Dashboard scoped to a GCP
project ID, GKE cluster name, or target ML workload ID. Uses the
`configure-and-launch-dashboard` action and `launch-workload-dashboard` workflow
to verify local dashboard health, discover cluster capacity, trigger live scans,
and generate deep-linked URLs.

## Command Usage

-   `/dashboard [project_id]` (e.g. `/dashboard my-gcp-project`)
-   `/dashboard [cluster_name]` (e.g. `/dashboard my-tpu-cluster`)
-   `/dashboard [workload_id]` (e.g. `/dashboard my-tpu-workload`)
-   `/dashboard` (defaults to active gcloud project or ambient configuration)

> [!NOTE] The local web UI daemon (`_dashboard/server.py`) is an optional
> companion component. In external checkouts where `_dashboard/` is not bundled,
> use `/troubleshoot` and `_wiki/tools/scripts/python_glue_runner.py` directly.

## Execution Flow & Actions

1.  **Target Identification & Scoping**:
    -   If a `project_id` is supplied, all active GKE clusters within the
        project are discovered and populated into
        `_dashboard/clusters_config.json`.
    -   If a `cluster_name` is supplied, its location, generation, and
        reservation are resolved and added to `_dashboard/clusters_config.json`.
    -   If a `workload_id` is supplied, the underlying cluster and region are
        resolved via `[[validate-workload]]` before configuring the dashboard.
    -   If omitted, the active gcloud project or existing configured clusters
        are monitored.
2.  **Server Health & Background Daemon**:
    -   Verifies whether `http://localhost:8080/api/health` responds 200 OK.
    -   If not running, launches `_dashboard/server.py --port 8080 --no-browser`
        as a detached background daemon and waits for health readiness.
    -   If the port is held by a server that does not answer, reports
        `server_status: UNRESPONSIVE`; stop that process before retrying.
    -   A newly started daemon runs AI triage on the launching harness (Claude
        Code here). The result's `agent_backend` names the backend the running
        server uses. If `agent_backend_warning` is set, a reused server uses the
        other harness: stop it and relaunch.
3.  **Live Diagnostics Scan**:
    -   Triggers `POST /api/scan` to immediately initiate a diagnostic scan
        across the monitored clusters.
4.  **Deep-Linked URL Generation**:
    -   Scopes the browser URL with query parameters (`cluster`, `workload`,
        `gen`, or `project`).
    -   Deep-linking triggers automatic table filtering in `#workload-search` on
        the frontend.
5.  **URL Verification (`[[verify-dashboard-url]]`)**:
    -   Resolves the URL's hostname; an unresolvable FQDN (e.g. `*.local`) falls
        back to `localhost`.
    -   Probes both the deep-linked page and `/api/health` on that host several
        times while the scan runs. A single health check is not enough: the
        server can answer once and then freeze.
    -   Never report a URL whose `url_status` is not `VERIFIED` or `DEGRADED`;
        report `DEGRADED` with a latency caveat (isolated slow probes under host
        load). On `UNRESPONSIVE`, follow `url_hint`, restart the server, and
        re-verify with `python3
        _wiki/tools/scripts/configure_and_launch_dashboard.py --verify-url
        '<url>'`.

## Primary Workflow & Actions

-   `[[launch-workload-dashboard]]`
-   `[[configure-and-launch-dashboard]]`
-   `[[gcloud-login]]`
-   `[[resolve-cluster-metadata]]`
-   `[[validate-workload]]`
-   `[[verify-dashboard-url]]`

## Related Playbooks

-   `[[general-ml-debugging]]`
-   `[[workload-disruption-orchestration]]`

## Execution Command

```bash
python3 _wiki/tools/scripts/configure_and_launch_dashboard.py "$@"
```

--------------------------------------------------------------------------------

## Main Coordinator Delegation Invariant

The main coordinator agent acts strictly as a high-level orchestrator and has
**zero authorization to execute standalone shell commands, direct script
invocations, log queries, file edits, or GCS uploads directly**.

All execution MUST be delegated to the designated specialized wiki subagents via
the `Agent` tool (`subagent_type`; `SendMessage` to continue one):

-   **Workflow & Playbook Routing**: Delegate to `router`.
-   **Diagnostic & Action Execution**: Delegate to `practitioner` (for
    `/troubleshoot` milestones, Cloud Logging, GCS bucket scaffolding, empirical
    packaging, uploads, and background monitoring).
-   **Hypothesis & Gap Evaluation**: Delegate to `advisor`.
-   **Adversarial Verification & Audit**: Delegate to `reviewer` (for the
    `ai-verify-failure-attribution` prompt task before publication).
-   **Code, Script & Schema Edits**: Delegate to `maintainer`.

The main agent manages user approvals, coordinates subagent dispatches, and
presents final synthesized reports.

--------------------------------------------------------------------------------

## Connections

-   Antigravity counterpart: `_agents/skills/dashboard/SKILL.md`
-   `_wiki/workflows/launch-workload-dashboard.json`
-   `_wiki/actions/configure-and-launch-dashboard.json`
-   `_wiki/actions/gcloud-login.json`
-   `_wiki/actions/resolve-cluster-metadata.json`
-   `_wiki/actions/validate-workload.json`
-   `_wiki/actions/verify-dashboard-url.json`
-   `_wiki/playbooks/general-ml-debugging.md`
-   `_wiki/playbooks/workload-disruption-orchestration.md`

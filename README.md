
# ML Workload Debugging Wiki

**https://github.com/google/ml-workload-debugging-wiki**

An AI-powered troubleshooting assistant and operational knowledge base for
large-scale ML training and inference workloads on Cloud TPU, GPU, Pathways,
JAX, XLA, and GKE. Instead of manually digging through thousands of worker logs
and restart attempts, you give the assistant a workload ID and it automatically
finds the earliest real error, identifies faulty GKE nodes, and recommends a
fix.

Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0.

> This is not an officially supported Google product. We share it as a
> reference implementation of agent-executable operational knowledge.

---

## 1-Minute Setup

You need Python 3.9+ and the Google Cloud SDK (`gcloud` and `kubectl`)
authenticated against the project you are debugging.

```bash
git clone https://github.com/google/ml-workload-debugging-wiki
cd ml-workload-debugging-wiki

python3 -m venv _wiki/.venv && _wiki/.venv/bin/pip install -r _wiki/requirements.txt
gcloud auth login --update-adc
```

Launch your coding agent (Claude Code or Antigravity) from the repository root.

## Diagnosing a Workload with `/troubleshoot`

Run `/troubleshoot` with your workload or JobSet name:

```text
/troubleshoot <workload_id>
```

*(Plain English such as `"troubleshoot workload my-tpu-workload"` works too.)*

### What `/troubleshoot` does

1.  **Checks access first**: Confirms log access up front and stops with
    `NO_DATA` if permissions are missing rather than guessing.
2.  **Finds the earliest real error**: Scans Cloud Logging, GKE events, and node
    logs across all pods, looking past downstream barrier timeouts to find the
    worker or node that failed first.
3.  **Delivers a clear report**: Returns a **Verdict** (`CONFIRMED`, `REFUTED`,
    `INCONCLUSIVE`, or `NO_DATA`), quoted log lines, the **Culprit GKE Node**, a
    **Failure Table** across runs, and the **Recommended Fix**. Any step that
    modifies cluster state requires your explicit approval.

## Other Commands at a Glance

| Command                           | What it does                             |
| :-------------------------------- | :--------------------------------------- |
| `/dashboard                       | Launches the optional local GKE Capacity |
: [project|cluster|workload]`       : & Triage web UI                          :
:                                   : (`http\://localhost\:8080`).             :
| `/handoff <workload_id> [case_id] | Packages `/troubleshoot` findings and    |
: --bucket=<gs\://...>`             : failure tables into a GCS bucket to      :
:                                   : share with Google Cloud Support.         :

## Browsing & Validating the Wiki

All diagnostic knowledge lives under `_wiki/`, organized by page type and
cataloged in [`_wiki/INDEX.md`](_wiki/INDEX.md):

| Type       | Directory    | What it holds                                                             |
| :--------- | :----------- | :------------------------------------------------------------------------ |
| `symptom`  | `symptoms/`  | A failure mode, its log signatures, and how to verify it                  |
| `playbook` | `playbooks/` | Narrative troubleshooting guidance and reasoning rules (see               |
:            :              : [`_wiki/playbooks/rca-invariants.md`](_wiki/playbooks/rca-invariants.md)) :
| `workflow` | `workflows/` | An automated diagnostic procedure: ordered steps, as JSON                 |
| `action`   | `actions/`   | One atomic diagnostic step, as JSON                                       |
| `tool`     | `tools/`     | A diagnostic tool or script and how to run it                             |
| `concept`  | `concepts/`  | How a system component works (e.g. the `libtpu` lifecycle)                |
| `entity`   | `entities/`  | Platform entities (TPU, GPU, JAX, XLA)                                    |

To validate all schemas, links, and formatting across the wiki (~1 second):

```bash
python3 _wiki/wiki_linter.py
```

A healthy tree reports `"status": "PASS"` with `0` issues. Read
[`_wiki/SCHEMA.md`](_wiki/SCHEMA.md) before adding or editing pages.

## Getting Help

Open an issue at https://github.com/google/ml-workload-debugging-wiki/issues.

## Contributing

See [docs/contributing.md](docs/contributing.md). Contributions require a Google
CLA and follow the [Community Guidelines](https://opensource.google/conduct/).
New or updated pages must conform to [`_wiki/SCHEMA.md`](_wiki/SCHEMA.md) and
pass `python3 _wiki/wiki_linter.py`.

## License

Apache License 2.0. See [LICENSE](LICENSE).

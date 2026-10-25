<!-- mdformat global-off -->
# Directory Dictionary: Workload Diagnostic Handoff Repository

This document establishes the canonical directory layout, component ownership boundaries, schema definitions, and sanitization invariants for Cloud TPU multi-job diagnostic evidence handoff repositories at `gs://<bucket>/<storage_key>/`.

---

## 1. Storage Root Architecture

Target GCS Root:
```text
gs://<bucket>/<storage_key>/
```
- **Storage Key (`<storage_key>`)**: Uniquely identifies the multi-job incident workspace (normally corresponding to the primary Support Case ID).
- **Access Boundary**: Strictly governed by uniform bucket-level IAM policies restricted to authenticated Google Support / CE teams (`@google.com`) and authorized partner engineering accounts (authorized customer partner domain). Public access prevention is enforced.

---

## 2. Exhaustive Directory Layout & Ownership Dictionary

```text
gs://<bucket>/<storage_key>/
|-- BUCKET_STRUCTURE.md                     # Exhaustive directory dictionary and governance map (this document)
|-- INCIDENT_OVERVIEW.md                    # Root incident summary (cluster, zone, shared infrastructure health)
|-- INCIDENT_MANIFEST.json                  # Machine-readable root manifest (Schema v2.2) indexing all child jobs
|-- README.md                               # Incident overview mirror for web console browsing
|-- infrastructure/                         # Shared cluster-level diagnostics (Google-synthesized, customer-sanitized)
|   |-- Level_A_Infrastructure.md           # Backward-compatible Level A mirror for legacy automation
|   |-- subblocks/                          # TPU subblock health telemetry, allocation maps, and host health logs
|   |-- dcn_optical/                        # Optical Torus & DCN transceiver attenuation metrics
|   |-- reservation/                        # Contracted vs ready TPU slice reservation allocations
|   |-- links/                              # Shared monitoring links, dashboards, and topology references
|   `-- findings/                           # Immutable infrastructure-level diagnostic findings
|       |-- _infra_latest.md                # Materialized latest Level A Infrastructure Failure Table
|       `-- google/                         # Immutable timestamped Google infrastructure triage findings
|-- jobs/                                   # Container directory for isolated workload workspaces
|   `-- <workload_name>/                    # Workload workspace (e.g. jobs/example-workload-1/)
|       |-- JOB_README.md                   # Workload context, parameters, and active storage summary
|       |-- JOB_MANIFEST.json               # Machine-readable workload manifest (Schema v2.2)
|       |-- README.md                       # Workload context mirror for web console browsing
|       |-- findings/                       # Partitioned, immutable findings tracks
|       |   |-- customer/                   # Customer-authored findings track
|       |   |   `-- _customer_latest.md     # Materialized latest customer symptom & reproduction report
|       |   `-- google/                     # Google-synthesized findings track
|       |       |-- _google_latest.md       # Materialized latest Level B Job Progression Failure Table
|       |       `-- finding_*.md            # Immutable timestamped Google job diagnostic records
|       |-- failure_tables/                 # Shared failure table directory
|       |   `-- Level_B_Job_Progression.md  # Backward-compatible Level B mirror
|       |-- data/                           # Raw empirical runtime telemetry
|       |   |-- baseline/                   # Clean / healthy window telemetry (logs, metrics)
|       |   |-- degraded/                   # Degraded / failing window telemetry
|       |   |   `-- xla_dumps/              # Sanitized textual HLO graphs and compiler traces
|       |   |-- orbax/                      # Orbax checkpoint barrier timings and GCS latency logs
|       |   |-- barriers/                   # Megascale collective communication and sflag stall logs
|       |   `-- memory/                     # HBM allocation traces, fragmentation maps, and OOM dumps
|       |-- theories/                       # Falsifiable hypothesis cards (THEORY.md)
|       `-- links/                          # Workload-specific dashboard links and logging queries
`-- links/                                  # Incident-level shared links and dashboard references
```

---

## 3. Directory Sub-Path Details & Security Invariants

| Path | Purpose | Primary Writer | Expected File Types | Sanitization / Governance |
| :--- | :--- | :--- | :--- | :--- |
| `BUCKET_STRUCTURE.md` | Storage dictionary and taxonomy specification | Automation (`scaffold_bucket.sh`) | Markdown (`.md`) | Parameterized with active bucket name and storage key. |
| `INCIDENT_MANIFEST.json` | Top-level incident manifest indexing child jobs | Google Support / Automation | JSON (Schema v2.2) | Strict Schema v2.2 compliance; zero Google-internal physical rack/cluster names. |
| `INCIDENT_OVERVIEW.md` | Top-level human-readable incident status summary | Google Support / Automation | Markdown (`.md`) | 7-bit ASCII typography; renders missing data as `NOT PROVIDED`. |
| `infrastructure/subblocks/` | GKE node and TPU subblock health records | Google Support | JSON, Text | Customer-sanitized GKE logical node coordinates only. |
| `infrastructure/dcn_optical/` | Inter-slice optical torus and collective fabric metrics | Google Support | JSON, CSV, Text | Transceiver attenuation (dBm) and optical path health. |
| `infrastructure/findings/` | Materialized & timestamped Level A failure tables | Google Support | Markdown (`.md`) | Level A Infrastructure Failure Table; initial status `PENDING_GOOGLE_CORRELATION`. |
| `jobs/<workload>/findings/customer/` | External customer workload symptom reports | Customer Engineering | Markdown (`.md`), Text | Customer empirical ground truth; customers do not write failure tables. |
| `jobs/<workload>/findings/google/` | Materialized & timestamped Level B failure tables | Google Support | Markdown (`.md`) | Level B Job Progression Failure Table synthesized from customer data. |
| `jobs/<workload>/data/degraded/xla_dumps/`| Textual HLO graphs and compiler traces | Customer / Automation | Text, `.hlo`, `.ir` | **Mandatory Redaction**: Customer source file paths, tensor names, and dimensions redacted. |
| `jobs/<workload>/data/orbax/` | Checkpoint barrier wait logs and storage traces | Customer / Automation | Text, JSON | Captures GCS upload latencies, commit stalls, and barrier durations. |
| `jobs/<workload>/theories/` | Hypotheses and falsification criteria | Both (Customer & Google) | Markdown (`.md`) | Structured cards citing proven items (P) and unproven gaps (C). |

---

## 4. Redaction & Node Replacement Safety Invariants

1. **Proprietary HLO Redaction**:
   All artifacts placed into `xla_dumps/` must be sanitized through `redact_diagnostic_payload.py` prior to publication. Proprietary variable names, vocabulary dimensions, and local source file paths are stripped while mathematical graph topology (`all-gather`, `reduce-scatter`, `custom-call`) is preserved.

2. **Node Replacement Safety Gate**:
   No TPU tray, subblock, cube, or node may be cordoned, drained, or replaced autonomously by AI agents without explicit human operator confirmation.

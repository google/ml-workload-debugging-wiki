<!-- mdformat global-off -->
# Workload Diagnostic Context: {{WORKLOAD}}

**Partner Issue**: {{ISSUE_REF}} | **Support Case**: #{{CASE_ID}} | **Storage
Key**: {{STORAGE_KEY}} **Customer Issue Tracker**:
[Partner Issue Tracker]({{ISSUE_URL}})
**Workload / JobSet**: `{{WORKLOAD}}` **Storage Root**: `{{JOB_STORAGE_ROOT}}`
**Handoff Specification Version**: 2.2.0 **Accelerator Type**:
{{ACCELERATOR_TYPE}} **Source**: Debug Wiki /troubleshoot session
{{TROUBLESHOOT_SOURCE}}

---

## 1. Workload Identity and Scale

* **Scale**: {{CHIP_COUNT}} chips, {{NODE_COUNT}} nodes, {{SLICE_COUNT}} slices
* **GKE Cluster / Zone**: `{{CLUSTER}}` ({{ZONE}}) | **Project**: `{{PROJECT}}`
* **Framework Versions**: JAX {{JAX_VERSION}} | jaxlib {{JAXLIB_VERSION}} | libtpu {{LIBTPU_VERSION}} | Python {{PYTHON_VERSION}} | Orbax {{ORBAX_VERSION}}
* **Customer DRI**: {{CUSTOMER_DRI}}
* **Customer Primary Investigator**: {{PRIMARY_INVESTIGATOR}}
* **Google Support Lead**: {{GOOGLE_LEAD}}
* **Google Customer Engineer Lead**: {{GOOGLE_CE}}

---

## 2. Temporal Windows

* **Clean / Nominal Baseline Window**: `{{GOOD_WINDOW}}` (Mean step: {{BASELINE_STEP}})
* **Degraded / Failing Window**: `{{BAD_WINDOW}}` (Mean step: {{DEGRADED_STEP}})

---

## 3. Active Storage & Orbax Checkpoint Context

* **Active Checkpoint Bucket**: `{{CHECKPOINT_BUCKET}}`
* **Active Dataset / Pipeline Bucket**: `{{DATASET_BUCKET}}`
* **Orbax Barrier Wait Latency**: `{{ORBAX_LATENCY}}`

---

## 4. Two-Level Failure Tables
Triage failure tables are maintained under the strict two-level standard synthesized exclusively by Google Support from customer empirical data. Customers do NOT author failure tables; customers provide empirical traces while Google synthesizes Level A and Level B failure tables.

Failure tables uploaded for this workload:
{{FAILURE_TABLES_LIST}}

---

## 5. Uploaded Job Artifacts
Only files actually uploaded under this job workspace are listed.
{{JOB_ARTIFACTS_LIST}}

## 6. Customer Next Steps (from the troubleshoot run)
{{CUSTOMER_NEXT_STEPS}}

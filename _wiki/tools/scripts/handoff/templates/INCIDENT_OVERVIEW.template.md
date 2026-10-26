<!-- mdformat global-off -->
# Incident Diagnostic Overview: Support Case {{CASE_ID}}

**Partner Issue**: {{ISSUE_REF}} | **Support Case**: #{{CASE_ID}} | **Storage
Key**: {{STORAGE_KEY}} **Customer Issue Tracker**:
[Partner Issue Tracker]({{ISSUE_URL}})
**Storage Root**: `{{STORAGE_ROOT}}` *(Restricted to authenticated Googlers and
customer partner domain)* **Handoff Specification Version**: 2.2.0 **Accelerator
Type**: {{ACCELERATOR_TYPE}} **Source**: Debug Wiki /troubleshoot session
{{TROUBLESHOOT_SOURCE}}

---

## 1. Ownership and Engineering Contacts

* **Customer DRI (Directly Responsible Individual)**: {{CUSTOMER_DRI}}
* **Customer Primary Investigator**: {{PRIMARY_INVESTIGATOR}}
* **Google Support Lead**: {{GOOGLE_LEAD}}
* **Google Customer Engineer (CE) Lead**: {{GOOGLE_CE}}

### Ownership Handover Ledger
{{OWNERSHIP_HISTORY}}

---

## 2. Infrastructure Identity

* **Customer GCP Project**: `{{PROJECT}}` ({{PROJECT_NUMBER}})
* **Target GKE Cluster**: `{{CLUSTER}}`
* **GCP Zone / Reservation**: `{{ZONE}}` ({{RESERVATION}})
* **Disruption Category**: `{{ISSUE_TYPE}}`

---

## 3. Shared Infrastructure Status (`infrastructure/`)

* **Subblocks (`infrastructure/subblocks/`)**: Customer-sanitized GKE slice and node health summaries.
* **DCN / Optical (`infrastructure/dcn_optical/`)**: Inter-slice collective fabric health summaries.
* **Reservation (`infrastructure/reservation/`)**: Contracted vs ready GKE TPU slice reservation allocations.
* **Findings (`infrastructure/findings/`)**: Immutable Level A infrastructure failure tables synthesized by Google Support using customer-visible GKE coordinates.

---

## 4. Workload Isolation (`jobs/`)
Each impacted workload possesses an isolated workspace under `jobs/[job_name]/`:

* **Active Jobs**:
{{JOBS_LIST}}

---

## 5. Uploaded Artifacts
Only files actually uploaded under the storage root are listed.
{{ARTIFACTS_LIST}}

## 6. Report Discrepancies
Fields the troubleshoot report stated inconsistently are left unset and listed here.
{{DISCREPANCIES}}

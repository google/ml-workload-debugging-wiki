<!-- mdformat global-off -->
# Large-Scale Workload Diagnostic Handoff: Support Case {{CASE_ID}}

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

## 2. Workload and Cluster Identity

* **Customer Project**: `{{PROJECT}}` ({{PROJECT_NUMBER}})
* **Target GKE Cluster**: `{{CLUSTER}}`
* **GCP Zone / Reservation**: `{{ZONE}}` (`{{RESERVATION}}`)
* **Workload / JobSet**: `{{WORKLOAD}}`
* **Scale**: {{CHIP_COUNT}} chips, {{NODE_COUNT}} nodes, {{SLICE_COUNT}} slices
* **Disruption Category**: `{{ISSUE_TYPE}}`

---

## 3. Temporal Windows

* **Clean / Nominal Baseline Window**: `{{GOOD_WINDOW}}` (Mean step: {{BASELINE_STEP}})
* **Degraded / Failing Window**: `{{BAD_WINDOW}}` (Mean step: {{DEGRADED_STEP}})

---

## 4. What Customer Telemetry Established (verbatim from the troubleshoot run)
{{PROVES_LIST}}

## 5. What Customer Telemetry Could Not Determine (verbatim from the troubleshoot run)
{{CANNOT_PROVE_LIST}}

---

## 6. Shared Failure Tables
Triage failure tables are maintained under a strict two-level standard synthesized exclusively by Google Support from customer empirical data. Customers do NOT author failure tables; customers provide empirical traces while Google synthesizes Level A and Level B failure tables.

Failure tables uploaded:
{{FAILURE_TABLES_LIST}}

---

## 7. Customer Next Steps (from the troubleshoot run)
{{CUSTOMER_NEXT_STEPS}}

---

## 8. Uploaded Artifacts
Only files actually uploaded under the storage root are listed.
{{ARTIFACTS_LIST}}

## 9. Report Discrepancies
{{DISCREPANCIES}}

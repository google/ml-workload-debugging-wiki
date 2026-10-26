<!-- mdformat global-off -->
TPU Workload Diagnostic Handoff: {{WORKLOAD_NAME}}
Partner Issue: {{ISSUE_REF}} | Support Case: {{CASE_ID}} | Storage Key: {{STORAGE_KEY}}
Source: Debug Wiki /troubleshoot session {{TROUBLESHOOT_SOURCE}} (customer-visible telemetry only)
Infrastructure Root Cause Status: {{INFRA_CORRELATION_STATUS}}

# (A) Bug Report

## 1. Issue Category
{{CATEGORY_CHECKLIST}}

Candidate disruption category: {{DISRUPTION_CATEGORY}}

## 2. System Information
- **JAX Version:** {{JAX_VERSION}}
- **Jaxlib Version:** {{JAXLIB_VERSION}}
- **libtpu Version:** {{LIBTPU_VERSION}}
- **Hardware/Accelerator:** {{ACCELERATOR}}
- **Environment (PROD/DEV):** {{ENVIRONMENT}}
- **Python Version:** {{PYTHON_VERSION}}
- **Workload Type:** {{WORKLOAD_TYPE}}
- **Workload / JobSet:** {{WORKLOAD_NAME}}
- **GCP Project:** {{PROJECT_ID}} ({{PROJECT_NUMBER}})
- **GKE Cluster / Zone:** {{CLUSTER}} ({{ZONE}})

## 3. Reproduction Steps / Minimal Example
Time windows investigated in the troubleshoot run (UTC):
- Degraded window: {{DEGRADED_WINDOW}} (step time: {{DEGRADED_STEP}})
- Healthy baseline window: {{BASELINE_WINDOW}} (step time: {{BASELINE_STEP}})

Restart attempts observed:
{{ATTEMPTS_LIST}}

Cloud Logging filters used in the troubleshoot run (verbatim):
{{LOG_FILTERS}}

## 4. Upload Artifacts
{{ARTIFACT_CHECKLIST}}

## 5. Additional Context

Customer Hypothesis:
- Hypothesis: {{CUSTOMER_HYPOTHESIS}}
- Stated by: {{HYPOTHESIS_AUTHOR}}
- Confidence: {{HYPOTHESIS_CONFIDENCE}}
- Supported by: {{HYPOTHESIS_SUPPORT_REFS}}
- Not yet explained: {{HYPOTHESIS_GAP_REFS}}
- Would be disproved if: {{HYPOTHESIS_FALSIFIER}}

Established by Customer Telemetry (from the troubleshoot run):
{{PROVES_LIST}}

Not Determinable from Customer Telemetry (from the troubleshoot run):
{{CANNOT_PROVE_LIST}}

Questions for Google (each cites a C item above):
{{QUESTIONS_FOR_GOOGLE}}

Customer Next Steps (from the troubleshoot run):
{{CUSTOMER_NEXT_STEPS}}

Report Discrepancies (fields left unset rather than guessed):
{{DISCREPANCIES}}

Artifact Locations:
- Storage Root: {{STORAGE_CONSOLE_URL}}
{{ARTIFACT_LOCATIONS}}

Contacts:
- Google CE Lead: {{GOOGLE_CE_LEAD}}
- Google Support Lead: {{GOOGLE_SUPPORT_LEAD}}
- Customer DRI: {{CUSTOMER_DRI}}
- Customer Primary Investigator: {{CUSTOMER_PRIMARY_INVESTIGATOR}}

Human Ownership Attribution:
- Verified Human of Record: {{VERIFIED_HUMAN_OPERATOR}}
- Issue Tracker: Partner Issue Tracker at {{ISSUE_URL}}

Actionable Node Remediation:
{{REMEDIATION_BLOCK}}

Uploaded Evidence Artifacts (this sync):
{{UPLOADED_FILES_LIST}}

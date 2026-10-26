<!-- mdformat global-off -->
# Customer Hypothesis {{HYPOTHESIS_ID}}: {{WORKLOAD_NAME}}

**Partner Issue**: {{ISSUE_REF}} | **Support Case**: {{CASE_ID}} | **Storage Key**: {{STORAGE_KEY}} | **Recorded**: {{TIMESTAMP_UTC}}
**Stated by**: {{HYPOTHESIS_AUTHOR}}
**Status**: {{HYPOTHESIS_STATUS}}  (OPEN, CONFIRMED, DISPROVED)
**Confidence**: {{HYPOTHESIS_CONFIDENCE}}  (low, medium, high)

## Hypothesis
{{CUSTOMER_HYPOTHESIS}}

## Supported by
Items from "What customer telemetry proves" in the troubleshoot run, by ID:
{{HYPOTHESIS_SUPPORT_REFS}}

## Not yet explained
Items from "What customer telemetry cannot prove" that this hypothesis depends on, by ID:
{{HYPOTHESIS_GAP_REFS}}

## Would be disproved if
{{HYPOTHESIS_FALSIFIER}}

## Source
Troubleshoot run: {{TROUBLESHOOT_SOURCE}}
Workflow trace: jobs/{{WORKLOAD_NAME}}/workflow_trace.md

## History
| Timestamp (UTC) | Change | By |
| :--- | :--- | :--- |
| {{TIMESTAMP_UTC}} | Recorded | {{HYPOTHESIS_AUTHOR}} |

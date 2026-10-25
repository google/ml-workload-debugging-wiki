<!-- mdformat global-off -->
# Investigation Workflow Trace: {{WORKLOAD}}

**Incident**: Support Case #{{CASE_ID}}
**Workload / JobSet**: `{{WORKLOAD}}`
**Customer GCP Project**: `{{PROJECT_ID}}`
**Source session**: Debug Wiki `/troubleshoot` (customer telemetry only)
**Generated (UTC)**: {{GENERATED_AT}}
**Recorded by**: {{AUTHOR_ROLE_EMAIL}}

---

## 1. Time Windows Investigated (UTC)
| Window | Start | End |
| :--- | :--- | :--- |
| Degraded | {{DEGRADED_START}} | {{DEGRADED_END}} |
| Healthy baseline | {{BASELINE_START}} | {{BASELINE_END}} |

## 2. Workflows and Actions Executed (in order)
| # | Workflow | Action | execution_mode | Inputs (parameters) | Output summary | Result count |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | {{WORKFLOW}} | {{ACTION}} | command / ai / manual | `{{PARAMS_JSON}}` | {{OUTPUT_SUMMARY}} | {{COUNT}} |

## 3. Cloud Logging Filters (verbatim)
### Query 1: {{PURPOSE}}
* Time range: {{START}} to {{END}}
* Result count: {{COUNT}} (list zero-result queries too)
```text
{{FILTER}}
```

## 4. Findings Proven from Customer Telemetry
Each finding cites the query number above and a quoted log line (timestamp, worker index, GKE node).
* {{FINDING}} (Query {{N}})

## 5. Open Questions for Google Support
Items that could not be determined from customer telemetry and need correlation on the Google side:
* {{OPEN_QUESTION}}

---
*This trace records only what was executed in the source session. Customer-visible GKE/GCP coordinates only.*

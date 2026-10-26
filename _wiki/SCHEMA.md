
# SCHEMA.md

## Domain

This wiki covers the common issues and investigation methods for ML training
workloads.

## Conventions

-   **Naming:** File names must be lowercase, hyphens, no spaces (e.g.,
    `pathways-architecture.md`)
-   **Metadata:** Every wiki page starts with YAML frontmatter (see below)
-   **Routing:** Every new page must be added to `INDEX.md` under the correct
    section type
-   **Updates:** When updating a page, always bump the `updated` date
-   **Logging:** Every mutating action must be appended to `LOG.md`
-   **Workflows & Playbooks:** Files in `workflows/` must be structured JSON.
    Heuristic/mindset guides must be in `playbooks/` as Markdown.
-   **Contradictions:** When ingesting conflicting information, verify
    timestamps (newer overwrites older). If unresolvable, document both
    positions citing sources/dates, set `contested: true` and
    `contradictions: [...]` in frontmatter, and generate a CL for human review.
-   **Canonical Key Ordering:** Metadata keys across all Markdown pages, source
    documents, workflows, and actions must follow the canonical key order
    defined below. The linter (`wiki_linter.py` / `/lint`) automatically
    enforces and sorts metadata keys into this order on every run. Any
    additional keys present in a file are sorted alphabetically following the
    canonical keys.

    ## Taxonomy Tags

    Only use the tags defined below. New tags must be added to this schema
    before they can be applied.

-   **Hardware:** `tpu`, `gpu`

-   **Workloads:** `training`, `inference`, `eval`, `benchmark`

-   **Failure Domains:** `out_of_memory`, `timeout`, `hardware_fault`

-   **Cloud & Orchestration:** `gke`, `kubernetes`, `kueue`, `networking`,
    `dns`, `storage`, `auth`, `gcp`, `gcloud`

-   **Observability:** `observability`, `logging`, `monitoring`

-   **Other:** `hangs`, `methodology`, `triage`, `metrics`, `goodput`,
    `analysis`, `ironwood`, `scheduling`, `pathways`, `capacity`,
    `datacenter`, `mxla`, `collective_communication`, `workflow`, `approval`

## Canonical Metadata Ordering

The wiki enforces consistent, canonical ordering for metadata keys. The linter
automatically sorts keys in-place if they are out of order:

-   **Markdown Pages** (`concepts/`, `symptoms/`, `playbooks/`, `tools/`,
    `entities/`): `["title", "description", "type", "tags", "created",
    "updated", "contested", "contradictions"]`
-   **Workflow JSON** (`workflows/*.json`): `["title", "type", "description",
    "tags", "created", "updated", "entrypoints", "summary_fields", "steps",
    "step_notes", "step_fallbacks"]`
-   **Action JSON** (`actions/*.json`): `["title", "type", "description",
    "tags", "created", "updated", "entrypoints", "execution_mode",
    "instruction", "prompt", "params", "outputs", "summary_fields",
    "requires_approval", "approval_exempt_params", "timeout_seconds",
    "deprecated", "superseded_by", "questionnaire", "fallback"]`

Any additional unrecognized or custom keys present in a file are placed after
the canonical keys in alphabetical order.

## Frontmatter Template

```yaml
---
title: "<page title>"
description: "<brief page description>"
type: concept | symptom | playbook | workflow | action | tool | entity
tags: [...]
created: YYYY-MM-DD
updated: YYYY-MM-DD

# Contradiction Metadata
contested: true | false              # set when the page has unresolved contradictions
contradictions: [other-page-name.md] # pages this one is conflicted with
---
```

## Workflows vs. Playbooks

To support both automated execution and heuristic troubleshooting, the wiki
divides procedural knowledge into two categories:

1.  **Workflows (`workflows/` - JSON only)**:

    *   **Purpose**: Rigid, automated Standard Operating Procedures (SOPs) for
        specific, well-understood failure modes.
    *   **Format**: Structured JSON file containing a sequence of parameterized
        steps (which can be atomic Actions or sub-Workflows).
    *   **Target Audience**: Test/Runtime agents that execute them
        deterministically.
    *   **Design Rule**: Create a JSON workflow when the diagnostic steps are
        fully codified, linear, and rely on specific, parameters-driven tool
        execution.

2.  **Playbooks (`playbooks/` - Markdown only)**:

    *   **Purpose**: Flexible, heuristic guides, flowcharts, troubleshooting
        mindsets, and lessons learned.
    *   **Format**: Markdown file with narrative instructions and concepts.
    *   **Target Audience**: Human operators and agents operating in "heuristic
        mode" when standard workflows fail.
    *   **Design Rule**: Create a Markdown playbook for high-level
        troubleshooting strategies, philosophies, decision-tree flowcharts,
        gotchas, or when steps cannot be fully automated or parameters-driven.

## Page Types

The wiki supports the following page types:

-   **concept**: Explains a system component, architecture, or terminology
   .
-   **symptom**: Describes a failure mode, its log signatures, and how to verify
    it (e.g., [[megascale-hang]], [[sigill-worker-kill]]).
-   **playbook**: Heuristic, narrative guide for troubleshooting (e.g.,
    [[general-ml-debugging]], [[debugging-gotchas]]).
-   **workflow**: Rigid, automated SOP in JSON format (under `workflows/`).
-   **action**: Atomic command execution step in JSON format (under `actions/`).
-   **tool**: Documents a diagnostic tool or script (e.g.,
    [[python-glue-runner]]).
-   **entity**: Represents a hardware or software platform entity (e.g.,
    [[cloud-tpu]], [[jax]]).

### Playbook Template (`playbooks/`)

```markdown
# [Playbook Name]

## Overview
[Narrative guide or troubleshooting philosophy]

## Diagnostic Steps (or Flowchart)
[High-level strategies and decision points]

## Connections
- **Workflows**:
  - [[workflows/...]]
- **Actions**:
  - [[actions/...]]
- **Playbooks**:
  - [[playbook-name]]
- **Symptoms**:
  - [[symptom-name]]
- **Concepts**:
  - [[concept-name]]
- **Entities**:
  - [[entity-name]]
- **Tools**:
  - [[tool-name]]
```

### Concept Template (`concepts/`)

```markdown
# [Concept Name]

## Overview
[Definition and core idea of the concept]

## Architecture & Design
[How it fits into the system, key components, and interactions]

## Operational Impact (or Debugging Significance)
[Why this concept matters when troubleshooting; key implications for operators]

## Connections
- **Workflows**:
  - [[workflows/...]]
- **Actions**:
  - [[actions/...]]
- **Playbooks**:
  - [[playbook-name]]
- **Symptoms**:
  - [[symptom-name]]
- **Concepts**:
  - [[concept-name]]
- **Entities**:
  - [[entity-name]]
- **Tools**:
  - [[tool-name]]
```

### Symptom Template (`symptoms/`)

```markdown
# [Symptom Name]

## Overview
[Summary of the symptom and its user-facing impact]

## Log Signatures
[Specific logs, error codes, stack traces, or metrics patterns that uniquely identify this symptom]

## Verification Steps
[Step-by-step instructions on how to confirm this is the active symptom using tools or logs]

## Playbooks & Root Causes
[Links to playbooks or concepts explaining the root causes and how to resolve them]

## Connections
- **Workflows**:
  - [[workflows/...]]
- **Actions**:
  - [[actions/...]]
- **Playbooks**:
  - [[playbook-name]]
- **Symptoms**:
  - [[symptom-name]]
- **Concepts**:
  - [[concept-name]]
- **Entities**:
  - [[entity-name]]
- **Tools**:
  - [[tool-name]]
```

### Entity Template (`entities/`)

```markdown
# [Entity Name]

## Overview
[Platform, hardware, or software service summary]

## Technical Specifications (or Architecture)
[Details of the entity, configuration, topology, and limits]

## Common Failure Modes
[Summaries of and links to symptoms or concepts associated with this entity]

## Connections
- **Workflows**:
  - [[workflows/...]]
- **Actions**:
  - [[actions/...]]
- **Playbooks**:
  - [[playbook-name]]
- **Symptoms**:
  - [[symptom-name]]
- **Concepts**:
  - [[concept-name]]
- **Entities**:
  - [[entity-name]]
- **Tools**:
  - [[tool-name]]
```

### Tool Template (`tools/`)

```markdown
# [Tool Name]

## Overview
[What the tool is, when to use it, and its main features]

## Usage & Commands
[Examples of commands, options, inputs/outputs, and how to run them]

## Troubleshooting Scenarios
[How this tool is applied in specific playbooks or workflows]

## Connections
- **Workflows**:
  - [[workflows/...]]
- **Actions**:
  - [[actions/...]]
- **Playbooks**:
  - [[playbook-name]]
- **Symptoms**:
  - [[symptom-name]]
- **Concepts**:
  - [[concept-name]]
- **Entities**:
  - [[entity-name]]
- **Tools**:
  - [[tool-name]]
```

### Workflow Template (`workflows/` - JSON only)

Workflows represent a sequence of diagnostic steps. The `"steps"` list can
contain references to both atomic Actions (defined in `actions/`) and other
Workflows (defined in `workflows/`), allowing workflows to be nested and
composed. Every workflow requires a `"description"` *(string, required)*
defining its purpose and diagnostic scope. Optionally, workflows can define
`"summary_fields"` to aggregate and report metrics upon completion, and
`"step_notes"` *(object, optional)* mapping step or phase names to operational
context, routing rationale, or execution notes. Direct symptom references are
encouraged for triage routing: referencing a workflow directly from matching
symptom pages in `symptoms/` ensures discoverability during incident diagnosis
(workflows not referenced by any symptom will receive an advisory warning during
linting).

#### Step Fallbacks

-   `"step_fallbacks"` *(object, optional)*: Maps a step name (e.g.
    `"resolve-workload-infra-specs"`) to a list of structured fallback branch
    objects triggered when that step fails or returns incomplete data. Each
    branch object specifies:
    -   `when` *(list of strings, required)*: Non-empty list of trigger tokens
        from `{"access_denied", "empty_result", "incomplete_coverage",
        "timeout", "auth_expired", "command_error"}`.
    -   `skip_steps` *(list of strings, optional)*: Downstream workflow steps
        (`"[[action-name]]"`) to skip when this fallback branch is taken.
    -   `replace_with` *(list of strings, required)*: Ordered list of actions or
        workflows (`"[[action-name]]"`) to execute instead.
    -   `reason` *(string, required)*: Operational explanation for why and when
        this fallback branch applies.

```json
{
  "title": "[Workflow Title]",
  "type": "workflow",
  "description": "[Brief description of workflow purpose and diagnostic scope]",
  "tags": [
    "[tag1]",
    "[tag2]"
  ],
  "created": "YYYY-MM-DD",
  "updated": "YYYY-MM-DD",
  "entrypoints": ["troubleshoot"],
  "summary_fields": [
    "total_found",
    "deduplicated_count",
    "error_logs_count",
    "output_file"
  ],
  "steps": [
    "[[action-name-1]]",
    "[[workflow-name-1]]"
  ],
  "step_notes": {
    "action-name-1": "Operational context or execution rationale for this step"
  },
  "step_fallbacks": {
    "action-name-1": [
      {
        "when": ["access_denied", "empty_result"],
        "replace_with": ["[[workflow-name-1]]"],
        "reason": "Fallback to alternate public step when primary step fails."
      }
    ]
  }
}
```

### Action Template (`actions/` - JSON only)

Actions represent atomic diagnostic execution steps. They can be standard
command-line actions, LLM-operated actions, or manual guidelines. Actions can
optionally define `"summary_fields"` to specify which metrics/parameters to
print out when execution concludes. Direct symptom references are encouraged for
triage routing: standalone actions (actions that are not part of any workflow's
steps) should be referenced directly by matching symptom pages in `symptoms/`
(standalone actions neither part of a workflow nor referenced by any symptom
will receive an advisory warning during linting).

Every action must specify an `"execution_mode"`. The three supported execution
modes are:

-   `"command"`: Used for deterministic, bash-executable scripts and CLI tools.
    Dispatched via `python3 _wiki/tools/scripts/python_glue_runner.py`.
-   `"ai"`: Used for complex, heuristic tasks that require LLM reasoning and
    prompt interpolation. **Dispatched as an agent prompt task (a spawned
    subagent), NOT through the glue runner.** Such actions MUST be named with an
    `ai-` prefix (e.g. `ai-hw-analyzer`).
-   `"manual"`: Used for steps that cannot be automated and require human
    interaction or proprietary UI navigation. Surfaced to the user as
    instruction text.

**The `ai-` filename prefix and `"execution_mode": "ai"` must always agree.**
The prefix is the at-a-glance signal to a reading agent that the action is an
agent prompt task; `execution_mode` is the authoritative field. An action with
one but not the other is a schema violation.

#### Entry Point Ownership (`entrypoints`)

Both Workflows and Actions carry `entrypoints` *(list of strings, required)*:
which operator-facing slash command / skill can lead to executing the artifact,
for example `"entrypoints": ["troubleshoot", "dashboard"]`.

Allowed values are `troubleshoot`, `handoff`, and `dashboard`. The list must be
non-empty, free of duplicates, and drawn only from that vocabulary; the linter
enforces all three as errors.

This is **not** a second tag namespace. `tags` describe the problem domain
(`tpu`, `hangs`, `hardware_fault`); `entrypoints` describe reachability from an
entry point. An artifact routinely has several tags and exactly one entrypoint.

Sharing is expressed as a list rather than forced into a single owner. Three
artifacts are reachable from both commands today — `validate-workload`,
`configure-and-launch-dashboard` and `launch-workload-dashboard` — because
`workload-disruption-orchestration` invokes the dashboard steps as part of a
triage run. Splitting the corpus into per-command directories would have forced
an arbitrary choice for exactly these three; a list states the truth instead.

#### Operational Action Flags

-   `requires_approval` *(boolean, optional / required for mutating commands &
    prompts)*: Safety guardrail indicating whether the action mutates
    environment, cluster, or workload state (e.g., cordoning/tainting nodes,
    changing priority, restarting pods, patching configs). Must be `true` for
    all state-changing actions. The glue runner enforces it: a gated Action is
    refused (exit code 4) until re-run with `--approved`.
-   `approval_exempt_params` *(object, optional)*: For a `requires_approval`
    action with a read-only mode, maps parameter names to the values that select
    that mode (e.g. `{"check_only": true}`). When every listed parameter
    matches, the glue runner runs the action without `--approved`. Use only for
    modes that change no state.
-   `timeout_seconds` *(integer, optional)*: Maximum execution timeout for
    streaming commands.
-   `deprecated` *(boolean, optional)*: Indicates an action is deprecated.
-   `superseded_by` *(string, optional)*: Wikilink to the replacing action.
-   `questionnaire` *(object, optional)*: Interactive prompt questionnaire to
    collect missing or unconfirmed parameters before execution. See
    [Interactive Action Questionnaire (`questionnaire`)](#interactive-action-questionnaire-questionnaire).
-   `fallback` *(array of objects, optional)*: Structured fallback recovery
    chains triggered on specific failure or coverage conditions. See
    [Action-Level Fallbacks (`fallback`)](#action-level-fallbacks-fallback).

#### Interactive Action Questionnaire (`questionnaire`)

Actions that require runtime user input or confirmation before execution can
optionally define a `questionnaire` block. Orchestrator agents (such as
`/troubleshoot` or Claude Code) inspect this block when declared parameters are
missing from context or require user verification, prompt the operator
interactively, and dispatch the action with the collected values:

-   `questionnaire.title` *(string, optional)*: Title for the interactive
    questionnaire modal/form.
-   `questionnaire.description` *(string, optional)*: Explanatory text providing
    context to the user.
-   `questionnaire.fields` *(array of objects, required)*:
    -   `name` *(string, required)*: Parameter name matching a declared key in
        `parameters`.
    -   `question` *(string, required)*: The prompt question shown to the user.
    -   `placeholder` *(string, optional)*: Example or placeholder value.
    -   `required` *(boolean, optional, default false)*: Whether the field must
        be supplied.
    -   `required_if_missing` *(boolean, optional, default false)*: Whether the
        field is only prompted if not already resolved from context.
    -   `input_mode` *(string, optional)*: `"text"` | `"select"` |
        `"select_then_input"` | `"select_or_input"`. Defaults to `"text"`.
    -   `options` *(array of strings or objects, optional)*: Available choices.
        If objects:
    -   `label` *(string, required)*: Display label for option.
    -   `value` *(string, optional)*: Fixed value if selected.
    -   `prefix` *(string, optional)*: Prefix prepended to the user's typed
        input (e.g. `"gs://"`, `"projects/"`).
    -   `prompt` *(string, optional)*: Follow-up question shown to the user to
        type their specific value after selection.

#### Action-Level Fallbacks (`fallback`)

Actions can optionally declare a `fallback` list to codify deterministic
recovery chains when an action encounters permissions errors, timeouts, empty
results, or incomplete coverage:

-   `action` *(string, required)*: Wikilink to the fallback action or workflow
    (e.g. `"[[lookup-workload-timestamps]]"`).
-   `when` *(list of strings, required)*: Non-empty list of trigger conditions
    from `{"access_denied", "empty_result", "incomplete_coverage", "timeout",
    "auth_expired", "command_error"}`.
-   `reason` *(string, required)*: Explanation of when and why to execute this
    fallback.
-   `param_mapping` *(object, optional)*: Maps target fallback parameter names
    to template expressions interpolated from the current action's parameters or
    outputs (e.g. `{"project": "${project}", "cluster": "${cluster_name}"}`).

-   `auto_execute` *(boolean, optional, default false)*: Whether
    `python_glue_runner.py --auto-fallback` may automatically chain to this
    fallback when triggered.

```json
{
  "fallback": [
    {
      "action": "[[fallback-action-name]]",
      "when": ["empty_result", "incomplete_coverage"],
      "param_mapping": {
        "project": "${project}",
        "cluster": "${cluster_name}"
      },
      "reason": "Fallback to secondary action when primary query returns empty or partial coverage.",
      "auto_execute": false
    }
  ]
}
```

#### Format 1: Standard Command Action

```json
{
  "title": "[Action Title]",
  "type": "action",
  "tags": ["[tag1]"],
  "created": "YYYY-MM-DD",
  "updated": "YYYY-MM-DD",
  "entrypoints": ["troubleshoot"],
  "description": "[Brief description of standard action]",
  "requires_approval": false,
  "summary_fields": [
    "total_found",
    "deduplicated_count",
    "error_logs_count",
    "output_file"
  ],
  "parameters": {
    "[parameter_name]": { "type": "string" }
  },
  "instruction": "[command sequence with parameter placeholders, e.g. kubectl get pods | grep ${workload_name}]",
  "execution_mode": "command"
}
```

#### Format 2: LLM-Operated Action (Agent Prompt Task)

For actions executed by spawning a specialized subagent, name the file with an
`ai-` prefix, set `instruction` to `"ai-action"`, set `execution_mode` to
`"ai"`, and provide a `prompt` template. Batch sizing and worker concurrency are
governed by built-in runtime defaults in `python_glue_runner.py`.

The executing agent MUST interpolate the action's declared parameters into the
`prompt` template — supporting both the `${parameter_name}` and
`<parameter_name>` placeholder styles — and then dispatch that interpolated
prompt as a subagent task. It MUST NOT pass the action to the glue runner, and
MUST NOT substitute its own ad-hoc reasoning for the prompt.

```json
{
  "title": "[AI Action Title]",
  "type": "action",
  "tags": ["[tag1]"],
  "created": "YYYY-MM-DD",
  "updated": "YYYY-MM-DD",
  "entrypoints": ["troubleshoot"],
  "description": "[Brief description of AI action]",
  "requires_approval": false,
  "parameters": {
    "[parameter_name]": { "type": "string" }
  },
  "instruction": "ai-action",
  "execution_mode": "ai",
  "prompt": "[Prompt template for LLM subagent. Interpolate variables using ${parameter_name}. Request JSON output formatting explicitly, e.g. Output a JSON list of objects...]"
}
```

#### Format 3: Manual Action

For manual or informational actions that do not support automated execution, set
`execution_mode` to `"manual"`. The `instruction` field should contain a
detailed, step-by-step instruction describing exactly what the user or the
reading agent should do.

```json
{
  "title": "[Manual Action Title]",
  "type": "action",
  "tags": ["[tag1]"],
  "created": "YYYY-MM-DD",
  "updated": "YYYY-MM-DD",
  "entrypoints": ["troubleshoot"],
  "description": "[Brief summary of the action scope]",
  "instruction": "[Detailed, step-by-step instruction of what the operator must perform]",
  "execution_mode": "manual"
}
```

## Portability & Environment Setup

To execute diagnostic Python scripts (such as `pull_logs.py`) directly, set up a
local virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

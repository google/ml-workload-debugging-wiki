
# ML Workload Debugging Wiki External - Claude Code Guide

## Workspace & Wiki Architecture

The **ML Workload Debugging Wiki** is a structured, LLM-executable knowledge
base and diagnostic framework for ML training and inference workload issues
(Cloud TPU, GPU, Pathways, JAX, XLA, GKE).

### Workspace Directory Structure

-   `.claude/`: Claude Code configurations, slash commands, and subagent prompt
    definitions.
    -   `commands/`: Slash commands (`troubleshoot.md`, `handoff.md`,
        `dashboard.md`).
    -   `agents/`: Subagent instruction prompts (`router.md`, `practitioner.md`,
        `maintainer.md`, `reviewer.md`, `advisor.md`).
    -   `settings.json`: Workspace configuration and execution rules.
-   `_wiki/`: Core wiki knowledge base, schemas, workflows, and tools.
    -   `SCHEMA.md`: Canonical schema definition for page types, frontmatter
        requirements, tags, workflows, and actions.
    -   `INDEX.md`: Master entry index mapping concepts, symptoms, playbooks,
        tools, entities, workflows, actions, and cases.
    -   `LOG.md`: Audit log tracking all mutations, diagnostic sessions, and
        lint runs.
    -   `concepts/`: High-level concepts and system component overviews.
    -   `symptoms/`: Failure modes, error signatures, log patterns, and
        verification steps.
    -   `playbooks/`: Narrative, heuristic troubleshooting guides and
        flowcharts.
    -   `workflows/`: Structured, automated SOPs in JSON format.
    -   `actions/`: Atomic diagnostic execution steps in JSON format.
    -   `tools/`: Diagnostic tools, scripts, and utility documentations.
    -   `entities/`: Hardware, software, and platform entities (TPU, GPU, JAX,
        XLA).
    -   `cases/`: Narrative and failure table case studies of past incidents.
    -   `tools/scripts/`: Diagnostic runner and synchronization utilities.
-   `_agents/`: Antigravity multi-agent definitions and configurations.

--------------------------------------------------------------------------------

## Slash Commands

Claude Code slash commands are located under `.claude/commands/`:

-   **`/troubleshoot <workload_id>`** (`.claude/commands/troubleshoot.md`):
    Executes an end-to-end multi-agent diagnostic sequence for a target ML
    workload ID using strictly public tools and data sources (Cloud Logging, GKE
    cluster events, Kubelet/Node logs) without requiring Google-internal
    permissions or tools. Orchestrates `router`, `practitioner`, `reviewer`, and
    `advisor`.

-   **`/handoff <workload_id> [case_id] [--bucket=<name>]`**
    (`.claude/commands/handoff.md`): Packages `/troubleshoot` findings into a
    customer GCS bucket (`gs://<bucket>/<storage_key>/`), publishes the
    manifests, failure tables and READMEs, verifies the bucket holds real
    content (`verify-handoff-package` must return `VERIFIED`), stages the
    partner issue tracker comment for approval, and arms bucket monitoring. Also
    `--resume`, `--watch` and `--resolve` modes. There is no default bucket.

-   **`/dashboard [target]`** (`.claude/commands/dashboard.md`): Configures and
    launches the GKE Capacity & Triage Dashboard daemon scoped to a GCP project
    ID, GKE cluster name, or target ML workload ID.

### Antigravity counterparts

The same entry points exist for Antigravity as **skills** under
`_agents/skills/<name>/SKILL.md`, registered by `_agents/skills.json`:
`troubleshoot/`, `handoff/`, `dashboard/`.

They are not slash commands. Antigravity has no command registry — a skill is
selected by its `description` frontmatter, which is why each description carries
explicit `Use when ...` triggers (including the literal `/troubleshoot`,
`/handoff`, `/dashboard` strings) and a `Don't use for ...` clause pointing at
its siblings. Weakening a description silently stops the skill being selected.

Keep the two harnesses in step: a command added to `.claude/commands/` without
its skill counterpart simply does not exist for Antigravity users, and vice
versa.

Both sides are maintained **by hand** — there is no generator. Beyond the name
sets, keep these in step:

-   every profile carries all seven **Grounding** rules, in both harnesses;
-   paired profiles reference the **same set of wiki artifacts** (actions,
    workflows, pages), so wording may differ but the Actions an agent is told to
    run may not;
-   paired commands and skills reference the same set too.

Wording is deliberately allowed to diverge — one side is Markdown, the other a
JSON string — but a rule or an Action reference added to one side and not the
other is a test failure, not a style difference.

Tool mapping when porting between them:

| Claude Code                       | Antigravity                             |
| :-------------------------------- | :-------------------------------------- |
| `Agent` tool with `subagent_type` | `invoke_subagent` with a `wiki_*` profile |
| `Bash` | `run_command` (persistent terminal; reuse `TerminalID`) |
| `Read` / `Grep` / `Glob` | `view_file` / `grep_search` / `find_by_name` |
| `.claude/agents/<name>.md`        | `_agents/agents/wiki_<name>/agent.json` |

### Shared Grounding Rules (Enforced Across All Subagents)

Every subagent (`router`, `practitioner`, `maintainer`, `reviewer`, `advisor`)
enforces these 7 core Grounding rules (and carries them inline in its subagent
system prompt because subagent invocations run in isolated context windows
verified by the internal test suite):

1.  **Cite the artifact**: Name the wiki page, action, or workflow backing each
    recommendation using `[[name]]`.
2.  **Check the citation fits**: A cited action must apply to the target
    platform (e.g., GKE-managed TPUs use `isolate-node-via-rr-label` via
    `node-isolation-and-rr-api`).
3.  **Label what the wiki does not support**: Mark operator reasoning not backed
    by a wiki page as `unsupported - operator judgement`.
4.  **Flag circular grounding**: Note when a cited page has `sources: []` and
    was created from the same single incident.
5.  **Never invent**: Never fabricate action names, parameter names, host names,
    file paths, or metric values.
6.  **Distinguish quoted from inferred**: State which log line or command output
    supports a claim vs. derived values.
7.  **Truncation is not absence**: Any query hitting its entry cap
    (`original_count == limit`) or read via `head`/`tail` must be re-run
    narrowed before concluding absence.

--------------------------------------------------------------------------------

## Supported Command-Line Tools

-   **Wiki Linter**: `python3 _wiki/wiki_linter.py` Takes no arguments. Audits
    wiki structure, frontmatter schemas, link integrity, orphan pages, taxonomy
    tags, source hashes, portability, blocking commands, and duplicate actions;
    appends the session lint entry to `_wiki/LOG.md` and rotates it past 500
    entries.

-   **Action / Python Glue Runner**: `python3
    _wiki/tools/scripts/python_glue_runner.py <action_name> '<parameters_json>'`
    Executes `execution_mode: "command"` diagnostic actions and Python glue
    scripts. **Not** for `ai-` actions — see below.

-   **Agent Backend Probe**: `python3 _wiki/tools/scripts/agent_dispatch.py`
    Reports which agent backend is reachable from this machine (Antigravity SDK
    / `agentapi`, or the `claude` CLI) and why the others are not.
    `agent_dispatch.py` is the single place that knows how to reach a model; the
    glue runner, the eval harness, and the dashboard's AI triage all go through
    it. **Anything started from Claude Code dispatches to Claude Code**
    (detected via `CLAUDECODE=1`), even where Antigravity is also installed;
    `/dashboard` pins the daemon's backend the same way and reports it as
    `agent_backend`. `WIKI_AGENT_BACKEND=antigravity|claude|none` overrides
    this; without either, detection prefers Antigravity.
    `_wiki/query_interface.py` is Antigravity-only and exits with code 3 under
    Claude Code. Where no backend is reachable, an `ai-` action fails with an
    explanation and exit code **3** rather than a traceback — that is the
    expected state under Claude Code, where `ai-` actions belong on the subagent
    path, not in the runner.

--------------------------------------------------------------------------------

## Subagent Roles

Specialized subagents are defined under `.claude/agents/` (Claude Code) and
`_agents/agents/` (Antigravity):

-   **`router`** (`wiki_router` / `.claude/agents/router.md`): Entry point
    triage agent that retrieves application/telemetry logs and matches error
    signatures against `_wiki/symptoms/` and `_wiki/INDEX.md`.
-   **`practitioner`** (`wiki_practitioner` / `.claude/agents/practitioner.md`):
    Diagnostic execution agent that interprets JSON workflows and executes
    action steps via `python3 _wiki/tools/scripts/python_glue_runner.py`.
-   **`maintainer`** (`wiki_maintainer` / `.claude/agents/maintainer.md`):
    Maintenance agent running `python3 _wiki/wiki_linter.py`, auto-updating
    `_wiki/INDEX.md`, rotating `_wiki/LOG.md`, and enforcing zero-dependency
    portability.
-   **`reviewer`** (`wiki_reviewer` / `.claude/agents/reviewer.md`): Quality
    control and schema auditor verifying strict compliance against
    `_wiki/SCHEMA.md`, link integrity, and `_wiki/LOG.md` entries.
-   **`advisor`** (`wiki_advisor` / `.claude/agents/advisor.md`): Recommendation
    agent resolving query ambiguities and guiding workflow selection when
    multiple matches exist.

--------------------------------------------------------------------------------

## Action Execution Modes

Every Action in `_wiki/actions/` declares an `execution_mode`. **Always read it
before executing — the three modes dispatch differently, and an Action is not
necessarily a shell command.**

| `execution_mode` | Naming convention | Dispatch |
| ---------------- | ---------- | ----------------------------------------- |
| `"command"` | any | `python3 _wiki/tools/scripts/python_glue_runner.py <action_name> '<parameters_json>'` |
| `"ai"` | **`ai-` prefix** | **Agent prompt task** — spawn a subagent with the Action's interpolated `prompt` |
| `"manual"` | any | Surface the `instruction` text to the user; do not automate |

**`ai-` prefixed Actions are agent prompt tasks, not shell commands.** They
always carry `"instruction": "ai-action"` plus a `prompt` template. To execute
one: read the `prompt` verbatim, interpolate every declared parameter (both
`${name}` and `<name>` placeholder styles), then dispatch it as a subagent task
via the `Agent` tool — honouring the Action's `model`, `batch_size`, and
`max_concurrent_workers` hints. Do **not** route them through the glue runner,
and do **not** improvise an equivalent analysis in place of the prompt; if a
prompt task cannot be dispatched, say so explicitly rather than hand-rolling a
substitute.

The `ai-` filename prefix and `"execution_mode": "ai"` must always agree — the
prefix is the at-a-glance signal, `execution_mode` is authoritative.

--------------------------------------------------------------------------------

## Development & Logging Guidelines

-   **Grouped LOG.md Entries**: Every mutating operation (diagnostic session,
    workflow execution, edit) must append a structured entry to `_wiki/LOG.md`
    (`## [YYYY-MM-DD] action | subject`). Multiple operations from the same
    session must be grouped together under a single descriptive header.
-   **Strict Frontmatter**: All Markdown pages must begin with valid YAML
    frontmatter matching `_wiki/SCHEMA.md`.
-   **File Hygiene**: Temporary files, scratch scripts, or debug logs must be
    placed exclusively inside `_wiki/scratch/`. Never write un-schema'd one-off
    scripts to the root of the workspace.
-   **Portability & Relative Paths**: All paths in documentation, commands,
    agents, and prompts must use relative paths from the workspace root (e.g.
    `_wiki/...`, `.claude/...`). Hardcoded workspace absolute paths (e.g.
    `/workspace/`, `/workspace/internal/`) are strictly forbidden.
-   **Virtual Environment Setup**: When executing python tools requiring
    dependencies, use `_wiki/.venv` (built from `_wiki/requirements.txt`).

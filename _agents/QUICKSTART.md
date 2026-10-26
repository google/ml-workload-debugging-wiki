# Quickstart — Antigravity

Setup and dispatch for the ML Workload Debugging Wiki under Antigravity. For
what the diagnosis actually does and how to judge it, read `_wiki/TUTORIAL.md`.

## Setup

1.  Authenticate to Google Cloud (`gcloud auth login --update-adc`) — most
    actions read Cloud Logging.
2.  Build the environment once:

    ```bash
    python3 -m venv _wiki/.venv && _wiki/.venv/bin/pip install -r _wiki/requirements.txt
    ```
3.  Antigravity dispatch needs the `agentapi` CLI and a reachable local agent
    daemon (`ANTIGRAVITY_LS_ADDRESS`, default `127.0.0.1:8081`). Without them
    the diagnostic agents cannot be spawned at all — that is the first thing to
    check if nothing happens.

## Diagnose a workload

```
/troubleshoot <workload_id>
```

## Selection is by description, not by name

Antigravity has **no slash-command registry**. Skills live in
`_agents/skills/<name>/SKILL.md` and are registered by `_agents/skills.json`;
which one runs is decided by the `description` frontmatter.

That has a practical consequence: typing `/troubleshoot` works because the
description names that trigger explicitly, and plain English works equally well
("why did jobset my-tpu-workload keep restarting?"). It also means **editing a
description changes routing** — weaken the `Use when ...` triggers or drop the
`Don't use for ...` clause and the skill quietly stops being selected, or starts
being selected for things it should decline.

## How dispatch works here

-   **Subagents** are profiles in `_agents/agents/wiki_*/agent.json`, invoked
    with `invoke_subagent`: `wiki_router`, `wiki_practitioner`, `wiki_advisor`,
    `wiki_maintainer`, `wiki_reviewer`.
-   **`command` actions** run via `run_command`:

    ```bash
    python3 _wiki/tools/scripts/python_glue_runner.py <action_name> '<parameters_json>'
    ```
-   **`ai-` actions** are *not* shell commands. Interpolate the action's
    `prompt` and dispatch it with `invoke_subagent`.

## Two harness details that matter

-   **Use one persistent terminal.** Start it with `RunPersistent: true` and
    reuse its `TerminalID` for every command so environment variables and active
    credentials persist across invocations.
-   **`_agents/scripts/boundary_gate.py`** is a `PreToolUse` hook on the file
    and command tools. If it blocks something, re-scope the call into the wiki
    rather than working around it.

## Permissions

Actions that change cluster state declare `requires_approval: true` and will not
run until you approve. Read the command first — the set includes node tainting,
cordoning, and cluster upgrades.

## Verify your setup

```bash
_wiki/.venv/bin/python3 _wiki/wiki_linter.py     # expect: status PASS, 0 issues
```

The linter takes no arguments; any argument exits 2.

## Where things live

Path                                         | What
:------------------------------------------- | :---
`_agents/skills/`                            | Skill definitions (`SKILL.md` per directory)
`_agents/agents/`                            | Subagent profiles (`agent.json`)
`_agents/skills.json`, `_agents/agents.json` | Registration
`_wiki/actions/`, `_wiki/workflows/`         | Executable procedure
`_wiki/INDEX.md`                             | Catalogue of everything
`_wiki/TUTORIAL.md`                          | The diagnostic walkthrough

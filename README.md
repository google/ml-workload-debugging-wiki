# ML Workload Debugging Wiki

**https://github.com/google/ml-workload-debugging-wiki**

Diagnose failures in large-scale ML training and inference workloads — Cloud
TPU, GPU, Pathways, JAX, XLA and GKE — using a knowledge base that both people
and coding agents can execute.

Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0.

> This is not an officially supported Google product. We share it as a
> reference implementation of agent-executable operational knowledge.

---

Debugging a stalled 2,048-chip training run is mostly an exercise in knowing
where to look: which log holds the first real error, which of 56 restart
attempts matters, whether a `DATA_LOSS` message names the culprit host or just
a device that noticed. That knowledge usually lives in the heads of a few
on-call engineers and in scattered incident docs.

This project writes it down in a form you can run. Prose pages explain a
failure mode; JSON workflows execute the diagnostic steps against your
cluster.

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Example: diagnosing a workload](#example-diagnosing-a-workload)
- [How actions run](#how-actions-run)
- [Using it with a coding agent](#using-it-with-a-coding-agent)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Getting help](#getting-help)
- [Contributing](#contributing)
- [License](#license)

## What you get

Around 250 pages, organised by **page type**, each with a schema defined in
[`_wiki/SCHEMA.md`](_wiki/SCHEMA.md):

| Type | Directory | What it holds |
| :--- | :--- | :--- |
| `concept` | `concepts/` | How a component works (e.g. the libtpu program lifecycle) |
| `symptom` | `symptoms/` | A failure mode, its log signatures, how to verify it |
| `playbook` | `playbooks/` | Narrative troubleshooting guidance |
| `entity` | `entities/` | Platform entities (TPU, GPU, JAX, XLA) |
| `tool` | `tools/` | A diagnostic tool and how to drive it |
| `workflow` | `workflows/` | An automated procedure: ordered steps, as JSON |
| `action` | `actions/` | One atomic diagnostic step, as JSON |

Pages cross-reference each other with `[[wikilink]]` syntax, and
[`_wiki/INDEX.md`](_wiki/INDEX.md) indexes all of them.

The distinction that matters most is **playbook vs. workflow**. Reach for a
playbook when you are reasoning heuristically and the picture is unclear. Reach
for a workflow when you want the same parameterised steps to run the same way
every time.

## Quick start

You need Python 3.9 or newer (tested on 3.13). Actions that query live
infrastructure also need the Google Cloud SDK — `gcloud` and `kubectl` —
authenticated against the project you are debugging.

```bash
git clone https://github.com/google/ml-workload-debugging-wiki
cd ml-workload-debugging-wiki

python3 -m venv .venv
source .venv/bin/activate
pip install -r _wiki/requirements.txt
```

Confirm the wiki is intact:

```bash
python3 _wiki/wiki_linter.py
```

The linter takes no arguments. It validates frontmatter against the schema,
link integrity, orphaned pages, tag vocabulary and duplicate actions, printing
a JSON report. A healthy tree reports `"status": "PASS"` with zero issues. Run
it before you send a change.

## Example: diagnosing a workload

Start from the symptom. If a training job hangs and the logs show workers
stuck on a synchronisation flag, open
[`_wiki/symptoms/wait-timeout-on-sflag.md`](_wiki/symptoms/wait-timeout-on-sflag.md).
Its `## Connections` section names the concepts and workflows that apply.

Run one step of that investigation directly:

```bash
python3 _wiki/tools/scripts/python_glue_runner.py \
  resolve-workload-nodes '{"workload_name": "my-job", "project_id": "my-project"}'
```

The runner reads `_wiki/actions/resolve-workload-nodes.json`, interpolates your
parameters into its command, runs it, and prints the action's declared
`summary_fields` — here the node set and whether it is complete.

Read the action file before running anything unfamiliar. Actions that change
cluster state carry `"requires_approval": true`, and several of them cordon
nodes or remove hosts from service.

## How actions run

An action is not necessarily a shell command. Dispatching one the wrong way is
the most common integration mistake:

| `execution_mode` | Naming | How it runs |
| :--- | :--- | :--- |
| `command` | any | Through the glue runner, as above |
| `ai` | **`ai-` prefix** | As an agent prompt task, using the action's `prompt` |
| `manual` | any | Surfaced to a person; never automated |

The `ai-` filename prefix and `"execution_mode": "ai"` always agree. The prefix
is the at-a-glance signal; the field decides.

## Using it with a coding agent

The wiki ships agent definitions for two harnesses: `.claude/agents/` for
Claude Code, and `_agents/agents/` for the Antigravity harness. A test fails if
the two drift apart.

| Agent | Role |
| :--- | :--- |
| `router` | Match an error signature to a symptom page and workflow |
| `practitioner` | Interpret a workflow and execute its actions |
| `advisor` | Resolve ambiguity when several workflows match |
| `reviewer` | Audit schema compliance and link integrity |
| `maintainer` | Run the linter, keep the index and log current |

Under Claude Code, `/troubleshoot <workload_id>` runs the whole sequence.

Every profile carries the same **grounding rules**: cite the page behind each
claim, label whatever the wiki does not support, never invent an action or host
name, and separate what you read from what you inferred. Those rules exist
because a confident but unsupported diagnosis is worse than none — the reader
cannot tell the difference.

## Repository layout

```
_wiki/              Knowledge base, schema, tooling
  SCHEMA.md           Page types, required frontmatter, tag taxonomy
  INDEX.md            Index of every page
  TUTORIAL.md         End-to-end walkthrough
  LOG.md              Append-only record of changes and lint runs
  wiki_linter.py      Structural and schema validation
  query_interface.py  Programmatic search over the wiki
  tools/scripts/      Diagnostic runners, including the glue runner
_agents/            Agent definitions for the Antigravity harness
.claude/            Agent definitions and slash commands for Claude Code
docs/               Contributing guide and code of conduct
```

## Documentation

- [`_wiki/SCHEMA.md`](_wiki/SCHEMA.md) — page types, frontmatter, tags. Read
  this before adding a page.
- [`_wiki/INDEX.md`](_wiki/INDEX.md) — the index; the best way to browse.
- [`_wiki/TUTORIAL.md`](_wiki/TUTORIAL.md) — a walkthrough of diagnosing a
  failing workload end to end.
- [`_wiki/LOG.md`](_wiki/LOG.md) — what changed, when, and why.

`_wiki/query_interface.py` offers a programmatic way to search the wiki when
you would rather not grep.

## Getting help

Open an issue at
https://github.com/google/ml-workload-debugging-wiki/issues. For a page that
contradicts reality, say which page and what you observed instead — that is the
most useful bug report this project receives.

## Contributing

See [docs/contributing.md](docs/contributing.md). Contributions need a Google
CLA and follow the
[Community Guidelines](https://opensource.google/conduct/).

The most valuable changes: a symptom page for a failure mode nobody has written
up, a workflow that codifies steps people still do by hand, or a correction
where the documented behaviour has drifted from the real system. New pages must
satisfy [`_wiki/SCHEMA.md`](_wiki/SCHEMA.md) and pass the linter.

## License

Apache License 2.0. See [LICENSE](LICENSE).

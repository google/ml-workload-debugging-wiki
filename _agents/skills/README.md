# Antigravity Skills

Registration point for this workspace's Antigravity skills.
`_agents/skills.json` declares this directory, mirroring how
`_agents/agents.json` declares `_agents/agents/` and `_agents/hooks.json`
references `_agents/scripts/`.

## Skills in this workspace

Skill           | Mirrors                            | Purpose
:-------------- | :--------------------------------- | :------
`troubleshoot/` | `.claude/commands/troubleshoot.md` | End-to-end multi-agent diagnosis of a workload id (public telemetry)
`handoff/`      | `.claude/commands/handoff.md`      | Package `/troubleshoot` findings into a GCS handoff bucket and issue comment
`dashboard/`    | `.claude/commands/dashboard.md`    | Configure and launch GKE Capacity & Triage Dashboard

Each is the Antigravity counterpart of the Claude Code slash command it mirrors.
The table above lists the skills available in this workspace. The diagnostics
themselves still live in the `wiki_*` agents under `_agents/agents/` and in the
wiki's own Actions and Workflows; a skill is the entry point that sequences
them, not a second copy of the logic.

**Selection is by description, not by name.** Antigravity has no slash-command
registry: the `description` frontmatter is what decides whether a skill is
chosen. That is why each description spells out its triggers, including the
literal `/<name>` string of the command it mirrors — so a user typing the Claude
Code form still lands on the right skill — and a `Don't use for ...` clause
pointing at its siblings. Edit those descriptions with care; weakening them
silently stops the skill from being selected.

**No BUILD files.** The v1 skills carry `agent_skill` / `validate_skill_test`
Bazel rules, but this workspace has no BUILD packages under `_agents/` and
registers skills at runtime through `_agents/skills.json` instead. Adding BUILD
rules here would not match how the workspace loads them.

## Adding a skill

One directory per skill, each containing a `SKILL.md` with YAML frontmatter:

```markdown
---
name: my-skill
description: >-
  What it does, and when to use it. Include explicit "Use when ..." and
  "Don't use for ..." triggers -- this text drives skill selection.
---

# My Skill

...instructions...
```

## Related

-   Paths here must stay relative to the workspace root. Hardcoded workspace
    absolute paths (`/path/to/workspace/`, `/workspace/internal/`) are forbidden
    — see the portability rules in `CLAUDE.md`.

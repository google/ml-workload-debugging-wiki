# TUTORIAL.md

A walkthrough for diagnosing a failing ML workload with this wiki.

Written for the **operator**: you have a workload that hung, crashed, restarted
or was allegedly preempted, and you want a root cause you can act on. It is
harness-neutral — for setup and the exact way to invoke things, read the
quickstart for whichever harness you use, then come back here:

-   Claude Code → `.claude/QUICKSTART.md`
-   Antigravity → `_agents/QUICKSTART.md`

For the schema and page taxonomy see `SCHEMA.md`; for the catalogue of every
concept, symptom, action and workflow see `INDEX.md`.

--------------------------------------------------------------------------------

## 1. What this wiki actually is

Two things at once, and it helps to keep them apart:

-   **Knowledge** — Markdown pages describing concepts, symptoms, entities and
    past cases. Prose, for a human or an agent to read.
-   **Executable procedure** — JSON *actions* (one diagnostic step) composed
    into *workflows* (an ordered sequence). These run; they are not
    descriptions.

A diagnosis is an agent reading the knowledge to decide which procedure to run,
running it, and interpreting what came back. The wiki's value is less the
commands — you could type those yourself — and more the accumulated record of
**how these investigations go wrong**.

## 2. Before you start

-   **Python environment.** `_wiki/.venv`, built from `_wiki/requirements.txt`.
    If its interpreter link is dangling (built on another platform), use
    `_wiki/.venv-<platform>` instead, e.g. `_wiki/.venv-darwin` on macOS.
    Several scripts need `pyyaml` and the Cloud Logging client.
-   **What you need to hand**: the workload/jobset id, the GCP project, and
    roughly when it went wrong. The time window matters more than you would
    expect — see the warning in 4.1.

## 3. The short path

```
/troubleshoot <workload_id>
```

That runs the whole sequence. If you want to know what it did and whether to
believe it, keep reading.

## 4. What happens, in order

The command runs four milestones. Each has a subagent whose role is narrow on
purpose — the one that runs commands is not the one that decides whether the
conclusion is sound.

### 4.1 Reconnaissance

Establishes that the workload exists, recovers its cluster and location, pulls
application and telemetry logs, and sweeps for crash signatures across the
**whole run window**.

> **The window is mandatory, not optional.** A crash sweep with no explicit
> `start`/`end` falls back to the logging default of one day and returns nothing
> for any older incident — while still exiting successfully. Zero results and
> "could not look" are indistinguishable unless you passed a window.

The sweep runs *independently of any restart timeline*, and that is deliberate:
a timeline derived from restarts cannot see a disruption the workload
**recovered from**, and those are often the most informative events in the run.

### 4.2 Diagnostics

An advisor resolves which workflow fits when several symptoms match, then a
practitioner executes it step by step.

Every action declares an `execution_mode`, and they dispatch differently:

| `execution_mode`   | What it is               | How it runs              |
| :----------------- | :----------------------- | :----------------------- |
| `command`          | A shell command          | Through the glue runner  |
| `ai` (always named `ai-*`) | A prompt for an agent | Dispatched as a subagent task |
| `manual` | Instructions for a human | Shown to you; never automated |

An `ai-` action is **not** a shell command. Handing one to the glue runner, or
paraphrasing its prompt instead of running it, produces confident output with
nothing behind it.

**Mutating actions stop and ask.** Anything that changes cluster or fleet state
— cordoning a node, tainting it, upgrading a cluster, triggering a repair —
declares `requires_approval: true` and will not run until you say so. If you are
ever asked to approve something you did not expect, that is the system working;
read the command before agreeing.

### 4.3 Baseline disruption evidence

Whenever a disruption occurred, four steps run regardless of which
symptom-specific workflow was chosen, because they are what makes an attribution
defensible:

1.  **Resolve the node set.** Confirm `status: COMPLETE`. If it says
    `INCOMPLETE`, every "no hardware event" conclusion downstream is
    *inconclusive*, not negative.
2.  **Quote the culprit hosts** from the logs that name them literally.
3.  **Query the audit log, scoped to those nodes.**
4.  **Query in-node health events, scoped to those nodes.**

Steps 3 and 4 need *different* scoping clauses, and they are not interchangeable
— one matches a field that exists only on audit entries, the other a field that
exists only on node logs. Using the wrong one silently returns other workloads'
events on other clusters.

### 4.4 Review

A linter run, then an adversarial check that re-derives every attribution from
primary logs and flags anything inferred rather than quoted. A failure table is
not published until that check approves it.

## 5. Knowing when it could not help you

The honest failure modes, and how to spot them:

-   **`status: INCOMPLETE`** on the node set — the scope is partial, so
    negatives are inconclusive.
-   **`original_count == limit`** — the query saturated. Narrow the filter and
    re-run before believing a zero.
-   **No symptom match** — a legitimate outcome. It means the signature is not
    in the catalogue yet, not that nothing is wrong. Better than a page
    stretched to fit.
-   **An unscoped verdict** — a bare "No VM Termination" with no statement of
    what was checked. A sound verdict names its scope: *"no VM termination
    across all 65 nodes in 4 node groups, 07:45–10:05"*.

## 6. A worked example

Consider a real multi-slice TPU incident (see [[sigill-worker-kill]] and
[[elastic-vs-permanent-failure]]) where the first answer was wrong four separate
times before it was right.

The run: 128 chips over 2 slices, 65 nodes. Three workers killed by `SIGILL`
with `si_code=0` — external kills, not CPU faults. The workload absorbed the
first and died on the other two, and the important finding is *why*: not because
those faults were worse, but because elastic recovery failed in two different
ways. Both hard failures were recovery-path defects, and the recovered event is
what proved it.

One host killed a worker twice, 49 minutes apart, with different pods scheduled
onto it — which is what justified recommending it be taken out of service, on
distinct incidents rather than raw event counts. Counting log lines instead
would have inflated 3 real failures into 29 and nominated 7 healthy nodes.

## 7. Where to go next

-   `INDEX.md` — every concept, symptom, action and workflow.
-   `SCHEMA.md` — page types and frontmatter rules.
-   `symptoms/` — signatures, and how to verify each one.
-   `playbooks/` — diagnostic strategies and operational runbooks.

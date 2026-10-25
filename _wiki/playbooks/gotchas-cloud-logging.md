---
title: Cloud Logging Gotchas
type: playbook
internal_only: false
tags: [tpu, training]
created: 2026-06-13
updated: 2026-10-06
---


## Cloud Logging Gotchas & Tips

<!-- Gotchas 1-2 removed as generic; numbering kept because other pages cite Gotcha 9-11 by number. -->

### 3. Inexact string matching

If a phrase search fails, search individual keywords (e.g. `scope` AND
`authorized`).

### 4. Timezone offsets

User reports may use local time; Cloud Logging uses UTC. If nothing is found,
shift the window by common offsets.

### 5. Exception message comes after the traceback

`gcloud logging read` returns newest first. Python prints the exception message
(e.g. `AbortedError`) *after* the traceback, so it has a newer timestamp. An
upper time bound set at the traceback time cuts it off; widen the window past
it.

### 6. Keyword over-filtering

Searching only for "fatal" or "hang" misses errors worded differently (e.g.
"Evicted", quota errors). Start with a time window and one worker and read raw
logs.

### 7. Cloud Logging before `kubectl`

With a workload name, query `gcloud logging read` with
`resource.labels.pod_name:"${workload_name}-"` instead of `kubectl get pods`. It
works post-mortem and avoids cluster connection/auth timeouts.

### 8. Newest-first order plus `--limit` hides older runs

For a series of runs (e.g. `my-training-job-4` to `-8`), a prefix query `my-training-job-`
with `--limit` returns only the latest run. Query the exact suffix (e.g.
`my-training-job-4-`) or narrow the timestamp.

### 9. Critical signals are logged at INFO

`Wait timeout on sflag` (JAX collective timeouts) and GKE VM preemption events
are logged at `INFO`. Don't filter `severity>=ERROR` early; read the unfiltered
stream around the failure first (see [[cloud-logging]]).

### 10. Earliest exit code gets truncated in large clusters

With 1024+ workers and a limit smaller than the cluster, unsorted or DESC
queries drop the first worker to exit (the culprit). Sort ascending
(`find-earliest-failure` action or `| sort | head`). The first failure often has
a different code (e.g. 1 or

134\) than the cascade teardown (all 143).

### 11. Stale errors from earlier runs or retries

Errors earlier in the window (a recovered TPU OOM, a previous run with the same
pod names) are red herrings. Check that the error timestamp sits right before
the exit code log.

### 12. `logName` needs contains, not equality

Log names are full paths (`projects/PROJECT_ID/logs/events`), so
`logName="events"` returns 0 results. Use `logName:events`.

### 13. Audit logs: recover a deleted JobSet's spec from `jobs.create`

A deleted JobSet's affinity, node selectors and queue are still in the GKE audit
log, but `jobsets.create` may not be there. Query
`protoPayload.methodName="io.k8s.batch.v1.jobs.create"` from the
jobset-controller in the run's creation window and read
`protoPayload.request.spec.template.spec`. Never assume a run had the same spec
as its siblings.

### 14. Keep queries small and scoped

-   Scope every query by namespace, pod-name prefix of the exact run, and that
    run's time window; exclude `kube-system` (device-plugin noise matches words
    like `HANG`) and shell trace lines (`+ ...`) that echo flags.
-   Use `--format="value(timestamp,resource.labels.pod_name,textPayload)"` for
    large reads; `--format=json` with thousands of entries can produce hundreds
    of MB.
-   `kubectl get events` keeps only about 1 h; read the `events` log in Cloud
    Logging for anything older (Scheduled events grouped by node pool show where
    each run's pods landed).

## Connections

-   **Playbooks**:
    -   [[debugging-gotchas]]
-   **Tools**:
    -   [[cloud-logging]]

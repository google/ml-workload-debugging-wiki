---
title: '/bin/bash: argument list too long at 8k-chip scale (TPU_PROCESS_ADDRESSES)'
description: 'Containers of a large JobSet (8k chips) never start: `exec /bin/bash:
  argument list too long`. GKE-injected TPU_PROCESS_ADDRESSES (~200 KB at 8k chips)
  exceeds Linux MAX_ARG_STRLEN (128 KiB). Fix: shorten names, 2*len(jobset)+len(replicatedJob)
  <= 17 at 8192 chips (e.g. jobset `bigrun`, replicated job `slice`).'
type: symptom
internal_only: false
tags:
- tpu
- gke
- training
- scheduling
created: 2026-10-03
updated: 2026-10-03
---


# /bin/bash: argument list too long at 8k-chip scale

## Overview

GKE injects `TPU_PROCESS_ADDRESSES` into every TPU container: one
`{jobset}-{replicatedJob}-{jobIndex}-{podIndex}.{subdomain}:{port}` entry per
TPU process (2 per host), `subdomain` = JobSet name unless overridden. At 8k
chips it reaches ~200 KB, above the per-string limit `MAX_ARG_STRLEN` (128 KiB;
not `ARG_MAX`), so `exec` of the entrypoint fails. Not a code or image bug.

## Log Signatures

```
exec /bin/bash: argument list too long
/bin/bash: argument list too long
```

Pods are created and scheduled, but the container exits immediately at start, on
every pod, on every retry. Smaller runs of the same YAML work.

## Verification Steps

1.  Chip count >= ~4k and long JobSet / replicated-job names in the YAML.
2.  Measure the variable on a pod that got that far, or on a smaller run of the
    same YAML: [[inspect-pod-env-variables]], or `kubectl get pod <pod> -o json
    | jq -r '.spec.containers[].env[] | select(.name=="TPU_PROCESS_ADDRESSES") |
    .value' | wc -c`. More than 131072 bytes confirms it.
3.  Name budget (computed estimate, subdomain = JobSet name):
    `2*len(jobset_name) + len(replicatedJob_name)` <= 17 at 8192 chips, <= 50 at
    4096, <= 114 at 2048.

## Playbooks & Root Causes

-   **Fix (user):** shorten names to fit the budget, e.g. JobSet `bigrun` (6
    chars) and replicated job `slice` (5 chars) for 8k chips. Re-submit; no
    cluster change needed.
-   `TPU_WORKER_HOSTNAMES` grows the same way (deprecated).

## Connections

-   **Actions**:
    -   [[inspect-pod-env-variables]]
-   **Symptoms**:
    -   [[workload-startup-crash]]
-   **Concepts**:
    -   [[jobset-yaml-structure]]

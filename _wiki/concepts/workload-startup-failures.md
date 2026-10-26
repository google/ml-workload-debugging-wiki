---
title: Workload Startup and Configuration Failures
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-06-13
updated: 2026-10-03
---

## Overview

A [[workload-startup-crash]] is admitted by Kueue and scheduled, then dies
within seconds to minutes (typically <5 min) before any training step. It is a
user-program or config error and fails 100% of the time on retry until the
config or launch script changes.

## Signatures

*   Short duration; many or all pods exit with `EXIT_CODE=1`; a Python/C++
    traceback at the end of container `stderr` (startup crashes always print
    one, unlike hangs).
*   Patterns: `ValueError: ...`, `KeyError: ...`,
    `omegaconf.errors.ConfigAttributeError`.
*   **Empty argument bug (OmegaConf / MaxText)**: a trailing space in XPK
    `--args` (e.g. `--args="run_name=foo "`) becomes an empty argument `""` at
    the container entrypoint. `OmegaConf.from_cli` reads it as key `""` with
    value `True` and raises `ValueError: '' not in [...]`. Passing a key that is
    not in the base config also fails.

## Troubleshooting

1.  Skip JAX barrier timeouts and communication warnings. Query the **earliest
    logs** (first 60 seconds) of Worker 0:

    ```bash
    gcloud logging read 'resource.type="k8s_container" \
      AND resource.labels.pod_name:"<WORKLOAD_NAME>-slice-job-0-0-"' \
      --project <PROJECT_ID> \
      --limit 200
    ```
2.  Read the last 50 lines of container stderr for the stack trace.
3.  Check the exact process arguments in the container spec or start logs for
    trailing whitespace or `""`.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Symptoms**:
    -   [[stopiteration-disk-full]]
    -   [[workload-startup-crash]]
-   **Concepts**:
    -   [[exit-codes]]
-   **Entities**:
    -   [[maxtext]]

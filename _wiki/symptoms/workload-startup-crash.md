---
title: Workload Startup Crash (Exit Code 1)
type: symptom
internal_only: false
tags:
- tpu
- pathways
- training
created: 2026-06-13
updated: 2026-10-16
---


## Symptom signature

Job dies within 2-3 minutes of launch; several pods exit `EXIT_CODE=1` (not
`143` or `134`); no step or throughput output. Exit 1 is a strong hint of a
software startup error, not proof. Common errors:

-   OmegaConf config key:

    ```text
    ValueError: 'some_config_key' not in ['valid_key1', 'valid_key2', ...]
    ```

-   Empty argument from a trailing space in `--args`:

    ```text
    ValueError: '' not in ['emb_dim', 'mlp_dim', 'moe_mlp_dim', ...]
    ```

-   `ModuleNotFoundError: No module named '...'`

### Grain Elastic Iterator Local Device Batch Divisibility Failure

In [[maxtext]] workloads using Pathways Colocated Python ([[pathways-sidecar]])
with `grain_use_elastic_iterator=True`, `packing=False`, and
`grain_worker_count > 0` (e.g., `example-elastic-workload` /
`example-elastic-workload`), multiprocessing Grain workers can yield a per-host
batch slice whose leading dimension is not evenly divisible by the host's local
TPU device count (such as shape `(2, 8192)` across `4` local TPU v6e cores on
`v6e-8`). When `multihost_dataloading.py:form_global_array_colocated_python`
calls `np.split(array, len(devices), axis=0)`, the head pod (`pathways-head`)
crashes with `EXIT_CODE=1` and emits the following 3-part signature:

```text
ValueError: array split does not result in an equal division
...
  File "/deps/src/maxtext/input_pipeline/multihost_dataloading.py", line 212, in form_global_array_colocated_python
ValueError: Unable to put to devices shape (2, 8192) with local device count 4 at ['inputs']
...
While processing examples for unknown_output_specs_and_push_result.1527570490078541891 [type.googleapis.com/ifrt.UserContext='7841080109532081157'] [type.googleapis.com/pathwaysutils.PyException='']
```

Distinguish this data-loader batch-slicing `ValueError` from
`jax._src.sharding.IndivisibleError` ([[elastic-reshard-pytree-mismatch]]),
which occurs when a non-power-of-2 elastic device mesh cannot evenly partition a
model weight tensor dimension.

## Diagnostic Steps

1.  Confirm exit codes; several `EXIT_CODE=1` and a job under 5 minutes =
    startup crash:

    ```bash
    gcloud logging read 'resource.type="k8s_container" \
      AND resource.labels.pod_name:"<WORKLOAD_NAME>-" \
      AND textPayload:"EXIT_CODE"' \
      --project <PROJECT_ID>
    ```

2.  Get the stack trace from worker 0 (or JAX process 0 / `pathways-head`) near
    start time; search `Traceback (most recent call last):` or `ValueError`:

    ```bash
    gcloud logging read 'resource.type="k8s_container" \
      AND resource.labels.pod_name:"<WORKLOAD_NAME>-slice-job-0-0-"' \
      --project <PROJECT_ID> \
      --limit 200
    ```

3.  Check the command line for empty strings `""` or trailing spaces in
    `--args`.

4.  If `ValueError: Unable to put to devices shape (B, S) with local device
    count N at ['inputs']`
    (`ValueError: array split does not result in an equal division`) appears on
    `pathways-head`, inspect the MaxText config for
    `grain_use_elastic_iterator=True`, `packing=False`, and
    `grain_worker_count > 0`, and verify that `0` GCE VM terminations and `0`
    node health faults occurred (do **not** isolate/replace any nodes).

## Mitigation

-   Fix the MaxText/HuggingFace config key typo.
-   XPK: the `--args` string must not end with a space (`--args="run_name=foo "`
    -> `--args="run_name=foo"`).
-   **Grain elastic iterator batch divisibility:** Set `grain_worker_count=0`
    when running with `grain_use_elastic_iterator=True` and `packing=False` (or
    ensure the per-host batch size is a multiple of the local TPU device count
    `len(devices)` / enable `packing=True`). Do **not** isolate/replace
    hardware.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Symptoms**:
    -   [[elastic-reshard-pytree-mismatch]]
-   **Concepts**:
    -   [[exit-codes]]
    -   [[pathways-sidecar]]
    -   [[workload-startup-failures]]
-   **Entities**:
    -   [[maxtext]]

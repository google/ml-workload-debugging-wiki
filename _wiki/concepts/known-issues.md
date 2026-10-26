---
title: Known Issues and Workarounds
type: concept
internal_only: false
tags: [out_of_memory, timeout, hangs, triage]
created: 2026-05-13
updated: 2026-06-16
---

## Overview

This page documents known issues in the TPU ML software stack, including JAX and
Jobset, along with their workarounds and resolution status.

## Memory Leak on Host RAM (Trillium)

-   **Background**: Multi-slice TPU workloads slowly leak host RAM memory,
    ultimately causing the workload to OOM (Out of Memory). This effect is
    exponential over time and increases with the number of chips used.
-   **Affected Versions**: JAX versions between 0.4.36 and before 0.5.
-   **Mitigation**: Set the following environment variable:

    ```bash
    --megascale_grpc_enable_xor_tracer=false
    ```
-   **Status**: Fixed in JAX 0.5.
-   **Reference Bug**: Large memory leak on host 0, pod 0 from
    `--megascale_grpc_enable_xor_tracer`.

## PCIe Overload (Trillium)

-   **Background**: Cloud TPU training jobs consistently fail after
    approximately 5 to 99 steps due to [[deadline-exceeded]] errors and
    [[slice-failure]].
-   **Affected Versions**: JAX versions between 0.4.36 and before 0.5.1.
-   **Mitigation**: Pass `xla_tpu_iova_dma_chunk_size_bytes` through
    `LIBTPU_INIT_ARGS`:

    ```bash
    LIBTPU_INIT_ARGS='--xla_tpu_iova_dma_chunk_size_bytes=104857'
    ```
-   **Status**: Fixed in JAX 0.5.1.

## Jobset Manager Out of Memory (Cluster Toolkit)

-   **Background**: The default memory limits for the jobset manager are too
    small for large Trillium workloads (specifically those using more than 16
    v6e-256 pods in a GKE cluster).
-   **Affected Versions**: Jobset version 0.7.2.
-   **Mitigation**: Remove the memory limits for the jobset manager.
    1.  Go to the `jobset-controller-manager` page in the Google Cloud Console.
    2.  Navigate to Kubernetes Engine -> Workloads -> (Select Cluster) ->
        `jobset-controller-manager` -> Edit.
    3.  Scroll down to "containers" in the YAML file and remove the memory limit
        lines.
-   **Status**: Fixed in Jobset 0.8.0.
-   **Reference Bug**:
    `Jobset manager running out of memory around > 10 slices, not sure why`.

## Connections

-   **Playbooks**:
    -   [[general-ml-debugging]]
-   **Concepts**:
    -   [[network-and-tpu-stacks]]

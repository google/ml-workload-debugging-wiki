---
title: libtpu
description: The low-level TPU runtime driver library that schedules compiled programs
  onto the device and emits the runtime logs debugging depends on.
type: entity
internal_only: false
tags: [tpu, pathways, logging, analysis]
created: 2026-09-15
updated: 2026-10-15
---

# libtpu

## Overview

`libtpu.so` is the TPU runtime driver library loaded by the host framework —
JAX, PyTorch/XLA, or Pathways — to drive the accelerator. It receives compiled
[[xla]] executables, schedules them onto the device **continuation queue**, and
emits the low-level runtime logs that most TPU hang analysis is built on.

## Technical Specifications

-   **Asynchronous dispatch**: programs are enqueued onto the continuation queue
    and tracked by program ID, HLO fingerprint and launch ID;
    `remaining programs=0` means the queue has drained and the device is about
    to idle. See [[libtpu-program-lifecycle]].
-   **Configuration**: runtime flags are passed via `LIBTPU_INIT_ARGS` (for
    example `--xla_tpu_iova_dma_chunk_size_bytes=104857`), and log verbosity via
    `TPU_VMODULE`, `TPU_MIN_LOG_LEVEL`, `TPU_STDERR_LOG_LEVEL` — see
    [[tpu-logging-flags]] and [[enable-tpu-verbose-logging]].
-   **Log destinations**: the driver writes to `/tmp/tpu_logs/` on the VM, which
    does not survive pod deletion; `TPU_STDERR_LOG_LEVEL=0` is what forwards
    those logs to Cloud Logging for post-mortem analysis.
-   **Version-dependent output**: newer libtpu versions print GKE instance names
    for both sender and receiver in communication errors, older ones do not
    ([[exit-codes]]) — check the version before concluding a log field is
    missing. Large customers upgrade libtpu slowly, which is why RapidEye moved
    its analysis off-node into a managed service.

## Common Failure Modes

-   **Uncorrectable HBM ECC**: aborts the libtpu process outright, terminating
    the job with no orderly shutdown. See [[ecc-errors]].
-   **Program fingerprint divergence**: workers running different HLO
    fingerprints for the same step indicate inconsistent compilation across the
    job, a known precursor to collective hangs ([[megascale-hang]],
    [[inspect-hlo-dumps]]).
-   **Host dispatch stalls**: a growing gap between enqueue and completion on
    one worker points at the host, not the device; quantified in
    [[libtpu-enqueue-timing-analysis]].
-   **DMA/bus pressure**: mitigated by reducing the IOVA DMA chunk size through
    `LIBTPU_INIT_ARGS` — see [[known-issues]].

## Connections

-   **Actions**:
    -   [[enable-tpu-verbose-logging]]
-   **Playbooks**:
    -   [[libtpu-enqueue-timing-analysis]]
-   **Concepts**:
    -   [[libtpu-program-lifecycle]]
    -   [[tpu-logging-flags]]
-   **Entities**:
    -   [[jax]]
    -   [[tensorcore]]
    -   [[xla]]

---
title: TPU Verbose Logging Environment Variables
type: concept
internal_only: false
tags: [tpu, analysis, metrics]
created: 2026-06-16
updated: 2026-10-06
---


## Overview: flags

```bash
export TPU_VMODULE=slice_configuration=1,real_program_continuator=1
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

python3 train.py ...
```

Set these in the JobSet manifest or launch script before the Python process
starts.

*   **`TPU_VMODULE`** (glog `--vmodule` for `libtpu.so`):
    *   `slice_configuration=1`: slice building and stitching (topology, ICI
        link maps, chip stitching). Use for network handshake hangs or topology
        mismatches at startup.
    *   `real_program_continuator=1`: logs each HLO program launch and
        completion per TensorCore/SparseCore, so you can see which program was
        running when progress stopped.
*   **`TPU_STDERR_LOG_LEVEL=0`** (glog `--stderrthreshold`): mirrors all `INFO`
    and `VLOG` lines to stderr. This is what reaches GKE/Cloud Logging. Local
    files are lost when the pod dies, so without it you can't analyze a stall
    post-mortem.
*   **`TPU_MIN_LOG_LEVEL=0`** (glog `--minloglevel`): writes everything,
    including VLOG, to `/tmp/tpu_logs/` (e.g. `tpu_driver.INFO`) on the VM.

## Defaults: why error digests are missing

Without these variables libtpu uses `TPU_MIN_LOG_LEVEL=0` (everything is written
to the local files under `/tmp/tpu_logs`) and `TPU_STDERR_LOG_LEVEL=3` (only
`FATAL` reaches stderr, and therefore Cloud Logging). Megascale's diagnostic
lines (`Megascale detects a hang ...`, `Full error digest:`) are logged at
`ERROR`, so by default they are written only to the local files, which disappear
with the pod. The coordinator's final `FATAL` line still appears, which is how
you can tell the level was the default: the `FATAL` line is present but the
`ERROR` digest written just before it on the same thread is not.

-   Before searching Cloud Logging for a digest, check the workload's effective
    `TPU_STDERR_LOG_LEVEL` (JobSet `env` / `envFrom`, launcher scripts, image
    `ENV`). If it is unset or 3, "no digest found" is not evidence of "no hang":
    report it as an observability gap.
-   For large jobs, `TPU_STDERR_LOG_LEVEL=2` (`ERROR` and up) is enough to keep
    the digest and is far less noisy than `0`, which mirrors every `INFO` line
    from every process (thousands of hosts × several processes per host).

## Debugging: diagnostic runs

Verbose logging slows training, so do not leave it on. After a repeated or
unidentified stall, enable the diagnostic flags for the next run only, then turn
them off. `--megascale_grpc_enable_xor_tracer=true` also leaks host RAM on some
versions (see [[known-issues]]).

## Connections

-   **Actions**:
    -   [[enable-tpu-verbose-logging]]
-   **Entities**:
    -   [[cloud-tpu]]
    -   [[libtpu]]

---
title: Megascale Hang Detected
type: symptom
internal_only: false
tags:
- tpu
- pathways
- training
- timeout
- hardware_fault
created: 2026-06-08
updated: 2026-10-16
---


# Megascale Hang Detected

## Definition / Explanation

A Megascale hang occurs when a training worker has waited on a [[megascale]]
collective communication operation for 5 minutes (default threshold). When this
occurs, the coordinator logs emit a `Megascale detects a hang` or
`Megascale HANG_DETECTED` message. **Crucially, where you find this log depends
on your orchestrator:**

-   **Pathways**: Check the **`pathways-head`** node (e.g.,
    `-pathways-head-0-0-` / `resource_manager` job).

-   **mcjax (Multislice JAX)**: Check the **`0-0`** worker node (e.g.,
    `-worker-job0-0-` / MXLA coordinator task 0 slice 0).

-   **PyTorch / torch_xla multislice**: the coordinator runs in the first pod of
    the first slice (for example `-slice-0-0-`), with several processes per
    host; read every process's stderr from that pod.

The core log signature of a Megascale hang is the `Megascale detects a hang` or
`Megascale HANG_DETECTED` message emitted by the coordinator. While you may also
see `DEADLINE_EXCEEDED` errors preceding this, the Megascale detection log is
the definitive indicator and contains the crucial diagnostic payload.

**No digest is not proof of no hang.** The digest lines are logged at `ERROR`,
and libtpu's default `TPU_STDERR_LOG_LEVEL=3` sends only `FATAL` to stderr (and
therefore to Cloud Logging). If the workload did not set `TPU_STDERR_LOG_LEVEL`
to 2 or lower, the digest exists only in the pod's local `/tmp/tpu_logs`, which
is lost with the pod. Check this first ([[tpu-logging-flags]]) and report it as
an observability gap instead of searching further.

## Log Signatures & Error Messages

```text
Megascale detects a hang that is likely caused by bad TPU chips on the following hosts. Please remove the hosts from the fleet and restart the workload. If problem persists please contact Megascale XLA team.

The host that have bad TPUs are: <host_name>

Full error digest:
  Potential cause: <cause>
  Potential culprit workers: <host_name>
  First error timestamp: <timestamp>
  First error type: <error_type>
  TPU states:
    Launch ID: <launch_id>
    Module: <module_name> Fingerprint: <fingerprint>
      Sample worker: <task_name>@<host_name>:<tpu_chip>:<tpu_core>
    Tag:PC breakdown:
      <num_cores>@<location>(HLO): [<task_name>:<host_name>@<tpu_chip>:<tpu_core>, ...]
```

### Non-hang coordinator abort

When the first error the coordinator collects is not a hang, it logs the same
error digest and then aborts with:

```text
Aborting the coordinator as an unrecoverable error is reported from one or more
workers. All workers will also abort after they detect the coordinator is
shutdown.
```

This is a worker failure (for example a failed DCN send), not a hang: find the
worker that reported first and read its stack. With default log levels only this
`FATAL` line is visible. In the 2026-10-05 case the workers' stacks went through
`SendRpcContext::CompleteRpc` → `DebugManager::ErrorDetected` and then crashed
in `TpuChipCommonImpl::GetCore` on half of the ranks; every failing run had pods
on the same two DCN-suspect node pools.

## Root Cause: The 7 Error Categories & Diagnostics

Megascale hangs look for outliers in what each TPU worker is running. Based on
the error digest's `Potential cause`, the runtime classifies hangs into 7
distinct categories:

### 1. Bad TPU Chip (`BAD_TPU_CHIP` or `BAD_SC_CHIP`)

-   **Log Signature**:

    ```text
    Megascale detects a hang that is likely caused by bad TPU tensor core chips on the following hosts. Please remove the hosts from the fleet and restart the workload. If the problem persists please contact Megascale XLA team.
      The host that have bad TPUs are: <host_name>
    ```

    *(Or `bad TPU sparse core chips` / `Potential cause: Bad Sparse Core`)*

-   **Recidivism Rule**: An isolated occurrence can occasionally be caused by an
    XLA compiler issue or custom Pallas kernel bug. However, if the **same host
    appears more than 3 times** across multiple hang events, the TPU hardware on
    that host is confirmed faulty.

-   **Remediation**:

    -   **3P (Cloud GKE)**: Isolate the node via [[isolate-node-via-rr-label]]:
        `kubectl label nodes <NODE_NAME> cloud.google.com/fault-behavior=SDC`.

### 2. Networking Issues (`NETWORKING_ISSUE`)

-   **Log Signature**:

    ```text
    Megascale detects a hang that is likely caused by a networking issue. Please examine the underlying networking stack for the following hosts.
      The hosts are: <host_src>, <host_dst>
    ```

-   **Diagnostics**: Indicates communication packet loss or link stalls between
    pairs of hosts. If multi-link localization is available, isolate whether a
    single host is common to multiple degraded links.

-   **Remediation**: Check node network interfaces and VPC health. In 3P,
    replace or isolate degraded nodes.

### 3. Different Modules (`DIFFERENT_MODULE`)

-   **Log Signature**:

    ```text
    Megascale detects a hang that is likely caused by running different modules on different devices. Please confirm that all workers is running the exact same program. It can also be caused by a hang in a subset of devices and the unaffected devices have moved on to the next program.
    Example hosts that have different HLO modules: <host_a>, <host_b>
    TPU states:
      Module: jit_loss_and_grad
      ...
      Module: jit_optimizer_apply
    ```
-   **Diagnostics**: Workers are out of sync executing different HLO modules
    (e.g., `jit_loss_and_grad` vs `jit_optimizer_apply`). Check for process 0
    divergence: JAX process 0 often executes distinct application logic
    (evaluations, checkpoint writes) while other workers wait at a barrier. It
    also occurs when unaffected devices advance while a lagging subset hangs.

### 4. Fingerprint Mismatch for HLO Module (`FINGERPRINT_MISMATCH`)

-   **Log Signature**:

    ```text
    Megascale detects a hang that is likely caused by inconsistent HLO module compilation across workers. This is likely a bug in JAX tracing or XLA compiler. Please inspect the HLO dumps to confirm the root cause.
      Example hosts that have different HLO fingerprints: <host_a>, <host_b>
    ```
-   **Diagnostics**: Workers are executing the same function name, but the
    compiled XLA module fingerprints differ. Collect HLO dumps from culprit
    workers using `--xla_dump_to` to compare graph structures and report
    compiler tracing bugs. Refer to [[inspect-hlo-dumps]].

### 5. Data Input Stall (`DATA_INPUT_STALL`)

-   **Log Signature**:

    ```text
    Megascale detects a hang that is likely caused by data input stall on the following hosts. Please check the workers to make sure the data input pipeline is working properly.
      The host that have data input stalls are: <host_name>
    ```
-   **Diagnostics**: Devices launched the step but stalled waiting for input
    tensors. The outlier TPU typically exhibits a Program Counter of `PC=0` (or
    `no HLO mapping` with `tag=1`). Check input data loaders, storage bandwidth
    (e.g. GCS rate limits), and disk I/O.

### 6. Unrecoverable Error (`UNRECOVERABLE_ERROR`)

-   **Log Signature**:

    ```text
    Some workers have halted with an unrecoverable error:
      <worker>:<host> : {error_message}
      Please inspect the error log of these workers: <worker>:<host>
    ```
-   **Diagnostics**: Triggered by host-side runtime failures, such as
    host-to-device (H2D) or device-to-host (D2H) data transfer stalls
    (`d2h_transfer_latency`). Inspect the application logs of the identified
    culprit workers for underlying driver or runtime stack traces.

### 7. Unknown Error (`UNKNOWN_CAUSE` / `Unknown`)

-   **Log Signature**:

    ```text
    Megascale detects a hang but cannot determine the root cause. Please inspect the full digest below.
    ```
-   **Diagnostics**: The error aggregator detected a timeout but found no
    outlier TPU state or module discrepancy. Fall back to manual **TPU State &
    Launch Alignment Analysis**:

    1.  Enable sync flags/verbose logging
        (`--vmodule=real_program_continuator=1`).
    2.  Run [[diagnose-sflag-hang]] or [[analyze-sflag-timeouts]] across all
        worker logs for `"Wait timeout on sflag"` and
        `"Sflag wait timeout on op"` messages with `--order=ASC`.
    3.  Check both **minority outlier groups** (`< 20%` of the majority `sflag`
        or SparseCore `tag:pc` group) and **silent/missing pods** (comparing the
        set of pods that reported `sflag` timeouts against the expected
        `<slice>-<worker>` pod topology to spot silent culprits that never
        emitted timeout logs).
    4.  If all reporting pods appear stuck on the same `sflag` and cross-slice
        DCN action states (`NETWORK_SEND` / `NETWORK_RECEIVE` vs `D2H_START` /
        `WAIT_BARRIER`) do not isolate a single host, run
        [[analyze-libtpu-timing]] to inspect `Enqueueing program` and
        `last_finished_run_id updated` timestamps per `launch_id` and identify
        straggler or missing workers. Refer to
        [[tpu-state-analysis-methodology]] and [[tpu-state-analysis]].

For a complete step-by-step investigation protocol, follow the
[[megascale-hang-triage]] playbook.

### `ErrorAggregator` Priority Cascade

`ErrorAggregator`
(`megascale_error_aggregator.cc`) evaluates potential root causes in a
deterministic 8-step priority cascade after building the topologically sorted
launch history (`BuildLaunchInfos`) and consolidating in-flight collective
action graphs (`Aggregator`):

1.  **`LAUNCHES_ORDER_INCONSISTENT`**: `BuildLaunchInfos` detected a cycle in
    the topological sort of `launch_id`s across workers' `recent_completions`
    ring buffers.

    -   *Diagnosis*: Inspect the full error digest launch history to see which
        workers executed programs out of order.

2.  **`NETWORKING_ISSUE`**: Cross-slice collective hang caused by a
    DCN/inter-host networking stall.

    -   *Diagnosis*: `Aggregator::GetBottlenecks()` (`graph_utils.cc`)
        finds incomplete `MegaScaleAction` nodes (`device_to_host`,
        `host_to_device`, `network_send`, `network_receive`, `compute`) whose
        dependencies are all complete (`!node->action_complete &&
        node->completed_dependencies == node->total_dependencies &&
        !node->missing_dependency && !node->missing_dependent`), and
        `GetErrorDigest()` pairs incomplete `network_send` / `network_receive`
        actions into `suspicious_networking_pair`. When a single source or
        destination worker is common across all faulty links, `NETWORKING_ISSUE`
        is reported. Examine the underlying networking stack for the identified
        hosts.

3.  **`UNRECOVERABLE_ERROR`**: One or more workers halted with a fatal gRPC
    transport or runtime error (`MegaScaleRuntimeError::UNRECOVERABLE_ERROR`,
    e.g., stream failures exceeding `send_rpc_timeout`, `ABORTED`, `DATA_LOSS`,
    `INTERNAL`, `UNAVAILABLE`).

    -   *Diagnosis*: Inspect the specific error messages and logs of the culprit
        workers provided in the digest. (Note: a single
        `"Cannot enqueue command as the command queue is not opened."` error is
        promoted to `BAD_TPU_CHIP`, and `"Error in send for transfer:..."`
        messages sharing a common worker are promoted to `NETWORKING_ISSUE`.)

4.  **`FINGERPRINT_MISMATCH` vs. `DIFFERENT_MODULE`**: Workers at the same
    active launch have multiple `ProgramInfo` entries
    (`launch.programs.size() > 1`).

    -   *Diagnosis*: If all programs share the same `module_name` but have
        different fingerprints, `FINGERPRINT_MISMATCH` is reported (likely a bug
        in JAX tracing or XLA compiler config drift; inspect HLO dumps). If the
        `module_name`s differ, `DIFFERENT_MODULE` is reported (confirm all
        workers run the exact same program, especially JAX process 0).

5.  **`DATA_INPUT_STALL`**: Workers waiting on the host CPU data input pipeline
    on the oldest active launch
    (`tpu_state.hlo_name.empty() && tpu_state.tag == 1`, or `pc == 0` in worker
    logs).

    -   *Diagnosis*: The core has not yet entered an HLO instruction and is
        waiting on host data loading or an entry barrier. Check the data input
        pipeline on the culprit workers and run [[analyze-libtpu-timing]] to
        confirm host dispatch gaps.

6.  **`PROGRAM_NOT_QUEUED`**: The oldest active launch has only a single TPU
    state equal to `"(program completed)"`.

    -   *Diagnosis*: Check whether the application on the culprit hosts is
        blocked in Python/host logic and failing to enqueue the next
        JIT-compiled TPU program.

7.  **`BAD_TPU_CHIP` / `BAD_SC_CHIP`**: Faulty hardware or a stalled device core
    causing `< 3` outlier workers to stall mid-execution.

    -   *Diagnosis*: `BAD_TPU_CHIP` is attributed when the oldest active launch
        has multiple TPU states and `< 3` distinct single-host outlier workers
        with a non-empty `hlo_name`. If no TensorCore outlier is found,
        `BAD_SC_CHIP` is attributed when `< 3` distinct workers are stuck at an
        outlier SparseCore `tag:pc` on a `call-start`/`call-done` offload HLO
        (`offloaded_sc_tag_pc_to_workers`). **Important**: You need at least 2
        data samples of hardware failures from the *same chip* to be confident
        enough to isolate/R&R. If a series of restarts repeatedly reports bad
        chips on *different* machines every time, it is likely a software
        failure.

8.  **`ICI_ERROR` $\rightarrow$ `UNKNOWN_CAUSE`**: If a single slice has all
    workers reporting errors on a single ICI collective HLO (`all-reduce`,
    `all-gather`, `all-to-all`, `reduce-scatter`, `collective-permute`),
    `ICI_ERROR` is reported; otherwise, `UNKNOWN_CAUSE` is returned.

### Elastic Scale-Up `Snapshotter: CloneRefs` Hang (`Potential Cause: Program not queued`) & `WatchDog Timebomb for MegaScale cancellation`

In Pathways elastic snapshot runs (`elastic_backup_kind=snapshot`), if a
background `Snapshotter` worker thread issues a `CloneRefs` collective RPC
(`launch_id=<N>`) immediately after scaling up to 2 slices while the newly
joined slice never queues the `CloneRefs` program and instead enters
`jit_train_step` (`launch_id=<M>`), `CloneRefs` stalls for 5 minutes and fails
on `pathways-head` with:

```text
Received error for CloneRefs rpc_uid=... triggering device=... DEADLINE_EXCEEDED: Megascale hang detected: Timed out waiting for 24 graphs to complete at launch_id 4476. Already completed: 0. StepGloballyInProgress: false Timeout: 5m
Snapshotter: error during CloneRefs: DEADLINE_EXCEEDED: Megascale hang detected: Timed out waiting for 24 graphs to complete at launch_id 4476...
```

The Megascale digest reports **`Potential Cause: Program not queued`** on the
slice that did not queue the `CloneRefs` launch. Exactly **3 minutes** after
`DEADLINE_EXCEEDED`, the cancellation watchdog expires on the workers of that
slice and aborts all 16 worker pods with `SIGABRT` (`si_code=-6`):

```text
(CRITICAL) WatchDog Timebomb for MegaScale cancellation for launch_id=4476 sequencer_id=4476. This could indicate either an infra or a user bug. timed out after 3m seconds.
*** SIGABRT received by PID 14 (TID ...) on cpu ..., si_code=-6, from PID 14; ***
```

Do **not** attribute the `Potential Cause: Program not queued` digest or the
subsequent 16-worker `WatchDog Timebomb for MegaScale cancellation` `SIGABRT`
(`si_code=-6`) storm to faulty TPU hardware: it is an application-level
`Snapshotter` / collective launch desynchronization across slices (fixed in
`AI-Hypercomputer/maxtext#5260`), which then triggers an elastic scale-down on
the coordinator.

## Handling "Unknown" Causes & Suppressed Digest (`TpuDiagnosticCoordinator::HarvestTelemetry` `RET_CHECK`)

If the coordinator's hang error digest reports `Potential cause: Unknown` (or is
empty/missing a clear culprit host), the hang is typically due to either a
**silent/missing worker** (a worker that crashed, hung in host code, or stalled
before emitting `MegaScaleRuntimeError` / `sflag` timeouts) or a uniform
multi-slice deadlock.

In Pathways runs (e.g., `example-elastic-4x1k` Attempt 0), a `pathways-worker`
pod may first log:

```text
Slow PjRt TPU operation detected: start_time=... host_id=141 index_on_host=6 generation=0 num_unknown_pending_deps=0 description=SystemIssuedEvent Execute Op:ProcessExamples:jit_train_step... launch_id=279 sequencer_id=279
2x6x8_TC1: Program 11 failed with status: DEADLINE_EXCEEDED: Megascale hang detected: Timed out waiting for 12 graphs to complete at launch_id 279. Already completed: 0. StepGloballyInProgress: false Timeout: 5m ... ABORTED: Runtime cancelled the launch due to detected MXLA error on another device
TpuDiagnosticCoordinator::HarvestTelemetry failed: INTERNAL: RET_CHECK failure (third_party/tensorflow/compiler/xla/hlo/ir/hlo_instruction.cc:543) DynCast<HloRecvInstruction>(operands(0)) != nullptr RecvDone must take the context operand from Recv
```

When `TpuDiagnosticCoordinator::HarvestTelemetry` crashes with `INTERNAL:
RET_CHECK failure ... DynCast<HloRecvInstruction>(operands(0)) != nullptr`, the
MegaScale `Potential cause` / `Potential culprit workers` digest is **suppressed
before emission**. Furthermore, `ABORTED: Runtime cancelled the launch due to
detected MXLA error on another device` proves that the pod logging these lines
(e.g. `worker-job0-1-195-mh2zc` on `gke-tpu-node-xxxx`) is a **reporter**, not a
confirmed hardware culprit (**Unknown Case — Reporter Only**; do **not** isolate
the reporting host).

In this scenario, bypass the standard digest parsing and perform **TPU State &
Launch Alignment Analysis**.

### Worker Not Queuing Programs to TPU — Orbax `array_metadata_store` Colocated Python Scale-Up Deadlock

When a MegaScale hang shows `StepGloballyInProgress: false` and
`Already completed: 0` on `pathways-head` (with
`copy_to_host_buffer: DEADLINE_EXCEEDED` and **no** `BAD_TPU_CHIP` culprit), the
coordinator explicitly emits:

```text
copy_to_host_buffer: DEADLINE_EXCEEDED: DEADLINE_EXCEEDED: Megascale hang detected: Timed out waiting for 24 graphs to complete at launch_id 1920. Already completed: 0. StepGloballyInProgress: false Timeout: 5m:{0}
Megascale detects a hang that is caused some worker not queuing programs to TPU. Check if your application is blocked/crashing and preventing JAX to queue the next TPU program (jitted functions).
```

In Pathways elastic snapshot / scale-up workloads using Colocated Python
([[pathways-sidecar]]) and [[orbax]](e.g., `example-elastic-2x64`), this
signature occurs during **dynamic scale-up** (`1x64 -> 2x64`) after an earlier
slice-loss event (such as an external user-space `SIGILL (si_code=0)` on Slice
0's `worker-job0-0-0`; see [[sigill-worker-kill]]):

-   **Mechanism**: In MaxText `train_utils.py`, if
    `ocp_pathways.register_type_handlers()` is called **after**
    `create_orbax_checkpoint_manager` (or if `array_metadata_store` is left
    configured as `array_metadata_store_lib.Store(primary_host=None)`), Orbax's
    `ArrayHandler` defaults `primary_host=0` (process index 0). During scale-up
    (`1x64 -> 2x64`), non-zero Colocated Python sidecar processes
    (`colocated_python_sidecar`) on the newly joining slice (Slice 1, e.g.
    `worker-job1-0-*` on `gke-tpu-node-*`) enter
    `orbax/checkpoint/_src/metadata/array_metadata_store.py`
    `_maybe_create_base_dir()` and block for up to `600s` waiting for sidecar
    worker 0 to create the `array_metadatas/` directory. Because worker 0 is not
    participating in that asymmetric sidecar barrier during scale-up, the Slice
    1 workers never return to JAX to queue the next jitted TPU program at
    `launch_id 1920`, causing MegaScale's 5-minute collective launch timer
    (`300s < 600s`) to expire first.
-   **Role-vs-Host Confound on Slice 1 Distributor (`worker-job1-0-0`)**: On
    Slice 1, the index-0 pod (`worker-job1-0-0-t2466` on `gke-tpu-node-xxxx`)
    acts as the slice's `first level distributor` in `MegaScaleHostContext`
    (`Selected tasks ... as first level distributors`) and therefore logs the
    most `RPC failed: UNAVAILABLE` / cancelled-program lines during teardown.
    This is a **role confound** (`isolate_recommended: []`) — do **not** isolate
    `gke-tpu-node-xxxx` or the CPU head node (`gke-tpu-node-pool-xxxx` /
    `host-xxxx`).
-   **Fix / Mitigation**: In MaxText `train_utils.py` (fixed in commit
    `71633707dec161897d7b342d31df286413f03a6d`), call
    `ocp_pathways.register_type_handlers()` **before**
    `checkpointing.create_orbax_checkpoint_manager(...)` and pass
    `array_metadata_store=None` when creating the Pathways checkpoint manager so
    Colocated Python sidecars do not block on primary-host `array_metadatas/`
    directory creation during scale-up.

## Verification and Corroboration

Do not rely solely on the Program Counter (PC) value to diagnose a bad chip:

-   **Check for Recidivism**: Verify if the same host fails repeatedly (>3
    occurrences) across runs.

-   **Corroborate with System Events**: Check for hardware alerts or node
    conditions in Kubernetes events.

-   **Rule Out Software**: If PC is 0, verify if data loading is actually stuck
    by checking data loader logs or metrics.

## Common Fix: Isolate the Bad Node


In live-debugging tests the usual resolution of a Megascale hang was to identify
the culprit host from the digest and isolate it, then restart
([[node-isolation-methodology]]; GKE: `isolate-node-via-rr-label` via
[[node-isolation-and-rr-api]]). Isolate only with the evidence bar above (same host
or chip in 2+ incidents, or a hardware record on it); a different "bad" host on
every restart points to software.

## Not a Signature: `Packet spurious retx`

An automated RCA cited `Packet spurious retx` lines (equal to `Packet retx`) as
network evidence for a cluster-wide stall. The only code emitting that text is
the gRPC TCP tracer warning `Invalid packet count metrics: Packet sent: ...
Packet retx: ... Packet spurious retx: ... Skip reporting to stats plugins.` (at
most every 60 s, when counter deltas are invalid). It is metrics bookkeeping,
not packet loss: do not use it as ICI/DCN evidence (single observation,
attribution by code search).

## Related Pages

-   [[megascale-hang-triage]]
-   [[megascale]]
-   [[cloud-tpu]]
-   [[debugging-megascale-hangs]]
-   [[tpu-state-analysis]]
-   [[libtpu-timing-outlier-analysis]]

-   [[node-isolation-methodology]]

-   [[elastic-vs-permanent-failure]]

-   [[pathways-sidecar]]

-   [[orbax]]

-   [[sigill-worker-kill]]

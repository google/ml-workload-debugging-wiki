<!-- mdformat global-off -->

# Wiki Index


> [!IMPORTANT] **Before starting any diagnostic task**, you MUST read
> [[rca-invariants]]: access preflight -> NO_DATA, earliest error across all
> pods, barriers are victims, workload scoping, verdict outcomes. Also read
> [[multi-workload-triage]] if triaging multiple consecutive runs.
> [[debugging-mindset]] is optional background.

> Router catalog: one line per page (title, path, then the signature or when to
> use it).
> Read top-down and stop at the first match: **Symptoms** (match exact error
> strings, event reasons, monitor alert names) -> its **Playbook** ->
> **Workflow** (JSON steps name the actions). Tools, Cases, Concepts and
> Entities are background; **Actions** come last and are normally reached from
> a workflow.
> Last updated: 2026-10-23 | Total pages: 285

## Symptoms
- [TPU Provisioner Slice Recreation (JobSet Suspended at readyCount N-1/N)](symptoms/tpu-provisioner-slice-recreation.md): `tpu-provisioner-controller-manager` `recreation condition matched: FAILED`, JobSet suspended `readyCount: 7, expectedCount: 8`, resumed on `All Slices are Ready`; culprit is the slice's nodes, not the code.
- [Worker Hardware Fatal Error](symptoms/worker-hardware-fatal-error.md): `!!!! FATAL ERROR !!!! observed errors are: [..., HARDWARE]` on a worker; dozens of hosts in the same second means slice-over-live-pods, not N bad chips.
- [Workload Stuck in Queue](symptoms/stuck-in-queue.md): Workload Pending / Suspended / `QuotaReserved` without Admitted, or `Requeued` loops; Kueue quota, topology and webhook causes.
- [Forever Suspended JobSet](symptoms/forever-suspended-jobset.md): JobSet stays `Suspended` with no pods and no container logs; usual cause: JobSet name > 22 chars for Pathways child DNS labels; check `jobset-controller-manager` logs.
- [/bin/bash: argument list too long at 8k-chip scale (TPU_PROCESS_ADDRESSES)](symptoms/argument-list-too-long.md): `exec /bin/bash: argument list too long` on all pods of an ~8k-chip JobSet: TPU_PROCESS_ADDRESSES exceeds MAX_ARG_STRLEN (128 KiB); shorten names (JobSet <= 6, replicated job <= 5 chars).
- [Megascale Hang Detected](symptoms/megascale-hang.md): A worker waited > 5 min (default) on a Megascale collective: "Megascale detects a hang" / `HANG_DETECTED`; read `Potential cause:` (BAD_TPU_CHIP, DATA_INPUT_STALL, DIFFERENT_MODULE...). Usual fix: isolate the culprit host; `Packet spurious retx` is not a signature.
- [Wait timeout on sflag](symptoms/wait-timeout-on-sflag.md): `Wait timeout on sflag` (TensorCore waiting on SparseCore / peer); find the earliest worker and its `hlo:` op; usually a victim of one stalled chip.
- [SIGILL Worker Kill (si_code=0 / SI_USER)](symptoms/sigill-worker-kill.md): `pathways-worker` killed by `SIGILL` while idle; read `si_code`: 0 = SI_USER (external kill, isolate candidate), -6 = self-abort teardown storm.
- [False Preemption Signal (SIGTERM reported as preemption)](symptoms/false-preemption-sigterm.md): Logs say "Resource manager is being preempted" but no `compute.instances.preempted`: Pathways RM reports any `SIGTERM` (eviction, kubelet stop) as preemption.
- [Incomplete Checkpoint Blocks Restore](symptoms/incomplete-checkpoint-restore.md): Restart fails in Orbax restore with `ValueError`: latest step dir is partial (no `_CHECKPOINT_METADATA` / `commit_success.txt`) after a crash mid-save.
- [Pathways Expected Topology Mismatch](symptoms/pathways-expected-topology-mismatch.md): Pathways worker `FAILED_PRECONDITION` "newly added instance does not match with the expected instances"; GKE >= 1.35.0-gke.2479000 injects `TPU_ACCELERATOR_TYPE` / `TPU_TOPOLOGY_WRAP`.
- [Elastic Reshard Pytree / Sharding Mismatch](symptoms/elastic-reshard-pytree-mismatch.md): After a slice drops, elastic re-trace raises a pytree / `in_shardings` structure error (`ValueError: One of pjit outputs ...`); detection worked, recovery path is the defect.
- [HBM Out Of Memory (HBM OOM / Error Code E1000)](symptoms/hbm-oom.md): Accelerator High Bandwidth Memory (HBM) capacity exhaustion during XLA compilation or runtime execution on TPUs, emitting RESOURCE_EXHAUSTED errors.
- [SDC Checksum Mismatch](symptoms/sdc-checksum-mismatch.md): Symptom signature for on-device Silent Data Corruption (SDC) events, checksum mismatches during repeated HLO execution, and SparseCore vs TensorCore divergence.
- [Session Failing with Status](symptoms/session-failing.md): Session master fails the session on one worker's error (`session_master.cc`, SLICE_FAILURE_*); find the reporting worker and its first error.
- [Workload Startup Crash (Exit Code 1)](symptoms/workload-startup-crash.md): `EXIT_CODE=1` within 2-3 min of launch with a Python `Traceback` / `ValueError`: config or `--args` error (e.g. trailing space in run_name).
- [KubeDNS Resolution Failure and Timeouts](symptoms/kubedns-resolution-failure.md): DNS timeouts in pods (`169.254.169.254:53: i/o timeout`, `dial UDP: operation was canceled`), conntrack `insert_failed`; can block slice acquisition.
- [StopIteration due to Disk Full](symptoms/stopiteration-disk-full.md): Some workers log `StopIteration` (false data exhaustion), others `Wait timeout on sflag`; pod `Evicted` for ephemeral-storage: disk full.
- [GKE Metadata Server Crashloop](symptoms/gke-metadata-server-crashloop.md): `Failed to retrieve credentials from the Google Metadata Server` / GCS access denied; `gke-metadata-server` in `CrashLoopBackOff` in kube-system.
- [Proxy Connection Failed](symptoms/proxy-connection-failed.md): `Unable to initialize backend 'proxy': UNAVAILABLE` / `CANCELLED` / `INVALID_ARGUMENT` connecting to ifrt_proxy: proxy down, DNS/ILB timeout, second client session conflict, or protocol version mismatch.
- [Data Loss Pipe Closed](symptoms/data-loss-pipe-closed.md): `DATA_LOSS: Pipe from scheduler to aggregator ... closed unexpectedly`: a host died; find the first failing worker, check `Evicted` / `OOMKilling`.
- [Step Timebomb Watchdog](symptoms/step-timebomb-watchdog.md): `step-timebomb` watchdog plus `Error print traceback for thread:`: distributed blocking hang, often a MaxText checkpoint save.
- [IFRT Proxy Terminated](symptoms/ifrt-proxy-terminated.md): `IFRT proxy server was terminated: UNAVAILABLE: Socket closed` or `GrpcClientSession: writes no longer allowed`: downstream of a Pathways worker or RM crash; skip proxy logs.
- [TPU Coordinator OOM](symptoms/tpu-coordinator-oom.md): Coordinator pod (`pod-0-0`) exits `137` (OOMKilled) after MFU drops and host RAM climbs; check `--megascale_grpc_enable_xor_tracer`.
- [PRE_STOP_SESSION_BARRIER Timeout](symptoms/pre-stop-session-barrier-timeout.md): `PRE_STOP_SESSION_BARRIER` timed out in `session_manager.cc` during shutdown: a secondary effect; look for the earlier failure.
- [GRPC_ERROR START_SESSION failed](symptoms/grpc-error-start-session-failed.md): `MaybeInitializeSliceBuilder failed: INTERNAL: GRPC_ERROR ... START_SESSION failed`: bad ICI link on the cube (all VMs) or one bad machine (isolate).
- [GCS 429/503 Errors during Data Loading](symptoms/gcs-429-503-data-loading.md): GCS `429 Rate limit exceeded` / `503 GCS Service Unavailable` while loading data at large scale; slow startup or stalled input; cache or spread reads.
- [Kueue TAS: "allows to fit only N out of M slice(s)" / Job Waits While Cubes Look Free](symptoms/kueue-tas-fit-only-n-of-m.md): Pending TPU job while cubes look free: decode `excluded: affinity / resource / taint` into cubes, compute capacity for this job (own NotIn pools, cubes held by others), recover deleted specs from audit `jobs.create`.
- [Job Cancelled Due To](symptoms/job-cancelled-unknown.md): Pathways head logs "Job cancelled due to <reason>"; read the reason and the first worker error before it (often "Megascale detects a hang").
- [Slow Step Time (Performance Degradation)](symptoms/slow-step-time.md): Job progresses but step time rises vs baseline (MFU drop, 15% TPU duty cycle drop after 15m warmup gate triggering ML Diagnostics `PERFORMANCE_DEGRADATION` across 11 subsystems and AutoCaptured XProf sessions, flatter step-count graph); compare per-worker `Enqueueing program` timing.
- [Deadline Exceeded (Timeout)](symptoms/deadline-exceeded.md): Generic `DEADLINE_EXCEEDED` timeout: almost always a victim of a hang or slow node elsewhere; routes to megascale-hang.
- [Missing Exit Codes due to Eviction](symptoms/missing-exit-codes-eviction.md): Far fewer exit codes than pods in `exit_codes.txt`: kubelet eviction (ephemeral disk pressure) kills before the wrapper prints; check `Evicted` events.
- [MXLA Hardware Fault](symptoms/mxla-hardware-fault.md): Multi-slice MXLA hang traced to `HBM ECC uncorrectable error` / `Fatal TPU PCIe bus error` on one host; find it, then isolate.
- [SLICE_FAILURE (Chip Driver Error / SW Inject Error)](symptoms/slice-failure.md): SliceBuilder ICI slice failure:

## Playbooks
- [RCA Invariants (Read First)](playbooks/rca-invariants.md): MUST read first. `PERMISSION_DENIED` / `logging.logViews.access` -> stop, NO_DATA; earliest error across all pods, not worker 0; `Barrier timed out` is a victim; scope to the workload; host = cluster machine; CONFIRMED / REFUTED / INCONCLUSIVE / NO_DATA.
- [Elastic Training: Acceptable Slice Loss vs Permanent Hardware Failure](playbooks/elastic-vs-permanent-failure.md): `placed -> unplaced` / `1/2 Pathways Slices Now Ready` / DATA_LOSS in elastic runs: resumed and no host record = acceptable; same host in 2+ primary incidents or HardwareFatal / hostError = permanent, isolate.
- [Kueue Stuck Termination Recovery](playbooks/kueue-stuck-termination-recovery.md): Pods stuck `Terminating` on an unreachable node after RIP: enable Kueue FailureRecoveryPolicy + safe-to-forcefully-delete annotation (`KueueForcefullyTerminated`); never force-delete on Ready nodes.
- [Debugging Workload Submission Failures](playbooks/debugging-submission-failures.md): Submitted but not running on AI Hypercomputer: Suspended / Requeued / Admitted checks, `kubectl describe slice`, name > 49 chars, 8k `UnexpectedAdmissionError` (`tpu-device-plugin` `Allocate` OOM / `get pods` timeout, `backoffLimitPerIndex: 3` + `maxFailedIndexes: 0` or `backoffLimit: 16`, `pods/resize`).
- [Pod Analyzer Playbook](playbooks/pod-analyzer-playbook.md): 2-stage pod SOP before reading app logs: scheduling/admission (QuotaReserved, Inadmissible, FailedScheduling, UntoleratedTaint) then startup (ContainerCreating, ImagePullBackOff).
- [Node Replacement Methodology](playbooks/node-isolation-methodology.md): How to remove a bad TPU host: R&R `cloud.google.com/fault-behavior` label, `kubectl taint`, node replacement safety rules and delegation.
- [Workload Disruption Orchestration & Triage Playbook](playbooks/workload-disruption-orchestration.md): Orchestrator SOP for restarts/reshards/crashes: timeline gaps, investigation invariants (count incidents, read si_code), subagent dispatch contracts.
- [Hang Diagnosis Methodology](playbooks/hang-diagnosis-methodology.md): Hang triage: coordinator `Megascale detects a hang` + `Potential cause:`, sflag timeouts, `--order=ASC`, find the silent culprit worker.
- [TPU State Analysis Methodology](playbooks/tpu-state-analysis-methodology.md): Find the culprit chip in a multi-chip hang from TC/SC state logs: `--xla_tpu_enable_log_recorder`, `sc_sflag_wait_timeout_ms=150000`, DmaWaitTag, `sync_wait`.
- [Cloud Logging Gotchas](playbooks/gotchas-cloud-logging.md): Cloud Logging traps: broad pod prefixes match old runs, ASC/DESC truncation, sflag errors logged at INFO, scope not authorized.
- [Kubernetes & GKE Gotchas](playbooks/gotchas-kubernetes-gke.md): GKE/kubectl traps: "Provided scope(s) are not authorized", `--dns-endpoint` credentials, SIGTERM from eviction, QuotaReserved quirks.
- [Log Query Verification Methodology](playbooks/query-verification-methodology.md): An empty log query is never "no logs": test the query on a known string, widen pod_name filters, check fluentbit.
- [Multi-Workload Triage Playbook](playbooks/multi-workload-triage.md): Several consecutive failing runs: build a Failure Table (run, first error, exit codes, events, hosts) to separate one-offs from repeat causes.
- [libtpu Enqueue and Execution Timing Analysis Playbook](playbooks/libtpu-enqueue-timing-analysis.md): Straggler triage from libtpu `Enqueueing program` / `last_finished_run_id updated` logs: device execution vs host dispatch gaps per worker.
- [TPU SDK Debugging Overview and Reference Guide](playbooks/tpu-sdk-debugging-overview.md): Comprehensive reference guide and navigation index for the 11-module Cloud TPU SDK debugging curriculum, connecting high-level JAX troubleshooting down to low-level hardware diagnostics.
- [HBM Out Of Memory (HBM OOM) Triage Playbook](playbooks/hbm-oom-triage.md): Playbook for investigating and resolving TPU High Bandwidth Memory (HBM) exhaustion using XProf Memory Viewer, buffer analysis, and optimization scenarios.
- [Megascale Hang Triage](playbooks/megascale-hang-triage.md): Triage methodology for diagnosing and resolving Megascale multislice hangs across 1P and 3P environments using error aggregator digests and HLO inspection.
- [Megascale Performance Tuning and MapDMA Buffer Triage](playbooks/megascale-performance-tuning.md): Playbook for diagnosing Megascale performance degradation, steady-state MapDMA buffer shortages, network transfer latency vs compute latency, and RPC latency timelines using XProf.
- [ICI Straggler Triage](playbooks/ici-straggler-triage.md): Triage methodology for detecting and localizing degraded or abnormally slow ICI links using active `ici-sdc-test` testing and XLA delay wave propagation analysis.
- [Numerics Error Triage Playbook](playbooks/numerics-error-triage.md): Standard operating procedure for isolating and resolving numerical divergence, intermediate tensor drift, and precision-related errors using HLO diff, comparison tools, and host-assisted logging.
- [TPU On-Device Debugging and Crash Triage](playbooks/on-device-debugging-triage.md): Heuristic playbook for diagnosing on-device TPU crashes, hangs, core dump register states, hierarchical progress tracking, and LLO bounds check failures.
- [SparseCore Debugging and Mismatch Triage Playbook](playbooks/sparsecore-debugging-triage.md): Diagnostic playbook for isolating SparseCore offloading errors, numerical mismatches, LLVM bundle profiling, and assert level verification.
- [Pallas and Mosaic Kernel Debugging Playbook](playbooks/pallas-mosaic-kernel-debugging.md): Architecture, compilation pipeline, custom kernel design patterns, and debugging practices for JAX Pallas and Mosaic IR on TensorCore and SparseCore.
- [HLO and LLO Dump Analysis Playbook](playbooks/hlo-llo-dump-analysis.md): Comprehensive guide for configuring, generating, and analyzing XLA High-Level Optimizer (HLO) and Low-Level Optimizer (LLO) dumps, compiler pipeline passes, bundle formatting, and fingerprint hashing on Cloud TPUs.
- [Reporting JAX and XLA Compile-Time Crashes on TPU](playbooks/compile-time-crash-reporting.md): Standard operating procedure for isolating, reporting, and escalating JAX/XLA compiler crashes to the XLA:TPU team, capturing C++ stack traces, minimal reproducers, HLO dumps, and applying optimization barriers.
- [Data Non-Determinism and SDC Triage Playbook](playbooks/data-non-determinism-triage.md): Heuristic playbook for detecting, localizing, and diagnosing data non-determinism, silent data corruptions (SDC), unexpected NaNs/Infs, and SparseCore vs TensorCore divergences using XSC and mismatch detectors.
- [Launch Workload Dashboard](playbooks/launch-workload-dashboard.md): Configure and launch the GKE Capacity & Triage Dashboard for a workload/cluster (`clusters_config.json`, `DASHBOARD_DATA_DIR`).
- [Pathways Debugging Principles](playbooks/pathways-debugging-principles.md): Pathways on GKE: container roles (`pathways-rm`, `pathways-worker`, `pathways-proxy`, `main`), XLA flags on proxy, GCS lifecycle 50-prefix limit, protocol version mismatch, SIGTERM + RESOURCE_EXHAUSTED OOM.
- [Layered Debugging Framework (Triage across the Stack)](playbooks/layered-debugging-framework.md): Four-layer triage (application, JobSet/pods, Kueue/node pools, GCE/hardware) with scenarios: classify the failure layer before debugging.
- [KubeDNS and Cluster DNS Troubleshooting Playbook](playbooks/kubedns-troubleshooting.md): KubeDNS / CoreDNS / NodeLocalDNS on GKE: `/etc/resolv.conf` and `ndots`, `kube-dns` Service and autoscaler, `dig` / `nslookup` checks.
- [Multiprong Approach to Multi-chip Hangs](playbooks/multi-chip-hang-investigation.md): Multi-prong hang/slowdown method for 1000s of chips: per-chip signals (`tensorcore_util`, `ici_slow_signal`, `hlo_queue_size`) and `tag:pc` state comparison.
- [Workload Diagnostic Handoff Playbook](playbooks/workload-diagnostic-handoff.md): Standard operating procedure for bi-directional diagnostic evidence exchange, failure table synchronization, and issue tracker coordination between Customer ML Engineering teams and Google Cloud Support / CE on Cloud TPU multi-slice clusters.
- [Run Forensics Report Template (Multi-Run Scheduling and Crash Triage)](playbooks/run-forensics-report-template.md): Report format for a JobSet series that won't schedule or keeps failing across runs: bottom line, per-run exclusion/placement/cube-ledger table, story, time-aligned evidence with run scope, system/observability gaps, ranked fixes with owners.
- [Distributed Training Gotchas](playbooks/gotchas-distributed-training.md): JAX/MaxText traps: `Barrier sync ... timed out` victims, two Worker 0s (`[process=0]` vs job0-0), global index math, false `StopIteration`.
- [General ML Debugging Steps for TPUs](playbooks/general-ml-debugging.md): Baseline TPU workload triage: TPU log-level env vars, worker 0 first, grep `Fatal Python error` / `Traceback` / `SIGABRT` / `Out of memory`.
- [MCJax Debugging Principles](playbooks/mcjax-debugging-principles.md): MCJax on GKE: pod naming `{workload_name}-slice-job-{slice_index}-{worker_index}-...`, `[process=0]` logs, checkpoint logger, DEADLINE_EXCEEDED victims.
- [Workload Debugging Flowchart](playbooks/debugging-flowchart.md): Flowchart: worker 0 of slice 0 first, verify error uniqueness across workers, then hardware vs software, record in a Failure Table.
- [Hardware Failure Modes](playbooks/hardware-failure-modes.md): TPU hardware signatures seen in the field: HBM ECC uncorrectable, chip driver errors, v6e PCIe DMA overload (`xla_tpu_iova_dma_chunk_size_bytes`).
- [Provisioning Failure Analysis](playbooks/provisioning-failure-analysis.md): non-AI Hypercomputer TPU provisioning failures by layer: project quota, regional TPU capacity, cluster jobs manager (`WAITING_FOR_RESOURCES`), GKE/GCE.
- [The Expert Debugging Mindset](playbooks/debugging-mindset.md): Optional background on rigor: verify error uniqueness, temporal ordering, don't stop at the first plausible error.
- [Investigation Report Template](playbooks/investigation-report-template.md): Template for an investigation write-up: worker groups, timeline, root cause, evidence.
- [Trace Program Execution Times](playbooks/trace-program-execution-times.md): Per-program step times from `real_program_continuator.cc` completion logs; find the unique slow worker.
- [Tail-End Log Analysis Workflow](playbooks/log-analysis.md): Tail-end log analysis for unknown crashes/hangs: filter workers, convert process IDs, compute step times from `run completion` logs.
- [Debugging Gotchas](playbooks/debugging-gotchas.md): Misc gotchas for distributed TPU debugging; see the cloud-logging, kubernetes-gke and distributed-training gotchas pages.

## Workflows
- [Workload Disruption Orchestration](workflows/workload-disruption-orchestration.json): Automated multi-layered triangulation timeline workflow.
- [Kueue Stuck Termination Recovery Workflow](workflows/kueue-stuck-termination-recovery.json): Mutating: Kueue FailureRecoveryPolicy + pod annotation for JobSets stuck Terminating; only when the node is gone.
- [Debugging Megascale Hangs](workflows/debugging-megascale-hangs.json): Megascale hang: locate the hang timestamp, read coordinator/Cloud Monitoring digest, run diagnose-sflag-hang and analyze-libtpu-timing, then map hosts with telemetry.
- [Pathways Topology Mismatch Triage Workflow](workflows/pathways-topology-mismatch-triage.json): Triage workflow for GKE TPU expected vs physical topology mismatches.
- [libtpu Timing Outlier Analysis Workflow](workflows/libtpu-timing-outlier-analysis.json): Workflow for analyzing libtpu program enqueue and completion timing across workers to detect outliers and stragglers.
- [TPU State Analysis Workflow](workflows/tpu-state-analysis.json): List pods, analyze sflag timeouts, map physical ids to GKE instances.
- [ICI Straggler Triage Workflow](workflows/ici-straggler-triage.json): Standard diagnostic workflow for isolating and remediating degraded or abnormally slow ICI links using XLA `ici-sdc-test` delay wave propagation analysis.
- [HBM Out of Memory (Error 1000) Triage Workflow](workflows/hbm-oom-triage.json): End-to-end diagnostic workflow for triaging TPU HBM Out of Memory (`RESOURCE_EXHAUSTED: Out of memory in memory space hbm` Error E1000) and disambiguating compile-time static HBM OOM vs runtime HBM fragmentation vs host RAM OOM (exit code 137). Directly addresses [[hbm-oom]] and [[tpu-coordinator-oom]].
- [SDC Checksum Mismatch and Numerics Error Triage Workflow](workflows/sdc-and-numerics-triage.json): Standard diagnostic workflow for triaging Silent Data Corruption (`XLA_SDC_CHECKER` checksum mismatches), LLO deterministic replay failures, SparseCore offload mismatches, and NaN/Inf numerical divergence across TPU slices. Directly addresses [[sdc-checksum-mismatch]] and [[mxla-hardware-fault]].
- [General ML Debugging](workflows/general-ml-debugging.json): General TPU failure triage: verbose logging, crash signatures, physical id -> GKE instance, VM termination events.
- [Workload Run Forensics (Multi-Run Scheduling and Crash Triage)](workflows/workload-run-forensics.json): Read-only forensics for a JobSet series that won't schedule ('allows to fit only N out of M') or keeps crashing across runs: recover specs from audit jobs.create, group Scheduled events by pool, decode TAS buckets into cubes, compute per-run cube ledger and known-bad cube visibility, check TPU_STDERR_LOG_LEVEL observability, and verify attribution.
- [MXLA Hardware Triage Workflow](workflows/mxla-hardware-triage.json): General multi-slice MXLA hardware fault isolation SOP workflow for tracing TPU execution hangs across any topology size to uncorrectable HBM ECC memory faults...
- [Exit Code Investigation](workflows/exit-code-investigation.json): Identify the root cause of a workload failure by analyzing the exit codes (see [[exit-codes]]) returned by various workers.
- [Launch Workload Dashboard](workflows/launch-workload-dashboard.json): Verifies GCP authentication, resolves cluster, project, TPU generation, and backing reservations for a target workload, seeds the dashboard configuration...
- [Workload Diagnostic Handoff](workflows/workload-diagnostic-handoff.json): End-to-end diagnostic handoff workflow that executes /troubleshoot to gather findings, verifies destination bucket access, converts triage state, scaffolds GCS repositories, generates manifests and Two-Level Failure Tables, sanitizes payloads, publishes the package and verifies the bucket holds real content, stages partner issue updates, and arms automated background monitoring.
- [Hardware Exclusion Test Workflow](workflows/exclusion-test-workflow.json): Mutating: taint suspect nodes, rerun, isolate the host if failures persist.
- [Restart in Place and Repair & Replace (R&R) Workflow](workflows/repair-in-place.json): Mutating: resolve culprit nodes and apply the R&R repair-in-place label.
- [Workflow for Node Replacement and Using the R&R API](workflows/node-isolation-and-rr-api.json): Mutating: identify culprit GKE TPU hosts and apply R&R fault-behavior labels / taints (human-approved).
- [Active Monitoring of MCJax GKE Workloads](workflows/monitoring-progress.json): Check that a running MCJax job progresses: pod status, then coordinator log tail.
- [AI Triage, Classify and Mask Logs](workflows/ai-triage-classify-and-mask-logs.json): Workflow that queries workload timestamps and mapping from Cloud Logging, pulls and formats logs, and runs AI-based batch classification of log severities.
- [Exit Code Multi-Pod Analysis Workflow](workflows/exit-code-multi-pod-analysis.json): Multi-pod exit code analysis SOP workflow for isolating primary failure exit codes across large scale multi-pod TPU workloads.
- [Fixing Stuck Kueue Workload](workflows/fixing-stuck-kueue-workload.json): Mutating: deactivate/activate a Kueue workload; only for QuotaReserved loops with 0 pods (see preconditions).
- [AI Hypercomputer Specific Investigation Playbook](workflows/ai-hypercomputer-investigation-playbook.json): AI Hypercomputer capacity checks: describe reservation and sub-blocks, list node-pool nodes.
- [KubeDNS Troubleshooting Workflow](workflows/kubedns-troubleshooting-workflow.json): KubeDNS troubleshooting SOP workflow for diagnosing DNS resolution timeouts, resolv.conf misconfigurations, and container log errors.
- [Log Ingestion Pipeline](workflows/log-ingestion-pipeline.json): Shared pipeline block to look up workload timestamps and mappings, and pull/format raw workload logs.
- [Log Processing and Classification Pipeline](workflows/log-processing-pipeline.json): Shared pipeline block to build log graphs, mask sensitive tokens, classify log severities, and unmask for readability.
- [MCJax Workload Debugging Playbook](workflows/mcjax-debugging-playbook.json): MCJax subset: exit codes and workload logs; full triage is workload-disruption-orchestration.
- [Pathways on Cloud Workload Debugging Playbook](workflows/pathways-debugging-playbook.json): Pathways checks: JobSet status, system pods, JobSet events, workload and leader logs.
- [Remediate KubeDNS Resolution Failure](workflows/remediate-kubedns-resolution-failure.json): Remediation workflow for resolving KubeDNS resolution timeouts and Kueue slice acquisition failures in multi-slice TPU clusters.
- [Shared Cluster Usage Instructions](workflows/shared-cluster-usage.json): Kueue / shared-cluster capacity and queue usage checks for scheduling delays.
- [Verifying Kubernetes Workload Abstractions](workflows/k8s-workload-verification.json): Basic JobSet / job / pod health listing.
- [Verifying Pathways Cluster Readiness](workflows/pathways-cluster-readiness.json): Check Pathways RM and IFRT proxy startup and connection.
- [Workflow for Debugging Failed Provisioning Requests](workflows/debugging-failed-provisioning.json): non-AI Hypercomputer provisioning: list/describe Queued Resources and reservations.
- [Workflow for Debugging Failed Provisioning Requests in AI Hypercomputer](workflows/debugging-failed-provisioning-ai-hypercomputer.json): AI Hypercomputer provisioning: describe reservation and sub-blocks.

## Tools
- [Cloud Logging](tools/cloud-logging.md): `gcloud logging read` patterns for GKE TPU workloads: pod_name prefixes, severity filters, coordinator vs worker queries.
- [XProf (XLA Profiler)](tools/xprof.md): Accelerator profiling and visualization suite for XLA, JAX, and PyTorch, featuring Trace Viewer and Memory Viewer for TPU and GPU execution.
- [Mask Segment Tool](tools/mask-segment-tool.md): `mask_segment_tool.py`: mask hostnames, IPs and numbers in log lines for log-clustering pipeline simulation.
- [Python Glue Runner](tools/python-glue-runner.md): `python_glue_runner.py`: executes an action JSON with `${variable}` interpolation.
- [libtpu Timing Analysis Script](tools/analyze-libtpu-timing-tool.md): `tools/scripts/analyze_libtpu_timing.py`: parse libtpu enqueue/completion timestamps, flag outlier workers.
- [Orbax Tips and Tricks](tools/orbax.md): Orbax checkpoint tips: Pathways distributed persistence (`ENABLE_PATHWAYS_PERSISTENCE=1`), slow in-memory checkpoints (`TPU_PREMAPPED_BUFFER_SIZE`), incomplete checkpoint restore.
- [Workload Diagnostic Handoff Tool](tools/workload-diagnostic-handoff-tool.md): `_wiki/tools/scripts/handoff/`: provides an automated, secure diagnostic handoff boundary between Google Support / Customer Engineering and partner ML engineering teams for Cloud TPU workloads.
- [Triage Workloads Script](tools/triage-script.md): `triage_workloads_api.py`: scan a series of runs for exit codes, kubelet evictions, TpuOOM and Python errors.
- [ML Diagnostics CLI and Cluster Director API (gcloud alpha mldiagnostics)](tools/ml-diagnostics-cli.md): `gcloud alpha mldiagnostics` and `hypercomputecluster.googleapis.com/v1alpha` REST API for querying Cloud TPU `machineLearningRun` records, `monitoredEvents`, 12 `analyzerReports`, and on-demand XProf profiler sessions.

## Concepts
- [GKE/GCE TPU Naming Taxonomy](concepts/gke-identifier-taxonomy.md): Which name is which: pod, GKE node (`gke-tpu-...`), GCE instance, physical host, worker tasks; which actions take which id.
- [Restart in Place and Repair & Replace (R&R)](concepts/repair-in-place.md): AI Hypercomputer Restart in Place and Repair & Replace (R&R): `restartInPlaceAttempting` -> `successfulRestartInPlace`, R&R fault-behavior values (SDC, PERFORMANCE, unspecified), REPORTED_FAULTY_HOST.
- [AI Hypercomputer Infrastructure Health](concepts/ai-hypercomputer-infrastructure-health.md): AI Hypercomputer health signals: `gke-tpu-partition-4x4x4-state`, reservation sub-block `healthStatus` / `degradedHostCount`, upcoming maintenance.
- [Kueue and Throughput Quota Scheduling](concepts/kueue-tq-scheduling.md): Kueue topology-aware scheduling and throughput quota: LocalQueue / ClusterQueue, cohort borrowing.
- [Exit Codes](concepts/exit-codes.md): Exit code meanings for TPU workers (1 config, 134 abort, 137 SIGKILL, 143 SIGTERM) and the log lines that go with them.
- [Pathways Sidecars and Colocated Python](concepts/pathways-sidecar.md): Pathways Colocated Python (`jax.experimental.colocated_python`) sidecar container over gRPC for distributed data loading (`RemoteIterator`) and checkpointing.
- [Tpu Logging Flags](concepts/tpu-logging-flags.md): TPU verbose logging env vars: `TPU_VMODULE`, `TPU_MIN_LOG_LEVEL`, `TPU_STDERR_LOG_LEVEL`, `TF_CPP_MIN_LOG_LEVEL`.
- [Application Log Analysis](concepts/application-log-analysis.md): How the app-log analyzer agent queries runtime logs around a disruption and maps them to root-cause categories (CRASH_*, STALL_*, RESTART_*).
- [JobSet YAML Structure](concepts/jobset-yaml-structure.md): JobSet YAML for MCJax on GKE: topology annotations (`gke-tpu-slice-topology`, podset-required-topology), `LIBTPU_INIT_ARGS`, nodeSelector.
- [Workload Startup and Configuration Failures](concepts/workload-startup-failures.md): A workload startup crash occurs when a TPU workload is successfully scheduled but terminates within seconds or minutes of initialization (typically <5 minutes)...
- [Goodput and Recovery Metrics](concepts/goodput.md): Goodput = productive training time / wall-clock time; recovery metrics.
- [Pathways Fault Tolerance and Error Propagation](concepts/pathways-fault-tolerance.md): Pathways error poisoning, Error Aggregator, and resilient training modes: Suspend-Resume, Elastic Training (`pathwaysutils.elastic.manager.Manager`), and GKE JobSet Hot-Swap.
- [Known Issues and Workarounds](concepts/known-issues.md): Known TPU stack issues and workarounds (xor tracer flag, `xla_tpu_iova_dma_chunk_size_bytes`, jobset-controller OOM > 10 slices).
- [Pathways Architecture and Internals](concepts/pathways.md): Pathways single-controller JAX backend: IFRT proxy, RM, workers, sidecar, batch JobSet, interactive mode, multihost inference (`LeaderWorkerSet`), persistent cache, `ENABLE_PATHWAYS_PERSISTENCE=1`, MXLA metrics, XProf.
- [Pathways Cluster Readiness](concepts/pathways-cluster-readiness-checks.md): **Pathways Cluster Readiness** log signatures (`pathways-rm`, `pathways-proxy`, `pathways-worker`), virtual slices, `gcluster job logs`, and Cloud Monitoring MXLA latency metrics.
- [ICI Errors](concepts/ici-errors.md): Inter-Chip Interconnect (ICI) bandwidths (`400`–`1,200` GBps + 6x D2D on TPU7x), copper intra-cube vs OCS inter-cube optical links, ICI resiliency, twisted torus topologies, and link failure modes.
- [Workload Progress Monitoring](concepts/workload-monitoring.md): Progress log lines (`Enqueueing program`, `completed step`, `Barrier sync ... timed out`) and ML Diagnostics Workload Monitoring (GKE >= 1.36.0-gke.4447000, `libtpu >= 0.40.0`, 5 states, 12 analyzers, 1-min/10-sec system metrics).
- [Job Disruption Detection](concepts/job-disruption-detection.md): How the disruption detector finds restarts/reshards: `Logging before InitGoogle()` markers, pod create audit logs, WORKLOAD_NOT_FOUND.
- [Failure Tables as a Debugging Tool](concepts/failure-tables.md): A **Failure Table** is a powerful methodology for organizing and analyzing a complex set of failures in large-scale ML workloads.
- [libtpu Program Lifecycle and Continuation Queue](concepts/libtpu-program-lifecycle.md): Architecture and mechanics of libtpu program queuing, execution lifecycle, HLO fingerprints, launch IDs, and dual-interval timing analysis.
- [Project Quotas](concepts/project-quotas.md): Project Quotas are the first layer of gatekeeping in the TPU provisioning stack.
- [Queued Resources (QR)](concepts/queued-resources.md): **Queued Resources (QR)** enable users to request Cloud TPU resources in a queued manner.
- [OCS Errors](concepts/ocs-errors.md): Optical Circuit Switch (OCS) errors involve failures or degradations in the optical switching infrastructure used to route data dynamically between TPU cubes in a Superpod.
- [Superslicing and Subslicing](concepts/superslicing-subslicing.md): **Superslicing** and **Subslicing** are advanced resource allocation and scheduling strategies enabled by AI Hypercomputer to optimize the...
- [KubeDNS](concepts/kubedns.md): KubeDNS (or CoreDNS depending on GKE version/configuration) is the default centralized DNS engine used in GKE for internal service discovery and DNS resolution.
- [ECC Errors](concepts/ecc-errors.md): Error Correcting Code (ECC) errors occur when data corruption is detected in the TPU's memory systems, most notably in the High Bandwidth Memory (HBM) associated with the TPU ASIC.
- [AI Hypercomputer](concepts/ai-hypercomputer.md): AI Hypercomputer: direct, persistent GKE instance <-> physical hardware mapping that replaces Queued Resources and TPU CLH.
- [Network and TPU Stacks](concepts/network-and-tpu-stacks.md): TPU workloads rely on a complex stack of software and hardware for execution and communication.
- [Known-Bad Cubes (Healthy in GKE, Failing a Class of Workload)](concepts/known-bad-cubes.md): HEALTHY cubes that keep failing one workload class (e.g. DCN multislice) but serve others; record with evidence and expose to the scheduler instead of per-user NotIn lists.
- [GKE Details](concepts/gke-architecture.md): Google Kubernetes Engine (GKE) supports running TPU workloads. It introduces specific abstractions to manage these workloads.
- [Megascale](concepts/megascale.md): Megascale is a collective communication library and runtime component responsible for managing DCN-based communication operations on [[cloud-tpu]] systems.
- [Multi-Chip Hangs](concepts/multi-chip-hangs.md): Why one stalled chip hangs all chips at a sync point.
- [Multi-Pod Exit Code Analysis](concepts/multi-pod-exit-code-analysis.md): Exit codes across thousands of pods: find the outlier code and its first host.

## Entities
- [Google Cloud Storage (GCS)](entities/cloud-storage.md): Google Cloud Storage (GCS) is the object store that holds training data and checkpoints for TPU workloads.
- [Google Cloud TPU](entities/cloud-tpu.md): Hub page for Cloud TPU generations (`TPU7x` Ironwood dual-chiplet, `v6e` Trillium, `v5p`, `v5e`, `v4`), 2D/3D/twisted torus topologies, VM/NUMA specs, AI zones, and backends.
- [JAX](entities/jax.md): JAX is a domain-specific tracing Just-In-Time (JIT) compiler designed to generate high-performance accelerator code from pure Python and Numpy programs.
- [libtpu](entities/libtpu.md): `libtpu.so` is the TPU runtime driver library loaded by the host framework (JAX, PyTorch/XLA, or Pathways) to drive the accelerator.
- [MaxText](entities/maxtext.md): MaxText is the JAX large-model training framework that the majority of workloads documented here run.
- [MCJax (Multi-Controller JAX)](entities/mcjax.md): MCJax is the traditional multi-controller execution model for JAX on TPUs, where the Python program runs on every TPU VM and all workers execute the same program in lockstep.
- [SparseCore](entities/sparsecore.md): SparseCore (SC) dataflow processor for embeddings and sparse DMA (`4/chip` on TPU7x & v5p, `2/chip` on v6e, `0` on v5e) and sflag/sequencer hang triage.
- [TensorCore](entities/tensorcore.md): TensorCore (TC) dense compute unit (`128x128` MXUs on v4/v5e/v5p; `256x256` MXUs on v6e/TPU7x; dual-chiplet vs MegaCore) and program lifecycle logs.
- [XLA](entities/xla.md): XLA compiler infrastructure: MXU tiling (`128x128` vs `256x256`), static shapes, zero-padding overhead (`op_profile`), HLO fingerprints, and Pathways proxy compilation.

## Actions
- [AI Classify Log Severities](actions/ai-classify-log-severities.json): AI-operated batch action to classify log severities and identify unassigned host mentions.
- [AI Generate Failure Table](actions/ai-generate-failure-table.json): Synthesizes diagnostic arrays from hw_analyzer, pod-analyzer, and app_analyzer to categorize disruptions and output a Master Failure Table.
- [AI Match Symptoms](actions/ai-match-symptoms.json): AI-operated semantic matching between classified log entries and documented symptom signatures.
- [AI Verify Failure Attribution](actions/ai-verify-failure-attribution.json): Adversarial verification pass over a drafted failure table.
- [Analyze HBM Out of Memory (Error 1000)](actions/analyze-hbm-oom.json): Deterministic diagnostic action to analyze XLA TPU HBM OOM allocation tables (Error E1000), disambiguate compile-time static HBM OOM vs runtime fragmentation OOM vs host RAM OOM (exit code 137), break down peak heap consumers (temporaries, arguments, outputs, aliases), and recommend concrete rematerialization, buffer donation, and SPMD sharding mitigations.
- [Analyze ICI Delay Wave](actions/analyze-ici-wave.json): Deterministic diagnostic action to trace ICI SDC test iteration progression, identify minimum iteration N_min S0 cores with SEND sync flag stalls...
- [Analyze libtpu Program Timing and Outliers](actions/analyze-libtpu-timing.json): Analyzes libtpu runtime program enqueue and last_finished_run_id logs to calculate device execution time and host dispatch gaps across TPU workers...
- [Analyze sflag Timeouts](actions/analyze-sflag-timeouts.json): Analyze and group sflag timeouts from logs to identify stragglers.
- [Analyze SparseCore Offload and Sync Failures](actions/analyze-sparsecore-offload.json): Deterministic diagnostic action to triage SparseCore (SC) offload mismatches (`--xla_tpu_enable_sparse_core_mismatch_detector=true`), SparseCore HBM/SRAM/Smem OOMs, and TensorCore-to-SparseCore `sflag` synchronization timeouts.
- [Annotate Pod Templates for Failure Recovery](actions/annotate-pod-failure-recovery.json): Dynamically injects the version-aware safe-termination annotation into a Kueue workload's pod templates.
- [App Analyzer](actions/ai-app-analyzer.json): Scans live application logs in the 2-minute pre-crash windows bounding each interruption to trace fatal runtime exceptions (e.g., MXLA hangs or TPU runtime crashes).
- [Build and Mask Log Graph](actions/build-and-mask-log-graph.json): Builds a hierarchical log graph from a raw log file and pipes it directly into the masking tool.
- [Build Log Graph](actions/build-log-graph.json): Constructs a log hierarchical tree (graph) from lines (local file) or from Cloud Logging queries (pod/jobset) using indent prefix level sorting and stack push/pop traversal.
- [Change Workload Priority Class](actions/change-workload-priority.json): Change the priority class of a workload (JobSet) to test scheduling order.
- [Check Checkpoint Corruption](actions/check-checkpoint-corruption.json): Check for checkpoint corruption-induced hangs causing 'Jax Process 0 != McJax Coordinator Slice 0' on startup restarts.
- [Check Failure Observability and TPU Log Levels](actions/check-failure-observability.json): Read-only check of framework, effective TPU_STDERR_LOG_LEVEL / TPU_MIN_LOG_LEVEL, coordinator fatal branch, and whether a known bug's log signature is direct or circumstantial.
- [Check GKE Node Versions](actions/check-gke-version.json): Inspects the GKE cluster node versions to check if the nodes are running a version older than 1.35.0-gke.2479000.
- [Check LocalQueue Status](actions/check-localqueue-status.json): Inspect a Kueue LocalQueue to verify resource usage and flavors allocation.
- [Check Node DNS Resolv Config](actions/check-node-dns-resolv.json): Inspect /etc/resolv.conf on GKE node and pod instances to verify metadata server configuration and check for custom DNS overrides.
- [Check Pod Status](actions/check-pod-status.json): Check GKE pod status filtered by workload name.
- [Check Proxy Readiness](actions/check-proxy-readiness.json): Check Proxy logs for readiness messages.
- [Check RM Readiness](actions/check-rm-readiness.json): Check Resource Manager logs for readiness messages.
- [Classify Workload State Across Runs](actions/classify-workload-state.json): Read-only classification of a GKE TPU workload or JobSet series into NEVER_ADMITTED, ADMITTED_NOT_PLACED, PLACED_THEN_CRASHED, MIXED_UNSCHEDULED_AND_CRASHED, HUNG, or WORKLOAD_NOT_FOUND.
- [Compute Per-Run Cube Ledger and Known-Bad Cube Visibility](actions/cube-ledger.json): Read-only TPU7x cube ledger accounting (usable - DEGRADED if pinned HEALTHY - self-excluded pools - held by other queues = free cubes vs needed) and known-bad cube taint visibility check.
- [Configure and Launch Dashboard](actions/configure-and-launch-dashboard.json): Configures GKE cluster monitoring in the dashboard, verifies the local server daemon, triggers a live scan, and generates a deep-linked URL scoped to the target workload.
- [Configure Megascale Timeout Flags](actions/configure-megascale-timeout-flags.json): Adjust Megascale barrier, error reporting, and graph hang threshold flags to customize timeout behavior or gather more debugging information.
- [Configure Orbax Checkpointing](actions/configure-orbax-checkpointing.json): Transition training checkpoint code to use Orbax checkpointing, which uses a larger, more efficient default file size (200MB) to mitigate GCS performance issues.
- [Connect to Cluster](actions/connect-to-cluster.json): Connect to GKE cluster using gcloud.
- [Convert Triage to Findings](actions/convert-triage-to-findings.json): Bridges /troubleshoot triage state JSON into canonical troubleshoot_findings.json conforming to Schema v2.2.
- [Deactivate and Reactivate Kueue Workload](actions/deactivate-and-activate-kueue-workload.json): Forcefully re-admit a stuck GKE Kueue workload (e.g. after preemption) by toggling its spec.active property to false and back to true.
- [Decode Kueue TAS Pending Message History](actions/decode-tas-pending-message.json): Read-only decoder for Kueue Topology-Aware Scheduling ('allows to fit only N out of M slice(s) ... excluded: affinity A, resource R, taint T') and quota messages in ascending order.
- [Describe Queued Resource](actions/describe-queued-resource.json): Describe compute queued resource details.
- [Describe Reservation](actions/describe-reservation.json): Describe compute reservation details.
- [Describe Reservation Sub-blocks](actions/describe-reservation-sub-blocks.json): Describe compute reservation sub-blocks details.
- [Detect Flapping Machines](actions/detect-flapping-machines.json): Internal, read-only deterministic analysis that detects flapping TPU machines and mass events in one zone's GCE system-event timeline...
- [Diagnose sflag Hang](actions/diagnose-sflag-hang.json): Chronological ascending (`--order=ASC`) sflag/SparseCore timeout analysis, minority outlier grouping (< 20%), silent pod detection, and telemetry mapping via diagnose_sflag_hang.py.
- [Enable CoreDNS and NodeLocalDNS](actions/enable-coredns-nodelocaldns.json): Enable CoreDNS and NodeLocalDNS cache on GKE cluster to resolve high-concurrency DNS lookup bottlenecks.
- [Enable Kueue Failure Recovery Policy](actions/enable-kueue-failure-recovery.json): Patches the Kueue controller manager configuration to turn on the FailureRecoveryPolicy feature gate.
- [Enable TPU Verbose Logging](actions/enable-tpu-verbose-logging.json): Set verbose logging flags to enable TPU program execution status monitoring.
- [Extract Culprit Hosts](actions/extract-culprit-hosts.json): Extracts the culprit hosts that Pathways names explicitly in its own logs (`lost connection to peer at http://machine/<node>`, `<node> exiting`) and flags repeat offenders.
- [Find Earliest Failure](actions/find-earliest-failure.json): List exit codes sorted by timestamp to find the earliest failure (first failure in cascade).
- [GCloud Auth Login](actions/gcloud-login.json): Authenticates user credentials with Google Cloud CLI and refreshes Application Default Credentials (ADC) to resolve Context-Aware Access (CAA) or expired tokens.
- [Generate Handoff Manifests](actions/generate-handoff-manifests.json): Generates JSON Schema v2.2 manifests (INCIDENT_MANIFEST.json and JOB_MANIFEST.json) for Cloud TPU diagnostic handoff.
- [Generate Two-Level Failure Tables](actions/generate-two-level-failure-tables.json): Generates Level A (infrastructure/hardware) and Level B (job progression) failure tables strictly adhering to /troubleshoot schema standards.
- [Get Cluster Queues](actions/get-cluster-queues.json): Check cluster queue details.
- [Get JobSet Events](actions/get-jobset-events.json): Query events for a specific JobSet.
- [Get JobSet Leader Logs](actions/get-jobset-leader-logs.json): Get logs for completion-index=0 pods in a JobSet.
- [Get Node from Pod Logs](actions/get-node-from-pod-logs.json): Query k8s_pod logs to find the GKE node name where the pod was scheduled.
- [Get Node Nodepool](actions/get-node-nodepool.json): Get the nodepool name for a GKE node.
- [Get Nodepool Nodes](actions/get-nodepool-nodes.json): List all nodes in a GKE nodepool.
- [Get Nodes Taints](actions/get-nodes-taints.json): List all nodes with their taints.
- [Get Slices](actions/get-slices.json): Monitor slice creation for superslicing-subslicing.
- [Get System Pods](actions/get-system-pods.json): Get pods in jobset-system and kueue-system namespaces.
- [Get TPU Cube Partition State](actions/get-tpu-cube-partition-state.json): Read-only per-node cube view for TPU7x (Ironwood) on GKE on AI Hypercomputer: each node's node pool, cube partition state and report-and-replace status labels next to its STATUS column.
- [Group Scheduled Events by Run and Node Pool](actions/scheduled-events-by-pool.json): Read-only grouping of Kubernetes Scheduled events by (run, node_pool) to identify which runs placed pods on suspect or known-bad node pools.
- [Hardware Interruption Analyzer](actions/ai-hw-analyzer.json): Analyze low-level infrastructure logs to determine if a workload disruption was caused by an underlying hardware failure or VM termination.
- [Identify Outlier TPU Stack Trace](actions/identify-outlier-tpu-stack-trace.json): Enable stack traces on the workload (e.g., using --deploy-stacktrace-sidecar for XPK users) and query Cloud Logging for stack trace logs from the outlier TPU...
- [Inspect HLO Dumps](actions/inspect-hlo-dumps.json): Generate HLO compilation dumps, verify if there are inconsistent program compiles across TPU workers, and copy the HLO dump to GCS to share with Google.
- [Inspect Pod Environment Variables](actions/inspect-pod-env-variables.json): Extracts the environment variables of all containers in a specific pod to verify if TPU_ACCELERATOR_TYPE and other required topology variables are present.
- [Job Disruption Timeline](actions/job-disruption-timeline.json): Identify a clear chronological timeline of disruption events mapping to jobset changes, relying on GKE Metadata parsing and Head Node tracking.
- [List Jobs in JobSet](actions/list-jobset-jobs.json): List all Jobs managed by the JobSet.
- [List JobSet Runs Across Prefix](actions/list-jobset-runs.json): Read-only discovery of every JobSet run matching a name prefix (including deleted or recreated runs) with created (UTC and PDT), admitted, and outcome metadata.
- [List Pods in JobSet](actions/list-jobset-pods.json): List all Pods spawned by the JobSet.
- [List Queued Resources](actions/list-queued-resources.json): List compute queued resources in a zone.
- [List Reservation Sub-blocks Health](actions/list-reservation-sub-blocks-health.json): Read-only JSON listing of every sub-block in one GCE reservation block (for TPU7x one sub-block = one 4x4x4 cube = one GKE TPU node pool of 16 VMs).
- [List Unhealthy GCE Instances](actions/list-unhealthy-gce-instances.json): Read-only JSON list of the VMs in one zone of the project that are not RUNNING (e.g.
- [List Unique Started Pods](actions/list-unique-started-pods.json): List unique pods that logged startup milestones from Cloud Logging.
- [Locate Disruption Windows](actions/locate-disruption-windows.json): Reads a workload's JobSet/Job Kubernetes events (BackoffLimitExceeded, RestartJobSetFailurePolicyAction, SliceDeleted, suspend/resume) to find every restart and the final teardown, and emits padded, merged time windows. These events are small and cheap to query, so the windows let expensive container-log queries (node resolution, culprit extraction) be confined to where the disruptions happened. Does not replace the full-window crash sweep: a disruption the workload recovered from without a JobSet restart leaves no event here.
- [Locate Megascale Hang Timestamp](actions/locate-megascale-hang-timestamp.json): Locate the exact timestamp of the 'Megascale detects a hang' log message in the coordinator logs.
- [Lookup Project Number](actions/lookup-project-number.json): Resolve a GCP project ID to its numeric project number using gcloud projects describe.
- [Lookup Workload Timestamps](actions/lookup-workload-timestamps.json): Lookup first_occurrence and last_occurrence timestamps for a given workload using Cloud Logging.
- [Map Physical ID to GKE Instance](actions/map-physical-id-to-gke-instance.json): Map physical TPU ID to GKE instance name from logs.
- [Mask Log Segment](actions/mask-segment.json): Masks raw log strings or JSON segments using mask_segment_tool.py.
- [Mitigate and Escalate](actions/mitigate-and-escalate.json): Cordon the node to prevent new workloads from landing on it. File a hardware replacement ticket with the data center team.
- [Mitigate TPU Hardware Fault](actions/mitigate-tpu-hardware-fault.json): Restart the TPU workload. If the same TPU worker shows up as an outlier repeatedly, capture the host information and request Google to isolate the bad host.
- [Monitor Handoff Bucket](actions/monitor-handoff-bucket.json): Monitors target GCS handoff repository for updates across all tracks: customer telemetry or Google verdicts, evidence, and actionable workarounds.
- [Isolate Node via R&R Label](actions/isolate-node-via-rr-label.json): Label a GKE node to trigger the Report & Replace (R&R) repair process. (Mutating Action)
- [Package Handoff Artifacts](actions/package-handoff-artifacts.json): Collects diagnostic scratch artifacts, sanitizes HLO dumps, and uploads them into canonical handoff paths (upload is on by default). Artifacts with no source are listed in missing_artifacts instead of being written as placeholder files; exits non-zero on any upload failure.
- [Patch Workload with TPU Env Workaround](actions/patch-workload-tpu-env-workaround.json): Applies a temporary environment variable workaround to a workload (JobSet) by injecting TPU_ACCELERATOR_TYPE, TPU_CHIPS_PER_HOST_BOUNDS...
- [Pod Analyzer Manual Action](actions/pod-analyzer.json): Authoritatively determine whether all pods of the workload scheduled and started running correctly.
- [Post Partner Issue Comment](actions/post-partner-issue-comment.json): Stages the diagnostic handoff comment for the Partner Issue Tracker: renders the comment body and prints the issue link (or the new-issue link) for the operator to paste it. Does not post automatically. Requires explicit operator approval, which also clears the content for sharing with the partner.
- [Prompt Handoff Follow-up Questionnaire](actions/prompt-handoff-followup.json): Interactive questionnaire presented at the conclusion of /troubleshoot, asking the operator whether to initiate a structured diagnostic handoff via /handoff.
- [Query Coordinator Logs Around Timestamp](actions/query-coordinator-logs-around-timestamp.json): Query all logs from the coordinator pod in a 3-minute window around a target timestamp to capture the full failure context.
- [Query Crash Signatures](actions/query-crash-signatures.json): Broadly query for common software crash signatures (e.g. Fatal Python error, Traceback) to catch silent software failures.
- [Query Disruption Events](actions/query-disruption-events.json): Query GKE cluster events for VM staydown and machine group eviction disruption messages.
- [Query Exit Codes](actions/query-exit-codes.json): Query Cloud Logging for EXIT_CODE across all VMs in the workload.
- [Query GCE System Events by Zone](actions/query-gce-system-events-by-zone.json): Read-only, cluster-wide GCE system-event timeline: every Compute Engine `system_event` audit-log entry in ONE zone of the project since `start`...
- [Query ICI Diagnostic Logs](actions/query-ici-logs.json): Query Cloud Logging for XLA `ici-sdc-test` and ICI sync flag wait timeout logs (`XLA_SDC_CHECKER`, `ICI transfer wait`, `Wait timeout on sflag`) emitted by active `ici-sdc-test` diagnostic runs.
- [Query KubeDNS Logs](actions/query-kubedns-logs.json): Query Cloud Logging for kube-system KubeDNS and NodeLocalDNS container logs to check for i/o timeouts and resolution failures.
- [Query ML Diagnostics Monitored Events](actions/query-mldiagnostics-monitored-events.json): Queries the Cluster Director / ML Diagnostics Workload Monitoring API (`gcloud alpha mldiagnostics`, `hypercomputecluster.googleapis.com/v1alpha`) on GKE clusters (`>= 1.36.0-gke.4447000`) to cover silent metric-only `PERFORMANCE_DEGRADATION` bottlenecks that do not emit error signatures in Cloud Logging: (1) host cgroup resource saturation (`CPU Utilization Analyzer`, `Host Memory Utilization Analyzer`), (2) accelerator memory saturation (`HBM Capacity Analyzer`, `HBM Bandwidth Analyzer`), (3) silent hardware/fabric throttling (`Thermal Analyzer`, `TPU Throttling Analyzer`, `ICI Link Analyzer`, `ToR NIC Link Analyzer`, `ToR Uplink Congestion`), (4) port-`9431` Prometheus framework stalls (`PyGrain Data Analyzer`, `Async Checkpointing Analyzer`), (5) negative hardware rule-out (`not_detected_analyzers` across all 11 subsystems), and (6) AutoCaptured XProf `.xplane.pb` traces (`profiler_sessions`). Requires the 15-minute duty cycle warmup gate after each `RUNNING` transition (comparing backward-looking `[t-5m, t]` vs. `[t-15m, t]` windows); for crashes, hangs, restarts, preemptions, or `<15m` startup/restore stalls, use Debug Wiki's core Cloud Logging, Cloud Logging, and host maintenance telemetry/host maintenance telemetry workflows instead.
- [Query Node Health Events](actions/query-node-health-events.json): Query `k8s_node`-level logs (kubelet, node-problem-detector, TPU device plugin) for the nodes of a workload.
- [Query Pod Exit Codes](actions/query-pod-exit-codes.json): Query and aggregate exit codes across all pods in a multi-pod workload to identify non-zero primary failure exit codes.
- [Query VM Termination Events](actions/query-vm-termination-events.json): Deterministically queries GCP Audit Logs (Compute Engine System Activity) for host errors, terminations, preemptions, and automated repairs, scoped to a specific workload's nodes.
- [Query Workload Logs](actions/query-workload-logs.json): Query Cloud Logging for workload logs with optional filters and limits.
- [Recover JobSet Spec from GKE Audit Logs](actions/recover-jobset-spec-from-audit.json): Read-only recovery of per-run JobSet pod templates (nodeSelector, NotIn node-pool exclusions, 4x4x4-state pin, queue name, container env) from GKE audit io.k8s.batch.v1.jobs.create request bodies.
- [Redact Diagnostic Artifacts](actions/redact-diagnostic-artifacts.json): Sanitizes proprietary tensor names, shapes, and customer code paths from textual HLO graphs and stack traces.
- [Render Handoff READMEs](actions/render-handoff-readmes.json): Renders incident overviews, root READMEs, and job context markdown files from templates adhering to strict ASCII standards.
- [Report Host as Faulty](actions/report-host-as-faulty.json): Report a GPU VM host as faulty to trigger repair (GCE/GPU VM).
- [Resolve Cluster Metadata](actions/resolve-cluster-metadata.json): Inspects GKE cluster properties to determine TPU accelerator generation (e.g. tpu7x, v6e, v5p), backing GCE reservation, and region.
- [Resolve Workload Infra Specs](actions/resolve-workload-infra-specs.json): Extracts machine_type, accelerator_type, chips_per_host, and topology using kubectl node metadata, gcloud compute instances describe...
- [Resolve Workload Node Set](actions/resolve-workload-nodes.json): Resolves the COMPLETE set of GKE nodes that hosted a workload's pods by slicing the query window, and asserts completeness.
- [Run Cross Layer Analysis](actions/run-cross-layer-analysis.json): Coordinate automated log analysis, stack traces, and hardware diagnostics to debug multi-chip hangs cross-layer.
- [Run Single Slice Hardware Test](actions/run-single-slice-hardware-test.json): Scale down to a single slice and enable hardware PC counters to test for hardware faults (e.g., stuck sparse core) during startup.
- [Scaffold Handoff Bucket](actions/scaffold-handoff-bucket.json): Provisions the multi-job diagnostic incident structure on GCS (BUCKET_STRUCTURE.md directory dictionary, support case link). Never pre-creates empty directory markers (.keep), manifests, READMEs or failure tables; directories in GCS are created on-demand as real files are uploaded. Requires an explicit bucket.
- [Scale DNS Pod Replicas](actions/scale-dns-pod-replicas.json): Scale CoreDNS or KubeDNS autoscaler deployment replicas proportional to TPU worker node count for multi-slice workloads.
- [Tail Coordinator Logs](actions/tail-coordinator-logs.json): Tail logs for the coordinator pod.
- [Taint Node](actions/taint-node.json): Apply a taint to a GKE node. (Mutating Action)
- [Trace HBM ECC Errors](actions/trace-hbm-ecc-errors.json): Query host kernel and driver diagnostic logs for uncorrectable HBM ECC memory errors and PCIe bus faults on target TPU machines.
- [Triage SDC Checksum Mismatch and Numerics Errors](actions/triage-sdc-and-numerics.json): Deterministic diagnostic action to analyze XLA Silent Data Corruption (SDC / XSC) checksum mismatch logs (`XLA_SDC_CHECKER`), LLO deterministic replay and runtime bounds check failures, and NaN/Inf numerics divergence across retries and hosts to classify hardware defects vs XLA compiler bugs vs application non-determinism.
- [Triage Workload Signals](actions/triage-workload-signals.json): Triage workload signals and pull related raw logs using triage_workloads_api.py.
- [Unmask Logs](actions/unmask-logs.json): Unmasks a masked log file using the generated mappings JSON file.
- [Upgrade GKE Cluster Version](actions/upgrade-gke-cluster.json): Steps to upgrade the GKE cluster control plane and node pools to a version that natively includes the ICI topology wrap-around fix (>= 1.35.0-gke.2479000).
- [Upload Handoff Package](actions/upload-handoff-package.json): Publishes the rendered handoff package (manifests, failure tables, READMEs) from the local staging tree to gs://<bucket>/<storage_key>/. All-or-nothing: refuses to upload anything if a required file is missing or any staged file is empty, placeholder-only, or invalid JSON.
- [Validate Workload](actions/validate-workload.json): Verifies if a workload initialized, failed startup, or was never submitted based on Cloud Logging / Audit Logs within a Start and End timestamp window.
- [Verify Dashboard URL](actions/verify-dashboard-url.json): Verifies that a generated dashboard URL is actually usable: its hostname resolves, and both the deep-linked page and /api/health on that host return 200 across...
- [Verify Handoff Bucket Access](actions/verify-handoff-bucket-access.json): Pre-flight GCS prober verifying bucket existence, write/read connectivity via canary probe, and public access prevention.
- [Verify Handoff Package](actions/verify-handoff-package.json): Post-publication content check: lists gs://<bucket>/<storage_key>/ and fails (INCOMPLETE) if any required manifest, README or failure table is missing, or any file is 0 bytes, placeholder-only, or invalid JSON. A failed listing is VERIFY_FAILED, never an empty bucket.
- [Verify JobSet Status](actions/verify-jobset-status.json): Check if the JobSet exists and what its status is.
- [Verify Training Barrier API Usage](actions/verify-training-barrier-api-usage.json): Inspect training files to verify that collective barrier synchronization at the end of checkpoint/IO loading uses wait_at_barrier rather than MXLA all-reduce collectives.
- [Watch Pods For Failure](actions/watch-pods-for-failure.json): Watch pods in a live workload to see which one transitions from Running to Error first.

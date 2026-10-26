#!/usr/bin/env python3
"""Read-only run forensics analyzers for multi-run GKE TPU JobSet triage.

Provides deterministic subcommands for the `workload-run-forensics` workflow:
  - classify-workload-state
  - list-jobset-runs
  - recover-jobset-spec
  - scheduled-events-by-pool
  - decode-tas-message
  - cube-ledger
  - check-failure-observability

All operations are strictly read-only (`gcloud logging read`, `kubectl get`,
`kubectl logs`, or offline fixture files). No cluster mutations are performed.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
from typing import Any
import zoneinfo

NODES_PER_TPU7X_CUBE = 16
# FATAL (tensorflow/compiler/xla/runtime/common_lib.cc)
DEFAULT_STDERR_LOG_LEVEL = 3
DEFAULT_MIN_LOG_LEVEL = 0  # INFO (local /tmp/tpu_logs only)

_TAS_RE = re.compile(
    r'topology\s+"(?P<topology>[^"]+)"\s+allows\s+to\s+fit\s+only\s+'
    r"(?P<fit>\d+)\s+out\s+of\s+(?P<needed>\d+)\s+slice\(s\).*?"
    r"Total\s+nodes\s+(?P<total_nodes>\d+);\s*excluded:\s*(?P<excluded>.*)",
    re.DOTALL,
)
_QUOTA_RE = re.compile(
    r"insufficient\s+unused\s+quota\s+in\s+cohort\s+for\s+"
    r"(?P<resource>[^\s,]+)\s+in\s+flavor\s+(?P<flavor>[^\s,]+),\s*"
    r"(?P<needed_amount>\d+)\s+more\s+needed",
    re.IGNORECASE,
)
_ASSIGNED_NODE_RE = re.compile(
    r"Successfully\s+assigned\s+(?P<namespace>[^\s/]+)/(?P<pod>[^\s]+)\s+to\s+"
    r"(?P<node>[^\s]+)"
)


def parse_utc(ts: str) -> datetime.datetime:
  """Parses an ISO-8601 UTC timestamp string."""
  cleaned = ts.strip()
  if cleaned.endswith("Z"):
    cleaned = cleaned[:-1] + "+00:00"
  dt = datetime.datetime.fromisoformat(cleaned)
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=datetime.timezone.utc)
  return dt.astimezone(datetime.timezone.utc)


def format_pdt(ts: str) -> str:
  """Formats an ISO-8601 UTC timestamp string into America/Los_Angeles."""
  if not ts:
    return ""
  try:
    dt = parse_utc(ts).astimezone(zoneinfo.ZoneInfo("America/Los_Angeles"))
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
  except ValueError:
    return ts


def _load_json_file(path: str) -> Any:
  with open(path, "r", encoding="utf-8") as fh:
    content = fh.read().strip()
    if not content:
      return []
    return json.loads(content)


def _run_readonly_cmd(cmd: list[str]) -> str:
  """Executes a read-only gcloud or kubectl command and returns stdout."""
  res = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      check=False,
      timeout=120,
  )
  if res.returncode != 0:
    raise RuntimeError(
        f"Read-only command failed (exit {res.returncode}): {' '.join(cmd)}\n"
        f"{res.stderr.strip()}"
    )
  return res.stdout


def _fetch_logging_json(
    project_id: str,
    log_filter: str,
    limit: int = 1000,
    order: str = "asc",
) -> list[dict[str, Any]]:
  """Runs a read-only `gcloud logging read` command returning JSON entries."""
  out = _run_readonly_cmd([
      "gcloud",
      "logging",
      "read",
      log_filter,
      f"--project={project_id}",
      f"--limit={limit}",
      f"--order={order}",
      "--format=json",
  ])
  if not out.strip():
    return []
  data = json.loads(out)
  return data if isinstance(data, list) else []


def _write_output_and_summary(
    result: dict[str, Any],
    summary_fields: list[str],
    output_file: str,
) -> None:
  """Writes full JSON result to `output_file` and prints summary fields."""
  if output_file:
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
      json.dump(result, fh, indent=2)
    result["output_file"] = output_file

  print(json.dumps(result, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  for field in summary_fields:
    val = result.get(field)
    if isinstance(val, (dict, list)):
      print(f"  {field}: {json.dumps(val)}")
    else:
      print(f"  {field}: {val}")
  print("---------------------------------------")


def _extract_jobset_name(raw_name: str, workload_prefix: str) -> str:
  """Extracts the JobSet name (e.g. `train-8k-5`) from a Job or Pod name."""
  if not raw_name:
    return ""
  match = re.search(
      rf"^({re.escape(workload_prefix)}-\d+)\b",
      raw_name,
  )
  if match:
    return match.group(1)
  if raw_name.startswith(workload_prefix):
    parts = raw_name.split("-")
    prefix_parts = workload_prefix.split("-")
    if len(parts) > len(prefix_parts):
      return "-".join(parts[: len(prefix_parts) + 1])
    return raw_name
  return raw_name


def extract_nodepool_from_node(node_name: str, cluster_name: str = "") -> str:
  """Extracts the GKE node pool name from a node hostname."""
  if not node_name:
    return ""
  name = node_name.strip()
  if cluster_name and name.startswith(f"gke-{cluster_name}-"):
    rest = name[len(f"gke-{cluster_name}-") :]
    parts = rest.split("-")
    if len(parts) >= 3:
      return "-".join(parts[:-2])
  # Generic GKE pattern: gke-<cluster>-<nodepool>-<8charhash>-<4charsuffix>
  match = re.fullmatch(
      r"gke-[a-z0-9-]+?-(np-[a-z0-9-]+)-[a-f0-9]{6,10}-[a-z0-9]+", name
  )
  if match:
    return match.group(1)
  parts = name.split("-")
  if len(parts) >= 4 and parts[0] == "gke":
    return "-".join(parts[2:-2])
  return name


def recover_specs_from_audit_entries(
    entries: list[dict[str, Any]],
    workload_prefix: str,
) -> dict[str, Any]:
  """Extracts per-run JobSet specs from `io.k8s.batch.v1.jobs.create` audit logs.

  Groups child Job creations belonging to the same JobSet instance (within a
  5-minute creation window) into a single run record, and distinguishes
  recreated JobSets that reuse the same name later.

  Args:
    entries: Audit log entries (`io.k8s.batch.v1.jobs.create`).
    workload_prefix: Workload name prefix (e.g. `train-8k`).

  Returns:
    Dict containing `runs` (ordered chronologically), `exclusion_counts_by_run`,
    `exclusion_list_drift`, and `pins_healthy_only`.
  """
  sorted_entries = sorted(entries, key=lambda entry: entry.get("timestamp", ""))
  raw_runs: list[dict[str, Any]] = []

  for entry in sorted_entries:
    ts = entry.get("timestamp", "")
    proto = entry.get("protoPayload") or {}
    req = proto.get("request") or {}
    meta = req.get("metadata") or {}
    labels = meta.get("labels") or {}
    annotations = meta.get("annotations") or {}

    jobset_name = (
        labels.get("jobset.sigs.k8s.io/jobset-name")
        or annotations.get("jobset.sigs.k8s.io/jobset-name")
        or _extract_jobset_name(meta.get("name", ""), workload_prefix)
    )
    if not jobset_name or not jobset_name.startswith(workload_prefix):
      continue

    spec = req.get("spec") or {}
    tmpl = spec.get("template") or {}
    tmpl_meta = tmpl.get("metadata") or {}
    tmpl_labels = tmpl_meta.get("labels") or {}
    tmpl_ann = tmpl_meta.get("annotations") or {}
    pod_spec = tmpl.get("spec") or {}

    queue_name = (
        labels.get("kueue.x-k8s.io/queue-name")
        or annotations.get("kueue.x-k8s.io/queue-name")
        or tmpl_labels.get("kueue.x-k8s.io/queue-name")
        or tmpl_ann.get("kueue.x-k8s.io/queue-name")
        or ""
    )
    node_selector = dict(pod_spec.get("nodeSelector") or {})

    excluded_pools: list[str] = []
    partition_states: list[str] = []
    affinity = (
        (pod_spec.get("affinity") or {})
        .get("nodeAffinity", {})
        .get("requiredDuringSchedulingIgnoredDuringExecution", {})
    )
    for term in affinity.get("nodeSelectorTerms") or []:
      for expr in term.get("matchExpressions") or []:
        key = expr.get("key", "")
        op = expr.get("operator", "")
        vals = list(expr.get("values") or [])
        if key == "cloud.google.com/gke-nodepool" and op == "NotIn":
          for val in vals:
            if val not in excluded_pools:
              excluded_pools.append(val)
        elif "partition" in key and key.endswith("-state") and op == "In":
          for val in vals:
            if val not in partition_states:
              partition_states.append(val)

    for label_key, label_val in node_selector.items():
      if (
          "partition" in label_key
          and label_key.endswith("-state")
          and label_val not in partition_states
      ):
        partition_states.append(label_val)

    pins_healthy_only = partition_states == ["HEALTHY"] or (
        "HEALTHY" in partition_states and "DEGRADED" not in partition_states
    )

    # Extract container env & image for observability cross-reference
    containers = pod_spec.get("containers") or []
    env_map: dict[str, str] = {}
    image = ""
    command_str = ""
    if containers:
      container_zero = containers[0]
      image = container_zero.get("image", "")
      cmd_parts = list(container_zero.get("command") or []) + list(
          container_zero.get("args") or []
      )
      command_str = " ".join(str(part) for part in cmd_parts)
      for env_item in container_zero.get("env") or []:
        if "name" in env_item and "value" in env_item:
          env_map[env_item["name"]] = str(env_item["value"])

    # Merge into an existing run if same jobset_name within 5 minutes
    merged = False
    for existing in reversed(raw_runs):
      if existing["jobset_name"] == jobset_name:
        try:
          delta = abs(
              (
                  parse_utc(ts) - parse_utc(existing["created_at"])
              ).total_seconds()
          )
        except ValueError:
          delta = 0.0
        if delta <= 300:
          for pool in excluded_pools:
            if pool not in existing["excluded_pools"]:
              existing["excluded_pools"].append(pool)
          existing["excluded_pools_count"] = len(existing["excluded_pools"])
          existing["child_jobs_count"] += 1
          merged = True
        break

    if not merged:
      raw_runs.append({
          "jobset_name": jobset_name,
          "created_at": ts,
          "created_pdt": format_pdt(ts),
          "queue_name": queue_name,
          "node_selector": node_selector,
          "excluded_pools": excluded_pools,
          "excluded_pools_count": len(excluded_pools),
          "partition_state_pin": partition_states or ["HEALTHY", "DEGRADED"],
          "pins_healthy_only": pins_healthy_only,
          "image": image,
          "command": command_str,
          "env": env_map,
          "child_jobs_count": 1,
      })

  # Disambiguate recreated runs that share a jobset_name
  name_totals: dict[str, int] = {}
  for run in raw_runs:
    name_totals[run["jobset_name"]] = name_totals.get(run["jobset_name"], 0) + 1

  name_seen: dict[str, int] = {}
  for run in raw_runs:
    jname = run["jobset_name"]
    idx = name_seen.get(jname, 0) + 1
    name_seen[jname] = idx
    if name_totals[jname] > 1:
      suffix = (
          "first" if idx == 1 else ("recreated" if idx == 2 else f"run-{idx}")
      )
      run["run_id"] = f"{jname} ({suffix})"
    else:
      run["run_id"] = jname

  exclusion_counts = {
      run["run_id"]: run["excluded_pools_count"] for run in raw_runs
  }
  distinct_lists = {tuple(sorted(run["excluded_pools"])) for run in raw_runs}
  drift = len(distinct_lists) > 1

  return {
      "runs_recovered": len(raw_runs),
      "runs": raw_runs,
      "exclusion_counts_by_run": exclusion_counts,
      "exclusion_list_drift": drift,
      "pins_healthy_only": any(run["pins_healthy_only"] for run in raw_runs),
  }


def group_scheduled_events_by_pool(
    events: list[dict[str, Any]],
    workload_prefix: str,
    cluster_name: str = "",
    suspect_pools: list[str] | None = None,
    runs_spec: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  """Groups Kubernetes `Scheduled` events by `(run_id, node_pool)`."""
  suspect_set = set(suspect_pools or [])
  pools_by_run: dict[str, dict[str, int]] = {}
  suspect_hits_by_run: dict[str, dict[str, int]] = {}

  sorted_events = sorted(events, key=lambda event: event.get("timestamp", ""))
  for ev in sorted_events:
    payload = ev.get("jsonPayload") or ev
    reason = payload.get("reason", "")
    if reason and reason != "Scheduled":
      continue
    msg = payload.get("message", "")
    obj = payload.get("involvedObject") or {}
    pod_name = obj.get("name", "")
    node_name = payload.get("node", "") or payload.get("reportingInstance", "")
    match = _ASSIGNED_NODE_RE.search(msg)
    if match:
      if not pod_name:
        pod_name = match.group("pod")
      node_name = match.group("node")
    if not pod_name or not pod_name.startswith(workload_prefix):
      continue

    base_jobset = _extract_jobset_name(pod_name, workload_prefix)
    run_id = base_jobset
    ev_ts = ev.get("timestamp", "")

    # Match to disambiguated run_id (e.g. `train-8k-6 (first)` vs recreated).
    if runs_spec and ev_ts:
      candidates = [
          run for run in runs_spec if run.get("jobset_name") == base_jobset
      ]
      if len(candidates) > 1:
        chosen = candidates[0]
        for cand in candidates:
          if cand.get("created_at", "") <= ev_ts:
            chosen = cand
        run_id = chosen["run_id"]
      elif len(candidates) == 1:
        run_id = candidates[0]["run_id"]

    pool = payload.get("nodepool") or extract_nodepool_from_node(
        node_name, cluster_name=cluster_name
    )
    if not pool:
      continue

    run_pools = pools_by_run.setdefault(run_id, {})
    run_pools[pool] = run_pools.get(pool, 0) + 1
    if pool in suspect_set:
      shits = suspect_hits_by_run.setdefault(run_id, {})
      shits[pool] = shits.get(pool, 0) + 1

  return {
      "runs_with_scheduled_pods": len(pools_by_run),
      "pools_by_run": pools_by_run,
      "suspect_pool_hits": suspect_hits_by_run,
  }


def decode_tas_messages(
    entries: list[dict[str, Any]],
    nodes_per_cube: int = NODES_PER_TPU7X_CUBE,
) -> dict[str, Any]:
  """Decodes Kueue TAS pending/QuotaReserved messages into 4x4x4 cube counts."""
  sorted_entries = sorted(entries, key=lambda e: e.get("timestamp", ""))
  history: list[dict[str, Any]] = []
  quota_messages: list[dict[str, Any]] = []
  tas_decoded: list[dict[str, Any]] = []

  for entry in sorted_entries:
    ts = entry.get("timestamp", "")
    payload = entry.get("jsonPayload") or entry
    msg = payload.get("message", "") or entry.get("textPayload", "")
    if not msg:
      continue

    qm = _QUOTA_RE.search(msg)
    if qm:
      rec = {
          "timestamp": ts,
          "type": "INSUFFICIENT_QUOTA",
          "resource": qm.group("resource"),
          "flavor": qm.group("flavor"),
          "needed_amount": int(qm.group("needed_amount")),
          "message": msg,
      }
      quota_messages.append(rec)
      history.append(rec)

    tm = _TAS_RE.search(msg)
    if tm:
      fit_slices = int(tm.group("fit"))
      needed_slices = int(tm.group("needed"))
      total_nodes = int(tm.group("total_nodes"))
      excl_str = tm.group("excluded")

      aff_match = re.search(r"affinity\s+(\d+)", excl_str)
      res_match = re.search(r"resource\s+[^\s,]+\s+(\d+)", excl_str)
      taint_matches = re.findall(r"taint\s+[^\s,]+\s+(\d+)", excl_str)
      if not taint_matches:
        taint_generic = re.search(r"taint\s+(\d+)", excl_str)
        taint_nodes = int(taint_generic.group(1)) if taint_generic else 0
      else:
        taint_nodes = sum(int(x) for x in taint_matches)

      aff_nodes = int(aff_match.group(1)) if aff_match else 0
      res_nodes = int(res_match.group(1)) if res_match else 0

      rec = {
          "timestamp": ts,
          "type": "TAS_TOPOLOGY_UNFIT",
          "topology": tm.group("topology"),
          "fit_slices": fit_slices,
          "needed_slices": needed_slices,
          "shortfall_cubes": needed_slices - fit_slices,
          "total_nodes": total_nodes,
          "excluded_affinity_nodes": aff_nodes,
          "excluded_affinity_cubes": aff_nodes // nodes_per_cube,
          "excluded_resource_nodes": res_nodes,
          "excluded_resource_cubes": res_nodes // nodes_per_cube,
          "excluded_taint_nodes": taint_nodes,
          "excluded_taint_full_cubes": taint_nodes // nodes_per_cube,
          "message": msg,
      }
      tas_decoded.append(rec)
      history.append(rec)

  latest_tas = tas_decoded[-1] if tas_decoded else {}
  two_pass_masking = bool(quota_messages and tas_decoded)

  return {
      "fit_slices": latest_tas.get("fit_slices", 0),
      "needed_slices": latest_tas.get("needed_slices", 0),
      "shortfall_cubes": latest_tas.get("shortfall_cubes", 0),
      "total_nodes": latest_tas.get("total_nodes", 0),
      "excluded_affinity_nodes": latest_tas.get("excluded_affinity_nodes", 0),
      "excluded_affinity_cubes": latest_tas.get("excluded_affinity_cubes", 0),
      "excluded_resource_nodes": latest_tas.get("excluded_resource_nodes", 0),
      "excluded_resource_cubes": latest_tas.get("excluded_resource_cubes", 0),
      "excluded_taint_nodes": latest_tas.get("excluded_taint_nodes", 0),
      "two_pass_quota_masking": two_pass_masking,
      "quota_messages_count": len(quota_messages),
      "tas_messages_count": len(tas_decoded),
      "history": history,
  }


def compute_cube_ledger(
    nodes_data: list[dict[str, Any]],
    excluded_pools: list[str],
    needed_cubes: int,
    pins_healthy_only: bool = False,
    known_bad_pools: list[str] | None = None,
    held_cubes_records: list[dict[str, Any]] | None = None,
    nodes_per_cube: int = NODES_PER_TPU7X_CUBE,
) -> dict[str, Any]:
  """Computes the per-run TPU7x cube ledger and known-bad cube visibility."""
  pools: dict[str, dict[str, Any]] = {}
  for node in nodes_data:
    meta = node.get("metadata") or {}
    labels = meta.get("labels") or {}
    spec = node.get("spec") or {}
    status = node.get("status") or {}

    pool_name = (
        labels.get("cloud.google.com/gke-nodepool")
        or node.get("nodepool")
        or ""
    )
    if not pool_name:
      continue

    part_state = (
        labels.get("cloud.google.com/gke-tpu-partition-4x4x4-state")
        or node.get("partition_state")
        or "HEALTHY"
    )
    taints = list(spec.get("taints") or node.get("taints") or [])
    unschedulable = bool(
        spec.get("unschedulable") or node.get("unschedulable", False)
    )
    ready = True
    for cond in status.get("conditions") or []:
      if cond.get("type") == "Ready":
        ready = cond.get("status") == "True"

    weight = int(node.get("node_count", 1))
    info = pools.setdefault(
        pool_name,
        {
            "pool": pool_name,
            "node_count": 0,
            "ready_nodes": 0,
            "partition_state": part_state,
            "tainted": False,
            "taints": [],
            "unschedulable": False,
        },
    )
    info["node_count"] += weight
    if ready:
      info["ready_nodes"] += weight
    if part_state in ("UNHEALTHY", "DEGRADED"):
      info["partition_state"] = part_state
    if taints:
      info["tainted"] = True
      info["taints"].extend(taints)
    if unschedulable:
      info["unschedulable"] = True

  healthy_pools: list[str] = []
  degraded_pools: list[str] = []
  broken_pools: list[str] = []

  for pname, pinfo in sorted(pools.items()):
    is_full = pinfo["ready_nodes"] >= nodes_per_cube
    state = pinfo["partition_state"]
    if (
        not is_full
        or state == "UNHEALTHY"
        or pinfo["tainted"]
        or pinfo["unschedulable"]
    ):
      broken_pools.append(pname)
    elif state == "DEGRADED":
      degraded_pools.append(pname)
    else:
      healthy_pools.append(pname)

  usable_set = set(healthy_pools) | set(degraded_pools)
  usable_cubes = len(usable_set)
  degraded_subtracted = len(degraded_pools) if pins_healthy_only else 0

  allowed_usable = set(healthy_pools) if pins_healthy_only else usable_set
  self_excluded_usable = [
      pool for pool in excluded_pools if pool in allowed_usable
  ]

  holders = list(held_cubes_records or [])
  if not holders:
    seen_holder_pools: set[str] = set()
    for node in nodes_data:
      hb = node.get("held_by")
      if isinstance(hb, dict):
        meta = node.get("metadata") or {}
        labels = meta.get("labels") or {}
        pname = (
            labels.get("cloud.google.com/gke-nodepool")
            or node.get("nodepool")
            or ""
        )
        if pname and pname not in seen_holder_pools:
          seen_holder_pools.add(pname)
          holders.append({"pool": pname, **hb})
  held_pools = [
      holder["pool"]
      for holder in holders
      if holder.get("pool") in allowed_usable
      and holder.get("pool") not in self_excluded_usable
  ]

  free_cubes = (
      usable_cubes
      - degraded_subtracted
      - len(self_excluded_usable)
      - len(held_pools)
  )
  fits = free_cubes >= needed_cubes

  # Check known-bad cube scheduler visibility
  known_bad = list(known_bad_pools or [])
  visibility: dict[str, dict[str, Any]] = {}
  invisible_known_bad: list[str] = []
  for kb_pool in known_bad:
    pinfo = pools.get(kb_pool, {})
    scheduler_visible = bool(pinfo.get("tainted") or pinfo.get("unschedulable"))
    visibility[kb_pool] = {
        "pool": kb_pool,
        "partition_state": pinfo.get("partition_state", "UNKNOWN"),
        "tainted": bool(pinfo.get("tainted")),
        "unschedulable": bool(pinfo.get("unschedulable")),
        "scheduler_visible": scheduler_visible,
        "only_avoided_by_user_notin": (
            not scheduler_visible and kb_pool in excluded_pools
        ),
    }
    if not scheduler_visible:
      invisible_known_bad.append(kb_pool)

  # "Would the remediation help?" math check
  only_known_bad_excluded = [
      pool for pool in known_bad if pool in allowed_usable
  ]
  free_if_only_known_bad_excluded = (
      usable_cubes
      - degraded_subtracted
      - len(only_known_bad_excluded)
      - len(held_pools)
  )
  free_if_holders_reclaimed = free_cubes + len(held_pools)

  holder_desc = (
      f"{len(held_pools)} held ({', '.join(held_pools)})"
      if held_pools
      else "0 held"
  )
  if self_excluded_usable:
    excl_desc = (
        f"{len(self_excluded_usable)} self-excluded"
        f" ({', '.join(self_excluded_usable)})"
    )
  else:
    excl_desc = "0 self-excluded"
  deg_desc = (
      f" - {degraded_subtracted} DEGRADED (job pins HEALTHY)"
      if degraded_subtracted
      else ""
  )
  cmp_sym = ">=" if fits else "<"
  formula = (
      f"{usable_cubes} usable ({len(healthy_pools)} HEALTHY + "
      f"{len(degraded_pools)} DEGRADED){deg_desc} - {excl_desc} - "
      f"{holder_desc} = {free_cubes} free {cmp_sym} {needed_cubes} needed"
  )

  recommendations: list[dict[str, Any]] = []
  if invisible_known_bad:
    recommendations.append({
        "action": "taint-node",
        "requires_approval": True,
        "approval_note": "Ask the user before running any taint or cordon.",
        "target_pools": invisible_known_bad,
        "reason": (
            "Known-bad pools report HEALTHY in GKE and have no cluster taint, "
            "so Kueue quota math still counts them while per-user NotIn lists "
            "hide the exclusion and drift between runs."
        ),
    })

  return {
      "total_cubes": len(pools),
      "broken_or_unhealthy_cubes": len(broken_pools),
      "broken_pools": broken_pools,
      "usable_cubes": usable_cubes,
      "healthy_cubes": len(healthy_pools),
      "degraded_cubes": len(degraded_pools),
      "degraded_pools": degraded_pools,
      "pins_healthy_only": pins_healthy_only,
      "degraded_subtracted": degraded_subtracted,
      "self_excluded_pools": self_excluded_usable,
      "self_excluded_cubes": len(self_excluded_usable),
      "held_by_other_queues": holders,
      "held_by_other_queues_cubes": len(held_pools),
      "free_cubes_for_run": free_cubes,
      "needed_cubes": needed_cubes,
      "fits": fits,
      "shortfall_cubes": max(0, needed_cubes - free_cubes),
      "formula": formula,
      "known_bad_cube_visibility": visibility,
      "invisible_known_bad_pools": invisible_known_bad,
      "remediation_math_check": {
          "free_if_holders_reclaimed": free_if_holders_reclaimed,
          "fits_if_holders_reclaimed": (
              free_if_holders_reclaimed >= needed_cubes
          ),
          "free_if_only_known_bad_excluded": free_if_only_known_bad_excluded,
          "fits_if_only_known_bad_excluded": (
              free_if_only_known_bad_excluded >= needed_cubes
          ),
      },
      "recommendations": recommendations,
  }


def evaluate_failure_observability(
    spec_data: dict[str, Any],
    log_lines: list[str] | None = None,
    known_bug_signature: str = "",
) -> dict[str, Any]:
  """Evaluates framework, effective TPU log levels, and coordinator branch."""
  image = str(spec_data.get("image", ""))
  command = str(spec_data.get("command", ""))
  env = dict(spec_data.get("env") or {})

  combined_spec = f"{image} {command} {json.dumps(env)}".lower()
  if "pathways" in combined_spec or "ifrt_proxy" in combined_spec:
    framework = "pathways"
    coordinator_pod_pattern = "pathways-rm / completion-index=0"
  elif (
      "torch" in combined_spec
      or "torchtitan" in combined_spec
      or "pjrt_device" in combined_spec
  ):
    framework = "pytorch_xla"
    coordinator_pod_pattern = (
        "<jobset>-<replicated-job>-0-0 (slice 0, worker 0)"
    )
  else:
    framework = "jax"
    coordinator_pod_pattern = (
        "<jobset>-<replicated-job>-0-0 (slice 0, worker 0)"
    )

  # Check env and command exports for TPU_STDERR_LOG_LEVEL and TPU_MIN_LOG_LEVEL
  stderr_raw = env.get("TPU_STDERR_LOG_LEVEL")
  if stderr_raw is None:
    m_stderr = re.search(r"TPU_STDERR_LOG_LEVEL\s*=\s*(\d+)", command)
    if m_stderr:
      stderr_raw = m_stderr.group(1)

  min_raw = env.get("TPU_MIN_LOG_LEVEL")
  if min_raw is None:
    m_min = re.search(r"TPU_MIN_LOG_LEVEL\s*=\s*(\d+)", command)
    if m_min:
      min_raw = m_min.group(1)

  effective_stderr = (
      int(stderr_raw) if stderr_raw is not None else DEFAULT_STDERR_LOG_LEVEL
  )
  effective_min = int(min_raw) if min_raw is not None else DEFAULT_MIN_LOG_LEVEL
  digest_visible = effective_stderr <= 2

  lines = list(log_lines or [])
  joined_logs = "\n".join(lines)
  if (
      "Megascale detects a hang" in joined_logs
      or "HANG_DETECTED" in joined_logs
  ):
    coordinator_branch = "HANG_DETECTED"
  elif (
      "Aborting the coordinator as an unrecoverable error is reported from one"
      " or more workers"
      in joined_logs
  ):
    coordinator_branch = "UNRECOVERABLE_WORKER_ERROR"
  else:
    coordinator_branch = "NONE_FOUND"

  worker_stack_signatures: list[str] = []
  for sig in (
      "SendRpcContext::CompleteRpc",
      "DebugManager::ErrorDetected",
      "TpuChipCommonImpl::GetCore",
      "raw_hash_map::at",
  ):
    if sig in joined_logs:
      worker_stack_signatures.append(sig)

  sig_hits = 0
  if known_bug_signature:
    sig_hits = sum(1 for ln in lines if known_bug_signature in ln)
    attribution_strength = (
        "DIRECT_LOG_MATCH" if sig_hits > 0 else "CIRCUMSTANTIAL"
    )
  else:
    attribution_strength = "NOT_CHECKED"

  observability_gap = ""
  if not digest_visible:
    observability_gap = (
        f"TPU_STDERR_LOG_LEVEL is {effective_stderr} (default 3=FATAL from "
        "tensorflow/compiler/xla/runtime/common_lib.cc). Megascale's "
        "'Megascale detects a hang' and 'Full error digest' lines are emitted "
        "with LOG(ERROR) (severity 2) in megascale_error_aggregator.cc before "
        "LOG(FATAL) in error_reporter.cc, so they are invisible in Cloud "
        "Logging while local /tmp/tpu_logs files are lost when pods terminate."
    )

  return {
      "framework": framework,
      "coordinator_pod_pattern": coordinator_pod_pattern,
      "tpu_stderr_log_level_configured": stderr_raw is not None,
      "effective_tpu_stderr_log_level": effective_stderr,
      "effective_tpu_min_log_level": effective_min,
      "megascale_digest_visible": digest_visible,
      "observability_gap": observability_gap,
      "coordinator_branch": coordinator_branch,
      "worker_stack_signatures": worker_stack_signatures,
      "known_bug_signature": known_bug_signature,
      "known_bug_signature_hits": sig_hits,
      "known_bug_attribution_strength": attribution_strength,
      "recommended_flags": {
          "TPU_STDERR_LOG_LEVEL": "2",
          "TPU_MIN_LOG_LEVEL": "0",
      },
  }


def classify_runs_and_timeline(
    workload_prefix: str,
    audit_entries: list[dict[str, Any]],
    event_entries: list[dict[str, Any]],
    suspect_pools: list[str] | None = None,
) -> dict[str, Any]:
  """Combines audit specs, Scheduled events, and TAS/JobSet events per run."""
  recovered = recover_specs_from_audit_entries(audit_entries, workload_prefix)
  runs = recovered["runs"]
  sched = group_scheduled_events_by_pool(
      event_entries,
      workload_prefix,
      suspect_pools=suspect_pools,
      runs_spec=runs,
  )
  tas = decode_tas_messages(event_entries)

  # Map failure/pending events to runs
  sorted_events = sorted(event_entries, key=lambda e: e.get("timestamp", ""))
  run_records: list[dict[str, Any]] = []
  started_count = 0
  unscheduled_count = 0

  for run in runs:
    rid = run["run_id"]
    jname = run["jobset_name"]
    pools_map = sched["pools_by_run"].get(rid, {})
    suspect_map = sched["suspect_pool_hits"].get(rid, {})
    placed_pods = sum(pools_map.values())

    # Determine outcome
    if placed_pods > 0:
      started_count += 1
      outcome = "PLACED_THEN_CRASHED"
    else:
      unscheduled_count += 1
      outcome = "NEVER_ADMITTED"

    # Find matching TAS message for this run if unscheduled
    tas_reason = ""
    for ev in sorted_events:
      payload = ev.get("jsonPayload") or ev
      msg = payload.get("message", "")
      obj = payload.get("involvedObject") or {}
      obj_name = obj.get("name", "")
      if jname in obj_name or jname in msg:
        if "allows to fit only" in msg:
          tas_reason = msg
        elif "Aborting the coordinator" in msg or "BackoffLimitExceeded" in msg:
          outcome = "PLACED_THEN_CRASHED"

    run_records.append({
        **run,
        "placed_pods": placed_pods,
        "placed_pools": pools_map,
        "suspect_pool_pods": suspect_map,
        "outcome": outcome,
        "tas_message": tas_reason,
    })

  if not run_records:
    overall_state = "WORKLOAD_NOT_FOUND"
  elif started_count > 0 and unscheduled_count > 0:
    overall_state = "MIXED_UNSCHEDULED_AND_CRASHED"
  elif started_count > 0:
    overall_state = "PLACED_THEN_CRASHED"
  else:
    overall_state = "NEVER_ADMITTED"

  return {
      "workload_state": overall_state,
      "total_runs": len(run_records),
      "runs_classified": len(run_records),
      "started_runs": started_count,
      "admitted_runs": started_count,
      "unscheduled_runs": unscheduled_count,
      "pending_runs": unscheduled_count,
      "run_names": [run["run_id"] for run in run_records],
      "runs": run_records,
      "tas_summary": {k: v for k, v in tas.items() if k != "history"},
  }


def _load_or_query_audit(args: argparse.Namespace) -> list[dict[str, Any]]:
  """Loads audit entries from `--audit-file` or queries Cloud Audit Logs."""
  arg_dict = vars(args)
  audit_file = arg_dict.get("audit_file", "")
  if audit_file and os.path.exists(audit_file):
    return _load_json_file(audit_file)
  time_clauses = []
  if arg_dict.get("start", ""):
    time_clauses.append(f'timestamp>="{args.start}"')
  if arg_dict.get("end", ""):
    time_clauses.append(f'timestamp<="{args.end}"')
  ns_clause = (
      f' AND protoPayload.resourceName:"namespaces/{args.namespace}/"'
      if arg_dict.get("namespace", "")
      else ""
  )
  t_str = (" AND " + " AND ".join(time_clauses)) if time_clauses else ""
  q = (
      'logName:"cloudaudit.googleapis.com%2Factivity" AND '
      'protoPayload.methodName="io.k8s.batch.v1.jobs.create" AND '
      f'protoPayload.resourceName:"{args.workload_prefix}"'
      f"{ns_clause}{t_str}"
  )
  return _fetch_logging_json(args.project_id, q, limit=500, order="asc")


def _load_or_query_events(args: argparse.Namespace) -> list[dict[str, Any]]:
  """Loads Kubernetes event entries from `--events-file` or Cloud Logging."""
  arg_dict = vars(args)
  events_file = arg_dict.get("events_file", "")
  if events_file and os.path.exists(events_file):
    return _load_json_file(events_file)
  time_clauses = []
  if arg_dict.get("start", ""):
    time_clauses.append(f'timestamp>="{args.start}"')
  if arg_dict.get("end", ""):
    time_clauses.append(f'timestamp<="{args.end}"')
  ns_clause = (
      f' AND jsonPayload.involvedObject.namespace="{args.namespace}"'
      if arg_dict.get("namespace", "")
      else ""
  )
  t_str = (" AND " + " AND ".join(time_clauses)) if time_clauses else ""
  q = (
      'resource.type="k8s_cluster" AND '
      f'jsonPayload.involvedObject.name:"{args.workload_prefix}"'
      f"{ns_clause}{t_str}"
  )
  return _fetch_logging_json(args.project_id, q, limit=2000, order="asc")


def _add_common_args(sub: argparse.ArgumentParser) -> None:
  """Adds common CLI flags to a subcommand parser."""
  sub.add_argument("--workload_prefix", required=True)
  sub.add_argument("--project_id", required=True)
  sub.add_argument("--cluster_name", default="")
  sub.add_argument("--namespace", default="default")
  sub.add_argument("--start", default="")
  sub.add_argument("--end", default="")
  sub.add_argument("--audit_file", default="")
  sub.add_argument("--events_file", default="")
  sub.add_argument("--output_file", default="")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Read-only multi-run forensics tool for GKE TPU workloads."
  )
  subparsers = parser.add_subparsers(dest="subcommand", required=True)

  # 0. classify-workload-state
  p_cls = subparsers.add_parser("classify-workload-state")
  _add_common_args(p_cls)

  # 1. list-jobset-runs
  p_list = subparsers.add_parser("list-jobset-runs")
  _add_common_args(p_list)

  # 2. recover-jobset-spec
  p_spec = subparsers.add_parser("recover-jobset-spec")
  _add_common_args(p_spec)

  # 3. scheduled-events-by-pool
  p_sched = subparsers.add_parser("scheduled-events-by-pool")
  _add_common_args(p_sched)
  p_sched.add_argument(
      "--suspect_pools",
      default="",
      help="Comma-separated list of suspect node pools (e.g. np-a,np-b).",
  )

  # 4. decode-tas-message
  p_tas = subparsers.add_parser("decode-tas-message")
  _add_common_args(p_tas)
  p_tas.add_argument("--nodes_per_cube", type=int, default=NODES_PER_TPU7X_CUBE)

  # 5 & 6. cube-ledger
  p_ledger = subparsers.add_parser("cube-ledger")
  _add_common_args(p_ledger)
  p_ledger.add_argument("--nodes_file", default="")
  p_ledger.add_argument("--holders_file", default="")
  p_ledger.add_argument(
      "--excluded_pools",
      default="",
      help="Comma-separated pools excluded by the run.",
  )
  p_ledger.add_argument(
      "--known_bad_pools",
      default="",
      help="Comma-separated known-bad pools (e.g. np-a,np-b).",
  )
  p_ledger.add_argument("--needed_cubes", type=int, default=128)
  p_ledger.add_argument(
      "--pins_healthy_only",
      default="false",
      help="Whether the run pins 4x4x4-state=HEALTHY only (true/false).",
  )

  # 8 & 10. check-failure-observability
  p_obs = subparsers.add_parser("check-failure-observability")
  _add_common_args(p_obs)
  p_obs.add_argument("--logs_file", default="")
  p_obs.add_argument("--known_bug_signature", default="")

  args = parser.parse_args()

  if args.subcommand in ("classify-workload-state", "list-jobset-runs"):
    audit_entries = _load_or_query_audit(args)
    event_entries = _load_or_query_events(args)
    res = classify_runs_and_timeline(
        args.workload_prefix, audit_entries, event_entries
    )
    if args.subcommand == "classify-workload-state":
      fields = [
          "workload_state",
          "runs_classified",
          "admitted_runs",
          "pending_runs",
          "output_file",
      ]
      out = args.output_file or "scratch/workload_state.json"
    else:
      fields = [
          "total_runs",
          "started_runs",
          "unscheduled_runs",
          "run_names",
          "output_file",
      ]
      out = args.output_file or "scratch/jobset_runs.json"
    _write_output_and_summary(res, fields, out)

  elif args.subcommand == "recover-jobset-spec":
    audit_entries = _load_or_query_audit(args)
    res = recover_specs_from_audit_entries(audit_entries, args.workload_prefix)
    fields = [
        "runs_recovered",
        "exclusion_counts_by_run",
        "exclusion_list_drift",
        "pins_healthy_only",
        "output_file",
    ]
    out = args.output_file or "scratch/recovered_jobset_specs.json"
    _write_output_and_summary(res, fields, out)

  elif args.subcommand == "scheduled-events-by-pool":
    audit_entries = (
        _load_or_query_audit(args) if vars(args).get("audit_file", "") else []
    )
    runs_spec = None
    if audit_entries:
      runs_spec = recover_specs_from_audit_entries(
          audit_entries, args.workload_prefix
      )["runs"]
    event_entries = _load_or_query_events(args)
    suspect = [
        pool.strip() for pool in args.suspect_pools.split(",") if pool.strip()
    ]
    res = group_scheduled_events_by_pool(
        event_entries,
        args.workload_prefix,
        cluster_name=args.cluster_name,
        suspect_pools=suspect,
        runs_spec=runs_spec,
    )
    fields = [
        "runs_with_scheduled_pods",
        "pools_by_run",
        "suspect_pool_hits",
        "output_file",
    ]
    out = args.output_file or "scratch/scheduled_events_by_pool.json"
    _write_output_and_summary(res, fields, out)

  elif args.subcommand == "decode-tas-message":
    event_entries = _load_or_query_events(args)
    res = decode_tas_messages(event_entries, nodes_per_cube=args.nodes_per_cube)
    fields = [
        "fit_slices",
        "needed_slices",
        "excluded_affinity_cubes",
        "excluded_resource_cubes",
        "excluded_taint_nodes",
        "two_pass_quota_masking",
        "output_file",
    ]
    out = args.output_file or "scratch/tas_decoded.json"
    _write_output_and_summary(res, fields, out)

  elif args.subcommand == "cube-ledger":
    raw_nodes: Any = {}
    if args.nodes_file and os.path.exists(args.nodes_file):
      raw_nodes = _load_json_file(args.nodes_file)
      nodes_list = (
          raw_nodes.get("items", [])
          if isinstance(raw_nodes, dict)
          else raw_nodes
      )
    else:
      kubectl_out = _run_readonly_cmd(["kubectl", "get", "nodes", "-o", "json"])
      raw_nodes = json.loads(kubectl_out) if kubectl_out.strip() else {}
      nodes_list = (
          raw_nodes.get("items", [])
          if isinstance(raw_nodes, dict)
          else raw_nodes
      )

    holders_list: list[dict[str, Any]] = []
    if args.holders_file and os.path.exists(args.holders_file):
      holders_list = _load_json_file(args.holders_file)
    elif isinstance(raw_nodes, dict) and isinstance(
        raw_nodes.get("holders"), list
    ):
      holders_list = raw_nodes["holders"]

    excl = [
        pool.strip() for pool in args.excluded_pools.split(",") if pool.strip()
    ]
    kbad = [
        pool.strip() for pool in args.known_bad_pools.split(",") if pool.strip()
    ]
    if not excl or not kbad:
      rec_runs: list[dict[str, Any]] = []
      try:
        audit_entries = _load_or_query_audit(args)
        rec = recover_specs_from_audit_entries(
            audit_entries, args.workload_prefix
        )
        rec_runs = rec.get("runs", [])
      except (OSError, RuntimeError, ValueError, KeyError):
        rec_runs = []
      if rec_runs:
        excl_lists: list[list[str]] = [
            list(run.get("excluded_pools", [])) for run in rec_runs
        ]
        max_excl: list[str] = max(excl_lists, key=len)
        min_excl: list[str] = min(excl_lists, key=len)
        if not excl and max_excl:
          excl = list(max_excl)
        if not kbad and len(max_excl) > len(min_excl):
          min_set = set(min_excl)
          kbad = [pool for pool in max_excl if pool not in min_set]
    pins_healthy = str(args.pins_healthy_only).lower() in ("true", "1", "yes")

    res = compute_cube_ledger(
        nodes_data=nodes_list,
        excluded_pools=excl,
        needed_cubes=args.needed_cubes,
        pins_healthy_only=pins_healthy,
        known_bad_pools=kbad,
        held_cubes_records=holders_list,
    )
    fields = [
        "usable_cubes",
        "self_excluded_cubes",
        "held_by_other_queues_cubes",
        "free_cubes_for_run",
        "needed_cubes",
        "formula",
        "invisible_known_bad_pools",
        "output_file",
    ]
    out = args.output_file or "scratch/cube_ledger.json"
    _write_output_and_summary(res, fields, out)

  elif args.subcommand == "check-failure-observability":
    audit_entries = _load_or_query_audit(args)
    recovered = recover_specs_from_audit_entries(
        audit_entries, args.workload_prefix
    )
    spec_data = recovered["runs"][-1] if recovered["runs"] else {}

    log_lines: list[str] = []
    if args.logs_file and os.path.exists(args.logs_file):
      with open(args.logs_file, "r", encoding="utf-8") as fh:
        log_lines = fh.read().splitlines()
    else:
      try:
        entries = _fetch_logging_json(
            args.project_id,
            'resource.type="k8s_container" AND '
            f'resource.labels.pod_name:"{args.workload_prefix}"',
            limit=200,
            order="desc",
        )
        for entry in entries:
          txt = entry.get("textPayload") or json.dumps(
              entry.get("jsonPayload") or {}
          )
          if txt:
            log_lines.append(txt)
      except RuntimeError:
        log_lines = []

    res = evaluate_failure_observability(
        spec_data=spec_data,
        log_lines=log_lines,
        known_bug_signature=args.known_bug_signature,
    )
    fields = [
        "framework",
        "effective_tpu_stderr_log_level",
        "megascale_digest_visible",
        "coordinator_branch",
        "known_bug_attribution_strength",
        "recommended_flags",
        "output_file",
    ]
    out = args.output_file or "scratch/failure_observability.json"
    _write_output_and_summary(res, fields, out)


if __name__ == "__main__":
  main()

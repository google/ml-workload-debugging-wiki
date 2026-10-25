#!/usr/bin/env python3
"""Deterministic Kueue failure recovery and TPU environment patching helper."""

import argparse
import json
import re
import subprocess
import sys
from typing import Any


def select_safe_termination_annotation(kueue_version: str) -> str:
  """Selects the Kueue safe-termination annotation key for the given version.

  Kueue KEP-6757 renamed `kueue.x-k8s.io/safe-to-forcefully-terminate` to
  `kueue.x-k8s.io/safe-to-forcefully-delete` in:
  - >= 0.17.0
  - 0.16.x >= 0.16.3
  - 0.15.x >= 0.15.6
  Versions 0.15.0-0.15.5 and 0.16.0-0.16.2 use `safe-to-forcefully-terminate`.

  Args:
    kueue_version: Version string (e.g. "v0.16.3" or "0.15.4").

  Returns:
    Annotation key string.
  """
  version_match = re.search(r"(\d+)\.(\d+)\.(\d+)", kueue_version or "")
  if not version_match:
    return "kueue.x-k8s.io/safe-to-forcefully-delete"
  major, minor, patch = (
      int(version_match.group(1)),
      int(version_match.group(2)),
      int(version_match.group(3)),
  )
  if (major, minor) == (0, 15) and patch <= 5:
    return "kueue.x-k8s.io/safe-to-forcefully-terminate"
  if (major, minor) == (0, 16) and patch <= 2:
    return "kueue.x-k8s.io/safe-to-forcefully-terminate"
  if (major, minor) < (0, 15):
    return "kueue.x-k8s.io/safe-to-forcefully-terminate"
  return "kueue.x-k8s.io/safe-to-forcefully-delete"


def _run_kubectl(args: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      ["kubectl"] + args,
      capture_output=True,
      text=True,
      check=False,
  )


def annotate_pods(
    workload_name: str,
    workload_type: str,
    namespace: str,
    kueue_version: str,
) -> dict[str, Any]:
  """Annotates a workload's pod templates with the version-aware Kueue key."""
  active_namespace = namespace or "default"
  workload_type_lower = (workload_type or "jobset").lower()
  annotation_key = select_safe_termination_annotation(kueue_version)
  annotation_applied = f'{annotation_key}: "true"'

  get_res = _run_kubectl([
      "get",
      workload_type_lower,
      workload_name,
      "-n",
      active_namespace,
      "-o",
      "json",
  ])
  if get_res.returncode != 0:
    return {
        "execution_status": f"ERROR: {get_res.stderr.strip()}",
        "annotation_applied": annotation_applied,
        "patched_objects": [],
    }

  try:
    obj = json.loads(get_res.stdout)
  except json.JSONDecodeError as exc:
    return {
        "execution_status": f"ERROR: Invalid JSON from kubectl: {exc}",
        "annotation_applied": annotation_applied,
        "patched_objects": [],
    }

  patched_objects: list[str] = []
  spec = obj.get("spec", {})
  if "replicatedJobs" in spec and isinstance(spec["replicatedJobs"], list):
    replicated_jobs = []
    for replicated_job in spec["replicatedJobs"]:
      replicated_job_copy = dict(replicated_job)
      tmpl = (
          replicated_job_copy.setdefault("template", {})
          .setdefault("spec", {})
          .setdefault("template", {})
      )
      meta = tmpl.setdefault("metadata", {})
      annotations = meta.setdefault("annotations", {})
      annotations[annotation_key] = "true"
      replicated_jobs.append(replicated_job_copy)
      rj_name = replicated_job_copy.get("name", "replicatedJob")
      patched_objects.append(f"{workload_type_lower}/{workload_name}:{rj_name}")
    patch_payload = {"spec": {"replicatedJobs": replicated_jobs}}
  else:
    patch_payload = {
        "spec": {
            "template": {"metadata": {"annotations": {annotation_key: "true"}}}
        }
    }
    patched_objects.append(f"{workload_type_lower}/{workload_name}")

  patch_res = _run_kubectl([
      "patch",
      workload_type_lower,
      workload_name,
      "-n",
      active_namespace,
      "--type=merge",
      "-p",
      json.dumps(patch_payload),
  ])
  if patch_res.returncode != 0:
    return {
        "execution_status": f"ERROR: {patch_res.stderr.strip()}",
        "annotation_applied": annotation_applied,
        "patched_objects": [],
    }

  return {
      "execution_status": "SUCCESS",
      "annotation_applied": annotation_applied,
      "patched_objects": patched_objects,
  }


def enable_feature_gate(
    cluster_name: str,
    location: str,
    kueue_version: str,
) -> dict[str, Any]:
  """Patches kueue-controller-manager to enable FailureRecoveryPolicy=true."""
  del cluster_name, location, kueue_version
  get_res = _run_kubectl([
      "get",
      "deployment",
      "kueue-controller-manager",
      "-n",
      "kueue-system",
      "-o",
      "json",
  ])
  if get_res.returncode != 0:
    return {
        "execution_status": f"ERROR: {get_res.stderr.strip()}",
        "affected_resources": [],
    }

  try:
    dep = json.loads(get_res.stdout)
  except json.JSONDecodeError as exc:
    return {
        "execution_status": f"ERROR: Invalid JSON from kubectl: {exc}",
        "affected_resources": [],
    }

  containers = (
      dep.get("spec", {})
      .get("template", {})
      .get("spec", {})
      .get("containers", [])
  )
  updated_containers = []
  for container in containers:
    container_copy = dict(container)
    args = list(container_copy.get("args", []))
    fg_updated = False
    for idx, arg in enumerate(args):
      if arg.startswith("--feature-gates="):
        if "FailureRecoveryPolicy=" in arg:
          args[idx] = re.sub(
              r"FailureRecoveryPolicy=(true|false)",
              "FailureRecoveryPolicy=true",
              arg,
          )
        else:
          args[idx] = f"{arg},FailureRecoveryPolicy=true"
        fg_updated = True
        break
    if not fg_updated and container_copy.get("name") == "manager":
      args.append("--feature-gates=FailureRecoveryPolicy=true")
    container_copy["args"] = args
    updated_containers.append(container_copy)

  patch_payload = {
      "spec": {"template": {"spec": {"containers": updated_containers}}}
  }
  patch_res = _run_kubectl([
      "patch",
      "deployment",
      "kueue-controller-manager",
      "-n",
      "kueue-system",
      "--type=merge",
      "-p",
      json.dumps(patch_payload),
  ])
  if patch_res.returncode != 0:
    return {
        "execution_status": f"ERROR: {patch_res.stderr.strip()}",
        "affected_resources": [],
    }

  return {
      "execution_status": "SUCCESS",
      "affected_resources": [
          "deployment/kueue-controller-manager (kueue-system)"
      ],
  }


def _inject_env_vars(
    containers: list[dict[str, Any]],
    env_map: dict[str, str],
) -> list[dict[str, Any]]:
  """Injects or updates TPU environment variables in container specs."""
  has_tpu_container = any(
      "google.com/tpu" in container.get("resources", {}).get("limits", {})
      or "google.com/tpu" in container.get("resources", {}).get("requests", {})
      for container in containers
  )
  updated = []
  for container in containers:
    container_copy = dict(container)
    limits = container_copy.get("resources", {}).get("limits", {})
    requests = container_copy.get("resources", {}).get("requests", {})
    if (
        not has_tpu_container
        or "google.com/tpu" in limits
        or "google.com/tpu" in requests
    ):
      env_list = [
          dict(env_entry) for env_entry in container_copy.get("env", [])
      ]
      existing_names = {
          env_entry.get("name"): i for i, env_entry in enumerate(env_list)
      }
      for env_key, env_val in env_map.items():
        if not env_val:
          continue
        existing_idx = existing_names.get(env_key)
        if existing_idx is not None:
          env_list[existing_idx]["value"] = env_val
        else:
          env_list.append({"name": env_key, "value": env_val})
      container_copy["env"] = env_list
    updated.append(container_copy)
  return updated


def patch_tpu_env(
    workload_name: str,
    workload_type: str,
    namespace: str,
    tpu_accelerator_type: str,
    tpu_chips_per_host_bounds: str,
    tpu_host_bounds: str,
) -> dict[str, Any]:
  """Patches TPU topology environment variables into workload pod templates."""
  active_namespace = namespace or "default"
  workload_type_lower = (workload_type or "jobset").lower()
  env_map = {
      "TPU_ACCELERATOR_TYPE": tpu_accelerator_type,
      "TPU_CHIPS_PER_HOST_BOUNDS": tpu_chips_per_host_bounds,
      "TPU_HOST_BOUNDS": tpu_host_bounds,
  }

  get_res = _run_kubectl([
      "get",
      workload_type_lower,
      workload_name,
      "-n",
      active_namespace,
      "-o",
      "json",
  ])
  if get_res.returncode != 0:
    return {
        "execution_status": f"ERROR: {get_res.stderr.strip()}",
        "patched_objects": [],
    }

  try:
    obj = json.loads(get_res.stdout)
  except json.JSONDecodeError as exc:
    return {
        "execution_status": f"ERROR: Invalid JSON from kubectl: {exc}",
        "patched_objects": [],
    }

  patched_objects: list[str] = []
  spec = obj.get("spec", {})
  if "replicatedJobs" in spec and isinstance(spec["replicatedJobs"], list):
    replicated_jobs = []
    for replicated_job in spec["replicatedJobs"]:
      replicated_job_copy = dict(replicated_job)
      pod_spec = (
          replicated_job_copy.setdefault("template", {})
          .setdefault("spec", {})
          .setdefault("template", {})
          .setdefault("spec", {})
      )
      pod_spec["containers"] = _inject_env_vars(
          pod_spec.get("containers", []), env_map
      )
      replicated_jobs.append(replicated_job_copy)
      rj_name = replicated_job_copy.get("name", "replicatedJob")
      patched_objects.append(f"{workload_type_lower}/{workload_name}:{rj_name}")
    patch_payload = {"spec": {"replicatedJobs": replicated_jobs}}
  else:
    pod_spec = spec.get("template", {}).get("spec", {})
    updated_containers = _inject_env_vars(
        pod_spec.get("containers", []), env_map
    )
    patch_payload = {
        "spec": {"template": {"spec": {"containers": updated_containers}}}
    }
    patched_objects.append(f"{workload_type_lower}/{workload_name}")

  patch_res = _run_kubectl([
      "patch",
      workload_type_lower,
      workload_name,
      "-n",
      active_namespace,
      "--type=merge",
      "-p",
      json.dumps(patch_payload),
  ])
  if patch_res.returncode != 0:
    return {
        "execution_status": f"ERROR: {patch_res.stderr.strip()}",
        "patched_objects": [],
    }

  return {
      "execution_status": "SUCCESS",
      "patched_objects": patched_objects,
  }


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Kueue failure recovery and TPU env workaround helper."
  )
  parser.add_argument(
      "subcommand",
      nargs="?",
      choices=["annotate-pods", "enable-feature-gate", "patch-tpu-env"],
      default="annotate-pods",
      help="Action subcommand to execute.",
  )
  parser.add_argument("--workload_name", default="", help="Workload name.")
  parser.add_argument(
      "--workload_type", default="jobset", help="Workload type."
  )
  parser.add_argument(
      "--namespace", default="default", help="Kubernetes namespace."
  )
  parser.add_argument(
      "--kueue_version", default="", help="Kueue version string."
  )
  parser.add_argument("--cluster_name", default="", help="GKE cluster name.")
  parser.add_argument("--location", default="", help="GKE cluster location.")
  parser.add_argument(
      "--tpu_accelerator_type", default="", help="TPU_ACCELERATOR_TYPE value."
  )
  parser.add_argument(
      "--tpu_chips_per_host_bounds",
      default="",
      help="TPU_CHIPS_PER_HOST_BOUNDS value.",
  )
  parser.add_argument(
      "--tpu_host_bounds", default="", help="TPU_HOST_BOUNDS value."
  )
  args = parser.parse_args()

  if args.subcommand == "annotate-pods":
    res = annotate_pods(
        workload_name=args.workload_name,
        workload_type=args.workload_type,
        namespace=args.namespace,
        kueue_version=args.kueue_version,
    )
  elif args.subcommand == "enable-feature-gate":
    res = enable_feature_gate(
        cluster_name=args.cluster_name,
        location=args.location,
        kueue_version=args.kueue_version,
    )
  else:
    res = patch_tpu_env(
        workload_name=args.workload_name,
        workload_type=args.workload_type,
        namespace=args.namespace,
        tpu_accelerator_type=args.tpu_accelerator_type,
        tpu_chips_per_host_bounds=args.tpu_chips_per_host_bounds,
        tpu_host_bounds=args.tpu_host_bounds,
    )

  print(json.dumps(res, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())

#!/usr/bin/env python3
"""Resolves workload infrastructure specifications.

Extracts machine_type, accelerator_type, chips_per_host, topology, and node_pool
from kubectl node metadata, gcloud compute instances describe, gcloud container
node-pools describe, or log telemetry / node pool naming conventions.
"""

import argparse
import json
import os
import re
import subprocess
from typing import Any

_CHIPS_PER_HOST_PATTERN = re.compile(r"-(\d+)t(?:-tpu)?$")
_AUDIT_MACHINE_TYPE_PATTERN = re.compile(
    r'(?:protoPayload\.request\.machineType|machineType)["\':\s]+'
    r"([a-zA-Z0-9_\-\.\/]+)"
)
_JAX_HBM_PATTERN = re.compile(
    r"Total HBM memory:\s*([\d\.]+)\s*(?:GB|GiB|B)", re.IGNORECASE
)
_TOPOLOGY_PATTERN = re.compile(
    r"(?:topology|topology:)\s*([0-9]+x[0-9]+(?:x[0-9]+)?)", re.IGNORECASE
)


def run_cmd(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
  """Runs a shell command safely, returning (returncode, stdout, stderr)."""
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr
  except (subprocess.SubprocessError, OSError) as e:
    return -1, "", str(e)


def parse_kubectl_node(node_json: dict[str, Any]) -> dict[str, Any]:
  """Extracts infra specs from a kubectl node JSON object."""
  metadata = node_json.get("metadata", {})
  labels = metadata.get("labels", {})

  machine_type = labels.get("node.kubernetes.io/instance-type", "")
  accelerator_type = labels.get("cloud.google.com/gke-tpu-accelerator", "")
  topology = labels.get("cloud.google.com/gke-tpu-topology", "")
  node_pool = labels.get("cloud.google.com/gke-nodepool", "")

  chips_per_host = 0
  if machine_type:
    match = _CHIPS_PER_HOST_PATTERN.search(machine_type)
    if match:
      chips_per_host = int(match.group(1))

  return {
      "machine_type": machine_type,
      "accelerator_type": accelerator_type,
      "chips_per_host": chips_per_host,
      "topology": topology,
      "node_pool": node_pool,
  }


def parse_gcloud_compute_instance(raw_output: str) -> dict[str, Any]:
  """Parses output from gcloud compute instances describe machineType."""
  raw = raw_output.strip()
  if not raw:
    return {}
  # raw can be a full URL or just the machineType name.
  machine_type = raw.split("/")[-1].strip()
  chips_per_host = 0
  match = _CHIPS_PER_HOST_PATTERN.search(machine_type)
  if match:
    chips_per_host = int(match.group(1))

  # Infer accelerator type if possible
  accelerator_type = ""
  if "tpu7x" in machine_type or "tpu7x" in machine_type:
    accelerator_type = "tpu-v6e-slice"
  elif "v6e" in machine_type or "ct6e" in machine_type:
    accelerator_type = "tpu-v6e-slice"
  elif "v5p" in machine_type or "ct5p" in machine_type:
    accelerator_type = "tpu-v5p-slice"
  elif (
      "v5e" in machine_type
      or "ct5e" in machine_type
  ):
    accelerator_type = "tpu-v5-lite-podslice"

  return {
      "machine_type": machine_type,
      "accelerator_type": accelerator_type,
      "chips_per_host": chips_per_host,
  }


def parse_gcloud_nodepool(nodepool_json: dict[str, Any]) -> dict[str, Any]:
  """Parses output from gcloud container node-pools describe JSON."""
  config = (
      nodepool_json.get("config", {})
      if "config" in nodepool_json
      else nodepool_json
  )
  machine_type = config.get("machineType", "")
  if "/" in machine_type:
    machine_type = machine_type.split("/")[-1]

  accelerators = config.get("accelerators", [])
  accelerator_type = ""
  if accelerators and isinstance(accelerators, list):
    accelerator_type = accelerators[0].get("acceleratorType", "")

  chips_per_host = 0
  if machine_type:
    match = _CHIPS_PER_HOST_PATTERN.search(machine_type)
    if match:
      chips_per_host = int(match.group(1))

  if not accelerator_type and machine_type:
    if (
        "tpu7x" in machine_type
        or "tpu7x" in machine_type
        or "v6e" in machine_type
        or "ct6e" in machine_type
    ):
      accelerator_type = "tpu-v6e-slice"
    elif "v5p" in machine_type or "ct5p" in machine_type:
      accelerator_type = "tpu-v5p-slice"
    elif (
        "v5e" in machine_type
        or "ct5e" in machine_type
    ):
      accelerator_type = "tpu-v5-lite-podslice"

  return {
      "machine_type": machine_type,
      "accelerator_type": accelerator_type,
      "chips_per_host": chips_per_host,
  }


def parse_log_fallback(
    node_pool_name: str = "",
    log_content: str = "",
) -> dict[str, Any]:
  """Extracts infra specs using naming conventions and log patterns."""
  machine_type = ""
  accelerator_type = ""
  chips_per_host = 0
  topology = ""

  combined_text = f"{node_pool_name}\n{log_content}"

  # 1. GCE audit logs: protoPayload.request.machineType
  audit_match = _AUDIT_MACHINE_TYPE_PATTERN.search(combined_text)
  if audit_match:
    val = audit_match.group(1).split("/")[-1]
    if "standard" in val or "tpu" in val or "highgpu" in val:
      machine_type = val

  # 2. JAX HBM allocation: Total HBM memory: 65.5GB or similar
  if not machine_type:
    hbm_match = _JAX_HBM_PATTERN.search(combined_text)
    if hbm_match:
      hbm_gb = float(hbm_match.group(1))
      # 32GB per chip on v6e/v5p, 16GB on v5e, 65.5GB or 64GB on tpu7x/v6e
      if hbm_gb > 60:
        machine_type = "tpu7x-standard-4t"
        accelerator_type = "tpu-v6e-slice"
        chips_per_host = 4

  # 3. Node pool naming convention: tpu7x-... -> tpu7x-standard-4t
  if not machine_type and node_pool_name:
    np_lower = node_pool_name.lower()
    if "tpu-7x" in np_lower or "tpu7x" in np_lower or "tpu7x" in np_lower:
      machine_type = "tpu7x-standard-4t"
      accelerator_type = "tpu-v6e-slice"
      chips_per_host = 4
    elif "v6e" in np_lower or "ct6e" in np_lower:
      machine_type = "ct6e-standard-4t"
      accelerator_type = "tpu-v6e-slice"
      chips_per_host = 4
    elif "v5p" in np_lower or "ct5p" in np_lower:
      machine_type = "ct5p-hightpu-4t"
      accelerator_type = "tpu-v5p-slice"
      chips_per_host = 4
    elif "v5e" in np_lower or "ct5e" in np_lower:
      machine_type = "ct5e-standard-4t"
      accelerator_type = "tpu-v5-lite-podslice"
      chips_per_host = 4

  # Extract topology if mentioned
  topo_match = _TOPOLOGY_PATTERN.search(combined_text)
  if topo_match:
    topology = topo_match.group(1)

  if machine_type and not chips_per_host:
    chips_match = _CHIPS_PER_HOST_PATTERN.search(machine_type)
    if chips_match:
      chips_per_host = int(chips_match.group(1))

  if machine_type and not accelerator_type:
    if (
        "tpu7x" in machine_type
        or "tpu7x" in machine_type
        or "v6e" in machine_type
        or "ct6e" in machine_type
    ):
      accelerator_type = "tpu-v6e-slice"
    elif "v5p" in machine_type or "ct5p" in machine_type:
      accelerator_type = "tpu-v5p-slice"
    elif (
        "v5e" in machine_type
        or "ct5e" in machine_type
    ):
      accelerator_type = "tpu-v5-lite-podslice"

  return {
      "machine_type": machine_type,
      "accelerator_type": accelerator_type,
      "chips_per_host": chips_per_host,
      "topology": topology,
  }


def resolve_workload_infra_specs(
    workload_name: str = "",
    project_id: str = "",
    cluster_name: str = "",
    location: str = "",
    node_name: str = "",
    node_pool: str = "",
    log_content: str = "",
) -> dict[str, Any]:
  """Resolves workload infra specs through multi-tiered non-internal queries."""
  del workload_name  # Unused.
  result: dict[str, Any] = {
      "status": "UNRESOLVED",
      "machine_type": "",
      "accelerator_type": "",
      "chips_per_host": 0,
      "topology": "",
      "node_pool": node_pool,
      "source": "",
  }

  # Strategy a) Try kubectl if node_name is provided
  if node_name:
    rc, stdout, _ = run_cmd(["kubectl", "get", "node", node_name, "-o", "json"])
    if rc == 0 and stdout.strip():
      try:
        node_json = json.loads(stdout)
        parsed = parse_kubectl_node(node_json)
        if parsed.get("machine_type") or parsed.get("accelerator_type"):
          result.update(parsed)
          result["source"] = "kubectl"
          result["status"] = "RESOLVED"
          return result
      except (json.JSONDecodeError, KeyError, TypeError):
        pass

  # Strategy b) Try gcloud compute instances if node_name & project_id provided
  if node_name and project_id:
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        node_name,
        f"--project={project_id}",
    ]
    if location:
      cmd.append(f"--zone={location}")
    cmd.append("--format=value(machineType)")

    rc, stdout, _ = run_cmd(cmd)
    if rc == 0 and stdout.strip():
      parsed = parse_gcloud_compute_instance(stdout)
      if parsed.get("machine_type"):
        result.update(parsed)
        result["source"] = "gcloud_compute"
        result["status"] = "RESOLVED"
        return result

  # Strategy c) Try gcloud container node-pools if node_pool & cluster provided
  if node_pool and cluster_name and project_id:
    cmd = [
        "gcloud",
        "container",
        "node-pools",
        "describe",
        node_pool,
        f"--cluster={cluster_name}",
        f"--project={project_id}",
    ]
    if location:
      cmd.append(f"--location={location}")
    cmd.append("--format=json(config.machineType, config.accelerators)")

    rc, stdout, _ = run_cmd(cmd)
    if rc == 0 and stdout.strip():
      try:
        nodepool_json = json.loads(stdout)
        parsed = parse_gcloud_nodepool(nodepool_json)
        if parsed.get("machine_type") or parsed.get("accelerator_type"):
          result.update(parsed)
          result["source"] = "gcloud_nodepool"
          result["status"] = "RESOLVED"
          return result
      except (json.JSONDecodeError, KeyError, TypeError):
        pass

  # Strategy d) Fallback from log telemetry / node pool name
  fallback = parse_log_fallback(
      node_pool_name=node_pool,
      log_content=log_content,
  )
  if fallback.get("machine_type") or fallback.get("accelerator_type"):
    result.update({k: v for k, v in fallback.items() if v})
    result["source"] = "log_telemetry"
    result["status"] = "RESOLVED"
    return result

  return result


def print_summary_report(data: dict[str, Any]) -> None:
  """Prints formatted summary report block adhering to glue runner conventions."""
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"  machine_type: {data.get('machine_type', '')}")
  print(f"  accelerator_type: {data.get('accelerator_type', '')}")
  print(f"  chips_per_host: {data.get('chips_per_host', 0)}")
  print(f"  topology: {data.get('topology', '')}")
  print(f"  source: {data.get('source', '')}")
  print(f"  status: {data.get('status', 'UNRESOLVED')}")
  print("---------------------------------------")


def main() -> None:
  """Runs the CLI to resolve workload infrastructure specifications."""
  parser = argparse.ArgumentParser(
      description="Resolve workload infrastructure specifications."
  )
  parser.add_argument(
      "--workload_name", default="", help="Workload or JobSet name"
  )
  parser.add_argument("--project_id", default="", help="GCP project ID")
  parser.add_argument("--cluster_name", default="", help="GKE cluster name")
  parser.add_argument("--location", default="", help="Zone or region")
  parser.add_argument("--node_name", default="", help="Target GKE node name")
  parser.add_argument("--node_pool", default="", help="GKE node pool name")
  parser.add_argument(
      "--log_content",
      default="",
      help="Log snippet or file path for fallback parsing",
  )
  parser.add_argument(
      "--output_file", default="", help="Optional JSON output file path"
  )

  args = parser.parse_args()

  log_text = args.log_content
  if log_text and os.path.isfile(log_text):
    try:
      with open(log_text, "r", encoding="utf-8") as f:
        log_text = f.read()
    except (OSError, UnicodeDecodeError):
      pass

  result = resolve_workload_infra_specs(
      workload_name=args.workload_name,
      project_id=args.project_id,
      cluster_name=args.cluster_name,
      location=args.location,
      node_name=args.node_name,
      node_pool=args.node_pool,
      log_content=log_text,
  )

  if args.output_file:
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

  print(json.dumps(result, indent=2))
  print_summary_report(result)


if __name__ == "__main__":
  main()

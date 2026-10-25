#!/usr/bin/env python3
"""Resolves GKE cluster metadata including TPU generation and GCE reservations.

Used by the /dashboard workflow to inspect cluster properties prior to
configuring the dashboard daemon.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from typing import Any

try:
  google_auth_exc = importlib.import_module("google.auth.exceptions")
  _GOOGLE_AUTH_EXCEPTIONS = (google_auth_exc.GoogleAuthError,)
except ImportError:
  _GOOGLE_AUTH_EXCEPTIONS = ()
try:
  google_api_core_exc = importlib.import_module("google.api_core.exceptions")
  _GOOGLE_API_EXCEPTIONS = (google_api_core_exc.GoogleAPICallError,)
except ImportError:
  _GOOGLE_API_EXCEPTIONS = ()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)
try:
  cld = importlib.import_module("configure_and_launch_dashboard")
  _DATA_DIR = cld.resolve_data_dir()
except (ImportError, OSError, AttributeError):
  if os.environ.get("DASHBOARD_DATA_DIR"):
    _DATA_DIR = os.environ["DASHBOARD_DATA_DIR"]
  else:
    _DATA_DIR = os.path.join(WIKI_DIR, "scratch", "dashboard")
DEFAULT_CACHE_PATH = os.path.join(_DATA_DIR, "clusters_cache.json")


def detect_generation(machine_type: str, cluster_name: str = "") -> str:
  """Infers TPU generation from machine type or cluster name."""
  combined = f"{machine_type} {cluster_name}".lower()
  if "tpu7x" in combined or "tpu7x" in combined:
    return "tpu7x"
  if "v6e" in combined or "ct6e" in combined:
    return "v6e"
  if "v5p" in combined or "ct5p" in combined:
    return "v5p"
  if "v5e" in combined or "ct5e" in combined:
    return "v5e"
  if "v4" in combined or "ct4" in combined:
    return "v4"
  if any(g in combined for g in ("h100", "a100", "gpu", "h200")):
    return "gpu"
  return "other"


def resolve_cluster_metadata(
    cluster_name: str,
    project_id: str,
    location: str = "us-central1",
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict[str, Any]:
  """Extracts cluster metadata, generation, and reservation."""
  res: dict[str, Any] = {
      "status": "UNKNOWN",
      "cluster_name": cluster_name,
      "project_id": project_id,
      "location": location,
      "generation": "other",
      "reservation": "",
  }

  if not cluster_name:
    res["status"] = "ERROR_MISSING_CLUSTER"
    return res

  # 1. Try local cache
  if os.path.exists(cache_path):
    try:
      with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
      cluster_list = cache.get(project_id, [])
      for c in cluster_list:
        if isinstance(c, dict) and c.get("name") == cluster_name:
          res["status"] = "SUCCESS"
          res["location"] = c.get("location", location)
          res["generation"] = c.get(
              "generation", detect_generation("", cluster_name)
          )
          res["reservation"] = c.get("reservation", "")
          return res
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
      # A corrupt or foreign-shaped cache just falls through to the API.
      pass

  # 2. Try GKE Python Client (ADC)
  try:
    container_v1 = importlib.import_module("google.cloud.container_v1")
    google_auth = importlib.import_module("google.auth")
    google_auth_requests = importlib.import_module(
        "google.auth.transport.requests"
    )

    creds, _ = google_auth.default()
    creds.refresh(google_auth_requests.Request())
    client = container_v1.ClusterManagerClient(credentials=creds)
    cluster = client.get_cluster(
        name=(
            f"projects/{project_id}/locations/{location}"
            f"/clusters/{cluster_name}"
        )
    )

    gen = "other"
    res_name = ""
    for np in getattr(cluster, "node_pools", []):
      cfg = getattr(np, "config", None)
      mt = getattr(cfg, "machine_type", "") if cfg else ""
      if gen == "other":
        detected_gen = detect_generation(mt, cluster_name)
        if detected_gen != "other":
          gen = detected_gen

      aff = getattr(cfg, "reservation_affinity", None) if cfg else None
      if aff and getattr(aff, "values", None):
        for val in aff.values:
          if val and not res_name:
            res_name = val.split("/")[-1] if "/" in val else val

    if gen == "other":
      gen = detect_generation("", cluster_name)

    res["status"] = "SUCCESS"
    res["location"] = getattr(cluster, "location", location)
    res["generation"] = gen
    res["reservation"] = res_name
    return res
  except (
      (ImportError, AttributeError, OSError)
      + _GOOGLE_AUTH_EXCEPTIONS
      + _GOOGLE_API_EXCEPTIONS
  ):
    # Missing client libraries, ADC and API errors all fall back to gcloud.
    pass

  # 3. Fallback: gcloud CLI describe
  try:
    cmd = [
        "gcloud",
        "container",
        "clusters",
        "describe",
        cluster_name,
        f"--project={project_id}",
        f"--location={location}",
        "--format=json(name,location,nodePools)",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=15, check=False
    )
    if proc.returncode == 0 and proc.stdout.strip():
      data = json.loads(proc.stdout)
      gen = "other"
      res_name = ""
      for np in data.get("nodePools", []):
        cfg = np.get("config", {})
        mt = cfg.get("machineType", "")
        if gen == "other":
          detected_gen = detect_generation(mt, cluster_name)
          if detected_gen != "other":
            gen = detected_gen
        aff = cfg.get("reservationAffinity", {})
        for val in aff.get("values", []):
          if val and not res_name:
            res_name = val.split("/")[-1] if "/" in val else val

      if gen == "other":
        gen = detect_generation("", cluster_name)

      res["status"] = "SUCCESS"
      res["location"] = data.get("location", location)
      res["generation"] = gen
      res["reservation"] = res_name
      return res
  except (
      OSError,
      subprocess.SubprocessError,
      json.JSONDecodeError,
      KeyError,
      TypeError,
  ):
    # Any gcloud or parse failure falls back to name-based inference.
    pass

  # 4. Fallback: inference from cluster name
  res["generation"] = detect_generation("", cluster_name)
  res["status"] = "INFERRED"
  return res


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Resolve GKE cluster metadata (TPU generation, reservation, region)."
      )
  )
  parser.add_argument("--cluster", required=True, help="GKE cluster name")
  parser.add_argument("--project", default="", help="GCP project ID")
  parser.add_argument(
      "--location", default="us-central1", help="Cluster region/zone"
  )
  parser.add_argument(
      "--output_file", default="", help="Path to write JSON result"
  )

  args = parser.parse_args()

  proj = args.project
  if not proj:
    try:
      p_res = subprocess.run(
          ["gcloud", "config", "get-value", "project"],
          capture_output=True,
          text=True,
          timeout=5,
          check=False,
      )
      if p_res.returncode == 0 and p_res.stdout.strip() != "(unset)":
        proj = p_res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
      pass

  result = resolve_cluster_metadata(
      cluster_name=args.cluster,
      project_id=proj,
      location=args.location,
  )

  if args.output_file:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Configures and launches the GKE Capacity & Triage Dashboard.

Supports targeting by project ID, cluster name, workload ID, or running
with default active configuration. Automatically provisions monitored clusters
in
clusters_config.json, ensures the local dashboard daemon is active, triggers a
fresh collection scan, and outputs deep-linked dashboard parameters.
"""

import argparse
from concurrent import futures
import datetime
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
  sys.path.insert(0, SCRIPT_DIR)
WIKI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(WIKI_DIR, ".."))
DASHBOARD_DIR = os.path.join(WORKSPACE_ROOT, "_dashboard")
DEFAULT_SCRATCH_DASHBOARD_DIR = os.path.join(
    WIKI_DIR, "scratch", "dashboard"
)


def resolve_data_dir() -> str:
  """Returns the dashboard state/config directory under _wiki/scratch/dashboard."""
  env_dir = os.environ.get("DASHBOARD_DATA_DIR")
  if env_dir:
    return env_dir
  return DEFAULT_SCRATCH_DASHBOARD_DIR


def resolve_log_dir() -> str:
  """Returns the dashboard log directory under _wiki/scratch/dashboard/logs."""
  env_dir = os.environ.get("PORTABLE_LOG_DIR")
  if env_dir:
    return env_dir
  return os.path.join(DEFAULT_SCRATCH_DASHBOARD_DIR, "logs")


STDERR_LOG_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_DATA_DIR = resolve_data_dir()
DEFAULT_CONFIG_PATH = os.environ.get(
    "DASHBOARD_CONFIG_PATH",
    os.path.join(DEFAULT_DATA_DIR, "clusters_config.json"),
)
DEFAULT_CLUSTERS_CACHE_PATH = os.environ.get(
    "DASHBOARD_CACHE_PATH",
    os.path.join(DEFAULT_DATA_DIR, "clusters_cache.json"),
)
SERVER_PY_PATH = os.path.join(DASHBOARD_DIR, "server.py")


def resolve_venv_python() -> str:
  """Picks a usable interpreter for the dashboard.

  A venv checked into the workspace may have been built on a different
  platform, leaving bin/python a dangling symlink. Prefer an explicit
  WIKI_VENV_PYTHON override, then _wiki/.venv, then the per-platform
  _wiki/.venv-<sys.platform> (e.g. .venv-darwin), then the current interpreter.

  Returns:
    Path to an executable Python interpreter.
  """
  override = os.environ.get("WIKI_VENV_PYTHON", "").strip()
  candidates = [override] if override else []
  candidates.append(os.path.join(WIKI_DIR, ".venv", "bin", "python"))
  candidates.append(
      os.path.join(WIKI_DIR, f".venv-{sys.platform}", "bin", "python")
  )
  for cand in candidates:
    if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
      return cand
  return sys.executable


DEFAULT_VENV_PYTHON = resolve_venv_python()


def ensure_environment_ready(
    venv_python: str | None = None,
) -> dict[str, Any]:
  """Ensures required packages, local CLI paths, and GCP auth tokens are prepared."""
  py_exec = venv_python or resolve_venv_python()
  res: dict[str, Any] = {
      "status": "READY",
      "packages_ready": True,
      "kubectl_available": False,
  }

  # 1. Checks if required packages (fastapi, uvicorn) can be imported
  # using venv_python
  req_check = subprocess.run(
      [py_exec, "-c", "import fastapi, uvicorn"],
      capture_output=True,
      text=True,
      check=False,
  )
  if req_check.returncode != 0:
    requirements_path = os.path.join(DASHBOARD_DIR, "requirements.txt")
    install_res = subprocess.run(
        [py_exec, "-m", "pip", "install", "-r", requirements_path],
        check=False,
    )
    if install_res.returncode != 0:
      res["packages_ready"] = False
      res["status"] = "WARNING"

  # 2. Checks if kubectl is available in PATH or _wiki/.venv/bin/ or
  # _dashboard/bin/. If in local bin, ensures PATH is exported.
  wiki_bin = os.path.join(WIKI_DIR, ".venv", "bin")
  dash_bin = os.path.join(DASHBOARD_DIR, "bin")

  current_path = os.environ.get("PATH", "")
  found_in_local = False
  for local_dir in (wiki_bin, dash_bin):
    if os.path.exists(os.path.join(local_dir, "kubectl")):
      path_dirs = current_path.split(os.pathsep)
      if local_dir not in path_dirs:
        current_path = f"{local_dir}{os.pathsep}{current_path}"
        os.environ["PATH"] = current_path
      found_in_local = True

  has_kubectl = (shutil.which("kubectl") is not None) or found_in_local
  res["kubectl_available"] = has_kubectl
  if not has_kubectl:
    res["status"] = "WARNING"

  # 3. Checks active GCP auth token via gcloud auth print-access-token
  # If CAA error or auth error is detected, records an advisory warning in
  # the returned dictionary ("auth_warning": "...") recommending
  # 'gcloud auth login --update-adc' and [DASHBOARD_LINK]
  try:
    auth_res = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    err_text = ((auth_res.stderr or "") + " " + (auth_res.stdout or "")).lower()
    if auth_res.returncode != 0 or "caa" in err_text or "error" in err_text:
      res["auth_warning"] = (
          "Active GCP auth token could not be verified or is blocked by"
          " Context-Aware Access (CAA). Recommend running 'gcloud auth login"
          " --update-adc'"
      )
      res["status"] = "WARNING"
  except (subprocess.SubprocessError, OSError) as e:
    res["auth_warning"] = (
        f"Active GCP auth token check failed: {e}. Recommend running 'gcloud"
        " auth login --update-adc'"
    )
    res["status"] = "WARNING"

  return res


def get_default_hostname() -> str:
  """Resolves the default public/host name for user-facing dashboard URLs."""
  try:
    fqdn = (socket.getfqdn() or "").strip()
    if fqdn:
      # getfqdn() can return an unresolvable name (e.g. localhost:8080 on a
      # laptop off-corp); only hand out a URL that actually resolves.
      socket.getaddrinfo(fqdn, None)
      return fqdn
  except OSError:
    pass
  return "localhost"


def get_active_gcloud_project() -> str | None:
  """Retrieves the active gcloud configured project ID if available."""
  try:
    res = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if res.returncode == 0:
      val = res.stdout.strip()
      if val and val != "(unset)":
        return val
  except (subprocess.SubprocessError, OSError):
    pass
  return None


def load_json_file(path: str) -> Any | None:
  """Safely loads a JSON file."""
  if os.path.exists(path):
    try:
      with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    except (OSError, ValueError, TypeError, KeyError):
      pass
  return None


def save_json_file(path: str, data: Any) -> bool:
  """Safely saves data to a JSON file."""
  try:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2)
    return True
  except (OSError, TypeError, ValueError) as e:
    sys.stderr.write(f"Error writing {path}: {e}\n")
    return False


def find_cluster_in_cache(
    cluster_name: str, cache_path: str = DEFAULT_CLUSTERS_CACHE_PATH
) -> dict[str, Any] | None:
  """Searches clusters_cache.json for a cluster across all projects."""
  cache = load_json_file(cache_path)
  if not isinstance(cache, dict):
    return None

  for project_id, cluster_list in cache.items():
    if isinstance(cluster_list, list):
      for c in cluster_list:
        if isinstance(c, dict) and c.get("name") == cluster_name:
          result = dict(c)
          result["project"] = project_id
          return result
  return None


def get_clusters_for_project(
    project_id: str,
    cache_path: str = DEFAULT_CLUSTERS_CACHE_PATH,
    use_gcloud: bool = True,
) -> list[dict[str, Any]]:
  """Retrieves clusters belonging to a specific GCP project."""
  clusters: list[dict[str, Any]] = []

  # 1. Try clusters_cache.json
  cache = load_json_file(cache_path)
  if isinstance(cache, dict) and project_id in cache:
    cached_list = cache[project_id]
    if isinstance(cached_list, list):
      for c in cached_list:
        if isinstance(c, dict):
          entry = {
              "name": c.get("name"),
              "project": project_id,
              "location": c.get("location", "us-central1"),
              "generation": c.get("generation", "other"),
          }
          if c.get("reservation"):
            entry["reservation"] = c.get("reservation")
          if c.get("is_nap"):
            entry["is_nap"] = True
          clusters.append(entry)
      if clusters:
        return clusters

  # 2. Try gcloud if allowed and cache had no clusters
  if use_gcloud:
    try:
      cmd = [
          "gcloud",
          "container",
          "clusters",
          "list",
          f"--project={project_id}",
          "--format=json(name,location,status)",
      ]
      res = subprocess.run(
          cmd, capture_output=True, text=True, timeout=90, check=False
      )
      if res.returncode == 0 and res.stdout.strip():
        data = json.loads(res.stdout)
        if isinstance(data, list):
          for cluster_item in data:
            cluster_name = cluster_item.get("name")
            cluster_loc = cluster_item.get("location", "us-central1")
            gen = "other"
            name_lower = str(cluster_name or "").lower()
            if "tpu7x" in name_lower or "7x" in name_lower:
              gen = "tpu7x"
            elif "v6e" in name_lower or "ct6e" in name_lower:
              gen = "v6e"
            elif "v5p" in name_lower or "ct5p" in name_lower:
              gen = "v5p"
            elif (
                "v5e" in name_lower
                or "ct5e" in name_lower
                or "ct5lp" in name_lower
                or "v5litepod" in name_lower
            ):
              gen = "v5e"
            elif "v4" in name_lower:
              gen = "v4"

            entry = {
                "name": cluster_name,
                "project": project_id,
                "location": cluster_loc,
                "generation": gen,
            }
            clusters.append(entry)
          if clusters:
            cache = load_json_file(cache_path) or {}
            if isinstance(cache, dict):
              cache[project_id] = clusters
              save_json_file(cache_path, cache)
            try:
              if (
                  DEFAULT_CLUSTERS_CACHE_PATH
                  and os.path.abspath(DEFAULT_CLUSTERS_CACHE_PATH)
                  != os.path.abspath(cache_path)
              ):
                global_cache = load_json_file(DEFAULT_CLUSTERS_CACHE_PATH) or {}
                if isinstance(global_cache, dict):
                  global_cache[project_id] = clusters
                  save_json_file(DEFAULT_CLUSTERS_CACHE_PATH, global_cache)
            except (OSError, ValueError, TypeError):
              pass
    except (
        subprocess.SubprocessError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):
      pass

  return clusters


def resolve_workload_details(
    workload_name: str,
    project_id: str | None = None,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
    lookback_days: int = 30,
    timeout_seconds: float = 25.0,
) -> tuple[str | None, str | None, str | None]:
  """Attempts to resolve cluster, location, and project for a workload ID."""
  try:
    sys.path.insert(0, SCRIPT_DIR)
    validate_workload_mod = importlib.import_module("validate_workload")
    validate_workload = validate_workload_mod.validate_workload

    proj = project_id or get_active_gcloud_project() or ""
    now = datetime.datetime.now(datetime.timezone.utc)
    end = end_timestamp or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    start = start_timestamp or (
        now - datetime.timedelta(days=lookback_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(
          validate_workload,
          workload=workload_name,
          start=start,
          end=end,
          project=proj,
      )
      res = future.result(timeout=timeout_seconds)
      if isinstance(res, dict) and res.get("cluster_name"):
        return res.get("cluster_name"), res.get("location"), proj
  except (
      futures.TimeoutError,
      ImportError,
      AttributeError,
      OSError,
      ValueError,
      KeyError,
      TypeError,
      RuntimeError,
  ):
    pass
  return None, None, project_id


def classify_target(
    target: str,
    cache_path: str = DEFAULT_CLUSTERS_CACHE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> str:
  """Classifies target as 'project', 'cluster', or 'workload'."""
  target = target.strip()
  if not target:
    return "empty"

  # Active gcloud project or all digits (GCP project number)
  active_project = get_active_gcloud_project()
  if (active_project and target == active_project) or target.isdigit():
    return "project"

  # Check if known project in cache
  cache = load_json_file(cache_path)
  if isinstance(cache, dict) and target in cache:
    return "project"

  # Check if known cluster in cache
  cluster_match = find_cluster_in_cache(target, cache_path=cache_path)
  if cluster_match:
    return "cluster"

  # Check if known cluster in current config
  config = load_json_file(config_path)
  config_entries = []
  if isinstance(config, dict):
    config_entries = [
        entry
        for entry in config.get("clusters", [])
        + config.get("monitored_clusters", [])
        if isinstance(entry, dict)
    ]
    for entry in config_entries:
      if entry.get("name") == target:
        return "cluster"

    # Check if known project in current config. Without this, a project ID that
    # is absent from the cache falls through to the "cluster" default below and
    # gets registered as a phantom cluster under the wrong project.
    for entry in config_entries:
      if entry.get("project") == target:
        return "project"

  # Workload signatures
  if (
      target.startswith(("pw-", "jobset-", "train-", "run-"))
      or bool(re.search(r"^v[0-9]+-", target))
      or bool(re.search(r"^p[0-9]+x", target))
      or "-jobset-" in target
      or target.endswith("-jobset")
  ):
    return "workload"

  # GCP project naming structure
  if re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", target):
    if (
        "-test-" in target
        or bool(re.search(r"-[0-9a-z]{4}$", target))
    ):
      return "project"

  # If it looks like a project id (e.g. contains hyphens and dev/prod,
  # or is alphanumeric with dashes)
  if "cluster" in target.lower():
    return "cluster"

  # Fallback assumption: if target contains words typically seen in project IDs
  if any(
      keyword in target
      for keyword in ["project", "dev", "prod", "corp", "cloud"]
  ):
    return "project"

  return "cluster"


def update_clusters_config(
    config_path: str,
    clusters_to_add: list[dict[str, Any]],
    replace_all: bool = False,
) -> list[dict[str, Any]]:
  """Updates clusters_config.json idempotently."""
  current_config = load_json_file(config_path) or {}
  existing_clusters = (
      current_config.get("monitored_clusters")
      or current_config.get("clusters")
      or []
  )
  if not isinstance(existing_clusters, list):
    existing_clusters = []

  if replace_all and clusters_to_add:
    updated_clusters = clusters_to_add
  else:
    updated_clusters = list(existing_clusters)
    for new_cluster in clusters_to_add:
      cluster_name = new_cluster.get("name")
      cluster_proj = new_cluster.get("project")
      existing_index = next(
          (
              index
              for index, existing_entry in enumerate(updated_clusters)
              if existing_entry.get("name") == cluster_name
              and existing_entry.get("project") == cluster_proj
          ),
          -1,
      )
      if existing_index >= 0:
        updated_clusters[existing_index].update(new_cluster)
      else:
        updated_clusters.append(new_cluster)

  new_config = {
      "clusters": updated_clusters,
      "monitored_clusters": updated_clusters,
  }
  save_json_file(config_path, new_config)
  return updated_clusters


def check_server_health(
    port: int, host: str = "127.0.0.1", timeout: float = 1.0
) -> bool:
  """Checks if the dashboard server responds 200 OK at /api/health."""
  url = f"http://{host}:{port}/api/health"
  try:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.status == 200
  except (
      urllib.error.URLError,
      urllib.error.HTTPError,
      OSError,
      TimeoutError,
      ValueError,
  ):
    return False


def get_server_agent_backend(
    port: int, host: str = "127.0.0.1", timeout: float = 2.0
) -> str:
  """The triage backend a running server reports via /api/settings, or ""."""
  url = f"http://{host}:{port}/api/settings"
  try:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
      return str(json.loads(resp.read()).get("agent_backend") or "")
  except (OSError, ValueError, TypeError, KeyError):
    return ""


def is_port_in_use(port: int, timeout: float = 1.0) -> bool:
  """Returns True if something is accepting TCP connections on the port."""
  try:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout):
      return True
  except OSError:
    return False


UNRESPONSIVE_HINT = (
    "The dashboard server accepts connections but does not answer. Known"
    " causes: (1) gRPC fork handlers freezing the whole process on every"
    " gcloud/kubectl spawn -- _dashboard/server.py must set"
    " GRPC_ENABLE_FORK_SUPPORT=0 before anything imports grpc; (2) scan"
    " thread over-subscription starving the HTTP loop of the GIL -- lower"
    " SCAN_CLUSTER_WORKERS. Stop the process holding the port (lsof -ti"
    " tcp:<port>) and rerun the dashboard launch."
)


DOWN_HINT = (
    "Nothing is listening on the dashboard port (connection refused). The"
    " server is not running or has exited; rerun the dashboard launch and"
    " check dashboard.log and server_stderr.log in the log_dir reported by"
    " the launcher if it keeps dying. A host reboot or sleep also kills the"
    " detached server."
)


DEGRADED_HINT = (
    "The dashboard answered the final probe round and most probes overall,"
    " but some probes were slow or timed out. This usually means host load"
    " (check `uptime`), not a hung server; the URL is usable but may be slow."
)


def probe_url(url: str, timeout: float = 5.0) -> tuple[bool, float, str]:
  """GETs a URL; returns (ok, latency_seconds, detail)."""
  start = time.time()
  try:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.status == 200, time.time() - start, f"HTTP {resp.status}"
  except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
    return False, time.time() - start, str(e)


def verify_dashboard_url(
    url: str,
    probes: int = 3,
    interval: float = 2.0,
    timeout: float = 5.0,
) -> dict[str, Any]:
  """Verifies a dashboard URL resolves and stays responsive across probes.

  A single health check is not enough: the server can answer once and then
  freeze as soon as a scan starts spawning subprocesses. Each probe fetches
  both /api/health and the deep-linked page itself on the URL's own host.

  Args:
    url: Deep-linked dashboard URL to verify.
    probes: Number of probe rounds.
    interval: Seconds to sleep between rounds.
    timeout: Per-request timeout in seconds.

  Returns:
    A dict with `url_status` (VERIFIED, DEGRADED, DOWN, UNRESPONSIVE or
    UNRESOLVABLE) plus `url_probes`, and `url_error`/`url_hint` on failure.
  """
  parsed = urllib.parse.urlparse(url)
  host = parsed.hostname or ""
  try:
    socket.getaddrinfo(host, None)
  except (socket.gaierror, OSError) as e:
    return {
        "url_status": "UNRESOLVABLE",
        "url_error": f"Hostname '{host}' does not resolve: {e}",
    }

  health_url = f"{parsed.scheme}://{parsed.netloc}/api/health"
  failures: list[str] = []
  refused = 0
  max_latency = 0.0
  total = 0
  for i in range(max(1, probes)):
    if i:
      time.sleep(interval)
    for target in (health_url, url):
      ok, latency, detail = probe_url(target, timeout=timeout)
      total += 1
      max_latency = max(max_latency, latency)
      if not ok:
        refused += "Connection refused" in detail
        failures.append(
            f"probe {i + 1}: {target} -> {detail} after {latency:.1f}s"
        )

  last_round = f"probe {max(1, probes)}:"
  last_round_ok = not any(f.startswith(last_round) for f in failures)
  if not failures:
    status = "VERIFIED"
  elif refused == len(failures):
    status = "DOWN"
  elif last_round_ok and len(failures) * 3 <= total:
    # Isolated slow probes on a loaded host; the server itself is alive.
    status = "DEGRADED"
  else:
    status = "UNRESPONSIVE"
  res: dict[str, Any] = {
      "url_status": status,
      "url_probes": (
          f"{total - len(failures)}/{total} OK, max latency {max_latency:.2f}s"
      ),
  }
  if failures:
    res["url_error"] = "; ".join(failures[:3])
    res["url_hint"] = {
        "DOWN": DOWN_HINT,
        "DEGRADED": DEGRADED_HINT,
    }.get(status, UNRESPONSIVE_HINT)
  return res


def pinned_agent_backend() -> str:
  """Returns the triage backend the daemon should use: the caller's harness.

  A dashboard launched from Claude Code triages with Claude Code, even on a
  host that also has Antigravity installed. An explicit WIKI_AGENT_BACKEND
  wins; "" leaves the choice to agent_dispatch detection.
  """
  explicit = os.environ.get("WIKI_AGENT_BACKEND", "").strip().lower()
  if explicit:
    return explicit
  try:
    agent_dispatch = importlib.import_module("agent_dispatch")
  except ImportError:
    return ""
  return agent_dispatch.host_harness()


def launch_server(
    port: int,
    host: str = "127.0.0.1",
    python_bin: str | None = None,
    server_py: str | None = None,
    max_wait_seconds: float = 5.0,
) -> str:
  """Spawns server.py as a detached process and waits for readiness."""
  if check_server_health(port=port, host=host):
    return "RUNNING"
  if is_port_in_use(port):
    # A hung server still holds the port; spawning another would just fail
    # to bind and surface as a misleading START_TIMEOUT.
    return "UNRESPONSIVE"

  py_exec = python_bin or resolve_venv_python()
  srv_path = server_py or SERVER_PY_PATH

  if not os.path.exists(srv_path):
    return f"ERROR: server script not found at {srv_path}"

  cmd = [
      py_exec,
      srv_path,
      "--port",
      str(port),
      "--host",
      str(host),
      "--no-browser",
  ]
  log_dir = resolve_log_dir()
  env = dict(
      os.environ,
      PORTABLE_LOG_DIR=log_dir,
      DASHBOARD_DATA_DIR=resolve_data_dir(),
      # Log lines already go to dashboard.log; keep stderr for crashes.
      DASHBOARD_QUIET_CONSOLE="1",
  )
  backend = pinned_agent_backend()
  if backend:
    env["WIKI_AGENT_BACKEND"] = backend
  try:
    agent_dispatch = importlib.import_module("agent_dispatch")

    # The daemon outlives the session that launched it.
    env = agent_dispatch.without_session_env(env)
  except ImportError:
    pass
  try:
    os.makedirs(log_dir, exist_ok=True)
    # Keep stderr so a crash at startup leaves a traceback behind.
    stderr_path = os.path.join(log_dir, "server_stderr.log")
    if (
        os.path.exists(stderr_path)
        and os.path.getsize(stderr_path) > STDERR_LOG_MAX_BYTES
    ):
      os.replace(stderr_path, stderr_path + ".1")
    with open(stderr_path, "ab") as stderr_f:
      subprocess.Popen(
          cmd,
          cwd=os.path.dirname(srv_path),
          env=env,
          stdout=subprocess.DEVNULL,
          stderr=stderr_f,
          start_new_session=True,
      )
  except (subprocess.SubprocessError, OSError) as e:
    return f"ERROR_SPAWN: {e}"

  # Poll health endpoint
  deadline = time.time() + max_wait_seconds
  while time.time() < deadline:
    time.sleep(0.25)
    if check_server_health(port=port, host=host):
      return "STARTED"

  return "START_TIMEOUT"


def trigger_scan(
    port: int, host: str = "127.0.0.1", timeout: float = 3.0
) -> bool:
  """Triggers an immediate collection scan via POST /api/scan."""
  url = f"http://{host}:{port}/api/scan"
  try:
    data = json.dumps({"scan_all_monitored": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.status in (200, 201, 202)
  except (
      urllib.error.URLError,
      urllib.error.HTTPError,
      OSError,
      TimeoutError,
      ValueError,
  ):
    return False


def configure_and_launch(
    target: str | None = None,
    project_id: str | None = None,
    cluster_name: str | None = None,
    location: str = "us-central1",
    workload_name: str | None = None,
    generation: str | None = None,
    reservation: str | None = None,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
    lookback_days: int = 30,
    timeout_seconds: float = 25.0,
    port: int = 8080,
    host: str = "127.0.0.1",
    no_server: bool = False,
    config_path: str = DEFAULT_CONFIG_PATH,
    cache_path: str = DEFAULT_CLUSTERS_CACHE_PATH,
    hostname: str | None = None,
    verify_probes: int = 3,
    verify_interval: float = 2.0,
) -> dict[str, Any]:
  """Configures clusters_config.json, launches dashboard, and returns structured result."""
  env_info = ensure_environment_ready()
  if not os.path.exists(config_path):
    legacy_cfg = os.path.join(DASHBOARD_DIR, "clusters_config.json")
    if os.path.exists(legacy_cfg):
      os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
      shutil.copy2(legacy_cfg, config_path)
  resolved_hostname = (
      hostname.strip()
      if hostname and hostname.strip()
      else get_default_hostname()
  )

  # 1. Target resolution
  resolved_project = project_id
  resolved_cluster = cluster_name
  resolved_workload = workload_name
  resolved_location = location
  resolved_generation = generation
  resolved_reservation = reservation

  if target:
    c_type = classify_target(
        target, cache_path=cache_path, config_path=config_path
    )
    if c_type == "project" and not resolved_project:
      resolved_project = target
    elif c_type == "cluster" and not resolved_cluster:
      resolved_cluster = target
    elif c_type == "workload" and not resolved_workload:
      resolved_workload = target

  # If workload is provided, resolve cluster & project if not already provided
  if resolved_workload and (not resolved_cluster or not resolved_project):
    c, loc, p = resolve_workload_details(
        resolved_workload,
        project_id=resolved_project,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        lookback_days=lookback_days,
        timeout_seconds=timeout_seconds,
    )
    if c and not resolved_cluster:
      resolved_cluster = c
    if loc and not resolved_location:
      resolved_location = loc
    if p and not resolved_project:
      resolved_project = p

  # If cluster is provided, resolve details from cache if missing
  if resolved_cluster:
    cached_info = find_cluster_in_cache(resolved_cluster, cache_path=cache_path)
    if not cached_info:
      cfg = load_json_file(config_path) or {}
      for c in cfg.get("clusters", []) + cfg.get("monitored_clusters", []):
        if isinstance(c, dict) and c.get("name") == resolved_cluster:
          cached_info = c
          break
    if not cached_info and resolved_project:
      proj_clusters = get_clusters_for_project(
          resolved_project, cache_path=cache_path
      )
      for c in proj_clusters:
        if c.get("name") == resolved_cluster:
          cached_info = c
          break
    if cached_info:
      if not resolved_project:
        resolved_project = cached_info.get("project")
      if not resolved_location or resolved_location == "us-central1":
        resolved_location = cached_info.get("location", resolved_location)
      if not resolved_generation:
        resolved_generation = cached_info.get("generation")
      if not resolved_reservation:
        resolved_reservation = cached_info.get("reservation")

  # If still no project, try active gcloud or config
  if not resolved_project:
    resolved_project = get_active_gcloud_project()

  if not resolved_project:
    existing_cfg = load_json_file(config_path) or {}
    for c in existing_cfg.get("clusters", []) + existing_cfg.get(
        "monitored_clusters", []
    ):
      if isinstance(c, dict) and c.get("project"):
        resolved_project = c.get("project")
        break

  if not resolved_project:
    resolved_project = ""

  # 2. Update clusters_config.json
  clusters_to_add: list[dict[str, Any]] = []
  replace_all = False

  if resolved_cluster:
    # Single cluster targeted
    c_entry: dict[str, Any] = {
        "name": resolved_cluster,
        "project": resolved_project,
        "location": resolved_location,
        "generation": resolved_generation or "other",
    }
    if resolved_reservation:
      c_entry["reservation"] = resolved_reservation
    clusters_to_add.append(c_entry)
  else:
    # Whole project targeted
    discovered = get_clusters_for_project(
        resolved_project, cache_path=cache_path
    )
    if discovered:
      clusters_to_add = discovered
      replace_all = True
    else:
      # If nothing discovered, keep existing matching clusters if present;
      # otherwise do not add placeholder
      existing_cfg = load_json_file(config_path) or {}
      existing_list = (
          existing_cfg.get("monitored_clusters")
          or existing_cfg.get("clusters")
          or []
      )
      matching = [
          c
          for c in existing_list
          if isinstance(c, dict) and c.get("project") == resolved_project
      ]
      if matching:
        clusters_to_add = matching
      else:
        clusters_to_add = []

  updated_clusters = update_clusters_config(
      config_path=config_path,
      clusters_to_add=clusters_to_add,
      replace_all=replace_all,
  )

  # 3. Server launch & scan
  server_status = "SKIPPED"
  if not no_server:
    server_status = launch_server(port=port, host=host)
    if server_status in ("RUNNING", "STARTED"):
      trigger_scan(port=port, host=host)

  # 4. Generate Dashboard URL
  query_params: list[str] = []
  if resolved_generation and resolved_generation != "overall":
    query_params.append(f"gen={urllib.parse.quote(resolved_generation)}")
  if resolved_cluster:
    query_params.append(f"cluster={urllib.parse.quote(resolved_cluster)}")
  if resolved_workload:
    query_params.append(f"workload={urllib.parse.quote(resolved_workload)}")
  elif not resolved_cluster and resolved_project:
    query_params.append(f"project={urllib.parse.quote(resolved_project)}")

  query_string = f"?{'&'.join(query_params)}" if query_params else "/"
  dashboard_url = (
      f"http://{resolved_hostname}:{port}/{query_string}"
      if not query_string.startswith("/")
      else f"http://{resolved_hostname}:{port}/"
  )

  # 5. Verify the URL actually works before handing it out.
  if no_server:
    verification: dict[str, Any] = {"url_status": "SKIPPED"}
  elif server_status not in ("RUNNING", "STARTED"):
    verification = {
        "url_status": "UNRESPONSIVE",
        "url_error": f"Dashboard server status: {server_status}",
        "url_hint": UNRESPONSIVE_HINT,
    }
  else:
    verification = verify_dashboard_url(
        dashboard_url, probes=verify_probes, interval=verify_interval
    )
    if (
        verification["url_status"] == "UNRESOLVABLE"
        and resolved_hostname != "localhost"
    ):
      fallback_url = dashboard_url.replace(
          f"//{resolved_hostname}:", "//localhost:", 1
      )
      fallback = verify_dashboard_url(
          fallback_url, probes=verify_probes, interval=verify_interval
      )
      fallback["url_fallback_reason"] = verification["url_error"]
      verification = fallback
      dashboard_url = fallback_url

  cluster_names = [
      c.get("name")
      for c in updated_clusters
      if isinstance(c, dict) and c.get("name")
  ]

  result: dict[str, Any] = {
      "status": (
          "SUCCESS"
          if verification["url_status"] in ("VERIFIED", "DEGRADED", "SKIPPED")
          else "FAILED"
      ),
      "dashboard_url": dashboard_url,
      "project_id": resolved_project,
      "cluster_name": (
          resolved_cluster
          or (cluster_names[0] if len(cluster_names) == 1 else "")
      ),
      "workload_name": resolved_workload or "",
      "clusters_count": len(updated_clusters),
      "clusters_list": cluster_names,
      "server_status": server_status,
      "log_dir": resolve_log_dir(),
      "data_dir": resolve_data_dir(),
  }
  if not no_server:
    # A reused server keeps the backend it was started with; report what it
    # actually uses and flag a mismatch with the caller's harness.
    running = get_server_agent_backend(port)
    result["agent_backend"] = running or "unknown"
    wanted = pinned_agent_backend()
    if running and wanted and running != wanted:
      result["agent_backend_warning"] = (
          f"The running dashboard triages with '{running}', but this session"
          f" runs under '{wanted}'. Stop it (`kill $(lsof -ti tcp:{port})`)"
          " and relaunch to switch."
      )
  result.update(verification)
  if env_info.get("auth_warning"):
    result["auth_warning"] = env_info["auth_warning"]

  return result


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Configure and launch GKE Capacity & Triage Dashboard."
  )
  parser.add_argument(
      "target",
      nargs="?",
      default=None,
      help="Target project ID, cluster name, or workload ID (optional).",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=None,
      help="GCP Project ID (optional).",
  )
  parser.add_argument(
      "--cluster",
      type=str,
      default=None,
      help="GKE Cluster Name (optional).",
  )
  parser.add_argument(
      "--workload",
      type=str,
      default=None,
      help="Workload name / JobSet ID (optional).",
  )
  parser.add_argument(
      "--location",
      type=str,
      default="us-central1",
      help="Cluster location/region (default: us-central1).",
  )
  parser.add_argument(
      "--generation",
      type=str,
      default=None,
      help="TPU generation (e.g. tpu7x, v6e, v5p).",
  )
  parser.add_argument(
      "--reservation",
      type=str,
      default=None,
      help="GCE backing reservation name (optional).",
  )
  parser.add_argument(
      "--port",
      type=int,
      default=8080,
      help="Localhost dashboard port (default: 8080).",
  )
  parser.add_argument(
      "--host",
      type=str,
      default="0.0.0.0",
      help="Dashboard host interface (default: 0.0.0.0).",
  )
  parser.add_argument(
      "--hostname",
      type=str,
      default=get_default_hostname(),
      help="Public hostname or FQDN for dashboard URL (default: system FQDN).",
  )
  parser.add_argument(
      "--no-server",
      action="store_true",
      help="Skip server launch and health check.",
  )
  parser.add_argument(
      "--start",
      type=str,
      default=None,
      help="Start UTC timestamp (YYYY-MM-DDTHH:MM:SSZ) for workload lookup.",
  )
  parser.add_argument(
      "--end",
      type=str,
      default=None,
      help="End UTC timestamp (YYYY-MM-DDTHH:MM:SSZ) for workload lookup.",
  )
  parser.add_argument(
      "--lookback-days",
      type=int,
      default=30,
      help=(
          "Workload lookup lookback window in days when --start is omitted"
          " (default: 30)."
      ),
  )
  parser.add_argument(
      "--timeout",
      type=float,
      default=25.0,
      help="Timeout in seconds for workload resolution query (default: 25.0).",
  )
  parser.add_argument(
      "--config-path",
      type=str,
      default=DEFAULT_CONFIG_PATH,
      help="Path to clusters_config.json.",
  )
  parser.add_argument(
      "--clusters-cache-path",
      type=str,
      default=DEFAULT_CLUSTERS_CACHE_PATH,
      help="Path to clusters_cache.json.",
  )
  parser.add_argument(
      "--verify-url",
      type=str,
      default=None,
      help=(
          "Only verify an existing dashboard URL (DNS + repeated page and"
          " health probes); exits non-zero unless VERIFIED or DEGRADED."
      ),
  )
  parser.add_argument(
      "--probes",
      type=int,
      default=3,
      help="Number of URL verification probes (default: 3).",
  )
  parser.add_argument(
      "--probe-interval",
      type=float,
      default=2.0,
      help="Seconds between URL verification probes (default: 2.0).",
  )

  args = parser.parse_args()

  if args.verify_url:
    verification = verify_dashboard_url(
        args.verify_url, probes=args.probes, interval=args.probe_interval
    )
    verification["dashboard_url"] = args.verify_url
    verification["status"] = (
        "SUCCESS"
        if verification["url_status"] in ("VERIFIED", "DEGRADED")
        else "FAILED"
    )
    print(json.dumps(verification, indent=2))
    sys.exit(0 if verification["status"] == "SUCCESS" else 1)

  result = configure_and_launch(
      target=args.target,
      project_id=args.project,
      cluster_name=args.cluster,
      location=args.location,
      workload_name=args.workload,
      generation=args.generation,
      reservation=args.reservation,
      start_timestamp=args.start,
      end_timestamp=args.end,
      lookback_days=args.lookback_days,
      timeout_seconds=args.timeout,
      port=args.port,
      host=args.host,
      no_server=args.no_server,
      config_path=args.config_path,
      cache_path=args.clusters_cache_path,
      hostname=args.hostname,
      verify_probes=args.probes,
      verify_interval=args.probe_interval,
  )

  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

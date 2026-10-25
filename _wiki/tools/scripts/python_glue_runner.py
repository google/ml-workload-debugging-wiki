#!/usr/bin/env python3
"""Executes and coordinates diagnostic workloads."""

import concurrent.futures
import functools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

import agent_dispatch

# What a batch LLM call can raise: agent_dispatch failures are RuntimeError
# (including DispatchUnavailable) or TimeoutError; JSON/ID parsing raises
# ValueError, KeyError or TypeError; file and subprocess I/O raises OSError.
_BATCH_ERRORS = (
    RuntimeError,
    TimeoutError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
)

_MARKDOWN_HEADER_RE = re.compile(r"(#+)\s+(.*)")
_TRIAGE_ENTRY_RE = re.compile(
    r"(?:^|\n)\s*\[(?:WARN|ERROR|FATAL|INFO)\][^\n]*"
    r"(?:\n(?!\s*\[(?:WARN|ERROR|FATAL|INFO)\]).*)*"
)
_LOG_ID_RE = re.compile(r"\[ID:\s*(\d+)\]")
_LOG_ID_SPLIT_RE = re.compile(r"\[ID:\s*(\d+)\]\s*")
_UNQUOTED_JSON_VALUE_RE = re.compile(r":\s*([a-zA-Z_][a-zA-Z0-9_]*)")
_PYTHON_WIKI_PREFIX_RE = re.compile(r"python3\s+_wiki/")


def parse_symptom_file(filepath: str) -> dict[str, Any]:
  """Parses a symptom file to extract title, description, and examples."""
  with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

  title = ""
  lines = content.splitlines()
  in_frontmatter = False
  yaml_lines = []
  body_start_idx = 0
  for idx, line in enumerate(lines):
    if line.strip() == "---":
      if not in_frontmatter:
        in_frontmatter = True
        continue
      else:
        in_frontmatter = False
        body_start_idx = idx + 1
        break
    if in_frontmatter:
      yaml_lines.append(line)

  for line in yaml_lines:
    if line.startswith("title:"):
      title = line.split("title:", 1)[1].strip().strip("\"'")
      break

  body_lines = lines[body_start_idx:]

  description = []
  examples = []

  current_header = None
  in_code_block = False
  current_code_block = []

  in_description = False
  in_example_section = False

  for line in body_lines:
    stripped = line.strip()
    header_match = _MARKDOWN_HEADER_RE.fullmatch(stripped)
    if header_match:
      in_code_block = False
      header_text = header_match.group(2).strip()
      current_header = header_text
      if current_header.lower() == "description":
        in_description = True
        in_example_section = False
      elif current_header.lower() in (
          "example error message/log",
          "verification steps & log signatures",
      ):
        in_description = False
        in_example_section = True
      else:
        in_description = False
        in_example_section = False
      continue

    if in_description:
      description.append(line)

    if in_example_section:
      if stripped.startswith("```"):
        if in_code_block:
          in_code_block = False
          if current_code_block:
            examples.append("\n".join(current_code_block))
            current_code_block = []
        else:
          in_code_block = True
      elif in_code_block:
        current_code_block.append(line)

  desc_text = "\n".join(description).strip()
  return {"title": title, "description": desc_text, "examples": examples}


# Names kept importable here because Actions, tests, and the interactive driver
# all reach for them through this module. The implementations live in
# agent_dispatch so both harnesses share one copy.
wait_for_conversation = agent_dispatch.wait_for_conversation
get_agent_json_path = agent_dispatch.get_agent_json_path
build_local_agent_config = agent_dispatch.build_local_agent_config
execute_sdk_conversation = agent_dispatch.execute_sdk_conversation
DispatchUnavailable = agent_dispatch.DispatchUnavailable


def invoke_llm_query(
    model: str,
    profile_name: str | None,
    prompt: str,
    root_dir: str | None = None,
) -> str:
  """Asks whichever agent backend is reachable, or fails saying none is."""
  return agent_dispatch.get_dispatch().invoke(
      prompt, model, profile=profile_name, root_dir=root_dir
  )


# Default wall-clock cap for a single command action. Generous enough for the
# slowest legitimate action (node resolution over a wide window takes ~80s)
# while still bounding a runaway stream.
DEFAULT_ACTION_TIMEOUT_SECONDS: int = 1800

# Agent profile used for `ai` actions that do not name one. `wiki_judge` is
# right for grading prompts and wrong for everything else -- under the Claude
# backend the profile becomes `--agent`, prepending the judge's system prompt
# to, say, a hardware-analysis prompt. It stays the default for compatibility;
# an Action should set `"profile"` to the profile its prompt was written for.
DEFAULT_AI_PROFILE: str = "wiki_judge"


_SUMMARY_BLOCK_HEADER = "SUMMARY REPORT (summary_fields)"
_SUMMARY_LINE_RE = re.compile(r"\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*)")


def parse_printed_summary(stdout: str | None) -> dict[str, Any]:
  """Extracts `key: value` pairs from a printed summary_fields block.

  Recognises the convention used by scripts that emit a human-readable report
  instead of JSON. Returns {} when no such block is present, so callers can
  merge unconditionally. Values are coerced to int where they look numeric,
  since summary fields are overwhelmingly counts.

  Args:
    stdout: Standard output text captured from a script execution.

  Returns:
    Dictionary of parsed summary key-value pairs.
  """
  out = {}
  in_block = False
  for line in (stdout or "").splitlines():
    if _SUMMARY_BLOCK_HEADER in line:
      in_block = True
      continue
    if not in_block:
      continue
    if line.startswith("---"):
      break
    m = _SUMMARY_LINE_RE.fullmatch(line)
    if not m:
      continue
    key, raw = m.group(1), m.group(2).strip()
    try:
      out[key] = int(raw)
    except ValueError:
      out[key] = raw
  return out


_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}|<(\w+)>|\{\{(\w+)\}\}")


def quote_context_at(template: str, index: int) -> str | None:
  r"""Shell quoting state at `index`: `"`, `'`, or None for bare.

  Walks the template rather than counting quote characters, because counting
  gets `--filter_extra="\\"${host_id}\\" AND ..."` backwards -- the escaped
  inner quotes are literal and do not open a region.

  Args:
    template: Command template string.
    index: Character offset inside `template`.

  Returns:
    Active quote character (`"` or `'`), or None if unquoted.
  """
  in_single = False
  in_double = False
  i = 0
  while i < index and i < len(template):
    c = template[i]
    if in_single:
      if c == "'":
        in_single = False
    elif in_double:
      if c == "\\":
        i += 1  # backslash escapes the next char inside double quotes
      elif c == '"':
        in_double = False
    else:
      if c == "\\":
        i += 1
      elif c == "'":
        in_single = True
      elif c == '"':
        in_double = True
    i += 1
  if in_single:
    return "'"
  if in_double:
    return '"'
  return None


def escape_for_context(value: str, context: str | None) -> str:
  """Neutralises shell metacharacters in `value` for its quoting context.

  Interpolation is textual and the result goes to a shell, so a value carrying
  a quote closes the one the template opened and the remainder runs as its own
  command. Workload and host names are read out of logs, so this needs no
  adversary -- an apostrophe or a `$` is enough to corrupt the command.

  The escaping is context-dependent because the templates are not uniform. Many
  placeholders sit inside a quoted region the template author already opened,
  and several are deliberate command fragments (`node_filter`, `filter_extra`,
  `severity_filter`); wrapping those in another layer of quotes breaks them. So:

    bare     -> shlex.quote, making the value exactly one argument
    '...'    -> close, escaped quote, reopen (the only escape sh honours there)
    "..."    -> backslash-escape the four characters the shell still reads

  A value containing no shell metacharacters is returned unchanged in every
  context, so ordinary invocations produce byte-identical command lines.

  Args:
    value: Parameter value string to escape.
    context: Active quote character (`"`, `'`, or None).

  Returns:
    Escaped value safe for interpolation into `context`.
  """
  if context == "'":
    return value.replace("'", "'\\''")
  if context == '"':
    out = []
    for ch in value:
      if ch in ('"', "\\", "$", "`"):
        out.append("\\")
      out.append(ch)
    return "".join(out)
  return shlex.quote(value)


def _replace_placeholder(
    template: str, params: dict[str, Any], match: re.Match[str]
) -> str:
  name = match.group(1) or match.group(2) or match.group(3)
  if name not in params:
    return match.group(0)
  context = quote_context_at(template, match.start())
  return escape_for_context(str(params[name]), context)


def interpolate_command(template: str, params: dict[str, Any]) -> str:
  """Substitutes `${name}`, `<name>`, and `{{name}}` with escaped values.

  One pass over the template, so a value that happens to contain something
  looking like another placeholder is not re-expanded. Names with no matching
  parameter are left verbatim, as before -- `<node>` in prose is not a slot.

  Args:
    template: Command template string containing placeholders.
    params: Parameter dictionary mapping placeholder names to values.

  Returns:
    Interpolated command string.
  """
  return _PLACEHOLDER_RE.sub(
      functools.partial(_replace_placeholder, template, params), template
  )


def terminate_process_group(
    proc: "subprocess.Popen[str]", grace_seconds: float = 5
) -> None:
  """Kills the whole process group, not just the shell we spawned.

  With `shell=True` the direct child is `/bin/sh`; the command it runs is a
  grandchild. Killing the shell leaves the grandchild running and holding the
  pipe, and the streaming commands the timeout exists for (`kubectl -w`,
  `logs -f`, `tail -f`) are precisely the ones that survive that. The group is
  the unit that has to die, which is why the process is started with
  `start_new_session`.

  Args:
    proc: Process to terminate.
    grace_seconds: Grace period in seconds.
  """
  if not hasattr(os, "killpg"):  # non-POSIX; best effort
    proc.kill()
    return
  try:
    pgid = os.getpgid(proc.pid)
  except (ProcessLookupError, OSError):
    return
  for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2)):
    try:
      os.killpg(pgid, sig)
    except (ProcessLookupError, OSError):
      return
    try:
      proc.wait(timeout=wait)
      return
    except subprocess.TimeoutExpired:
      continue


def run_command_bounded(
    command: str, cwd: str | None, timeout_seconds: float
) -> tuple[str, str, int, bool]:
  """Runs `command` under a shell, bounded by a timeout that actually bites.

  Returns `(stdout, stderr, returncode, timed_out)`. On timeout the partial
  output captured so far is returned rather than discarded, since a truncated
  result is still evidence.

  Args:
    command: Shell command string to execute.
    cwd: Working directory for the subprocess, or None.
    timeout_seconds: Maximum wall-clock duration in seconds.

  Returns:
    Tuple of (stdout, stderr, returncode, timed_out).
  """
  # terminate_process_group ends with SIGKILL, so the context manager's
  # final wait() on exit cannot hang after a timeout.
  with subprocess.Popen(
      command,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=cwd,
      start_new_session=True,
  ) as proc:
    try:
      out, err = proc.communicate(timeout=timeout_seconds)
      return out, err, proc.returncode, False
    except subprocess.TimeoutExpired:
      terminate_process_group(proc)
      try:
        out, err = proc.communicate(timeout=10)
      except subprocess.TimeoutExpired:
        out, err = "", ""
      return out or "", err or "", proc.returncode, True


def _script_of(text: str | None) -> str | None:
  for tok in (text or "").split():
    if tok.endswith(".py"):
      return os.path.basename(tok)
  return None


def script_level_mock(command: str, wiki_dir: str) -> str | None:
  """Fixture path for a `python3 <script>` mock matching `command`, else None.

  Some fixtures declare their mocks against the *script* rather than the
  binaries the script shells out to. `mock_driver.py` cannot serve those:
  interception works by shadowing a binary on PATH, and `python3` is
  deliberately never shadowed because doing so would intercept this runner.
  Such a mock is inert -- the script runs for real, reaches live Cloud Logging
  with a synthetic workload name, returns nothing, and the case is unpassable
  for reasons unrelated to the agent.

  Serving it here closes that gap. Only ever called from inside the test-context
  gate, so production runs never reach this path.

  Args:
    command: Interpolated command string.
    wiki_dir: Path to the `_wiki` directory.

  Returns:
    Resolved fixture file path if matched, or None.
  """
  del wiki_dir
  case_dir = os.environ.get("ACTIVE_TEST_CASE_DIR")
  if not case_dir:
    return None
  problem = os.path.join(case_dir, "problem.json")
  if not os.path.exists(problem):
    return None
  try:
    with open(problem, encoding="utf-8") as f:
      mocks = json.load(f).get("mock_commands", [])
  except (OSError, ValueError):
    return None

  # Compare on script basename: instructions use paths relative to the wiki
  # root ("tools/scripts/x.py") while fixtures were written with the "_wiki/"
  # prefix, so a literal prefix comparison never matches.
  want = _script_of(command)
  if not want:
    return None
  for mc in mocks:
    prefix = mc.get("command_prefix") or ""
    if not prefix.split()[:1] or prefix.split()[0] != "python3":
      continue
    if _script_of(prefix) != want:
      continue
    if any(tok not in command for tok in mc.get("args_contains") or []):
      continue
    mock_file = mc.get("mock_file", "")
    if not mock_file:
      continue
    # Fixtures do NOT live under the case dir. prepare_sandbox() excludes
    # `mock_data` from the sandbox copy and stages it outside the tree, at
    # ACTIVE_MOCK_DATA_DIR, so that the agent cannot read the answers straight
    # off disk. Resolving against the case dir therefore never found anything
    # and this returned None silently -- which is why the one case whose
    # fixtures are keyed to scripts rather than to gcloud/kubectl ran blind.
    # Case dir is kept as a fallback for any caller staging them in place.
    roots = [
        r
        for r in (os.environ.get("ACTIVE_MOCK_DATA_DIR"), case_dir)
        if r
    ]
    name = os.path.basename(mock_file)
    subdir = os.path.dirname(mock_file)
    for root in roots:
      # The staged copy renames the real payload to `.<name>.real` and leaves
      # an access warning at the original path; mock_driver.py prefers the
      # former, and so must this or the action is handed the warning text.
      for candidate in (
          os.path.join(root, subdir, f".{name}.real"),
          os.path.join(root, mock_file),
      ):
        if os.path.exists(candidate):
          return candidate
  return None


def _count_lines(filepath: str | None) -> int | str:
  """Counts non-empty lines in filepath, resolving relative to _wiki."""
  if not filepath:
    return "N/A"
  target = filepath
  if not os.path.exists(target) and not os.path.isabs(target):
    candidate = os.path.join(get_wiki_dir(), target.removeprefix("_wiki/"))
    if os.path.exists(candidate):
      target = candidate
  if not os.path.exists(target):
    return "N/A"
  try:
    with open(target, "r", encoding="utf-8") as f:
      return sum(1 for line in f if line.strip())
  except OSError:
    return "N/A"


def print_summary_report(
    action_name: str,
    action_schema: dict[str, Any],
    params: dict[str, Any],
    custom_metrics: dict[str, Any] | None = None,
) -> None:
  """Prints the formatted summary report block for an action execution."""
  if "summary_fields" not in action_schema:
    return
  title = action_schema.get("title", action_name)
  header = f"--- SUMMARY REPORT: {title} ({action_name}) ---"
  print(f"\n{header}", file=sys.stderr)

  in_file = params.get("input_file") or params.get("log_file")
  out_file = params.get("output_file")

  in_lines = _count_lines(in_file)
  out_lines = _count_lines(out_file)

  if (
      out_lines == "N/A"
      and custom_metrics
      and "_stdout_lines" in custom_metrics
  ):
    out_lines = custom_metrics["_stdout_lines"]

  for field in action_schema["summary_fields"]:
    val = "N/A"
    if custom_metrics and field in custom_metrics:
      val = custom_metrics[field]
    elif field in params:
      val = params[field]
    elif field == "execution_status":
      val = "SUCCESS"
    elif field == "affected_workload":
      val = params.get("workload_name", params.get("workload", "N/A"))
    elif field == "output_file":
      val = out_file if out_file else "scratch/output.txt"
    elif field in ("original_count", "extracted_log_count", "raw_lines_count"):
      val = in_lines
    elif field == "unmasked_entries":
      val = out_lines if out_lines != "N/A" else in_lines
    elif field in ("deduplicated_count", "combined_messages_count"):
      val = out_lines
    elif (
        (field == "masking_ratio" or field == "compression_ratio")
        and isinstance(in_lines, int)
        and isinstance(out_lines, int)
        and in_lines > 0
    ):
      ratio = (1.0 - (out_lines / in_lines)) * 100
      val = f"{ratio:.1f}% reduction"

    print(f"  {field}: {val}", file=sys.stderr)
  print("-" * len(header), file=sys.stderr)


def get_wiki_dir() -> str:
  """Dynamically discovers the _wiki directory for repo root or _wiki cwd."""
  if "WIKI_DIR" in os.environ and os.environ["WIKI_DIR"]:
    env_dir = os.path.abspath(os.environ["WIKI_DIR"])
    if os.path.exists(os.path.join(env_dir, "actions")):
      return env_dir
  script_dir = os.path.dirname(os.path.abspath(__file__))
  default_wiki_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
  if os.path.exists(os.path.join(default_wiki_dir, "actions")):
    return default_wiki_dir
  cwd = os.path.abspath(os.getcwd())
  if os.path.exists(os.path.join(cwd, "actions")):
    return cwd
  if os.path.exists(os.path.join(cwd, "_wiki", "actions")):
    return os.path.join(cwd, "_wiki")
  return default_wiki_dir


def resolve_action_path(action_name: str, wiki_dir: str) -> tuple[str, str]:
  """Resolves an action or workflow file from a variety of naming formats."""
  clean_name = action_name.strip("[]")
  clean_name = os.path.basename(clean_name)
  if clean_name.endswith(".json"):
    clean_name = clean_name[:-5]
  json_name = f"{clean_name}.json"

  candidates = [
      os.path.join(wiki_dir, "actions", json_name),
      os.path.join(wiki_dir, "workflows", json_name),
      os.path.join(os.path.dirname(wiki_dir), "_wiki", "actions", json_name),
      os.path.join(os.path.dirname(wiki_dir), "_wiki", "workflows", json_name),
  ]
  for cand in candidates:
    if os.path.exists(cand):
      return cand, clean_name
  return candidates[0], clean_name


def resolve_input_path(filepath: str | None, wiki_dir: str) -> str | None:
  """Resolves an input file path relative to cwd, wiki_dir, or repo root."""
  if not filepath:
    return filepath
  if os.path.isabs(filepath):
    return filepath
  if os.path.exists(filepath):
    return os.path.abspath(filepath)
  # Mirror resolve_output_path: a `_wiki/`-prefixed path is relative to the
  # repo root, not to wiki_dir, or it resolves to `_wiki/_wiki/...`.
  if filepath.startswith("_wiki/"):
    cand_prefixed = os.path.join(os.path.dirname(wiki_dir), filepath)
    if os.path.exists(cand_prefixed):
      return cand_prefixed
  cand_wiki = os.path.join(wiki_dir, filepath)
  if os.path.exists(cand_wiki):
    return cand_wiki
  cand_root = os.path.join(os.path.dirname(wiki_dir), filepath)
  if os.path.exists(cand_root):
    return cand_root
  return cand_wiki


def resolve_output_path(out_path: str | None, wiki_dir: str) -> str:
  """Resolves an output path, ensuring parent directories exist."""
  if not out_path:
    resolved = os.path.join(wiki_dir, "scratch", "output.txt")
  elif os.path.isabs(out_path):
    resolved = out_path
  elif out_path.startswith("_wiki/"):
    resolved = os.path.join(os.path.dirname(wiki_dir), out_path)
  else:
    resolved = os.path.join(wiki_dir, out_path)
  os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
  return resolved


APPROVAL_FLAG = "--approved"
ATTEMPTS_JSON_PREFIX = "--attempts-json="


def approval_exempt(
    action_schema: dict[str, Any], params: dict[str, Any]
) -> bool:
  """Checks if params select a declared read-only mode of a gated Action.

  `approval_exempt_params` maps parameter names to the values that make the
  Action non-mutating (e.g. gcloud-login's {"check_only": true}). Every listed
  parameter must match; values compare as lower-cased strings so a JSON `true`
  and a templated "True" agree.

  Args:
    action_schema: Parsed JSON schema dictionary for the Action.
    params: Runtime parameter values supplied for the invocation.

  Returns:
    True when all declared approval_exempt_params match the runtime params.
  """
  exempt = action_schema.get("approval_exempt_params")
  if not isinstance(exempt, dict) or not exempt:
    return False
  return all(
      k in params and str(params[k]).lower() == str(v).lower()
      for k, v in exempt.items()
  )


# Distinct from 1 (usage/schema errors), 3 (no agent backend) and 124
# (timeout) so callers can tell "needs a human" apart from "failed".
EXIT_APPROVAL_REQUIRED = 4

# Exit code 77 (EX_NOPERM) is emitted by log_query.py and Cloud Logging actions
# when IAM or authentication access is denied on the target GCP project.
EXIT_ACCESS_DENIED = 77
ACCESS_DENIED_CACHE_TTL_SECONDS = 900

CLOUD_LOGGING_ACTIONS = frozenset({
    "validate-workload",
    "lookup-workload-timestamps",
    "query-crash-signatures",
    "locate-disruption-windows",
    "job-disruption-timeline",
    "find-earliest-failure",
    "extract-culprit-hosts",
    "resolve-workload-nodes",
    "query-vm-termination-events",
    "query-node-health-events",
    "query-workload-logs",
    "get-jobset-events",
    "list-unhealthy-gce-instances",
    "query-gce-system-events-by-zone",
    "resolve-workload-infra-specs",
    "pod-analyzer",
})

AUTH_REFRESH_ACTIONS = frozenset({
    "gcloud-login",
})

_PARAM_ALIAS_MAP: dict[str, str] = {
    "workload": "workload_name",
    "workload_name": "workload",
    "jobset_name": "workload_name",
    "node_name_filter_clause": "node_filter",
    "node_filter": "node_name_filter_clause",
    "start": "start_timestamp",
    "start_timestamp": "start",
    "start_time": "start_timestamp",
    "end": "end_timestamp",
    "end_timestamp": "end",
    "end_time": "end_timestamp",
    "project": "project_id",
    "project_id": "project",
}

_PARAM_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("workload", "workload_name", "jobset_name"),
    ("node_name_filter_clause", "node_filter"),
    ("start", "start_timestamp", "start_time"),
    ("end", "end_timestamp", "end_time"),
    ("project", "project_id"),
)


def _expand_param_aliases(params: dict[str, Any]) -> None:
  """Populates canonical and alias keys across each parameter alias group."""
  for group in _PARAM_ALIAS_GROUPS:
    param_value = next(
        (
            params[alias_key]
            for alias_key in group
            if alias_key in params and params[alias_key] not in (None, "")
        ),
        None,
    )
    if param_value is not None:
      for alias_key in group:
        params.setdefault(alias_key, param_value)


_ACCESS_DENIED_MARKERS = (
    "PERMISSION_DENIED",
    "Permission denied",
    "ACCESS_DENIED",
    "access denied",
    "not authorized",
    "caller does not have permission",
    "unauthenticated",
    "UNAUTHENTICATED",
    "Reauthentication failed",
    "invalid_grant",
    "One or more resources require customer approval",
    "https://cloud.google.com/iam",
)


def get_access_denied_cache_path(wiki_dir: str) -> str:
  """Returns file path for Cloud Logging ACCESS_DENIED circuit breaker cache."""
  return os.path.join(
      wiki_dir, "scratch", ".cloud_logging_access_denied_cache.json"
  )


def get_auth_fingerprint() -> str:
  """Computes lightweight fingerprint of current gcloud/ADC environment."""
  cfg_dir = os.environ.get(
      "CLOUDSDK_CONFIG", os.path.expanduser("~/.config/gcloud")
  )
  adc_path = os.environ.get(
      "GOOGLE_APPLICATION_CREDENTIALS",
      os.path.join(cfg_dir, "application_default_credentials.json"),
  )
  active_cfg_path = os.path.join(cfg_dir, "active_config")
  parts = [
      f"cfg={cfg_dir}",
      f"acct={os.environ.get('CLOUDSDK_CORE_ACCOUNT', '')}",
      f"token={bool(os.environ.get('CLOUDSDK_AUTH_ACCESS_TOKEN'))}",
  ]
  for path in (active_cfg_path, adc_path):
    try:
      if os.path.exists(path):
        st = os.stat(path)
        parts.append(f"{os.path.basename(path)}:{int(st.st_mtime)}")
    except OSError:
      pass
  return "|".join(parts)


def _extract_project_key(params: dict[str, Any]) -> str:
  """Extracts the target GCP project identifier from action parameters."""
  proj = (
      params.get("project_id")
      or params.get("project")
      or os.environ.get("CLOUDSDK_CORE_PROJECT")
      or "default"
  )
  return str(proj).strip() or "default"


def is_access_denied_output(
    returncode: int, stdout: str | None, stderr: str | None
) -> bool:
  """Returns True if command failed due to GCP IAM or authentication denial."""
  if returncode == EXIT_ACCESS_DENIED:
    return True
  if returncode == 0:
    return False
  combined = f"{stderr or ''}\n{stdout or ''}"
  return any(marker in combined for marker in _ACCESS_DENIED_MARKERS)


def check_access_denied_circuit_breaker(
    action_name_clean: str,
    params: dict[str, Any],
    wiki_dir: str,
) -> dict[str, Any] | None:
  """Checks if Cloud Logging action should fast-fail on cached ACCESS_DENIED."""
  if action_name_clean not in CLOUD_LOGGING_ACTIONS:
    return None
  cache_path = get_access_denied_cache_path(wiki_dir)
  if not os.path.exists(cache_path):
    return None
  try:
    with open(cache_path, "r", encoding="utf-8") as f:
      cache = json.load(f)
  except (OSError, ValueError):
    return None
  if not isinstance(cache, dict):
    return None

  project_key = _extract_project_key(params)
  auth_fp = get_auth_fingerprint()
  cache_key = f"{project_key}::{auth_fp}"
  entry = cache.get(cache_key)
  if not isinstance(entry, dict):
    return None

  ts = entry.get("timestamp", 0)
  if time.time() - ts > ACCESS_DENIED_CACHE_TTL_SECONDS:
    return None
  return entry


def record_access_denied_circuit_breaker(
    action_name_clean: str,
    params: dict[str, Any],
    wiki_dir: str,
    reason: str = "",
) -> None:
  """Records Cloud Logging ACCESS_DENIED event in circuit breaker cache."""
  cache_path = get_access_denied_cache_path(wiki_dir)
  os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
  cache: dict[str, Any] = {}
  if os.path.exists(cache_path):
    try:
      with open(cache_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
      if isinstance(loaded, dict):
        cache = loaded
    except (OSError, ValueError):
      cache = {}

  project_key = _extract_project_key(params)
  auth_fp = get_auth_fingerprint()
  cache_key = f"{project_key}::{auth_fp}"
  cache[cache_key] = {
      "timestamp": time.time(),
      "project_id": project_key,
      "auth_fingerprint": auth_fp,
      "cached_from_action": action_name_clean,
      "reason": (reason or "").strip()[:400],
  }
  try:
    with open(cache_path, "w", encoding="utf-8") as f:
      json.dump(cache, f, indent=2)
  except OSError:
    pass


def clear_access_denied_circuit_breaker(
    params: dict[str, Any],
    wiki_dir: str,
    clear_all: bool = False,
) -> None:
  """Clears the Cloud Logging ACCESS_DENIED circuit breaker cache."""
  cache_path = get_access_denied_cache_path(wiki_dir)
  if not os.path.exists(cache_path):
    return
  if clear_all:
    try:
      os.remove(cache_path)
    except OSError:
      pass
    return
  try:
    with open(cache_path, "r", encoding="utf-8") as f:
      cache = json.load(f)
    if isinstance(cache, dict):
      project_key = _extract_project_key(params)
      auth_fp = get_auth_fingerprint()
      cache.pop(f"{project_key}::{auth_fp}", None)
      with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
  except (OSError, ValueError):
    pass


def _repair_unquoted_json_value(match: re.Match[str]) -> str:
  val = match.group(1)
  if val in ("true", "false", "null"):
    return match.group(0)
  return ": false"


def _parse_llm_json_output(response_content: str, batch_idx: int) -> Any:
  """Parses and repairs JSON output returned by an LLM batch call."""
  cleaned_content = response_content.strip()
  if cleaned_content.startswith("```json"):
    cleaned_content = cleaned_content[7:]
  if cleaned_content.endswith("```"):
    cleaned_content = cleaned_content[:-3]
  cleaned_content = cleaned_content.strip()

  try:
    return json.loads(cleaned_content)
  except json.JSONDecodeError as first_err:
    repaired = _UNQUOTED_JSON_VALUE_RE.sub(
        _repair_unquoted_json_value, cleaned_content
    )
    try:
      return json.loads(repaired)
    except json.JSONDecodeError as e:
      raise RuntimeError(
          f"Error parsing subagent output JSON for batch {batch_idx + 1}"
          f" (even after repair): {e}. Content: {cleaned_content}"
      ) from first_err


def _execute_match_batch(
    batch_idx: int,
    batch: list[dict[str, Any]],
    *,
    total_batches: int,
    prompt_template: str,
    symptoms_catalog_str: str,
    model: str,
    profile: str,
    wiki_dir: str,
) -> Any:
  """Executes a single LLM symptom-matching batch."""
  logs_str = "\n".join(item["raw"] for item in batch)
  prompt = prompt_template.replace("${logs}", logs_str).replace(
      "${symptoms}", symptoms_catalog_str
  )

  print(
      "[Glue Runner] Executing subagent conversation for batch"
      f" {batch_idx + 1}/{total_batches}...",
      file=sys.stderr,
  )
  response_content = invoke_llm_query(model, profile, prompt, root_dir=wiki_dir)
  return _parse_llm_json_output(response_content, batch_idx)


def _execute_classify_batch(
    batch_idx: int,
    batch: list[dict[str, Any]],
    *,
    total_batches: int,
    prompt_template: str,
    assigned_hosts_val: str,
    mappings_data: dict[str, Any],
    model: str,
    profile: str,
    wiki_dir: str,
) -> list[str]:
  """Executes a single LLM severity-classification batch."""
  logs_str = "\n".join(item["raw"] for item in batch)
  prompt = prompt_template.replace("${logs}", logs_str).replace(
      "${assigned_hosts}", assigned_hosts_val
  )

  print(
      "[Glue Runner] Executing subagent conversation for batch"
      f" {batch_idx + 1}/{total_batches}...",
      file=sys.stderr,
  )
  response_content = invoke_llm_query(model, profile, prompt, root_dir=wiki_dir)
  subagent_output = _parse_llm_json_output(response_content, batch_idx)

  parsed_map = {}
  for item in subagent_output:
    if "id" in item:
      parsed_map[int(item["id"])] = item

  batch_annotations = []
  for item in batch:
    info = parsed_map.get(item["id"])
    if info:
      severity = info.get("severity", "INFO")
      has_mentioned_host = info.get("has_mentioned_host", False)
      leakage_flag = " [CROSS_TALK_LEAKAGE]" if has_mentioned_host else ""
      annotated_line = f"[{severity}]{leakage_flag} {item['raw']}"
    else:
      annotated_line = f"[INFO] {item['raw']}"

    unmasked_line = annotated_line
    for placeholder, details in mappings_data.items():
      raw_val = details.get("value")
      if raw_val:
        unmasked_line = unmasked_line.replace(placeholder, raw_val)
    batch_annotations.append(unmasked_line)
  return batch_annotations


def _run_single_attempt(
    idx_and_item: tuple[int, Any],
    *,
    base_params: dict[str, Any],
    alias_map: dict[str, str],
    declared: set[str],
    action_name_clean: str,
    run_template: str,
    total_items: int,
    wiki_dir: str,
    timeout_seconds: float,
    in_test_context: bool = False,
) -> dict[str, Any]:
  """Runs a single attempt or batch item command for parallel execution."""
  idx, item = idx_and_item
  if not isinstance(item, dict):
    item = {"value": item}
  item_params = dict(base_params)
  item_params.update(item)
  if "first_occurrence" in item and "start" not in item:
    item_params["start"] = item["first_occurrence"]
    item_params["start_timestamp"] = item["first_occurrence"]
  if "last_occurrence" in item and "end" not in item:
    item_params["end"] = item["last_occurrence"]
    item_params["end_timestamp"] = item["last_occurrence"]
  if "start_time" in item and "start" not in item:
    item_params["start"] = item["start_time"]
    item_params["start_timestamp"] = item["start_time"]
  if "end_time" in item and "end" not in item:
    item_params["end"] = item["end_time"]
    item_params["end_timestamp"] = item["end_time"]
  _expand_param_aliases(item_params)
  for k, v in list(item_params.items()):
    if k in alias_map and alias_map[k] not in item_params:
      item_params[alias_map[k]] = v
  att_label = item.get("restart_attempt", item.get("host_id", f"idx{idx}"))
  if "output_file" not in item:
    item_params["output_file"] = (
        f"scratch/{action_name_clean}_attempt_{att_label}_output.json"
    )
  if "max_concurrent_workers" not in item_params:
    item_params["max_concurrent_workers"] = 4
  if "disruption_timestamp" not in item_params:
    item_params["disruption_timestamp"] = item_params.get(
        "start_timestamp", item_params.get("start", "")
    )
  for pname in declared:
    item_params.setdefault(pname, "")

  cmd = interpolate_command(run_template, item_params)
  norm_cmd = _PYTHON_WIKI_PREFIX_RE.sub("python3 ", cmd)
  print(
      f"[Glue Runner] Parallel [{idx + 1}/{total_items}]"
      f" (attempt={att_label}): {norm_cmd}",
      file=sys.stderr,
  )
  if in_test_context:
    mock_path = script_level_mock(norm_cmd, wiki_dir)
    if mock_path:
      with open(mock_path, encoding="utf-8") as mf:
        payload = mf.read()
      print(
          f"[Glue Runner] Served script-level mock {mock_path}",
          file=sys.stderr,
      )
      out_target = item_params.get("output_file")
      if out_target:
        out_path = resolve_output_path(out_target, wiki_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as of:
          of.write(payload)
      try:
        parsed_mock = json.loads(payload.strip())
      except (ValueError, TypeError):
        parsed_mock = payload.strip()
      return {
          "index": idx,
          "restart_attempt": item.get("restart_attempt"),
          "host_id": item.get("host_id"),
          "start": item_params.get("start", ""),
          "end": item_params.get("end", ""),
          "output_file": item_params.get("output_file", ""),
          "returncode": 0,
          "timed_out": False,
          "output": parsed_mock,
          "stderr": "",
      }
  out, err, rc, timed_out = run_command_bounded(
      norm_cmd, wiki_dir, timeout_seconds
  )
  if timed_out:
    rc = 124
  elif is_access_denied_output(rc, out, err) and (
      action_name_clean in CLOUD_LOGGING_ACTIONS or rc == EXIT_ACCESS_DENIED
  ):
    rc = EXIT_ACCESS_DENIED
  parsed_out = None
  if out:
    try:
      parsed_out = json.loads(out.strip())
    except (ValueError, TypeError):
      parsed_out = out.strip()
  return {
      "index": idx,
      "restart_attempt": item.get("restart_attempt"),
      "host_id": item.get("host_id"),
      "start": item_params.get("start", ""),
      "end": item_params.get("end", ""),
      "output_file": item_params.get("output_file", ""),
      "returncode": rc,
      "timed_out": timed_out,
      "output": parsed_out,
      "stderr": (err or "").strip()[:500] if rc != 0 else "",
  }


AUTO_FALLBACK_FLAG = "--auto-fallback"
_AUTH_EXPIRED_MARKERS = (
    "unauthenticated",
    "UNAUTHENTICATED",
    "Reauthentication failed",
    "invalid_grant",
    "Credentials have expired",
    "token has expired",
)


def _replace_raw_placeholder(
    match: re.Match[str], *, ctx: dict[str, Any]
) -> str:
  name = match.group(1) or match.group(2) or match.group(3)
  if name not in ctx or ctx[name] is None:
    return match.group(0)
  return str(ctx[name])


def _interpolate_raw_value(template: str, ctx: dict[str, Any]) -> str:
  """Interpolates ${name}, <name>, or {{name}} as raw unescaped text."""
  return _PLACEHOLDER_RE.sub(
      functools.partial(_replace_raw_placeholder, ctx=ctx), str(template)
  )


def evaluate_action_fallbacks(
    action_schema: dict[str, Any],
    params: dict[str, Any],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
  """Evaluates declared action-level fallbacks against execution outcomes.

  Args:
    action_schema: Parsed action JSON dictionary containing optional `fallback`.
    params: Resolved runtime parameter mapping used for command interpolation.
    returncode: Process exit code returned by the primary action command.
    stdout: Captured standard output string, or None (treated as empty string).
    stderr: Captured standard error string, or None (treated as empty string).
    metrics: Optional parsed output metrics dictionary; when None, metrics are
      derived from JSON stdout if available.

  Returns:
    Ordered list of triggered fallback entries with interpolated commands.
  """
  fallback_list = action_schema.get("fallback")
  if not isinstance(fallback_list, list) or not fallback_list:
    return []

  parsed_json: Any = None
  raw_out = (stdout or "").strip()
  if raw_out:
    try:
      parsed_json = json.loads(raw_out)
    except (ValueError, TypeError):
      parsed_json = None

  triggers: list[str] = []
  combined_text = f"{stderr or ''}\n{stdout or ''}"
  if returncode == EXIT_ACCESS_DENIED or is_access_denied_output(
      returncode, stdout, stderr
  ):
    triggers.append("access_denied")
    if any(m in combined_text for m in _AUTH_EXPIRED_MARKERS):
      triggers.append("auth_expired")
  elif returncode == 124:
    triggers.append("timeout")
  elif returncode != 0:
    triggers.append("command_error")
    if any(m in combined_text for m in _AUTH_EXPIRED_MARKERS):
      triggers.append("auth_expired")
  else:
    is_empty = False
    is_partial = False
    if not raw_out:
      is_empty = True
    elif isinstance(parsed_json, (list, dict)) and not parsed_json:
      is_empty = True
    elif isinstance(parsed_json, dict):
      status_val = str(parsed_json.get("status", "")).upper()
      if status_val in ("NOT_FOUND", "EMPTY", "NO_MATCH"):
        is_empty = True
      elif status_val in ("PARTIAL", "INCOMPLETE"):
        is_partial = True
      for count_key in (
          "matched_rows",
          "found_count",
          "total_entries_scanned",
          "unique_hosts_count",
      ):
        if count_key in parsed_json and parsed_json[count_key] == 0:
          is_empty = True
      if parsed_json.get("missing_nodes") or not parsed_json.get(
          "coverage_complete", True
      ):
        is_partial = True
    if metrics:
      for count_key in (
          "matched_rows",
          "found_count",
          "total_entries_scanned",
          "unique_hosts_count",
      ):
        if count_key in metrics and metrics[count_key] == 0:
          is_empty = True
    if is_empty:
      triggers.append("empty_result")
    if is_partial:
      triggers.append("incomplete_coverage")

  if not triggers:
    return []

  interp_ctx: dict[str, Any] = dict(params)
  if isinstance(parsed_json, dict):
    for k, v in parsed_json.items():
      if isinstance(v, (str, int, float, bool)):
        interp_ctx.setdefault(k, v)
  if metrics:
    for k, v in metrics.items():
      if isinstance(v, (str, int, float, bool)):
        interp_ctx.setdefault(k, v)

  matched_fallbacks: list[dict[str, Any]] = []
  for entry in fallback_list:
    if not isinstance(entry, dict):
      continue
    when_list = entry.get("when", [])
    if not isinstance(when_list, list):
      continue
    matched_when = [t for t in when_list if t in triggers]
    if not matched_when:
      continue
    raw_target = str(entry.get("action", "")).strip()
    clean_target = raw_target.strip("[]")
    if not clean_target:
      continue

    resolved_params = {k: v for k, v in params.items() if k != "output_file"}
    param_mapping = entry.get("param_mapping")
    if isinstance(param_mapping, dict):
      for target_param, expr in param_mapping.items():
        val = _interpolate_raw_value(str(expr), interp_ctx)
        if val and not _PLACEHOLDER_RE.search(val):
          resolved_params[target_param] = val

    matched_fallbacks.append({
        "action": clean_target,
        "when": matched_when,
        "reason": entry.get("reason", ""),
        "resolved_params": resolved_params,
        "auto_execute": bool(entry.get("auto_execute", False)),
    })

  return matched_fallbacks


def main() -> None:
  # `--approved`, `--auto-fallback`, and `--attempts-json=...` may appear
  # anywhere; positional arguments remain <action_name> '<parameters_json>'.
  approved = APPROVAL_FLAG in sys.argv[1:]
  auto_fallback = AUTO_FALLBACK_FLAG in sys.argv[1:]
  cli_attempts_arg = None
  argv = []
  i = 1
  while i < len(sys.argv):
    arg = sys.argv[i]
    if arg in (APPROVAL_FLAG, AUTO_FALLBACK_FLAG):
      i += 1
      continue
    if arg.startswith(ATTEMPTS_JSON_PREFIX):
      cli_attempts_arg = arg[len(ATTEMPTS_JSON_PREFIX) :]
      i += 1
      continue
    if arg == "--attempts-json" and i + 1 < len(sys.argv):
      cli_attempts_arg = sys.argv[i + 1]
      i += 2
      continue
    argv.append(arg)
    i += 1

  if len(argv) < 2:
    print(
        "Usage: python_glue_runner.py <action_name> '<parameters_json>'"
        f" [{APPROVAL_FLAG}] [{AUTO_FALLBACK_FLAG}]"
        " [--attempts-json='<attempts_json_or_@file>']",
        file=sys.stderr,
    )
    sys.exit(1)

  action_name = argv[0]
  params_str = argv[1]

  wiki_dir = get_wiki_dir()
  os.makedirs(os.path.join(wiki_dir, "scratch"), exist_ok=True)
  # Only shadow real binaries with the test mocks when a test context is
  # explicitly active. Prepending tests/bin unconditionally routes every
  # production gcloud/kubectl call through mock_driver.py, which emits
  # [DEBUG WRAPPER] noise into captured output and -- if a stale
  # .active_test_case marker is present -- silently serves fixtures or exits
  # 127 with "Command not mocked".
  # One predicate for every test-only behaviour below: PATH shadowing, the
  # script-level fixture, and the stdout->file bridge.
  in_test_context = bool(
      os.environ.get("ACTIVE_TEST_CASE_DIR")
      or os.environ.get("WIKI_USE_TEST_MOCKS")
      or os.path.exists(os.path.join(wiki_dir, ".active_test_case"))
  )
  if in_test_context:
    tests_bin_path = os.path.abspath(os.path.join(wiki_dir, "tests", "bin"))
    os.environ["PATH"] = (
        f"{tests_bin_path}{os.pathsep}{os.environ.get('PATH', '')}"
    )

  action_path, action_name_clean = resolve_action_path(action_name, wiki_dir)
  if not os.path.exists(action_path):
    print(
        f"Error: Action or workflow file {action_path} not found.",
        file=sys.stderr,
    )
    sys.exit(1)

  with open(action_path, "r") as f:
    action_schema = json.load(f)

  if params_str.startswith("@"):
    params_path = resolve_input_path(params_str[1:], wiki_dir)
    if not params_path or not os.path.exists(params_path):
      print(
          f"Error: parameters file '{params_str[1:]}' not found.",
          file=sys.stderr,
      )
      sys.exit(1)
    with open(params_path, "r", encoding="utf-8") as pf:
      params = json.load(pf)
  else:
    try:
      params = json.loads(params_str)
    except json.JSONDecodeError as e:
      print(f"Error: Invalid parameters JSON: {e}", file=sys.stderr)
      sys.exit(1)

  if isinstance(params, list):
    params = {"batch_params": params}

  if "instruction" not in action_schema:
    print("Error: Action schema has no 'instruction' field.", file=sys.stderr)
    sys.exit(1)
  if "execution_mode" not in action_schema:
    print(
        "Error: Action schema has no 'execution_mode' field.", file=sys.stderr
    )
    sys.exit(1)

  mode = action_schema["execution_mode"]
  run_template = action_schema["instruction"]

  if mode == "manual":
    print(f"[Glue Runner] Manual Action: {action_name}", file=sys.stderr)
    print(f"Instruction:\n{run_template}")
    print_summary_report(
        action_name,
        action_schema,
        params,
        {"execution_status": "MANUAL_PENDING"},
    )
    sys.exit(0)

  # Enforce `requires_approval` here rather than trusting every caller to: an
  # agent that skips the conversational approval step would otherwise run a
  # mutating Action unchallenged. The first line matches the approval
  # convention that query_interface.py watch for.
  if (action_schema.get("requires_approval") and not approved
      and not approval_exempt(action_schema, params)):
    preview = interpolate_command(run_template, params)
    print(
        f"Awaiting user approval for mutating action '{action_name}' with"
        f" command: '{preview}'"
    )
    print(
        f"[Glue Runner] '{action_name}' declares requires_approval. Ask the"
        " operator; only after an explicit APPROVED, re-run the same command"
        f" with {APPROVAL_FLAG}.",
        file=sys.stderr,
    )
    print_summary_report(
        action_name, action_schema, params,
        {"execution_status": "APPROVAL_REQUIRED"},
    )
    sys.exit(EXIT_APPROVAL_REQUIRED)

  # Check if it's an AI action
  if mode == "ai":
    # Policy: ai-actions are agent prompt tasks and should be dispatched as
    # subagents by the Practitioner (see .claude/agents/practitioner.md and
    # SCHEMA.md Format 2). This path remains for headless/CI use, but running
    # an ai-action through the runner bypasses the documented contract, so it
    # announces itself rather than doing so silently.
    print(
        f"[Glue Runner] WARNING: '{action_name}' is an ai-action (agent prompt"
        " task). The documented path is subagent dispatch; the runner's LLM"
        " path is a headless fallback and will not honour model/batch_size the"
        " same way.",
        file=sys.stderr,
    )
    print(
        f"[Glue Runner] Executing as LLM Action: {action_name}",
        file=sys.stderr,
    )

    if "triage_results_file" in params:
      raw_triage = params.get("triage_results_file")
      raw_symptoms = params.get("symptoms_dir")
      raw_out = params.get("output_file")

      if not raw_triage or not raw_symptoms or not raw_out:
        print(
            "Error: 'triage_results_file', 'symptoms_dir', and 'output_file'"
            " parameters are required.",
            file=sys.stderr,
        )
        sys.exit(1)

      triage_results_file = resolve_input_path(raw_triage, wiki_dir)
      symptoms_dir = resolve_input_path(raw_symptoms, wiki_dir)
      output_file = resolve_output_path(raw_out, wiki_dir)

      if not os.path.exists(triage_results_file):
        print(
            f"Error: triage_results_file {triage_results_file} not found.",
            file=sys.stderr,
        )
        sys.exit(1)

      with open(triage_results_file, "r", encoding="utf-8") as tf:
        triage_content = tf.read()
      # Entries are delimited by their `[ID: n]` marker, not by blank lines: a
      # single log entry containing a stack trace has blank lines inside it,
      # and splitting on those shreds one entry into several.
      raw_entries = [
          m.group(0) for m in _TRIAGE_ENTRY_RE.finditer(triage_content)
      ] or triage_content.split("\n\n")

      matching_logs = []
      for entry in raw_entries:
        entry = entry.strip()
        if not entry:
          continue
        if entry.startswith(("[WARN]", "[ERROR]", "[FATAL]")):
          id_match = _LOG_ID_RE.search(entry)
          if id_match:
            log_id = int(id_match.group(1))
            matching_logs.append({
                "id": log_id,
                "raw": entry
            })

      symptoms_catalog = []
      if os.path.exists(symptoms_dir):
        for filename in sorted(os.listdir(symptoms_dir)):
          if filename.endswith(".md"):
            filepath = os.path.join(symptoms_dir, filename)
            try:
              parsed = parse_symptom_file(filepath)
              symptom_page = os.path.splitext(filename)[0]
              symptoms_catalog.append({
                  "symptom_page": symptom_page,
                  "title": parsed["title"] or symptom_page,
                  "description": parsed["description"],
                  "examples": parsed["examples"]
              })
            except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
              print(
                  f"Warning: Failed to parse symptom file {filepath}: {e}",
                  file=sys.stderr,
              )

      symptoms_catalog_str = json.dumps(symptoms_catalog, indent=2)

      batch_size = 30
      profile = action_schema.get("profile", DEFAULT_AI_PROFILE)
      prompt_template = action_schema.get("prompt", "")
      model = action_schema.get("model", "flash_lite")

      batches = []
      for i in range(0, len(matching_logs), batch_size):
        batches.append(matching_logs[i : i + batch_size])

      results = [None] * len(batches)

      default_workers = os.cpu_count() or 2
      max_workers = max(1, min(len(batches), default_workers))

      # Exiting from inside the `with` propagates SystemExit through
      # __exit__ -> shutdown(wait=True), which blocks on every other in-flight
      # LLM call first. Collect failures and exit after the pool is closed.
      batch_failures = []
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max_workers
      ) as executor:
        futures = {
            executor.submit(
                _execute_match_batch,
                idx,
                batch,
                total_batches=len(batches),
                prompt_template=prompt_template,
                symptoms_catalog_str=symptoms_catalog_str,
                model=model,
                profile=profile,
                wiki_dir=wiki_dir,
            ): idx
            for idx, batch in enumerate(batches)
        }
        for future in futures:
          idx = futures[future]
          try:
            results[idx] = future.result()
          except _BATCH_ERRORS as e:
            batch_failures.append((idx + 1, e))
      for num, err in batch_failures:
        print(f"Batch {num} failed: {err}", file=sys.stderr)
      if batch_failures:
        sys.exit(1)

      consolidated_matches = []
      for batch_res in results:
        if isinstance(batch_res, list):
          consolidated_matches.extend(batch_res)

      log_id_to_raw = {log["id"]: log["raw"] for log in matching_logs}
      symptom_title_map = {
          item["symptom_page"]: item["title"] for item in symptoms_catalog
      }
      s_dir_abs = os.path.abspath(symptoms_dir)

      report_lines = [
          "# Semantic Symptom Matching Report",
          "",
          f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
          "",
          "## Summary of Matches",
          "",
      ]

      if not consolidated_matches:
        report_lines.append(
            "No matching symptoms were detected with HIGH or MEDIUM confidence."
        )
      else:
        report_lines.extend([
            "| Log ID | Symptom | Confidence | Reason |",
            "| --- | --- | --- | --- |",
        ])

        cleaned_matches = []
        for match in consolidated_matches:
          try:
            log_id = int(match.get("id"))
            symptom_page = match.get("symptom_page")
            confidence = match.get("confidence", "UNKNOWN")
            reason = match.get("reason", "")
            if symptom_page:
              cleaned_matches.append((log_id, symptom_page, confidence, reason))
          except (ValueError, TypeError):
            pass

        cleaned_matches.sort(key=lambda x: x[0])
        for log_id, symptom_page, confidence, reason in cleaned_matches:
          title = symptom_title_map.get(symptom_page, symptom_page)
          page_path = os.path.join(s_dir_abs, f"{symptom_page}.md")
          link = f"[{title}](file://{page_path})"
          report_lines.append(
              f"| {log_id} | {link} | {confidence} | {reason} |"
          )

        report_lines.append("")
        report_lines.append("## Matched Log Details")
        report_lines.append("")
        for log_id, symptom_page, confidence, reason in cleaned_matches:
          raw_log = log_id_to_raw.get(log_id, "Unknown log line")
          title = symptom_title_map.get(symptom_page, symptom_page)
          page_path = os.path.join(s_dir_abs, f"{symptom_page}.md")
          report_lines.extend([
              f"### Log ID {log_id} matched {title}",
              f"- **Symptom Playbook**: [{symptom_page}](file://{page_path})",
              f"- **Confidence**: {confidence}",
              f"- **Reason**: {reason}",
              "- **Log Line**:",
              "  ```",
              f"  {raw_log}",
              "  ```",
              "",
          ])

      os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
      with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(report_lines) + "\n")

      print(
          f"Consolidated markdown report successfully written to {output_file}",
          file=sys.stderr,
      )
      print_summary_report(
          action_name,
          action_schema,
          params,
          {
              "matched_symptoms_count": len(consolidated_matches),
              "primary_severity": "ERROR" if consolidated_matches else "INFO",
              "classification_output": output_file,
              "output_file": output_file,
          },
      )
      sys.exit(0)

    elif (
        "logs_file" in params
        and action_name_clean == "ai-classify-log-severities"
    ):
      # Batch [ID: N] severity classification. Other ai- actions may also take
      # a logs_file (e.g. analyze-ici-wave) and must reach the general
      # branch below rather than this masked-log pipeline.
      # Read logs_file
      raw_logs = params.get("logs_file")
      raw_hosts = params.get("assigned_hosts_file")
      raw_mappings = params.get("mappings_file")
      raw_out = params.get("output_file")

      if not raw_logs or not raw_out:
        print(
            "Error: 'logs_file' and 'output_file' parameters are required for"
            " ai-action.",
            file=sys.stderr,
        )
        sys.exit(1)

      logs_file = resolve_input_path(raw_logs, wiki_dir)
      assigned_hosts_file = (
          resolve_input_path(raw_hosts, wiki_dir) if raw_hosts else None
      )
      mappings_file = (
          resolve_input_path(raw_mappings, wiki_dir) if raw_mappings else None
      )
      output_file = resolve_output_path(raw_out, wiki_dir)

      if not os.path.exists(logs_file):
        print(f"Error: logs_file {logs_file} not found.", file=sys.stderr)
        sys.exit(1)

      logs_content = open(logs_file, "r", encoding="utf-8").read()
      parts = _LOG_ID_SPLIT_RE.split(logs_content)
      logs_list = []
      if len(parts) > 1:
        for i in range(1, len(parts), 2):
          log_id = int(parts[i])
          log_text = parts[i + 1].rstrip("\n")
          logs_list.append({
              "id": log_id,
              "raw": f"[ID: {log_id}] {log_text}",
              "content": log_text
          })

      reverse_map = {}
      mappings_data = {}
      if mappings_file and os.path.exists(mappings_file):
        try:
          with open(mappings_file, "r", encoding="utf-8") as mf:
            mappings_data = json.load(mf)
            for placeholder, details in mappings_data.items():
              if details.get("type") == "HOST_NAME":
                reverse_map[details.get("value")] = placeholder
        except (OSError, ValueError, KeyError, TypeError) as e:
          print(
              f"Warning: could not parse mappings_file: {e}",
              file=sys.stderr,
          )

      assigned_hosts_val = ""
      if assigned_hosts_file and os.path.exists(assigned_hosts_file):
        with open(assigned_hosts_file, "r", encoding="utf-8") as ahf:
          raw_hosts = ahf.read().strip().splitlines()
        assigned_hosts_placeholders = []
        for host in raw_hosts:
          host = host.strip()
          if not host:
            continue
          if host in reverse_map:
            assigned_hosts_placeholders.append(reverse_map[host])

        if assigned_hosts_placeholders:
          assigned_hosts_val = ", ".join(assigned_hosts_placeholders)
        else:
          assigned_hosts_val = ", ".join(
              h.strip() for h in raw_hosts if h.strip()
          )

      batch_size = 100
      profile = action_schema.get("profile", DEFAULT_AI_PROFILE)
      prompt_template = action_schema.get("prompt", "")
      model = action_schema.get("model", "flash_lite")

      batches = []
      for i in range(0, len(logs_list), batch_size):
        batches.append(logs_list[i : i + batch_size])

      results = [None] * len(batches)

      default_workers = os.cpu_count() or 2

      # Never spawn more threads than tasks (and guarantee at least 1 worker
      # to prevent ValueError).
      max_workers = max(1, min(len(batches), default_workers))
      # Exiting from inside the `with` propagates SystemExit through
      # __exit__ -> shutdown(wait=True), which blocks on every other in-flight
      # LLM call first. Collect failures and exit after the pool is closed.
      batch_failures = []
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max_workers
      ) as executor:
        futures = {
            executor.submit(
                _execute_classify_batch,
                idx,
                batch,
                total_batches=len(batches),
                prompt_template=prompt_template,
                assigned_hosts_val=assigned_hosts_val,
                mappings_data=mappings_data,
                model=model,
                profile=profile,
                wiki_dir=wiki_dir,
            ): idx
            for idx, batch in enumerate(batches)
        }
        for future in futures:
          idx = futures[future]
          try:
            results[idx] = future.result()
          except _BATCH_ERRORS as e:
            batch_failures.append((idx + 1, e))
      for num, err in batch_failures:
        print(f"Batch {num} failed: {err}", file=sys.stderr)
      if batch_failures:
        sys.exit(1)

      consolidated_annotations = []
      for batch_res in results:
        consolidated_annotations.extend(batch_res)

      os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
      with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("\n\n".join(consolidated_annotations) + "\n")

      print(
          f"Consolidated annotated logs successfully written to {output_file}",
          file=sys.stderr,
      )
      print_summary_report(
          action_name,
          action_schema,
          params,
          {
              "total_found": len(logs_list),
              "deduplicated_count": len(consolidated_annotations),
              "error_logs_count": sum(
                  1
                  for line in consolidated_annotations
                  if "[ERROR]" in line or "[FATAL]" in line
              ),
              "output_file": output_file,
          },
      )
      sys.exit(0)
    else:
      # General AI action execution (e.g., ai-hw-analyzer, ai-app-analyzer,
      # ai-analyze-ici-wave)
      if params.get("logs_file") and not params.get("logs"):
        resolved_logs = resolve_input_path(params["logs_file"], wiki_dir)
        if os.path.exists(resolved_logs):
          with open(resolved_logs, "r", encoding="utf-8") as lf:
            params["logs"] = lf.read()
      prompt = action_schema.get("prompt", "")
      model = action_schema.get("model", "flash")
      profile = action_schema.get("profile", DEFAULT_AI_PROFILE)
      for p, val in params.items():
        val_str = str(val)
        prompt = prompt.replace(f"${{{p}}}", val_str).replace(f"<{p}>", val_str)

      print(
          f"[Glue Runner] Executing general AI action ({action_name}) with"
          f" model {model}...",
          file=sys.stderr,
      )
      res_text = invoke_llm_query(
          model, profile, prompt, root_dir=wiki_dir
      )
      sys.stdout.write(res_text)
      if not res_text.endswith("\n"):
        sys.stdout.write("\n")

      output_file = params.get("output_file")
      if output_file:
        out_path = resolve_output_path(output_file, wiki_dir)
        with open(out_path, "w", encoding="utf-8") as out_f:
          out_f.write(res_text)
        print(
            f"General AI action output written to {out_path}", file=sys.stderr
        )
        print_summary_report(
            action_name, action_schema, params, {"output_file": out_path}
        )
      else:
        print_summary_report(action_name, action_schema, params, {})
      sys.exit(0)

  # Parameter alias expansion
  _expand_param_aliases(params)
  alias_map = _PARAM_ALIAS_MAP

  # Pick command template
  if action_schema.get("run_command") and "python3 " in action_schema.get(
      "run_command"
  ):
    run_template = action_schema["run_command"]
  else:
    run_template = action_schema["instruction"]

  # Apply defaults declared in the action schema for any parameter the caller
  # omitted. Without this, an omitted parameter leaks its literal "${name}"
  # placeholder into the command line, producing errors that look like tool
  # bugs (e.g. "--restart_attempt: invalid int value: ''") and forcing callers
  # to restate every optional parameter on every invocation.
  schema_params = action_schema.get("parameters", {})
  declared = set()
  if isinstance(schema_params, dict):
    for pname, pspec in schema_params.items():
      declared.add(pname)
      if pname not in params and isinstance(pspec, dict) and "default" in pspec:
        params[pname] = pspec["default"]
  elif isinstance(schema_params, list):
    for pspec in schema_params:
      if isinstance(pspec, dict) and pspec.get("name"):
        declared.add(pspec["name"])
        if pspec["name"] not in params and "default" in pspec:
          params[pspec["name"]] = pspec["default"]

  circuit_breaker_enabled = (
      not in_test_context
      or os.environ.get("WIKI_ENABLE_CIRCUIT_BREAKER") == "1"
  )

  # Fast-fail if Cloud Logging ACCESS_DENIED circuit breaker is tripped for this
  # (project_id, auth_fingerprint).
  if circuit_breaker_enabled:
    cb_entry = check_access_denied_circuit_breaker(
        action_name_clean, params, wiki_dir
    )
    if cb_entry is not None:
      recommended_fallbacks = evaluate_action_fallbacks(
          action_schema, params, EXIT_ACCESS_DENIED, "", "", {}
      )
      cb_payload = {
          "status": "ACCESS_DENIED",
          "circuit_breaker": True,
          "project_id": cb_entry.get(
              "project_id", _extract_project_key(params)
          ),
          "skipped_action": action_name_clean,
          "cached_from_action": cb_entry.get("cached_from_action", ""),
          "message": (
              "Cloud Logging / GCP API access is denied for this project"
              " (cached from prior exit 77). Skipping redundant query; use"
              ""
              " refresh credentials via gcloud-auth-login / gcloud-auth-login /"
              " gcloud-login."
          ),
          **(
              {"recommended_fallbacks": recommended_fallbacks}
              if recommended_fallbacks
              else {}
          ),
      }
      print(json.dumps(cb_payload))
      print(
          f"[Glue Runner] CIRCUIT BREAKER (exit {EXIT_ACCESS_DENIED}):"
          f" Skipping '{action_name_clean}' because Cloud Logging access for"
          f" project '{cb_payload['project_id']}' already failed with"
          f" ACCESS_DENIED during '{cb_payload['cached_from_action']}'.",
          file=sys.stderr,
      )
      if recommended_fallbacks:
        print(
            f"[Glue Runner] Recommended fallbacks for '{action_name_clean}':"
            f" {json.dumps(recommended_fallbacks)}",
            file=sys.stderr,
        )
      cb_metrics = {"execution_status": "ACCESS_DENIED_CIRCUIT_BREAKER"}
      if recommended_fallbacks:
        cb_metrics["recommended_fallbacks"] = [
            fb["action"] for fb in recommended_fallbacks
        ]
      print_summary_report(
          action_name,
          action_schema,
          params,
          cb_metrics,
      )
      sys.exit(EXIT_ACCESS_DENIED)

  # Check for parallel per-attempt / batch execution mode
  batch_items = None
  if cli_attempts_arg:
    raw_attempts_str = cli_attempts_arg.strip()
    if raw_attempts_str.startswith("@"):
      att_path = resolve_input_path(raw_attempts_str[1:], wiki_dir)
      if not att_path or not os.path.exists(att_path):
        print(
            f"Error: attempts file '{raw_attempts_str[1:]}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)
      with open(att_path, "r", encoding="utf-8") as af:
        loaded_att = json.load(af)
    else:
      try:
        loaded_att = json.loads(raw_attempts_str)
      except json.JSONDecodeError as e:
        print(f"Error: Invalid --attempts-json: {e}", file=sys.stderr)
        sys.exit(1)
    if isinstance(loaded_att, dict) and isinstance(
        loaded_att.get("attempts"), list
    ):
      batch_items = loaded_att["attempts"]
    elif isinstance(loaded_att, list):
      batch_items = loaded_att
    else:
      print(
          "Error: --attempts-json must be a JSON list or an object with an"
          " 'attempts' list.",
          file=sys.stderr,
      )
      sys.exit(1)
  elif isinstance(params.get("batch_params"), list):
    batch_items = params.pop("batch_params")
  elif isinstance(params.get("attempts"), list) and "attempts" not in declared:
    batch_items = params.pop("attempts")
  elif (
      isinstance(params.get("host_id"), str)
      and "," in params["host_id"]
  ):
    raw_machines = params.pop("host_id")
    batch_items = [
        {"host_id": m.strip()}
        for m in raw_machines.split(",")
        if m.strip()
    ]

  if batch_items is not None:
    if not batch_items:
      print(
          json.dumps({
              "action": action_name_clean,
              "parallel_count": 0,
              "results": [],
          })
      )
      sys.exit(0)

    timeout_seconds = action_schema.get(
        "timeout_seconds", DEFAULT_ACTION_TIMEOUT_SECONDS
    )
    max_workers = max(
        1,
        min(
            len(batch_items),
            int(params.get("max_concurrent_workers", 8) or 8),
        ),
    )

    worker_fn = functools.partial(
        _run_single_attempt,
        base_params=params,
        alias_map=alias_map,
        declared=declared,
        action_name_clean=action_name_clean,
        run_template=run_template,
        total_items=len(batch_items),
        wiki_dir=wiki_dir,
        timeout_seconds=timeout_seconds,
        in_test_context=in_test_context,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      batch_results = list(executor.map(worker_fn, enumerate(batch_items)))

    any_access_denied = any(
        r["returncode"] == EXIT_ACCESS_DENIED for r in batch_results
    )
    if any_access_denied and circuit_breaker_enabled:
      first_denied = next(
          r for r in batch_results if r["returncode"] == EXIT_ACCESS_DENIED
      )
      record_access_denied_circuit_breaker(
          action_name_clean,
          params,
          wiki_dir,
          reason=first_denied.get("stderr") or str(first_denied.get("output")),
      )

    combined_payload = {
        "action": action_name_clean,
        "parallel_count": len(batch_results),
        "status": "ACCESS_DENIED" if any_access_denied else "COMPLETE",
        "results": batch_results,
    }
    combined_out_file = params.get(
        "output_file", f"scratch/{action_name_clean}_parallel_output.json"
    )
    resolved_combined = resolve_output_path(combined_out_file, wiki_dir)
    with open(resolved_combined, "w", encoding="utf-8") as cf:
      json.dump(combined_payload, cf, indent=2)

    print(json.dumps(combined_payload))
    print_summary_report(
        action_name,
        action_schema,
        params,
        {
            "execution_status": (
                "ACCESS_DENIED" if any_access_denied else "SUCCESS"
            ),
            "output_file": resolved_combined,
        },
    )
    if any_access_denied:
      sys.exit(EXIT_ACCESS_DENIED)
    failed_rcs = [
        r["returncode"] for r in batch_results if r["returncode"] != 0
    ]
    sys.exit(failed_rcs[0] if failed_rcs else 0)

  interpolated_cmd = run_template

  # Auto-fill defaults for common missing parameters if referenced in template
  if "output_file" not in params:
    # Normalized name: action_name may be the "[[x]]" or "x.json" invocation
    # form, which would otherwise appear verbatim in the output path.
    params["output_file"] = f"scratch/{action_name_clean}_output.json"
  elif isinstance(params.get("output_file"), str) and params[
      "output_file"
  ].startswith("_wiki/"):
    params["output_file"] = params["output_file"].removeprefix("_wiki/")
  if "max_concurrent_workers" not in params:
    params["max_concurrent_workers"] = 4
  if "disruption_timestamp" not in params:
    params["disruption_timestamp"] = params.get(
        "start_timestamp", params.get("start", "")
    )

  # Any declared parameter still absent resolves to empty rather than leaking
  # its placeholder text into the shell.
  for pname in declared:
    params.setdefault(pname, "")

  # Replace ${var}, <var>, and {{var}}, escaping each value for the shell
  # context it lands in. See escape_for_context.
  interpolated_cmd = interpolate_command(interpolated_cmd, params)
  interpolated_cmd = re.sub(r"\s+AND\s+\(\s*\)", "", interpolated_cmd)

  # Normalize command paths for execution inside wiki_dir
  # (e.g. "python3 _wiki/tools/scripts/..." -> "python3 tools/scripts/...")
  normalized_cmd = _PYTHON_WIKI_PREFIX_RE.sub("python3 ", interpolated_cmd)
  print(f"[Glue Runner] Executing command: {normalized_cmd}", file=sys.stderr)

  # Serve a script-level fixture if this test case declares one. Gated on the
  # same test-context signal as the PATH shadowing above, so production runs
  # never take this branch.
  if in_test_context:
    mock_path = script_level_mock(normalized_cmd, wiki_dir)
    if mock_path:
      with open(mock_path, encoding="utf-8") as mf:
        payload = mf.read()
      print(
          f"[Glue Runner] Served script-level mock {mock_path}", file=sys.stderr
      )
      out_target = params.get("output_file")
      if out_target:
        out_path = resolve_output_path(out_target, wiki_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as of:
          of.write(payload)
      sys.stdout.write(payload)
      mock_metrics = {
          "_stdout_lines": len(
              [line for line in payload.splitlines() if line.strip()]
          )
      }
      try:
        parsed_payload = json.loads(payload.strip())
        if isinstance(parsed_payload, dict):
          for k, v in parsed_payload.items():
            mock_metrics.setdefault(k, v)
      except (ValueError, TypeError):
        pass
      for k, v in parse_printed_summary(payload).items():
        mock_metrics.setdefault(k, v)
      print_summary_report(
          action_name,
          action_schema,
          params,
          custom_metrics=mock_metrics,
      )
      sys.exit(0)

  # Hard timeout. The runner captures output to completion, so any action that
  # streams indefinitely (`kubectl -w`, `logs -f`, `tail -f`) would otherwise
  # hang the whole session with no way out. Actions that legitimately run long
  # can raise this via a `timeout_seconds` field in their schema.
  timeout_seconds = action_schema.get(
      "timeout_seconds", DEFAULT_ACTION_TIMEOUT_SECONDS
  )
  stdout, stderr, returncode, timed_out = run_command_bounded(
      normalized_cmd, wiki_dir, timeout_seconds
  )
  if timed_out:
    returncode = 124
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    print(
        f"\n[Glue Runner] ERROR: action '{action_name}' exceeded its "
        f"{timeout_seconds}s timeout and was terminated. If this action is "
        "expected to stream (e.g. `kubectl -w`), bound it in its own command; "
        'if it is legitimately slow, set "timeout_seconds" in its schema.',
        file=sys.stderr,
    )
    recommended_fallbacks = evaluate_action_fallbacks(
        action_schema, params, returncode, stdout, stderr, {}
    )
    if recommended_fallbacks:
      print(
          f"[Glue Runner] Recommended fallbacks for '{action_name_clean}':"
          f" {json.dumps(recommended_fallbacks)}",
          file=sys.stderr,
      )
  else:
    if is_access_denied_output(returncode, stdout, stderr) and (
        action_name_clean in CLOUD_LOGGING_ACTIONS
        or returncode == EXIT_ACCESS_DENIED
    ):
      returncode = EXIT_ACCESS_DENIED
      if circuit_breaker_enabled:
        record_access_denied_circuit_breaker(
            action_name_clean,
            params,
            wiki_dir,
            reason=stderr or stdout,
        )
    elif returncode == 0 and circuit_breaker_enabled:
      if action_name_clean in AUTH_REFRESH_ACTIONS:
        clear_access_denied_circuit_breaker(params, wiki_dir, clear_all=True)
      elif action_name_clean in CLOUD_LOGGING_ACTIONS:
        clear_access_denied_circuit_breaker(params, wiki_dir, clear_all=False)

    # Bridge stdout to the expected output file when a mocked command printed
    # its payload instead of writing it. Gated on the same test-context signal
    # as the PATH shadowing and the script-level fixture above: `output_file`
    # is auto-filled for every action, so ungated this writes a file for every
    # production run and can mask a script that genuinely failed to produce one.
    if (
        in_test_context
        and "output_file" in params
        and params["output_file"]
        and stdout
        and returncode == 0
    ):
      out_path = resolve_output_path(params["output_file"], wiki_dir)
      if not os.path.exists(out_path):
        with open(out_path, "w", encoding="utf-8") as out_f:
          out_f.write(stdout)
        print(
            "[Glue Runner Mock Fallback] Intercepted STDOUT bridged to missing"
            f" file {out_path}",
            file=sys.stderr,
        )

    sys.stdout.write(stdout)
    sys.stderr.write(stderr)

    metrics: dict[str, Any] = {}
    if returncode == 0:
      out_lines_mem = 0
      if stdout:
        out_lines_mem = len(
            [line for line in stdout.splitlines() if line.strip()]
        )
      metrics = {"_stdout_lines": out_lines_mem}
      # Prefer the action's own structured output. Scripts that print a JSON
      # object are reporting their summary_fields directly; without this the
      # report falls through to name-based heuristics and prints "N/A" for
      # fields the script just emitted.
      if stdout:
        try:
          parsed = json.loads(stdout.strip())
          if isinstance(parsed, dict):
            for k, v in parsed.items():
              metrics.setdefault(k, v)
        except (ValueError, TypeError):
          pass
        # Some scripts report their metrics as a printed block rather than JSON.
        # Without this the report falls back to counting lines in the output
        # file, which yields a number that contradicts the script's own summary
        # printed a few lines below it -- and, if a previous run left the file
        # behind, a number from that earlier run entirely.
        for k, v in parse_printed_summary(stdout).items():
          metrics.setdefault(k, v)

    recommended_fallbacks = evaluate_action_fallbacks(
        action_schema, params, returncode, stdout, stderr, metrics
    )
    if recommended_fallbacks:
      metrics["recommended_fallbacks"] = [
          fb["action"] for fb in recommended_fallbacks
      ]
      print(
          f"[Glue Runner] Recommended fallbacks for '{action_name_clean}':"
          f" {json.dumps(recommended_fallbacks)}",
          file=sys.stderr,
      )

    if returncode == 0:
      print_summary_report(
          action_name, action_schema, params, custom_metrics=metrics
      )

  if (
      auto_fallback
      and recommended_fallbacks
      and os.environ.get("WIKI_FALLBACK_DEPTH", "0") == "0"
  ):
    for fb in recommended_fallbacks:
      if not fb.get("auto_execute"):
        continue
      fb_path, fb_name = resolve_action_path(fb["action"], wiki_dir)
      if not os.path.exists(fb_path):
        continue
      try:
        with open(fb_path, "r", encoding="utf-8") as fbf:
          fb_schema = json.load(fbf)
      except (OSError, ValueError):
        continue
      if fb_schema.get("execution_mode") != "command":
        continue
      fb_params = fb.get("resolved_params", {})
      if fb_schema.get("requires_approval") and not approval_exempt(
          fb_schema, fb_params
      ):
        continue
      print(
          f"[Glue Runner] Auto-executing fallback action '{fb_name}'"
          f" (triggers={fb['when']})...",
          file=sys.stderr,
      )
      fb_env = dict(os.environ, WIKI_FALLBACK_DEPTH="1")
      fb_proc = subprocess.run(
          [
              sys.executable,
              os.path.abspath(__file__),
              fb_name,
              json.dumps(fb_params),
          ],
          capture_output=True,
          text=True,
          env=fb_env,
          cwd=wiki_dir,
          check=False,
      )
      if fb_proc.stdout:
        sys.stdout.write(fb_proc.stdout)
      if fb_proc.stderr:
        sys.stderr.write(fb_proc.stderr)
      sys.exit(fb_proc.returncode)

  sys.exit(returncode)

if __name__ == "__main__":
  try:
    main()
  except agent_dispatch.DispatchUnavailable as unavailable:
    # Distinct exit code so a caller can tell "no backend installed" apart from
    # "the action ran and failed" (1) and "the action timed out" (124).
    print(f"[Glue Runner] {unavailable}", file=sys.stderr)
    sys.exit(3)

#!/usr/bin/env python3
"""Harness-neutral dispatch for one-shot agent/LLM calls.

The wiki runs under two harnesses -- Claude Code and Antigravity -- and three
places need to ask a model a question and get text back: the glue runner's
`ai`-mode fallback and the interactive query driver.
Neither harness is guaranteed present: a Claude Code machine has no `agentapi`
and no Antigravity SDK, an Antigravity machine has no `claude` CLI. Left to
each caller, that produces three different cascades and three different ways
of giving up.

This module is the single place that knows how to reach a model:

    get_dispatch().invoke(prompt, model, profile) -> str

with three implementations and honest failure:

    AntigravityDispatch  - google.antigravity SDK, falling back to `agentapi`
    ClaudeDispatch       - `claude -p --output-format json`
    NullDispatch         - raises DispatchUnavailable naming what is missing

It also owns the cross-harness vocabulary (`MODEL_MAP`, `PROFILE_MAP`) and the
model-alias sanitizing that both the dispatchers and `_wiki/tests/
agent_backends.py` need, so the two harnesses cannot drift apart on what "pro"
means.

Backend selection, in order: an explicit name, `WIKI_AGENT_BACKEND`
(`antigravity` | `claude` | `none`, also how tests pin a backend), the harness
the caller is running under (`host_harness()`), then detection. A session
started from Claude Code therefore dispatches to Claude everywhere, even on a
machine that also has Antigravity installed.

Run `python3 _wiki/tools/scripts/agent_dispatch.py` to see what is reachable.
"""

import asyncio
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

# Antigravity model names -> Claude Code model names. Antigravity's names are
# the canonical spelling and this table translates outward.
MODEL_MAP: Dict[str, str] = {
    "flash_lite": "haiku",
    "flash": "sonnet",
    "pro": "opus",
}

# Antigravity profile names -> Claude Code agent names
# (`.claude/agents/<name>.md`). Keep in step with the profiles that exist in
# both harnesses.
PROFILE_MAP: Dict[str, str] = {
    "wiki_practitioner": "practitioner",
    "wiki_router": "router",
    "wiki_maintainer": "maintainer",
    "wiki_reviewer": "reviewer",
    "wiki_advisor": "advisor",
}

# Claude Code exports the *parent* session's model aliases to child processes,
# and an interactive session may be pinned to a bracketed context variant such
# as `claude-sonnet-4-5@20250929[1m]`. A child `claude -p` inherits that, sends
# the corresponding context-1m beta header, and gets a hard 400 back. The suffix
# is a launcher-level convenience, not part of any model id the API accepts, so
# strip it rather than hardcoding ids (which differ per environment).
_MODEL_ALIAS_VARS = (
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
)
_BRACKET_SUFFIX_RE = re.compile(r"\[[^\]]*\]\s*$")


def user_settings_env() -> Dict[str, str]:
  """The `env` block of the user-level settings file, or {}.

  Claude Code layers settings, and a value here overrides the inherited process
  environment -- so sanitizing `os.environ` alone does not dislodge an alias
  pinned in this file.

  Returns:
    The user-level env mapping, or an empty dictionary if missing or invalid.
  """
  path = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
  try:
    with open(path, encoding="utf-8") as f:
      return json.load(f).get("env") or {}
  except (OSError, ValueError):
    return {}


def strip_context_suffix(value: Optional[str]) -> str:
  """`claude-sonnet-4-5@20250929[1m]` -> `claude-sonnet-4-5@20250929`."""
  return _BRACKET_SUFFIX_RE.sub("", value or "").strip()


def sanitized_model_aliases(
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
  """Alias var -> unbracketed id, for every alias that carries a suffix.

  Considers both the process environment and the user settings file, so the
  result is what actually needs overriding regardless of which layer set it.

  Args:
    env: Environment to inspect. Defaults to `os.environ`.

  Returns:
    Mapping of variable name to unbracketed model identifier.
  """
  base = dict(os.environ if env is None else env)
  base.update(user_settings_env())
  fixed = {}
  for var in _MODEL_ALIAS_VARS:
    val = base.get(var)
    if not val:
      continue
    stripped = strip_context_suffix(val)
    if stripped and stripped != val:
      fixed[var] = stripped
  return fixed


def sanitize_model_env(
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
  """Copy of `env` with bracketed context variants stripped from model aliases.

  Returns the copy; never mutates the input. Silently a no-op when no alias
  carries a suffix, which is the common case outside a pinned session.

  Args:
    env: Environment to copy and sanitize. Defaults to `os.environ`.

  Returns:
    Sanitized environment dictionary copy.
  """
  out = dict(os.environ if env is None else env)
  out.update(sanitized_model_aliases(env))
  return out


# Variables that tie a process to the Claude Code session that spawned it. A
# detached dashboard daemon outlives that session, so neither it nor its
# `claude -p` children should inherit them. `CLAUDECODE` stays: it is how
# host_harness() recognises a Claude Code launch.
CLAUDE_SESSION_ENV_VARS = (
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
)


def without_session_env(env: Dict[str, str]) -> Dict[str, str]:
  """Copy of `env` minus CLAUDE_SESSION_ENV_VARS."""
  return {k: v for k, v in env.items() if k not in CLAUDE_SESSION_ENV_VARS}


def run_process_group(
    cmd: Sequence[str],
    *,
    input_data: Optional[str] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
  """subprocess.run() that kills the whole process group on timeout.

  `claude -p` runs kubectl/gcloud through its shell tool; killing only the
  CLI would leave those running after the caller has given up on them.

  Args:
    cmd: Command and arguments to execute.
    input_data: Text written to the child's stdin.
    cwd: Working directory for the child.
    env: Environment for the child; inherits the caller's when None.
    timeout: Seconds to wait before killing the process group.
    **kwargs: Accepts legacy 'input' keyword argument.

  Returns:
    The completed process with text stdout and stderr.

  Raises:
    subprocess.TimeoutExpired: The process group outlived `timeout`.
  """
  if "input" in kwargs:
    input_data = kwargs.pop("input")
  proc = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=cwd,
      env=env,
      start_new_session=True,
  )
  try:
    out, err = proc.communicate(input=input_data, timeout=timeout)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, OSError):
      proc.kill()
    proc.communicate()
    raise
  return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


class DispatchUnavailableError(RuntimeError):
  """No backend can reach a model from here.

  Carries the per-backend reasons so the caller can say *what* is missing
  rather than surfacing whichever low-level error happened to escape first.
  """


DispatchUnavailable = DispatchUnavailableError


# ---------------------------------------------------------------------------
# Antigravity transcript polling
# ---------------------------------------------------------------------------


def wait_for_conversation(
    conv_id: str, timeout_seconds: int = 120, poll_interval: float = 1
) -> str:
  """Blocks until an Antigravity conversation settles, returning its final text.

  `agentapi new-conversation` returns as soon as the conversation *starts*, so
  the only way to read the answer is to poll the transcript the daemon writes.

  Args:
    conv_id: The conversation ID to poll.
    timeout_seconds: Maximum seconds to wait before timing out.
    poll_interval: Seconds between poll attempts.

  Returns:
    The final text content from the conversation.

  Raises:
    RuntimeError: If the subagent crashed or reported an error.
    TimeoutError: If the conversation did not settle within timeout_seconds.
  """
  home = os.path.expanduser("~")
  transcript_base = os.environ.get(
      "AGENT_BRAIN_DIR",
      os.path.join(home, ".gemini", "brain"),
  )
  transcript_path = os.path.join(
      transcript_base,
      conv_id,
      ".system_generated",
      "logs",
      "transcript.jsonl",
  )

  start_time = time.time()
  while time.time() - start_time < timeout_seconds:
    time.sleep(poll_interval)
    if not os.path.exists(transcript_path):
      continue

    try:
      with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
      if not lines:
        continue

      planner_steps = []
      for line in lines:
        line = line.strip()
        if not line:
          continue
        try:
          step = json.loads(line)
          if step.get("type") == "ERROR_MESSAGE" or (
              step.get("status") == "ERROR" and step.get("error")
          ):
            raise RuntimeError(
                f"Subagent crashed: {step.get('error') or 'Unknown error'}"
            )
          if step.get("type") == "PLANNER_RESPONSE":
            planner_steps.append(step)
        except json.JSONDecodeError:
          pass

      if not planner_steps:
        continue

      last_planner = planner_steps[-1]
      if (
          last_planner.get("source") == "MODEL"
          and last_planner.get("status") == "DONE"
          and not last_planner.get("tool_calls")
      ):
        return last_planner.get("content", "")
    except (OSError, ValueError, TypeError, KeyError):
      continue

  raise TimeoutError(
      f"Subagent conversation {conv_id} timed out after {timeout_seconds}s"
  )


# ---------------------------------------------------------------------------
# Antigravity SDK
# ---------------------------------------------------------------------------


def get_agent_json_path(
    root_dir: Optional[str], profile_name: Optional[str]
) -> Optional[str]:
  """Locates the agent.json configuration file for a given profile."""
  if not root_dir or not profile_name:
    return None
  candidates = [
      os.path.join(root_dir, "_agents", "agents", profile_name, "agent.json"),
      os.path.join(
          os.path.dirname(os.path.abspath(root_dir)),
          "_agents",
          "agents",
          profile_name,
          "agent.json",
      ),
  ]
  for c in candidates:
    if os.path.exists(c):
      return c
  return None


def build_local_agent_config(
    agent_json_path: Optional[str], model: str, **kwargs: Any
) -> Any:
  """Loads a custom agent.json into a declarative LocalAgentConfig object."""
  try:
    ag_pkg = importlib.import_module("google.antigravity")
    local_agent_config_cls = ag_pkg.LocalAgentConfig
    types = ag_pkg.types
  except (ImportError, AttributeError):
    if "LocalAgentConfig" in kwargs and "types_mod" in kwargs:
      local_agent_config_cls = kwargs.pop("LocalAgentConfig")
      types = kwargs.pop("types_mod")
    else:
      return None

  system_instructions = None
  capabilities = None
  if agent_json_path and os.path.exists(agent_json_path):
    try:
      with open(agent_json_path, "r", encoding="utf-8") as f:
        agent_data = json.load(f)
      custom_agent = agent_data.get("config", {}).get("customAgent", {})

      # 1. Map System Instructions
      sections = []
      for sec in custom_agent.get("systemPromptSections", []):
        sections.append(
            types.SystemInstructionSection(
                title=sec.get("title", ""), content=sec.get("content", "")
            )
        )
      if sections:
        system_instructions = types.TemplatedSystemInstructions(
            sections=sections
        )

      # 2. Map Tools (SDK Enums)
      sdk_tool_mapping = {
          "invoke_subagent": types.BuiltinTools.START_SUBAGENT,
          "start_subagent": types.BuiltinTools.START_SUBAGENT,
          "view_file": types.BuiltinTools.VIEW_FILE,
          "list_dir": types.BuiltinTools.LIST_DIR,
          "grep_search": types.BuiltinTools.SEARCH_DIR,
          "find_by_name": types.BuiltinTools.FIND_FILE,
          "write_to_file": types.BuiltinTools.CREATE_FILE,
          "create_file": types.BuiltinTools.CREATE_FILE,
          "replace_file_content": types.BuiltinTools.EDIT_FILE,
          "multi_replace_file_content": types.BuiltinTools.EDIT_FILE,
          "edit_file": types.BuiltinTools.EDIT_FILE,
          "read_url_content": types.BuiltinTools.READ_URL_CONTENT,
          "search_web": types.BuiltinTools.SEARCH_WEB,
          "run_command": types.BuiltinTools.RUN_COMMAND,
          "ask_question": types.BuiltinTools.ASK_QUESTION,
          "generate_image": types.BuiltinTools.GENERATE_IMAGE,
          "finish": types.BuiltinTools.FINISH,
      }
      enabled_tools = []
      for tool_name in custom_agent.get("toolNames", []):
        if tool_name in sdk_tool_mapping:
          mapped_tool = sdk_tool_mapping[tool_name]
          if mapped_tool not in enabled_tools:
            enabled_tools.append(mapped_tool)
      if enabled_tools:
        capabilities = types.CapabilitiesConfig(enabled_tools=enabled_tools)
    except (OSError, ValueError, TypeError, KeyError) as e:
      print(
          "Warning: Failed to parse agent configuration from"
          f" {agent_json_path}: {e}",
          file=sys.stderr,
      )

  config_kwargs = {"model": model}
  if system_instructions is not None:
    config_kwargs["system_instructions"] = system_instructions
  if capabilities is not None:
    config_kwargs["capabilities"] = capabilities
  config_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
  return local_agent_config_cls(**config_kwargs)


def execute_sdk_conversation(
    config: Any, prompt: str, agent_cls: Optional[Any] = None
) -> str:
  """Runs an Antigravity SDK conversation, validating steps for crashes."""
  if agent_cls is None:
    try:
      ag_pkg = importlib.import_module("google.antigravity")
      agent_cls = ag_pkg.Agent
    except (ImportError, AttributeError) as err:
      raise DispatchUnavailable("google.antigravity is not available") from err

  async def _run():
    async with agent_cls(config) as agent:
      resp = await agent.chat(prompt)
      text = await resp.text()
      conv = getattr(agent, "conversation", None)
      conv_id = (
          conv.conversation_id
          if conv and hasattr(conv, "conversation_id")
          else None
      )
      if not conv_id:
        print(
            "Error: Could not obtain conversation_id from agent.conversation",
            file=sys.stderr,
        )
      trajectory = (
          list(conv.history) if conv and hasattr(conv, "history") else []
      )
      if not trajectory:
        print(
            "Error: Could not obtain step trajectory from agent.conversation"
            f" (conv_id={conv_id})",
            file=sys.stderr,
        )
      return text, conv_id, trajectory

  text, conv_id, trajectory = asyncio.run(_run())

  for step in trajectory:
    status = (
        step["status"]
        if isinstance(step, dict)
        else getattr(step, "status", "")
    )
    error = (
        step["error"] if isinstance(step, dict) else getattr(step, "error", "")
    )
    if str(status) == "ERROR" and error:
      raise RuntimeError(
          f"SDK Agent conversation {conv_id} failed with error: {error}"
      )

  return text


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------


class Dispatch:
  """One-shot "ask a model, get text back" interface."""

  name: str = "abstract"

  def available(self) -> bool:
    raise NotImplementedError

  def unavailable_reason(self) -> str:
    return "not available"

  def invoke(
      self,
      prompt: str,
      model: str,
      profile: Optional[str] = None,
      root_dir: Optional[str] = None,
      timeout: Optional[float] = None,
  ) -> str:
    raise NotImplementedError


class AntigravityDispatch(Dispatch):
  """google.antigravity SDK, falling back to the `agentapi` CLI."""

  name: str = "antigravity"

  def sdk_importable(self) -> bool:
    """True when the google.antigravity SDK can be imported."""
    try:
      importlib.import_module("google.antigravity")
      return True
    except ImportError:
      return False

  def available(self) -> bool:
    return self.sdk_importable() or shutil.which("agentapi") is not None

  def unavailable_reason(self) -> str:
    return (
        "neither the google.antigravity SDK nor the `agentapi` binary is "
        "present; Antigravity dispatch needs one of them (see "
        "_agents/QUICKSTART.md)"
    )

  def invoke(
      self,
      prompt: str,
      model: str,
      profile: Optional[str] = None,
      root_dir: Optional[str] = None,
      timeout: Optional[float] = None,
  ) -> str:
    # The SDK is preferred when importable; `USE_ANTIGRAVITY_SDK=true` makes it
    # mandatory, so an SDK failure raises instead of silently degrading to the
    # CLI and a different code path.
    sdk_required = os.environ.get("USE_ANTIGRAVITY_SDK", "").lower() in (
        "true",
        "1",
    )
    if sdk_required or self.sdk_importable():
      try:
        return self._invoke_sdk(prompt, model, profile, root_dir)
      except (OSError, RuntimeError, ValueError, TypeError, KeyError) as e:
        if sdk_required:
          raise RuntimeError(f"SDK LLM execution failed: {e}") from e
        print(
            f"Warning: SDK LLM query failed ({e}), attempting fallback to"
            " agentapi CLI...",
            file=sys.stderr,
        )
    return self._invoke_cli(prompt, model, profile)

  def _invoke_sdk(
      self,
      prompt: str,
      model: str,
      profile: Optional[str],
      root_dir: Optional[str],
  ) -> str:
    """Invokes Antigravity conversation execution via the Python SDK."""
    try:
      policy = importlib.import_module("google.antigravity.hooks.policy")
      policies = [policy.allow_all()]
    except (ImportError, AttributeError):
      policies = []

    target_dir = root_dir or os.environ.get("WIKI_DIR", ".")
    if target_dir == "." and not os.path.exists(os.path.join(".", "_agents")):
      print(
          "Warning: WIKI_DIR environment variable not set or invalid for"
          " declarative config lookup.",
          file=sys.stderr,
      )
    agent_json = get_agent_json_path(target_dir, profile)
    config = build_local_agent_config(
        agent_json, model, policies=policies
    )
    if config is None:
      raise RuntimeError("could not build a LocalAgentConfig")
    return execute_sdk_conversation(config, prompt)

  def _invoke_cli(self, prompt: str, model: str, profile: Optional[str]) -> str:
    """Invokes Antigravity conversation execution via agentapi CLI."""
    cmd = ["agentapi", "new-conversation", "--model", model]
    if profile:
      cmd += ["--profile", profile]
    cmd.append(prompt)
    # Let the exec fail and translate it, rather than pre-checking PATH: one
    # lookup instead of two, no TOCTOU window, and the mockable boundary stays
    # where callers expect it (`subprocess.run`).
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
      raise DispatchUnavailable(self.unavailable_reason()) from e
    try:
      output_json = json.loads(res.stdout)
      conv_id = output_json["response"]["newConversation"]["conversationId"]
    except (json.JSONDecodeError, KeyError) as e:
      raise RuntimeError(
          f"Error parsing agentapi response: {e}. Output: {res.stdout}"
      ) from e
    return wait_for_conversation(conv_id)


class ClaudeDispatch(Dispatch):
  """`claude -p --output-format json`, one prompt in, one answer out."""

  name: str = "claude"

  # Headless `claude -p` has nobody to ask for permission, so a prompt task
  # that tries to use a gated tool stalls rather than failing. These calls are
  # pure analysis over text supplied in the prompt, so no tools are needed at
  # all -- but the harness's own agents are read-only by definition, and
  # denying everything would break a profile that reads a wiki page to ground
  # its answer. Bypassing permissions is scoped to this non-interactive,
  # analysis-only path.
  PERMISSION_MODE: str = "bypassPermissions"

  def available(self) -> bool:
    return shutil.which("claude") is not None

  def unavailable_reason(self) -> str:
    return "the `claude` CLI was not found on PATH"

  def invoke(
      self,
      prompt: str,
      model: str,
      profile: Optional[str] = None,
      root_dir: Optional[str] = None,
      timeout: Optional[float] = None,
  ) -> str:
    if not self.available():
      raise DispatchUnavailable(self.unavailable_reason())
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        MODEL_MAP.get(model, model),
        "--permission-mode",
        self.PERMISSION_MODE,
    ]
    agent = PROFILE_MAP.get(profile, profile)
    if agent:
      cmd += ["--agent", agent]
    try:
      res = run_process_group(
          cmd,
          input=prompt,
          cwd=root_dir or None,
          env=without_session_env(sanitize_model_env()),
          timeout=timeout,
      )
    except subprocess.TimeoutExpired as e:
      raise RuntimeError(f"claude did not finish within {timeout}s") from e
    if res.returncode != 0:
      stderr_snippet = (res.stderr or "").strip()[:500]
      stdout_snippet = (res.stdout or "").strip()[:500]
      raise RuntimeError(
          f"claude exited with code {res.returncode}. "
          f"stderr: {stderr_snippet} stdout: {stdout_snippet}"
      )
    try:
      payload = json.loads(res.stdout or "")
    except (json.JSONDecodeError, TypeError) as e:
      raise RuntimeError(
          f"Error parsing claude response: {e} (returncode={res.returncode}). "
          f"stdout: {(res.stdout or '')[:300]} "
          f"stderr: {(res.stderr or '')[:300]}"
      ) from e
    if not isinstance(payload, dict):
      raise RuntimeError(f"Unexpected claude response shape: {type(payload)}")
    if payload.get("is_error") or payload.get("api_error_status"):
      raise RuntimeError(
          f"claude returned an error (api_error_status="
          f"{payload.get('api_error_status')}): "
          f"{str(payload.get('result'))[:300]}"
      )
    return payload.get("result") or ""


class NullDispatch(Dispatch):
  """No backend reachable. Fails with the reason, not a stack trace.

  Reaching this is the *normal* state for an `ai-` Action under Claude Code:
  those are agent prompt tasks the Practitioner dispatches as subagents, and
  the glue runner's LLM path is only a headless fallback. The error therefore
  names the supported path rather than the missing binary -- an exec failure
  reads as a broken tool, when the real answer is that the caller is on the
  wrong execution path.
  """

  name: str = "none"

  def __init__(self, reasons: Sequence[str] = ()) -> None:
    self.reasons: List[str] = list(reasons)

  def available(self) -> bool:
    return False

  def unavailable_reason(self) -> str:
    detail = "; ".join(self.reasons) if self.reasons else "no backend found"
    return f"no agent backend is reachable from here ({detail})"

  def invoke(
      self,
      prompt: str,
      model: str,
      profile: Optional[str] = None,
      root_dir: Optional[str] = None,
      timeout: Optional[float] = None,
  ) -> str:
    raise DispatchUnavailable(
        f"{self.unavailable_reason()}. `ai-` Actions are agent prompt tasks: "
        "dispatch them as a subagent from the Practitioner (Claude Code: the "
        "Agent tool; Antigravity: invoke_subagent) rather than through the "
        "glue runner. See CLAUDE.md 'Action Execution Modes'."
    )


_BACKENDS = {
    "antigravity": AntigravityDispatch,
    "claude": ClaudeDispatch,
    "none": NullDispatch,
}


def host_harness(env: Optional[Dict[str, str]] = None) -> str:
  """The harness this process was started from: "claude", or "".

  Claude Code sets `CLAUDECODE=1` in every process it spawns, including
  daemons started from its shell tool, so the marker survives into the
  dashboard server and the triage calls it makes.

  Args:
    env: Environment to inspect; defaults to `os.environ`.

  Returns:
    "claude" when started from Claude Code, otherwise "".
  """
  env = os.environ if env is None else env
  if env.get("CLAUDECODE") == "1":
    return "claude"
  return ""


def get_dispatch(name: Optional[str] = None) -> Dispatch:
  """Selects a dispatcher by name, `WIKI_AGENT_BACKEND`, host, or detection.

  Under Claude Code the Claude dispatcher is returned even when `claude` is
  missing from PATH: failing with that reason is better than silently
  switching harness. Otherwise detection prefers Antigravity: its profiles
  are what Action schemas name in their prompt fields, so where both
  harnesses are installed the Antigravity path is the one the schemas were
  written against.

  Args:
    name: Explicit backend name; overrides the environment and detection.

  Returns:
    The selected dispatcher instance.

  Raises:
    ValueError: The requested backend name is unknown.
  """
  requested = (
      name or os.environ.get("WIKI_AGENT_BACKEND") or host_harness() or ""
  )
  requested = requested.strip().lower()
  if requested:
    if requested not in _BACKENDS:
      raise ValueError(
          f"unknown agent backend '{requested}'; choose from"
          f" {sorted(_BACKENDS)}"
      )
    return _BACKENDS[requested]()

  reasons = []
  for cls in (AntigravityDispatch, ClaudeDispatch):
    inst = cls()
    if inst.available():
      return inst
    reasons.append(f"{inst.name}: {inst.unavailable_reason()}")
  return NullDispatch(reasons)


def invoke(
    prompt: str,
    model: str,
    profile: Optional[str] = None,
    root_dir: Optional[str] = None,
    backend: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
  """Convenience one-liner over `get_dispatch().invoke(...)`."""
  return get_dispatch(backend).invoke(
      prompt, model, profile=profile, root_dir=root_dir, timeout=timeout
  )


if __name__ == "__main__":
  selected = get_dispatch()
  print(f"selected     : {selected.name}")
  print(f"host harness : {host_harness() or '(not detected)'}")
  if not selected.available():
    print(f"reason       : {selected.unavailable_reason()}")
  for backend_cls in (AntigravityDispatch, ClaudeDispatch):
    probe = backend_cls()
    print(
        f"  {probe.name:<12} "
        f"{'available' if probe.available() else probe.unavailable_reason()}"
    )

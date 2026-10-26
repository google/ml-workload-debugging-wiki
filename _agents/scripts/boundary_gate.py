#!/usr/bin/env python3
"""Boundary gate hook enforcing agent safety and coordinator delegation."""

import json
import os
import re
import subprocess
import sys

# Pre-compiled regexes and constants for fast boundary checking
PROJECT_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)
WIKI_ROOT = os.path.join(PROJECT_ROOT, "_wiki")

# Fast prefix check tuples
ALLOWED_CWD_PREFIXES = (WIKI_ROOT, PROJECT_ROOT)
ALLOWED_CWD_SUBSTRINGS = (
    "/_wiki",
    "/.gemini/",
    "/scratch",
    "/debug_ml_workloads_wiki",
)

ALLOWED_MUTATION_PREFIXES = (WIKI_ROOT, PROJECT_ROOT)
ALLOWED_MUTATION_SUBSTRINGS = (
    "/_wiki/",
    "/.gemini/",
    "/scratch",
    "/debug_ml_workloads_wiki/",
)

ALLOWED_READ_PREFIXES = (WIKI_ROOT, PROJECT_ROOT)
ALLOWED_READ_SUBSTRINGS = (
    "/_wiki",
    "/scratch",
    "/tmp/",
    "/_agents/",
    "/.gemini/",
    "/debug_ml_workloads_wiki/",
)

BAD_CMD_REGEX = re.compile(
    r"os\.walk\(['\"]/(?:usr|etc|var|opt|home)|"
    r"find\s+/(?:usr|etc|var|opt|home)|grep\s+-r\s+/(?:usr|etc|var|opt|home)|"
    r"ls\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(?:usr|etc|var|opt|home)|"
    r"os\.listdir\(['\"]/(?:usr|etc|var|opt|home)"
)

PROTECTED_DIAGNOSTIC_SCRIPTS = (
    "pull_and_format_logs.py",
    "validate_workload.py",
    "extract_culprit_hosts.py",
    "resolve_workload_nodes.py",
    "query_interface.py",
    "triage_workloads_api.py",
    "diagnose_sflag_hang.py",
    "fetch_jobset_timeline.py",
    "analyze_libtpu_timing.py",
)


def main():
  try:
    payload = json.load(sys.stdin)
  except (json.JSONDecodeError, OSError, ValueError):
    print(json.dumps({"decision": "allow"}))
    return

  tool_name = payload.get("toolCall", {}).get("name", "")
  args = payload.get("toolCall", {}).get("args", {})

  # 1. Extract target path
  target_path = ""
  if tool_name in ("view_file", "list_dir"):
    target_path = args.get("AbsolutePath") or args.get("DirectoryPath") or ""
  elif tool_name in (
      "write_to_file",
      "replace_file_content",
      "multi_replace_file_content",
  ):
    target_path = args.get("TargetFile", "")
  elif tool_name == "run_command":
    cwd = args.get("Cwd", "")
    cmd_str = args.get("CommandLine", "")
    if cwd:
      resolved_cwd = os.path.realpath(cwd)
      if not (
          resolved_cwd.startswith(ALLOWED_CWD_PREFIXES)
          or any(s in resolved_cwd for s in ALLOWED_CWD_SUBSTRINGS)
      ):
        print(
            json.dumps({
                "decision": "deny",
                "reason": (
                    f"Command execution blocked: Cwd '{cwd}' must be inside"
                    " _wiki/ or project directories."
                ),
            })
        )
        return

    # Inspect CommandLine to block bypass scanning attempts
    if BAD_CMD_REGEX.search(cmd_str.lower()):
      print(
          json.dumps({
              "decision": "deny",
              "reason": (
                  "Command execution blocked: You are strictly bounded to the"
                  " project workspace (_wiki/). Do NOT run Python scripts (like"
                  " os.walk) or shell commands (like find/ls) to scan or access"
                  " parent directories outside the repository. Search inside"
                  " _wiki/ directly."
              ),
          })
      )
      return

    # Coordinator Delegation Guardrail
    is_diagnostic_cmd = any(
        s in cmd_str for s in PROTECTED_DIAGNOSTIC_SCRIPTS
    ) or bool(re.search(r"\bpython_glue_runner\.py\b", cmd_str))

    if is_diagnostic_cmd:
      conv_id = payload.get("conversationId")
      if conv_id:
        is_subagent = False
        try:
          out = subprocess.check_output(
              ["agentapi", "get-conversation-metadata", conv_id],
              stderr=subprocess.DEVNULL,
              text=True,
              timeout=5,
          )
          meta_json = json.loads(out)
          meta = (
              meta_json.get("response", {})
              .get("conversationMetadata", {})
              .get("metadata", {})
          )
          subagent_spec = meta.get("subagentSpec")
          if isinstance(subagent_spec, dict):
            type_name = (subagent_spec.get("typeName") or "").lower()
            role = (subagent_spec.get("role") or "").lower()
            if (
                "main coordinator" in role
                or "main coordinator" in type_name
                or role == "coordinator"
                or type_name == "coordinator"
            ):
              is_subagent = False
            elif type_name or role:
              is_subagent = True
          elif (
              meta.get("parentConversationId")
              and meta.get("nestingDepth", 0) > 0
          ):
            is_subagent = True
        except (
            subprocess.SubprocessError,
            json.JSONDecodeError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ):
          pass

        if not is_subagent:
          print(
              json.dumps({
                  "decision": "deny",
                  "reason": (
                      "Command execution blocked: The main coordinator agent"
                      " cannot execute diagnostic tools directly. You MUST"
                      " invoke the specialized subagent ('wiki_router' or"
                      " 'wiki_practitioner') using invoke_subagent."
                  ),
              })
          )
          return

    print(json.dumps({"decision": "allow"}))
    return

  if not target_path:
    print(json.dumps({"decision": "allow"}))
    return

  resolved_path = os.path.realpath(target_path)

  # 2. Hard Deny on Mutations
  if tool_name in (
      "write_to_file",
      "replace_file_content",
      "multi_replace_file_content",
  ):
    is_prompt_file = (
        (
            resolved_path.startswith(ALLOWED_MUTATION_PREFIXES)
            or "/debug_ml_workloads_wiki/" in resolved_path
        )
        and os.path.basename(resolved_path).startswith("prompt")
        and resolved_path.endswith(".md")
    )
    if "AGENTS.md" in resolved_path or not (
        resolved_path.startswith(WIKI_ROOT)
        or any(s in resolved_path for s in ALLOWED_MUTATION_SUBSTRINGS)
        or is_prompt_file
    ):
      print(
          json.dumps({
              "decision": "deny",
              "reason": (
                  "Security boundary violation: Forbidden from modifying"
                  f" '{resolved_path}'. Mutations are allowed strictly inside"
                  " _wiki/ and prompt files."
              ),
          })
      )
      return

  # 3. Read Boundary
  if tool_name in ("view_file", "list_dir"):
    is_wiki = resolved_path.startswith(WIKI_ROOT) or any(
        s in resolved_path for s in ["/_wiki", "/scratch", "/tmp/"]
    )
    is_prompt = os.path.basename(resolved_path).startswith("prompt")
    is_in_project_root = (
        os.path.dirname(resolved_path) == PROJECT_ROOT
        or "/debug_ml_workloads_wiki/" in resolved_path
    )
    is_rule = (
        "AGENTS.md" in resolved_path
        or "/_agents/" in resolved_path
        or "/.gemini/" in resolved_path
        or is_prompt
        or is_in_project_root
    )

    if not (is_wiki or is_rule):
      print(
          json.dumps({
              "decision": "deny",
              "reason": (
                  f"Read boundary violation: Cannot read '{resolved_path}'."
                  " Reads are bounded to _wiki/, prompt.md, and system rule"
                  " files."
              ),
          })
      )
      return

  print(json.dumps({"decision": "allow"}))

if __name__ == "__main__":
  main()

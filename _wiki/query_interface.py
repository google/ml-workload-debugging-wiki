#!/usr/bin/env python3
"""Interactive query interface for the ML Workload Troubleshooting Wiki."""

import argparse
import datetime
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

WIKI_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.abspath(os.path.join(WIKI_DIR, "..", "_agents"))
LOG_FILE = os.path.join(WIKI_DIR, "LOG.md")

try:
  agent_dispatch = importlib.import_module("tools.scripts.agent_dispatch")
except ImportError:

  _dispatch_path = os.path.join(
      WIKI_DIR, "tools", "scripts", "agent_dispatch.py"
  )
  if os.path.exists(_dispatch_path):
    _spec = importlib.util.spec_from_file_location(
        "agent_dispatch", _dispatch_path
    )
    if _spec and _spec.loader:
      agent_dispatch = importlib.util.module_from_spec(_spec)
      _spec.loader.exec_module(agent_dispatch)
    else:
      agent_dispatch = None
  else:
    agent_dispatch = None

# Default agentapi path fallback
AGENTAPI_PATH = (
    os.environ.get("AGENTAPI_PATH") or shutil.which("agentapi") or "agentapi"
)


def resolve_workspace_uri() -> str | None:
  cwd = os.getcwd()
  match = re.search(r"^(.*\/workspace)(\/.*)?", cwd)
  if match:
    return "file://" + match.group(1)
  return None


def get_project_config_path(full_env: dict[str, str]) -> str | None:
  """Returns the Antigravity project config JSON path if found."""
  project_id = full_env.get("ANTIGRAVITY_PROJECT_ID")
  home = os.path.expanduser("~")
  proj_dir = os.path.join(home, ".gemini", "config", "projects")

  if not project_id:
    if os.path.exists(proj_dir):
      files = os.listdir(proj_dir)
      json_files = [f for f in files if f.endswith(".json")]
      if json_files:
        project_id = json_files[0][:-5]

  if project_id:
    return os.path.join(proj_dir, f"{project_id}.json")
  return None


def inject_workspace(config_path: str, workspace_uri: str) -> bool:
  """Temporarily injects workspace_uri into the Antigravity project config."""
  if not os.path.exists(config_path):
    return False

  try:
    with open(config_path, "r") as f:
      data = json.load(f)
  except (OSError, ValueError, TypeError, json.JSONDecodeError):
    return False

  resources = (
      data.setdefault("projectResources", {}).setdefault("resources", [])
  )

  # Check if already exists
  exists = False
  for res in resources:
    if res.get("folderUri") == workspace_uri:
      exists = True
      break

  if exists:
    return False

  # Backup
  backup_path = config_path + ".bak"
  try:
    shutil.copy(config_path, backup_path)
    resources.append({"folderUri": workspace_uri})
    with open(config_path, "w") as f:
      json.dump(data, f, indent=2)
    return True
  except (OSError, ValueError, TypeError):
    # Cleanup backup if write failed
    if os.path.exists(backup_path):
      os.remove(backup_path)
    return False


def restore_config(config_path: str) -> bool:
  backup_path = config_path + ".bak"
  if os.path.exists(backup_path):
    try:
      shutil.move(backup_path, config_path)
      return True
    except (OSError, ValueError, TypeError):
      pass
  return False


def run_agentapi(
    cmd_args: list[str], env: dict[str, str] | None = None
) -> dict[str, Any]:
  """Run agentapi and return parsed JSON stdout or raise error."""
  full_env = os.environ.copy()
  if "ANTIGRAVITY_LS_ADDRESS" not in full_env:
    full_env["ANTIGRAVITY_LS_ADDRESS"] = "127.0.0.1:8081"
  if env:
    full_env.update(env)

  # Auto-discover project id if not present
  if "ANTIGRAVITY_PROJECT_ID" not in full_env:
    # Look for configs/users/
    home = os.path.expanduser("~")
    proj_dir = os.path.join(home, ".gemini", "config", "projects")
    if os.path.exists(proj_dir):
      files = os.listdir(proj_dir)
      json_files = [f for f in files if f.endswith(".json")]
      if json_files:
        full_env["ANTIGRAVITY_PROJECT_ID"] = json_files[0][:-5]

  is_new_conv = len(cmd_args) > 0 and cmd_args[0] == "new-conversation"
  injected = False
  config_path = None

  if is_new_conv:
    workspace_uri = resolve_workspace_uri()
    config_path = get_project_config_path(full_env)
    if workspace_uri and config_path:
      injected = inject_workspace(config_path, workspace_uri)

  try:
    # Run agentapi
    cmd = [AGENTAPI_PATH] + cmd_args
    res = subprocess.run(
        cmd, env=full_env, capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
      error_msg = res.stderr or res.stdout or "Unknown error"
      raise RuntimeError(
          f"agentapi execution failed with code {res.returncode}: {error_msg}"
      )

    try:
      return json.loads(res.stdout)
    except Exception as e:
      raise RuntimeError(
          f"Failed to parse agentapi JSON output: {e}. Output was:\n"
          f"{res.stdout}"
      ) from e
  finally:
    if injected and config_path:
      restore_config(config_path)


def poll_conversation_transcript(
    conv_id: str, timeout_seconds: int = 600
) -> tuple[str, list[str]]:
  """Poll conversation transcript and return the final model response."""
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
    time.sleep(2.0)
    if not os.path.exists(transcript_path):
      continue

    try:
      with open(transcript_path, "r") as f:
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
            err_detail = step.get("error") or "Unknown error"
            raise RuntimeError(f"Agent conversation crashed: {err_detail}")
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
      ):
        tool_calls = last_planner.get("tool_calls") or []
        if tool_calls:
          continue
        content = last_planner.get("content", "")
        if content:
          return content, lines
    except (IOError, json.JSONDecodeError):
      pass

  raise TimeoutError(
      f"Timeout waiting for conversation {conv_id} after {timeout_seconds}s"
  )


def read_agent_system_prompt(agent_name: str) -> str:
  """Read custom agent instructions from config."""
  agent_json_path = os.path.join(
      AGENTS_DIR, "agents", agent_name, "agent.json"
  )
  if not os.path.exists(agent_json_path):
    raise FileNotFoundError(
        f"Agent config not found for {agent_name} at {agent_json_path}"
    )
  with open(agent_json_path, "r") as f:
    data = json.load(f)
  return data["config"]["customAgent"]["systemPromptSections"][0]["content"]


def param_description(p_info: Any) -> str:
  """Normalizes a parameter spec to its description string.

  Action schemas use two styles: {"p": {"type", "description"}} and the
  shorthand {"p": "description"}. Assuming dict raises AttributeError on the
  shorthand, which aborts the run just before the parameter prompt.

  Args:
    p_info: Parameter specification dictionary or shorthand description string.

  Returns:
    Normalized description string.
  """
  if isinstance(p_info, dict):
    return p_info.get("description") or ""
  if isinstance(p_info, str):
    return p_info
  return ""


def collect_required_params(
    actions_data: dict[str, Any] | None,
) -> dict[str, str]:
  """Union of parameter names across actions, first description wins."""
  required = {}
  for _, act_obj in (actions_data or {}).items():
    params = (act_obj or {}).get("parameters")
    if not isinstance(params, dict):
      continue
    for p_name, p_info in params.items():
      if p_name not in required:
        required[p_name] = param_description(p_info)
  return required


def clean_json_response(text: str) -> str:
  """Remove code block syntax from JSON response."""
  match = re.search(
      r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
  )
  if match:
    return match.group(1).strip()
  return text.strip()


def append_to_log(
    entry_type: str, subject: str, details_list: list[str]
) -> None:
  """Append structured entry to LOG.md."""
  today = datetime.date.today().isoformat()
  entry_header = f"## [{today}] {entry_type} | {subject}\n"
  entry_body = ""
  for detail in details_list:
    entry_body += f"- {detail}\n"
  entry_body += "\n"

  # Append to LOG.md
  with open(LOG_FILE, "a") as f:
    f.write(entry_header + entry_body)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Wiki Query & Troubleshooting Interface"
  )
  parser.add_argument(
      "query", nargs="?", help="Symptom description or error log"
  )
  args = parser.parse_args()

  # This driver holds multi-turn agentapi conversations, so it only runs on
  # Antigravity. Under Claude Code, do not quietly start Antigravity sessions.
  if agent_dispatch and agent_dispatch.host_harness() == "claude":
    print(
        "query_interface.py drives Antigravity conversations and is not used"
        " under Claude Code. Use `/troubleshoot <workload_id>` (or"
        " `/troubleshoot`) instead.",
        file=sys.stderr,
    )
    sys.exit(3)

  query = args.query
  if not query:
    query = input("Enter error message or symptom query: ").strip()

  if not query:
    print("Error: Empty query.")
    sys.exit(1)

  print(f"\n[1/3] Invoking wiki_router to analyze query: '{query}'...")

  router_prompt = read_agent_system_prompt("wiki_router")
  prompt = f"""{router_prompt}

Please analyze the following error message/symptom query:
"{query}"

Respond ONLY with a JSON object containing:
{{
  "status": "CACHE_HIT" or "CACHE_MISS",
  "symptom_page": "symptoms/<page>.md",
  "workflow": "workflows/<workflow>.json",
  "potential_matches": ["workflows/<workflow1>.json", ...] (if multiple matches)
}}
"""

  steps = []

  # Start Router Conversation
  resp = run_agentapi(["new-conversation", "--model=flash", prompt])
  conv_id = resp["response"]["newConversation"]["conversationId"]
  print(f"Router conversation spawned: {conv_id}. Polling response...")

  router_output, _ = poll_conversation_transcript(conv_id)
  router_json_str = clean_json_response(router_output)

  try:
    router_data = json.loads(router_json_str)
  except (ValueError, TypeError, json.JSONDecodeError) as e:
    print(f"Failed to parse router output as JSON: {e}")
    print(f"Raw router output:\n{router_output}")
    sys.exit(1)

  print(f"Router output: {json.dumps(router_data, indent=2)}")

  status = router_data.get("status", "CACHE_MISS")
  symptom_page = router_data.get("symptom_page")
  workflow = router_data.get("workflow")
  potential_matches = router_data.get("potential_matches", [])

  steps.append({
      "name": "Triage (wiki_router)",
      "type": "triage",
      "details": (
          f"Status: {status}, Symptom: {symptom_page}, Workflow: {workflow}"
      ),
  })

  # Resolve Ambiguity if needed
  if len(potential_matches) > 1 or (not workflow and potential_matches):
    print("\n[Ambiguity Detected] Invoking wiki_advisor to resolve matches...")
    advisor_prompt = read_agent_system_prompt("wiki_advisor")

    adv_prompt = f"""{advisor_prompt}

The user query was: "{query}"
The router identified multiple potential matching workflows or symptoms:
{json.dumps(potential_matches, indent=2)}

Please guide the user to select the single correct workflow.
Analyze the differences and ask clarifying questions or output the chosen one.
If you know which one it is, output:
{{
  "workflow": "workflows/<selected>.json"
}}
"""
    resp = run_agentapi(["new-conversation", "--model=flash", adv_prompt])
    adv_conv_id = resp["response"]["newConversation"]["conversationId"]
    print(f"Advisor conversation spawned: {adv_conv_id}. Polling...")

    # Bounded advisor loop. poll_conversation_transcript returns as soon as the
    # transcript is DONE, so a reply that parses as JSON but carries no usable
    # "workflow" would otherwise re-poll the same finished transcript forever.
    advisor_attempts = 0
    max_advisor_attempts = 5
    while True:
      advisor_attempts += 1
      if advisor_attempts > max_advisor_attempts:
        print(
            "Advisor did not return a usable workflow in "
            f"{max_advisor_attempts} attempts; keeping the router's candidate."
        )
        break
      adv_output, _ = poll_conversation_transcript(adv_conv_id)
      adv_cleaned = clean_json_response(adv_output)

      try:
        adv_data = json.loads(adv_cleaned)
        # Do not clobber the router-supplied candidate with a missing/null key.
        selected = adv_data.get("workflow")
        workflow = selected or workflow
        if selected:
          print(f"Advisor selected workflow: {workflow}")
          steps.append({
              "name": "Resolve Ambiguity (wiki_advisor)",
              "type": "advisor_resolution",
              "details": f"Selected workflow: {workflow}"
          })
          break
      except (json.JSONDecodeError, ValueError):
        # If not JSON, it must be a clarifying question to the user
        print(f"\n[Advisor clarification request]: {adv_output}")
        user_ans = input("Your answer: ").strip()

        # Send reply to advisor
        run_agentapi(["send-message", adv_conv_id, user_ans])
        print("Sent answer to Advisor. Polling next response...")
        time.sleep(2.0)

  if not workflow:
    print(
        "\nNo matching troubleshooting workflow found in the Wiki"
        " (CACHE_MISS)."
    )
    append_to_log(
        "query", query, ["CACHE_MISS", "No matching workflow found."]
    )
    sys.exit(0)

  # Execute Workflow using wiki_practitioner
  print(f"\n[2/3] Executing selected workflow: {workflow}...")
  workflow_path = os.path.join(WIKI_DIR, workflow)
  if not os.path.exists(workflow_path):
    print(f"Error: Workflow file {workflow_path} does not exist.")
    sys.exit(1)

  with open(workflow_path, "r") as f:
    workflow_content = f.read()

  # Let's read the action files too to pass to practitioner
  actions_data = {}
  try:
    workflow_json = json.loads(workflow_content)
    all_step_links = (
        workflow_json.get("steps", []) + workflow_json.get("actions", [])
    )
    for act_link in all_step_links:
      act_name = act_link.replace("[[", "").replace("]]", "").strip()
      act_path = os.path.join(WIKI_DIR, "actions", f"{act_name}.json")
      if os.path.exists(act_path):
        with open(act_path, "r") as af:
          actions_data[act_name] = json.load(af)
  except (json.JSONDecodeError, OSError) as e:
    print(f"Warning: Failed to load actions for context: {e}")
  # Collect all required parameters from actions
  required_params = collect_required_params(actions_data)

  user_params = {}
  if required_params:
    print(
        "\n[Parameter Input Required] The workflow uses the following"
        " parameters:"
    )
    for p_name, p_desc in required_params.items():
      env_val = os.environ.get(f"WIKI_PARAM_{p_name}")
      if env_val:
        print(f"  Using parameter '{p_name}' from environment: {env_val}")
        user_params[p_name] = env_val
        continue
      desc_str = f" ({p_desc})" if p_desc else ""
      prompt_msg = (
          f"  Enter '{p_name}'{desc_str} [press Enter to skip/auto-discover]: "
      )
      try:
        val = input(prompt_msg).strip()
        if val:
          user_params[p_name] = val
      except EOFError:
        print(f"{prompt_msg}EOF (skipped)")

  practitioner_prompt = read_agent_system_prompt("wiki_practitioner")

  # Pass parameters in practitioner instructions
  params_str = ""
  if user_params:
    params_str = (
        "Initial Parameters Provided by User:\n"
        f"{json.dumps(user_params, indent=2)}\n\n"
    )

  # Instruct practitioner
  prac_instructions = (
      f"{practitioner_prompt}\n\n"
      "You are starting a troubleshooting execution session.\n"
      f"Workflow to run: '{workflow}'\n"
      "Workflow Configuration:\n"
      f"{workflow_content}\n\n"
      "Related Action Configurations:\n"
      f"{json.dumps(actions_data, indent=2)}\n\n"
      f"{params_str}Instructions:\n"
      "1. Go through each action step sequentially.\n"
      "2. If an action's description contains \"(Mutating Action)\" or is\n"
      "   state-mutating, you MUST pause and ask the user for approval.\n"
      "   Specifically, format the request exactly as:\n"
      '   "Awaiting user approval for mutating action \'<action_name>\' with '
      "command: '<command>'\"\n"
      "   Wait until you receive the user response.\n"
      '3. If you receive "APPROVED", execute the action command using\n'
      "   the run_command tool and proceed. For a python_glue_runner.py\n"
      "   invocation, append `--approved`; the runner refuses\n"
      "   requires_approval Actions without it.\n"
      '4. If you receive "DENIED", skip or abort, and report that it was\n'
      "   denied.\n"
      "5. For non-mutating actions, run them directly using the\n"
      "   run_command tool (or appropriate tools) and print the status.\n"
      "6. Once finished, output a summary of the diagnostic outcomes.\n"
      "7. Maintain a record of all workflows and actions executed during\n"
      "   this session. DO NOT re-execute or loop over any workflow or\n"
      "   action that has already been executed or attempted.\n"
  )

  resp = run_agentapi(["new-conversation", "--model=pro", prac_instructions])
  prac_conv_id = resp["response"]["newConversation"]["conversationId"]
  print(f"Practitioner conversation spawned: {prac_conv_id}. Executing...")

  # Poll practitioner, handle interactive approvals
  last_index = 0
  outcome_summary = ""
  # Wall-clock deadline. poll_conversation_transcript returns immediately once
  # the transcript is DONE, so its own timeout never fires afterwards; without
  # this an agent that finishes without emitting [RUNNER_FINISHED] would spin
  # on the same finished transcript indefinitely.
  practitioner_deadline = time.time() + 1800
  while True:
    if time.time() > practitioner_deadline:
      print("Practitioner loop exceeded its 1800s deadline; stopping.")
      outcome_summary = outcome_summary or "Practitioner deadline exceeded."
      break
    try:
      content, all_lines = poll_conversation_transcript(
          prac_conv_id, timeout_seconds=900
      )
    except TimeoutError:
      print("Timeout executing practitioner workflow.")
      outcome_summary = "Timeout executing workflow."
      break

    awaiting_approval = False
    action_name = ""
    cmd_to_run = ""
    # Print new lines in transcript to console for tracking progress
    for line in all_lines[last_index:]:
      try:
        step_obj = json.loads(line)
        # Display progress / messages
        if step_obj.get("source") == "MODEL":
          content_slice = step_obj.get("content", "")
          if content_slice:
            print(content_slice)
            approval_match = re.search(
                r"Awaiting user approval for mutating action '([^']+)' with "
                r"command: '([^']+)'",
                content_slice,
                re.IGNORECASE,
            )
            if approval_match:
              awaiting_approval = True
              action_name = approval_match.group(1)
              cmd_to_run = approval_match.group(2)
          # Track commands run
          tool_calls = step_obj.get("tool_calls") or []
          for tc in tool_calls:
            if tc.get("name") == "run_command":
              args = tc.get("args", {})
              cmd_run = args.get("CommandLine")
              print(f"  [Executing Tool]: {cmd_run}")
              steps.append({
                  "name": "Run Command",
                  "type": "tool_execution",
                  "details": f"Command: {cmd_run}",
                  "command": cmd_run,
              })
      except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    last_index = len(all_lines)

    # Check if awaiting approval
    if awaiting_approval:
      print("\n>>> [MUTATING ACTION APPROVAL REQUIRED] <<<")
      print(f"Action: {action_name}")
      print(f"Command: {cmd_to_run}")
      user_choice = input(
          "Do you approve this command? (y/N): "
      ).strip().lower()

      if user_choice == "y":
        print("Sending approval to practitioner...")
        run_agentapi(["send-message", prac_conv_id, "APPROVED"])
        # Attach the verdict to the step that actually ran the command. The
      # previous ternary had identical branches, so it always tagged the last
      # step -- which is the triage step when no run_command tool call was
      # observed for this command.
      target = next(
          (st for st in reversed(steps) if st.get("command") == cmd_to_run),
          None,
      )
      if target is not None:
        target["approval"] = "APPROVED"
      elif steps:
        steps[-1]["approval"] = "APPROVED"
      else:
        print("Sending denial to practitioner...")
        run_agentapi(["send-message", prac_conv_id, "DENIED"])
        # Attach the verdict to the step that actually ran the command. The
      # previous ternary had identical branches, so it always tagged the last
      # step -- which is the triage step when no run_command tool call was
      # observed for this command.
      target = next(
          (st for st in reversed(steps) if st.get("command") == cmd_to_run),
          None,
      )
      if target is not None:
        target["approval"] = "DENIED"
      elif steps:
        steps[-1]["approval"] = "DENIED"

      time.sleep(2.0)
      continue

    # Check if conversation is finished
    print(f"[DEBUG] practitioner content: {repr(content)}")
    if content and "[RUNNER_FINISHED]" in content:
      print("\nRunner finished.")
      outcome_summary = content.split("[RUNNER_FINISHED]")[0].strip()
      break

    time.sleep(5.0)

  print("\n[3/3] Execution Completed.")
  print("="*40 + " DIAGNOSTIC OUTCOMES " + "="*40)
  print(outcome_summary)
  print("="*101)

  steps.append({
      "name": "Execution Summary (wiki_practitioner)",
      "type": "execution_summary",
      "details": outcome_summary
  })

  # Log outcomes
  append_to_log("query", query, [
      f"Match: {symptom_page} | {workflow}",
      f"Steps Executed: {[s['name'] for s in steps]}"
  ])

  print("\nLogs recorded in LOG.md.")


if __name__ == "__main__":
  main()

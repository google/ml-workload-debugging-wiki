"""Audits the ML Workload Debugging Wiki for structural and content defects.

Checks frontmatter schemas, link integrity, orphan pages, taxonomy tags, source
hashes, portability, PII leakage, blocking commands and duplicate actions, then
appends a session entry to LOG.md (rotating it when needed) and prints a JSON
report. Takes no command-line arguments.
"""

import ast
from collections.abc import Collection, Sequence
import datetime
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import types
from typing import Any

import yaml

# The checks below deliberately degrade to "skip this file" on a read or parse
# failure so one malformed page cannot abort the whole lint run. These tuples
# name the failures that degrade; anything else is a linter bug and propagates.
#
# Reading a wiki file: missing/unreadable path, or bytes that are not UTF-8.
_READ_ERRORS = (OSError, UnicodeDecodeError)
# Reading and parsing a JSON action/workflow file.
_JSON_FILE_ERRORS = _READ_ERRORS + (json.JSONDecodeError,)
# yaml.safe_load on malformed frontmatter: syntax errors, plus ValueError /
# TypeError raised by scalar constructors (e.g. an impossible date such as
# 2026-02-30) and by sorting a mapping whose keys mix types.
_YAML_ERRORS = (yaml.YAMLError, ValueError, TypeError)

TYPE_TO_SECTION = types.MappingProxyType({
    "concept": "Concepts",
    "symptom": "Symptoms",
    "workflow": "Workflows",
    "playbook": "Playbooks",
    "action": "Actions",
    "tool": "Tools",
    "entity": "Entities",
})

# Router-first: an agent reads INDEX.md top-down and stops once it has a
# symptom -> playbook/workflow match, so the routing sections come first and
# the reference sections (concepts, entities, actions) go last.
SECTION_ORDER = (
    "Symptoms",
    "Playbooks",
    "Workflows",
    "Tools",
    "Concepts",
    "Entities",
    "Actions",
)

# Directories that are never wiki content. Every tree walk in this file shares
# this one set so the walks cannot drift apart as directories are added.
EXCLUDED_SCAN_DIRS = frozenset({
    ".agents",
    "_agents",
    "srcs",
    "tests",
    "scratch",
    "archive",
    ".venv",
    ".gemini",
    "__pycache__",
    ".pytest_cache",
})


def is_virtualenv_dir(path: str) -> bool:
  """Returns True if path is a Python virtualenv root.

  Matching the literal name ".venv" misses every other environment name a
  contributor might use (".venv-darwin", "venv", ".venv312"). A stray env is
  not merely noisy: the walks below index site-packages metadata as wiki pages
  and write it into INDEX.md. pyvenv.cfg marks a virtualenv whatever its
  directory is called.

  Args:
    path: Directory to test.

  Returns:
    True if `path` contains a `pyvenv.cfg` file.
  """
  return os.path.exists(os.path.join(path, "pyvenv.cfg"))


def prune_scan_dirs(root: str, dirs: list[str]) -> list[str]:
  """Filters os.walk's `dirs` in place to skip excluded and virtualenv dirs.

  Args:
    root: The directory os.walk is currently visiting.
    dirs: The `dirs` list yielded by os.walk for `root`; modified in place.

  Returns:
    The same `dirs` list, after filtering.
  """
  dirs[:] = [
      d
      for d in dirs
      if d not in EXCLUDED_SCAN_DIRS
      and not is_virtualenv_dir(os.path.join(root, d))
  ]
  return dirs


def parse_frontmatter(text: str) -> Any:
  """Parses a YAML frontmatter block.

  Args:
    text: The YAML text between the `---` fences.

  Returns:
    The parsed mapping, or an empty dict if the YAML is empty or invalid.
  """
  try:
    return yaml.safe_load(text) or {}
  except yaml.YAMLError:
    return {}


def extract_summary(
    content: str, is_json: bool, frontmatter: dict[str, Any]
) -> str:
  """Extracts a one-line summary of a page for its INDEX.md entry.

  Args:
    content: The full text of the page.
    is_json: Whether the page is a JSON workflow/action rather than markdown.
    frontmatter: The parsed frontmatter (or JSON document) of the page.

  Returns:
    The `description` field from frontmatter/JSON if present; otherwise for
    markdown, the first body line that is not a heading, HTML comment, fence,
    quote or list item, with provenance markers removed. Empty string if none is
    found.
  """
  if is_json:
    return frontmatter.get("description", "")

  if isinstance(frontmatter, dict) and frontmatter.get("description"):
    return str(frontmatter["description"]).strip()

  # For markdown, strip frontmatter first
  body = content
  fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
  if fm_match:
    body = content[fm_match.end() :]

  lines = body.strip().split("\n")
  for line in lines:
    line = line.strip()
    if (
        not line
        or line.startswith("#")
        or line.startswith("<!--")
        or line.startswith("```")
        or line.startswith(">")
        or line.startswith("-")
    ):
      continue
    # Remove provenance markers like ^[srcs/...]
    line = re.sub(r"\^\[[^\]]+\]", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()
  return ""


def update_index_file(
    wiki_dir: str, missing_pages: list[str], current_date_str: str
) -> None:
  """Adds entries for pages missing from INDEX.md and rewrites it in place.

  Does nothing if INDEX.md does not exist.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    missing_pages: Wiki-relative paths of pages not yet listed in INDEX.md.
    current_date_str: Today's date as `YYYY-MM-DD`.
  """
  index_path = os.path.join(wiki_dir, "INDEX.md")
  if not os.path.exists(index_path):
    return

  with open(index_path, "r") as f:
    content = f.read()

  lines = content.splitlines()
  header_lines = []
  sections = {}
  current_section = None

  for s in SECTION_ORDER:
    sections[s] = []

  for line in lines:
    stripped = line.strip()
    if stripped.startswith("## "):
      current_section = stripped[3:].strip()
      if current_section not in sections:
        sections[current_section] = []
    elif (
        stripped.startswith("- [") or stripped.startswith("* [")
    ) and current_section:
      match = re.fullmatch(
          r"\s*[-*]\s*\[((?:[^\[\]]|\[[^\]]*\])+)\]\(([^)]+)\)(?::\s*(.*))?",
          line,
      )
      if match:
        title = match.group(1).strip()
        path = match.group(2).strip()
        summary = match.group(3).strip() if match.group(3) else ""
        sections[current_section].append(
            {"title": title, "path": path, "summary": summary}
        )
    elif not current_section:
      header_lines.append(line)

  for page in missing_pages:
    full_path = os.path.join(wiki_dir, page)
    if not os.path.exists(full_path):
      continue
    with open(full_path, "r") as f:
      p_content = f.read()

    frontmatter = {}
    is_json = page.endswith(".json")
    if is_json:
      try:
        frontmatter = json.loads(p_content)
      except json.JSONDecodeError:
        pass
    else:
      fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", p_content, re.DOTALL)
      if fm_match:
        try:
          frontmatter = parse_frontmatter(fm_match.group(1))
        except _YAML_ERRORS:
          pass

    title = frontmatter.get(
        "title",
        os.path.splitext(os.path.basename(page))[0].replace("-", " ").title(),
    )
    p_type = frontmatter.get("type", "")

    section_name = TYPE_TO_SECTION.get(p_type)
    if not section_name:
      dir_name = os.path.dirname(page)
      if dir_name:
        section_name = dir_name.capitalize()
      else:
        section_name = "Other"

    if section_name not in sections:
      sections[section_name] = []

    summary = extract_summary(p_content, is_json, frontmatter)

    exists = False
    for entry in sections[section_name]:
      if entry["path"] == page:
        exists = True
        break
    if not exists:
      sections[section_name].append(
          {"title": title, "path": page, "summary": summary}
      )

  # Keep the curated in-section order (most useful entries first); new pages
  # were appended above, at the end of their section. Do not re-sort.
  total_pages = 0
  for section_entries in sections.values():
    for entry in section_entries:
      if os.path.exists(os.path.join(wiki_dir, entry["path"])):
        total_pages += 1

  updated_header_lines = []
  for line in header_lines:
    if "Total pages:" in line or "Last updated:" in line:
      line = re.sub(
          r"Last updated: \d{4}-\d{2}-\d{2}",
          f"Last updated: {current_date_str}",
          line,
      )
      line = re.sub(r"Total pages: \d+", f"Total pages: {total_pages}", line)
    updated_header_lines.append(line)

  # Drop trailing blanks before writing. The writer inserts a blank line
  # between the header and the first "## " section; on the next run everything
  # before that section is re-absorbed into header_lines, so the blank
  # accumulates once per lint. Left unchecked the header grows without bound
  # (observed: 71 blank lines in a 77-line header).
  while updated_header_lines and not updated_header_lines[-1].strip():
    updated_header_lines.pop()

  new_content_parts = []
  new_content_parts.append("\n".join(updated_header_lines))

  ordered_sections = [s for s in SECTION_ORDER if s in sections and sections[s]]
  for s in sections:
    if s not in SECTION_ORDER and sections[s]:
      ordered_sections.append(s)

  for s in ordered_sections:
    new_content_parts.append(f"\n## {s}")
    for entry in sections[s]:
      if entry["summary"]:
        new_content_parts.append(
            f"- [{entry['title']}]({entry['path']}): {entry['summary']}"
        )
      else:
        new_content_parts.append(f"- [{entry['title']}]({entry['path']})")

  with open(index_path, "w") as f:
    f.write("\n".join(new_content_parts) + "\n")


DEFAULT_MAX_LOG_ENTRIES = 200
_ROTATED_LOG_RE = re.compile(r"log-\d{4}(-\d+)?\.md")


def get_max_log_entries() -> int:
  """Returns the maximum LOG.md entry count before rotation is triggered."""
  try:
    return int(os.environ.get("WIKI_LOG_MAX_ENTRIES", DEFAULT_MAX_LOG_ENTRIES))
  except ValueError:
    return DEFAULT_MAX_LOG_ENTRIES


def rotate_log_if_needed(
    wiki_dir: str, current_date_str: str, max_entries: int | None = None
) -> None:
  """Moves LOG.md into archive/log-YYYY.md once it exceeds max_entries."""
  log_path = os.path.join(wiki_dir, "LOG.md")
  if not os.path.exists(log_path):
    return

  if max_entries is None:
    max_entries = get_max_log_entries()

  with open(log_path, "r", encoding="utf-8") as log_file:
    log_content = log_file.read()

  lines = log_content.splitlines()
  entry_count = sum(1 for line in lines if line.startswith("## ["))

  if entry_count > max_entries:
    archive_dir = os.path.join(wiki_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    current_year = current_date_str.split("-")[0]
    rotated_path = os.path.join(archive_dir, f"log-{current_year}.md")
    # Never clobber an existing archive. Crossing max_entries twice in one
    # calendar year would otherwise destroy up to max_entries of a file the
    # project documents as append-only.
    if os.path.exists(rotated_path):
      seq = 2
      while os.path.exists(
          os.path.join(archive_dir, f"log-{current_year}-{seq}.md")
      ):
        seq += 1
      rotated_path = os.path.join(archive_dir, f"log-{current_year}-{seq}.md")

    os.rename(log_path, rotated_path)

    leading_comments: list[str] = []
    for line in lines:
      stripped_line = line.strip()
      if stripped_line.startswith("<!--") and stripped_line.endswith("-->"):
        leading_comments.append(stripped_line)
      elif not stripped_line:
        continue
      else:
        break

    header_lines = [
        "> Chronological record of all wiki actions. Append-only.",
        "> Format: `## [YYYY-MM-DD] action | subject`",
        "> Actions: ingest, update, query, lint, create, archive, delete",
        (
            f"> When this file exceeds {max_entries} entries, rotate: rename to"
            " archive/log-YYYY.md, start fresh."
        ),
    ]

    with open(log_path, "w", encoding="utf-8") as log_file:
      if leading_comments:
        for comment_line in leading_comments:
          log_file.write(comment_line + "\n")
        log_file.write("\n")
      for line in header_lines:
        log_file.write(line + "\n")
      log_file.write("\n")


def append_lint_entry(
    wiki_dir: str, current_date_str: str, num_issues: int
) -> None:
  """Records this lint run in LOG.md.

  Creates LOG.md with its standard header if absent. If a lint entry for
  today already exists, the last one is updated in place rather than adding a
  duplicate; otherwise a new entry is appended.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    current_date_str: Today's date as `YYYY-MM-DD`.
    num_issues: Number of issues found by this run.
  """
  log_path = os.path.join(wiki_dir, "LOG.md")
  max_entries = get_max_log_entries()
  if not os.path.exists(log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
      log_file.write(
          "> Chronological record of all wiki actions. Append-only.\n"
      )
      log_file.write("> Format: `## [YYYY-MM-DD] action | subject`\n")
      log_file.write(
          "> Actions: ingest, update, query, lint, create, archive, delete\n"
      )
      log_file.write(
          f"> When this file exceeds {max_entries} entries, rotate: rename to"
          " archive/log-YYYY.md, start fresh.\n\n"
      )

  with open(log_path, "r", encoding="utf-8") as log_file:
    content = log_file.read()

  pattern = rf"## \[{re.escape(current_date_str)}\] lint \| \d+ issues found"
  matches = list(re.finditer(pattern, content))

  if matches:
    last_match = matches[-1]
    start, end = last_match.span()
    new_entry = f"## [{current_date_str}] lint | {num_issues} issues found"
    content = content[:start] + new_entry + content[end:]
  else:
    if not content.endswith("\n"):
      content += "\n"
    if not content.endswith("\n\n"):
      content += "\n"
    entry = f"## [{current_date_str}] lint | {num_issues} issues found\n"
    content += entry

  with open(log_path, "w", encoding="utf-8") as log_file:
    log_file.write(content)


def check_internal_only_boolean(
    page: str, frontmatter: Any, issues: list[str]
) -> None:
  """Enforces that internal_only is required and must be a boolean."""
  if isinstance(frontmatter, dict):
    if "internal_only" not in frontmatter:
      msg = f"Field 'internal_only' is required in '{page}'."
      if msg not in issues:
        issues.append(msg)
    elif not isinstance(frontmatter["internal_only"], bool):
      msg = f"Field 'internal_only' in '{page}' must be a boolean."
      if msg not in issues:
        issues.append(msg)


ALLOWED_QUESTIONNAIRE_INPUT_MODES = frozenset({
    "text",
    "select",
    "select_then_input",
    "select_or_input",
})

ALLOWED_QUESTIONNAIRE_OPTION_KEYS = frozenset({
    "label",
    "value",
    "prefix",
    "prompt",
})

ALLOWED_FALLBACK_TRIGGERS = frozenset({
    "access_denied",
    "empty_result",
    "incomplete_coverage",
    "timeout",
    "auth_expired",
    "command_error",
})

ALLOWED_ACTION_FALLBACK_KEYS = frozenset({
    "action",
    "when",
    "reason",
    "param_mapping",
    "internal_only",
    "auto_execute",
})

ALLOWED_WORKFLOW_FALLBACK_KEYS = frozenset({
    "when",
    "skip_steps",
    "replace_with",
    "reason",
})


_WIKILINK_TARGET_RE = re.compile(r"^\[\[(.*)\]\]$")


def _resolve_action_or_workflow_target(
    wiki_dir: str, ref: str
) -> tuple[str | None, str | None]:
  """Validates a [[name]] reference and resolves its target path.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    ref: Raw step reference string expected in `[[name]]` format.

  Returns:
    A `(step_name, target_path)` tuple where `target_path` is the existing JSON
    path or `None` if no matching action/workflow file exists, or `(None, None)`
    when `ref` is not a string or does not match the `[[name]]` format.
  """
  if not isinstance(ref, str):
    return None, None
  match = _WIKILINK_TARGET_RE.fullmatch(ref.strip())
  if not match:
    return None, None
  step_name = match.group(1)
  action_path = os.path.join(wiki_dir, "actions", f"{step_name}.json")
  workflow_path = os.path.join(wiki_dir, "workflows", f"{step_name}.json")
  if os.path.exists(action_path):
    return step_name, action_path
  if os.path.exists(workflow_path):
    return step_name, workflow_path
  return step_name, None


def _is_target_internal_only(target_path: str) -> bool:
  """Returns True if the target JSON file declares internal_only: true."""
  try:
    with open(target_path, "r", encoding="utf-8") as f:
      target_data = json.load(f)
      return (
          isinstance(target_data, dict)
          and isinstance(target_data.get("internal_only"), bool)
          and target_data["internal_only"]
      )
  except _JSON_FILE_ERRORS:
    return False


def validate_action_json(
    frontmatter: dict[str, Any], page: str, wiki_dir: str, issues: list[str]
) -> None:
  """Validates action JSON frontmatter and syntax constraints."""
  check_internal_only_boolean(page, frontmatter, issues)

  if "instruction" not in frontmatter:
    issues.append(f"Action '{page}' is missing 'instruction' field.")
  if "execution_mode" not in frontmatter:
    issues.append(f"Action '{page}' is missing 'execution_mode' field.")

  if "instruction" not in frontmatter or "execution_mode" not in frontmatter:
    return

  mode = frontmatter["execution_mode"]
  inst = frontmatter["instruction"]

  valid_modes = {"command", "manual", "ai"}
  if mode not in valid_modes:
    issues.append(
        f"Action '{page}' has invalid 'execution_mode' '{mode}'. Must be one of"
        f" {valid_modes}."
    )
    return

  # The 'ai-' filename prefix and execution_mode 'ai' must always agree. The
  # prefix is the at-a-glance signal that an action is an agent prompt task
  # rather than a shell command; execution_mode is the authoritative field.
  basename = os.path.basename(page)
  action_name = basename[:-5] if basename.endswith(".json") else basename
  has_ai_prefix = action_name.startswith("ai-")
  if mode == "ai" and not has_ai_prefix:
    issues.append(
        f"AI Action '{page}' has execution_mode 'ai' but its filename lacks "
        f"the required 'ai-' prefix. Rename to 'ai-{action_name}.json'."
    )
  elif mode != "ai" and has_ai_prefix:
    issues.append(
        f"Action '{page}' has the 'ai-' filename prefix but execution_mode "
        f"'{mode}'. The prefix is reserved for agent prompt tasks "
        "(execution_mode 'ai')."
    )

  if not isinstance(inst, str) or not inst.strip():
    issues.append(f"Action '{page}' 'instruction' must be a non-empty string.")
    return

  params = frontmatter.get("parameters", {})
  if isinstance(params, dict):
    expected_params = set(params.keys())
  elif isinstance(params, list):
    expected_params = {
        item.get("name")
        for item in params
        if isinstance(item, dict) and "name" in item
    }
  else:
    expected_params = set()

  if mode == "command":
    if re.search(r"^(?:1\.\s+|To\s+[a-z]|Ask\s+the\s+user\b)", inst.strip()):
      issues.append(
          f"Command Action '{page}' has prose/step-by-step 'instruction' text; "
          "use 'execution_mode': 'manual' instead."
      )
    angle_placeholders = re.findall(r"<([a-z][a-z0-9_]*)>", inst)
    for ap in angle_placeholders:
      if ap in expected_params:
        issues.append(
            f"Command Action '{page}' uses angle-bracket placeholder '<{ap}>' "
            f"in 'instruction'; use '${{{ap}}}' syntax instead."
        )

    placeholders = set(re.findall(r"\$\{([^}]+)\}", inst))
    for p in placeholders:
      if p not in expected_params:
        issues.append(
            f"Command Action '{page}' has placeholder ${{{p}}} in "
            "'instruction' but it is not defined in parameters."
        )

    script_match = re.search(r"python3\s+([^\s]+\.py)", inst)
    if script_match:
      script_rel = script_match.group(1)
      script_abs = os.path.join(wiki_dir, script_rel)
      root_script_abs = os.path.join(os.path.dirname(wiki_dir), script_rel)
      if not os.path.exists(script_abs) and not os.path.exists(root_script_abs):
        issues.append(
            f"Command Action '{page}' references non-existent python script "
            f"'{script_rel}'."
        )

    if "/google/src/cloud/" in inst or "/usr/local/google/home/" in inst:
      issues.append(
          f"Command Action '{page}' contains hardcoded absolute workspace path "
          "in 'instruction'."
      )

  if mode != "ai" and "prompt" in frontmatter:
    issues.append(
        f"Action '{page}' defines 'prompt' but has execution_mode '{mode}'; "
        "'prompt' is only valid when execution_mode is 'ai'."
    )

  if mode == "ai":
    if inst != "ai-action":
      issues.append(
          f"AI Action '{page}' 'instruction' must be 'ai-action'. Found:"
          f" '{inst}'"
      )
    if "prompt" not in frontmatter:
      issues.append(f"AI Action '{page}' is missing 'prompt' field.")
    elif (
        not isinstance(frontmatter["prompt"], str)
        or not frontmatter["prompt"].strip()
    ):
      issues.append(
          f"AI Action '{page}' 'prompt' field must be a non-empty string."
      )

    if "prompt" in frontmatter and isinstance(frontmatter["prompt"], str):
      placeholders = set(re.findall(r"\$\{([^}]+)\}", frontmatter["prompt"]))
      for p in placeholders:
        if p not in expected_params:
          issues.append(
              f"AI Action '{page}' has placeholder ${{{p}}} in 'prompt' but it"
              " is not defined in parameters."
          )

  if "questionnaire" in frontmatter:
    q = frontmatter["questionnaire"]
    if not isinstance(q, dict):
      issues.append(f"Action '{page}' 'questionnaire' field must be a dict.")
    elif "fields" in q:
      fields = q["fields"]
      if not isinstance(fields, list) or any(
          not isinstance(item, dict) for item in fields
      ):
        issues.append(
            f"Action '{page}' 'questionnaire.fields' must be a list of dicts."
        )
      else:
        for f in fields:
          if not isinstance(f, dict):
            continue
          if "name" not in f:
            issues.append(
                f"Action '{page}' questionnaire field is missing 'name'."
            )
          if "question" not in f:
            issues.append(
                f"Action '{page}' questionnaire field is missing 'question'."
            )
          field_name = f.get("name", "<unnamed>")
          if "name" in f and field_name not in expected_params:
            issues.append(
                f"Action '{page}': Questionnaire field '{field_name}' not"
                " declared in parameters"
            )

          if "input_mode" in f:
            input_mode = f["input_mode"]
            if (
                not isinstance(input_mode, str)
                or input_mode not in ALLOWED_QUESTIONNAIRE_INPUT_MODES
            ):
              issues.append(
                  f"Action '{page}' questionnaire field '{field_name}' has"
                  f" invalid input_mode '{input_mode}'. Allowed:"
                  f" {sorted(ALLOWED_QUESTIONNAIRE_INPUT_MODES)}"
              )

          if "options" in f:
            options = f["options"]
            if not isinstance(options, list):
              issues.append(
                  f"Action '{page}' questionnaire field '{field_name}'"
                  " 'options' must be a list."
              )
            else:
              for opt in options:
                if isinstance(opt, str):
                  continue
                elif isinstance(opt, dict):
                  unknown_keys = (
                      set(opt.keys()) - ALLOWED_QUESTIONNAIRE_OPTION_KEYS
                  )
                  if unknown_keys:
                    issues.append(
                        f"Action '{page}' questionnaire field '{field_name}'"
                        " option contains invalid keys:"
                        f" {sorted(unknown_keys)}. Allowed:"
                        f" {sorted(ALLOWED_QUESTIONNAIRE_OPTION_KEYS)}"
                    )
                else:
                  issues.append(
                      f"Action '{page}' questionnaire field '{field_name}'"
                      " options items must be strings or dicts."
                  )

  if "fallback" in frontmatter:
    fb_list = frontmatter["fallback"]
    if not isinstance(fb_list, list) or any(
        not isinstance(item, dict) for item in fb_list
    ):
      issues.append(f"Action '{page}' 'fallback' must be a list of dicts.")
    else:
      parent_is_public = (
          isinstance(frontmatter.get("internal_only"), bool)
          and not frontmatter["internal_only"]
      )
      for idx, entry in enumerate(fb_list):
        unknown_keys = set(entry.keys()) - ALLOWED_ACTION_FALLBACK_KEYS
        if unknown_keys:
          issues.append(
              f"Action '{page}' fallback[{idx}] contains invalid keys:"
              f" {sorted(unknown_keys)}. Allowed:"
              f" {sorted(ALLOWED_ACTION_FALLBACK_KEYS)}"
          )
        raw_target = entry.get("action")
        if not isinstance(raw_target, str):
          issues.append(
              f"Action '{page}' fallback[{idx}] is missing 'action' string."
          )
        else:
          target_name, target_path = _resolve_action_or_workflow_target(
              wiki_dir, raw_target
          )
          if target_name is None:
            issues.append(
                f"Action '{page}' fallback[{idx}] has invalid action reference"
                f" format: '{raw_target}'. Expected '[[name]]'"
            )
          elif target_path is None:
            issues.append(
                f"Action '{page}' fallback[{idx}] refers to non-existent action"
                f" or workflow '{target_name}'"
            )
          elif parent_is_public and _is_target_internal_only(target_path):
            if not (
                isinstance(entry.get("internal_only"), bool)
                and entry["internal_only"]
            ):
              issues.append(
                  f"Action '{page}' is public (internal_only: false) and"
                  f" fallback[{idx}] references internal target '{target_name}'"
                  " without setting 'internal_only': true on the fallback"
                  " entry."
              )

        when_val = entry.get("when")
        if not isinstance(when_val, list) or not when_val:
          issues.append(
              f"Action '{page}' fallback[{idx}] 'when' must be a non-empty"
              " list of trigger strings."
          )
        else:
          for trig in when_val:
            if (
                not isinstance(trig, str)
                or trig not in ALLOWED_FALLBACK_TRIGGERS
            ):
              issues.append(
                  f"Action '{page}' fallback[{idx}] has invalid trigger"
                  f" '{trig}'. Allowed: {sorted(ALLOWED_FALLBACK_TRIGGERS)}"
              )

        reason_val = entry.get("reason")
        if not isinstance(reason_val, str) or not reason_val.strip():
          issues.append(
              f"Action '{page}' fallback[{idx}] 'reason' must be a non-empty"
              " string."
          )

        if "param_mapping" in entry and not isinstance(
            entry["param_mapping"], dict
        ):
          issues.append(
              f"Action '{page}' fallback[{idx}] 'param_mapping' must be a dict."
          )
        if "internal_only" in entry and not isinstance(
            entry["internal_only"], bool
        ):
          issues.append(
              f"Action '{page}' fallback[{idx}] 'internal_only' must be a"
              " boolean."
          )
        if "auto_execute" in entry and not isinstance(
            entry["auto_execute"], bool
        ):
          issues.append(
              f"Action '{page}' fallback[{idx}] 'auto_execute' must be a"
              " boolean."
          )


def _validate_workflow_step_fallbacks_dict(
    fb_dict: Any,
    field_name: str,
    page: str,
    wiki_dir: str,
    forbid_internal_targets: bool,
    issues: list[str],
) -> None:
  """Validates a workflow's step_fallbacks or internal_step_fallbacks object."""
  if not isinstance(fb_dict, dict):
    issues.append(f"Workflow '{page}' '{field_name}' field must be a dict.")
    return

  for step_key, branches in fb_dict.items():
    if not isinstance(step_key, str) or not step_key.strip():
      issues.append(
          f"Workflow '{page}' '{field_name}' keys must be non-empty step name"
          " strings."
      )
      continue
    if (
        not isinstance(branches, list)
        or not branches
        or any(not isinstance(b, dict) for b in branches)
    ):
      issues.append(
          f"Workflow '{page}' '{field_name}[\"{step_key}\"]' must be a"
          " non-empty list of branch dicts."
      )
      continue

    for idx, branch in enumerate(branches):
      unknown_keys = set(branch.keys()) - ALLOWED_WORKFLOW_FALLBACK_KEYS
      if unknown_keys:
        issues.append(
            f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]' contains"
            f" invalid keys: {sorted(unknown_keys)}. Allowed:"
            f" {sorted(ALLOWED_WORKFLOW_FALLBACK_KEYS)}"
        )

      when_val = branch.get("when")
      if not isinstance(when_val, list) or not when_val:
        issues.append(
            f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]' 'when'"
            " must be a non-empty list of trigger strings."
        )
      else:
        for trig in when_val:
          if not isinstance(trig, str) or trig not in ALLOWED_FALLBACK_TRIGGERS:
            issues.append(
                f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]' has"
                f" invalid trigger '{trig}'. Allowed:"
                f" {sorted(ALLOWED_FALLBACK_TRIGGERS)}"
            )

      reason_val = branch.get("reason")
      if not isinstance(reason_val, str) or not reason_val.strip():
        issues.append(
            f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]' 'reason'"
            " must be a non-empty string."
        )

      replace_with = branch.get("replace_with")
      if not isinstance(replace_with, list) or not replace_with:
        issues.append(
            f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]'"
            " 'replace_with' must be a non-empty list of '[[name]]' references."
        )
      else:
        for ref in replace_with:
          target_name, target_path = _resolve_action_or_workflow_target(
              wiki_dir, ref
          )
          if target_name is None:
            issues.append(
                f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]' has"
                f" invalid replace_with reference format: '{ref}'. Expected"
                " '[[name]]'"
            )
          elif target_path is None:
            issues.append(
                f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]'"
                " refers to non-existent action or workflow"
                f" '{target_name}' in replace_with"
            )
          elif forbid_internal_targets and _is_target_internal_only(
              target_path
          ):
            issues.append(
                f"Workflow '{page}' public '{field_name}' references internal"
                f" target '{target_name}'. Internal fallback branches must be"
                " placed in 'internal_step_fallbacks'."
            )

      if "skip_steps" in branch:
        skip_steps = branch["skip_steps"]
        if not isinstance(skip_steps, list):
          issues.append(
              f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]'"
              " 'skip_steps' must be a list of '[[name]]' references."
          )
        else:
          for ref in skip_steps:
            target_name, target_path = _resolve_action_or_workflow_target(
                wiki_dir, ref
            )
            if target_name is None:
              issues.append(
                  f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]' has"
                  f" invalid skip_steps reference format: '{ref}'. Expected"
                  " '[[name]]'"
              )
            elif target_path is None:
              issues.append(
                  f"Workflow '{page}' '{field_name}[\"{step_key}\"][{idx}]'"
                  " refers to non-existent action or workflow"
                  f" '{target_name}' in skip_steps"
              )
            elif forbid_internal_targets and _is_target_internal_only(
                target_path
            ):
              issues.append(
                  f"Workflow '{page}' public '{field_name}' references internal"
                  f" target '{target_name}' in skip_steps. Internal fallback"
                  " branches must be placed in 'internal_step_fallbacks'."
              )


def validate_workflow_json(
    frontmatter: dict[str, Any], page: str, wiki_dir: str, issues: list[str]
) -> None:
  """Validates a workflow JSON document's step references.

  Checks `internal_only`, rejects the deprecated `actions` field, and verifies
  that every `steps` / `internal_steps` entry is a `[[name]]` reference to an
  existing action or workflow, and that public `steps` do not reference
  internal-only actions.

  Args:
    frontmatter: The parsed workflow JSON document.
    page: Wiki-relative path of the workflow, used in messages.
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
  """
  check_internal_only_boolean(page, frontmatter, issues)

  if "actions" in frontmatter:
    issues.append(
        f"Workflow '{page}' uses deprecated 'actions' field. Use 'steps'"
        " instead."
    )
  if "steps" not in frontmatter:
    issues.append(f"Workflow '{page}' is missing required 'steps' field.")
    return

  steps = frontmatter.get("steps")
  if not isinstance(steps, list):
    issues.append(f"Workflow '{page}' 'steps' field must be a list.")
    return

  for step_ref in steps:
    if not isinstance(step_ref, str):
      issues.append(
          f"Workflow '{page}' step reference must be a string. Found:"
          f" {step_ref}"
      )
      continue

    match = re.fullmatch(r"\[\[(.*)\]\]", step_ref)
    if not match:
      issues.append(
          f"Workflow '{page}' has invalid step reference format: '{step_ref}'."
          " Expected '[[name]]'"
      )
      continue

    step_name = match.group(1)
    action_path = os.path.join(wiki_dir, "actions", f"{step_name}.json")
    workflow_path = os.path.join(wiki_dir, "workflows", f"{step_name}.json")
    if not os.path.exists(action_path) and not os.path.exists(workflow_path):
      issues.append(
          f"Workflow '{page}' refers to non-existent action or workflow"
          f" '{step_name}'"
      )
    else:
      target_path = (
          action_path if os.path.exists(action_path) else workflow_path
      )
      if _is_target_internal_only(target_path):
        issues.append(
            f"Workflow '{page}' public 'steps' array references internal"
            " action"
            f" '{step_name}'. Internal actions must be placed in"
            " 'internal_steps'."
        )

  if "internal_steps" in frontmatter:
    internal_steps = frontmatter.get("internal_steps")
    if not isinstance(internal_steps, list):
      issues.append(f"Workflow '{page}' 'internal_steps' field must be a list.")
    else:
      for step_ref in internal_steps:
        if not isinstance(step_ref, str):
          issues.append(
              f"Workflow '{page}' internal_steps reference must be a string."
              f" Found: {step_ref}"
          )
          continue

        match = re.fullmatch(r"\[\[(.*)\]\]", step_ref)
        if not match:
          issues.append(
              f"Workflow '{page}' has invalid internal_steps reference"
              f" format: '{step_ref}'. Expected '[[name]]'"
          )
          continue

        step_name = match.group(1)
        action_path = os.path.join(wiki_dir, "actions", f"{step_name}.json")
        workflow_path = os.path.join(wiki_dir, "workflows", f"{step_name}.json")
        if not os.path.exists(action_path) and not os.path.exists(
            workflow_path
        ):
          issues.append(
              f"Workflow '{page}' refers to non-existent action or workflow"
              f" '{step_name}' in internal_steps"
          )

  workflow_is_public = (
      isinstance(frontmatter.get("internal_only"), bool)
      and not frontmatter["internal_only"]
  )
  if "step_fallbacks" in frontmatter:
    _validate_workflow_step_fallbacks_dict(
        frontmatter["step_fallbacks"],
        "step_fallbacks",
        page,
        wiki_dir,
        forbid_internal_targets=workflow_is_public,
        issues=issues,
    )
  if "internal_step_fallbacks" in frontmatter:
    _validate_workflow_step_fallbacks_dict(
        frontmatter["internal_step_fallbacks"],
        "internal_step_fallbacks",
        page,
        wiki_dir,
        forbid_internal_targets=False,
        issues=issues,
    )


def _has_any_match(
    keywords: Sequence[str], headers_list: Sequence[str]
) -> bool:
  """Returns True if any keyword is a case-insensitive substring of a header."""
  for h in headers_list:
    h_lower = h.lower()
    if any(kw.lower() in h_lower for kw in keywords):
      return True
  return False


def validate_markdown_format(
    body: str, page: str, p_type: str | None, issues: list[str]
) -> None:
  """Checks that a markdown page has the `##` sections its type requires.

  Args:
    body: The page body with frontmatter removed.
    page: Wiki-relative path of the page, used in messages.
    p_type: The page's frontmatter `type` (e.g. "case", "concept", "symptom").
    issues: List to which issue messages are appended.
  """
  headers = re.findall(r"^##\s+(.+)$", body, re.MULTILINE)
  headers_clean = [h.strip() for h in headers]

  if p_type == "case":
    if any("Failure Table" in h for h in headers_clean):
      required = [
          "Overview",
          "Failure Table",
          "Key Learnings",
          "Detailed Analysis & Reasoning",
          "Connections",
      ]
      for req in required:
        found = False
        for h in headers_clean:
          if req == h or (
              req == "Detailed Analysis & Reasoning"
              and "Analysis & Reasoning" in h
          ):
            found = True
            break
        if not found:
          issues.append(
              f"Case study '{page}' (Failure Table type) is missing required"
              f" header: '## {req}'"
          )
    else:
      if not _has_any_match(
          [
              "Summary",
              "Overview",
              "Introduction",
          ],
          headers_clean,
      ):
        issues.append(
            f"Case study '{page}' (Narrative type) is missing a"
            " summary/overview header ('## Summary' or '## Incident Overview')"
        )
      if not _has_any_match(
          [
              "Challenge",
              "Diagnosis",
              "Symptom",
              "Root Cause",
              "Resolution",
              "Timeline",
          ],
          headers_clean,
      ):
        issues.append(
            f"Case study '{page}' (Narrative type) is missing debugging details"
            " (must have at least one header like '## Diagnosis', '## Root"
            " Cause', etc.)"
        )
      if not _has_any_match(
          [
              "Takeaways",
              "Learnings",
              "Lessons",
          ],
          headers_clean,
      ):
        issues.append(
            f"Case study '{page}' (Narrative type) is missing takeaways header"
            " ('## Key Takeaways' or '## Lessons Learned')"
        )
      if "Connections" not in headers_clean:
        issues.append(
            f"Case study '{page}' (Narrative type) is missing '## Connections'"
            " header"
        )

  elif p_type == "concept":
    if not _has_any_match(
        [
            "Overview",
            "Summary",
            "Introduction",
            "Definition",
            "Description",
            "Reference",
            "Context",
            "Symptom",
            "Objective",
            "Goal",
            "Purpose",
        ],
        headers_clean,
    ):
      issues.append(
          f"Concept '{page}' is missing an Overview/Summary/Definition header."
      )
    if not _has_any_match(
        [
            "Architecture",
            "Design",
            "Technical",
            "Key Metrics",
            "Internal",
            "Structure",
            "Common Issues",
            "Failure Modes",
            "Usage",
            "Flags",
            "Specifics",
            "Field",
            "YAML",
            "Detail",
            "Example",
            "Reference",
            "Phase",
            "Flow",
            "Step",
            "Process",
            "Strategy",
            "Pattern",
            "Heuristic",
            "Rule",
            "Schema",
            "Column",
            "Workflow",
            "Concept",
            "Spec",
            "Specs",
            "Parameters",
            "Vars",
            "Abstractions",
            "Dynamics",
            "Priority",
            "Cell",
            "Pool",
            "Domain",
            "View",
            "Cube",
            "Pod",
            "Capacity",
            "Comparison",
            "Compare",
            "Mechanism",
            "Mechanisms",
        ],
        headers_clean,
    ):
      issues.append(
          f"Concept '{page}' is missing an Architecture/Design/Technical"
          " Details header."
      )
    if not _has_any_match(
        [
            "Operational",
            "Debugging",
            "Significance",
            "Why it matters",
            "Troubleshooting",
            "Impact",
            "Common Issues",
            "Usage",
            "Field",
            "Details",
            "Reference",
            "Stalls",
            "Rule",
            "Rules",
            "Best Practice",
            "Trouble",
            "Issues",
            "Problems",
            "Failures",
            "Faults",
            "Performance",
            "Behavior",
            "Label",
            "Labeling",
            "Marking",
            "Action",
            "Leak",
            "Overload",
            "OOM",
            "Crash",
            "Crashes",
            "Hang",
            "Hangs",
            "Timeout",
            "Timeouts",
            "Warning",
            "Warnings",
            "Error",
            "Errors",
            "Stuck",
        ],
        headers_clean,
    ):
      issues.append(
          f"Concept '{page}' is missing an Operational Impact/Debugging"
          " Significance header."
      )
    if not _has_any_match(
        [
            "Connections",
            "Related",
            "Reference",
        ],
        headers_clean,
    ):
      issues.append(f"Concept '{page}' is missing '## Connections' header.")

  elif p_type == "symptom":
    is_routing = _has_any_match(["Routing"], headers_clean)
    if not _has_any_match(
        [
            "Overview",
            "Description",
            "Summary",
            "Definition",
            "Context",
            "Symptom",
            "Objective",
            "Goal",
            "Purpose",
        ],
        headers_clean,
    ):
      issues.append(
          f"Symptom '{page}' is missing an Overview/Description header."
      )
    if not is_routing and not _has_any_match(
        [
            "Log",
            "Signature",
            "Error",
            "Message",
            "Pattern",
            "Example",
            "Filter",
            "Query",
            "Identify",
            "Identification",
        ],
        headers_clean,
    ):
      issues.append(
          f"Symptom '{page}' is missing a Log Signatures/Error Messages header."
      )
    if not _has_any_match(
        [
            "Verification",
            "Diagnos",
            "Steps",
            "Check",
            "How to confirm",
            "Actionable",
            "Triage",
            "Value",
            "Routing",
            "Identify",
            "Identification",
            "Checklist",
            "Procedure",
            "Flow",
            "Analysis",
            "Playbook",
            "Playbooks",
        ],
        headers_clean,
    ):
      issues.append(
          f"Symptom '{page}' is missing a Verification"
          " Steps/Diagnosis/Actionable Steps header."
      )
    if not is_routing and not _has_any_match(
        [
            "Playbook",
            "Playbooks",
            "Root Cause",
            "Resolution",
            "Action",
            "Mitigation",
            "Cause",
            "Potential Cause",
            "Value",
            "Routing",
            "Issues",
            "Problems",
            "Failures",
            "Faults",
        ],
        headers_clean,
    ):
      issues.append(
          f"Symptom '{page}' is missing a Playbooks/Root Cause/Potential Causes"
          " header."
      )
    if not _has_any_match(["Connections", "Related", "Routing"], headers_clean):
      issues.append(f"Symptom '{page}' is missing '## Connections' header.")

  elif p_type == "entity":
    if not _has_any_match(
        [
            "Overview",
            "Summary",
            "Description",
            "Context",
            "Symptom",
            "Objective",
            "Goal",
            "Purpose",
        ],
        headers_clean,
    ):
      issues.append(f"Entity '{page}' is missing an Overview/Summary header.")
    if not _has_any_match(
        [
            "Specifications",
            "Architecture",
            "Technical Specs",
            "Details",
            "Spec",
            "Specs",
            "Structure",
        ],
        headers_clean,
    ):
      issues.append(
          f"Entity '{page}' is missing a Technical Specifications/Architecture"
          " header."
      )
    if not _has_any_match(
        [
            "Failure Modes",
            "Common Failure",
            "Issues",
            "Symptom",
            "Problems",
            "Failures",
            "Faults",
        ],
        headers_clean,
    ):
      issues.append(
          f"Entity '{page}' is missing '## Common Failure Modes' header."
      )
    if not _has_any_match(["Connections", "Related"], headers_clean):
      issues.append(f"Entity '{page}' is missing '## Connections' header.")

  elif p_type == "tool":
    if not _has_any_match(
        [
            "Overview",
            "Summary",
            "Description",
            "Context",
            "Symptom",
            "Objective",
            "Goal",
            "Purpose",
            "Definition",
        ],
        headers_clean,
    ):
      issues.append(f"Tool '{page}' is missing an Overview/Summary header.")
    if not _has_any_match(
        [
            "Usage",
            "Command",
            "How to Run",
            "Examples",
            "Access",
            "Run",
            "Tips",
            "Tricks",
            "Cases",
            "Procedure",
            "How to",
        ],
        headers_clean,
    ):
      issues.append(f"Tool '{page}' is missing a Usage/Commands header.")
    if not _has_any_match(
        [
            "Troubleshooting",
            "Application",
            "Scenario",
            "Use Case",
            "Triage",
            "Tips",
            "Tricks",
            "Cases",
            "Issues",
            "Problems",
            "Failures",
            "Faults",
            "Debugging",
            "Analysis",
            "Output",
            "Diagnostic",
        ],
        headers_clean,
    ):
      issues.append(
          f"Tool '{page}' is missing a Troubleshooting/Application header."
      )
    if not _has_any_match(["Connections", "Related"], headers_clean):
      issues.append(f"Tool '{page}' is missing '## Connections' header.")

  elif p_type == "playbook":
    if not _has_any_match(
        [
            "Overview",
            "Summary",
            "Introduction",
            "Definition",
            "Description",
            "Gotchas",
            "Failure Modes",
            "Context",
            "Objective",
            "Goal",
            "Purpose",
        ],
        headers_clean,
    ):
      issues.append(f"Playbook '{page}' is missing an Overview/Summary header.")
    if not _has_any_match(
        [
            "Diagnos",
            "Triage",
            "Step",
            "Troubleshooting",
            "Investigation",
            "Methodology",
            "Approach",
            "Common Issues",
            "Gotchas",
            "Failure Modes",
            "Verification",
            "Checklist",
            "Check",
            "Identify",
            "Process",
            "Procedure",
            "Flow",
            "Analysis",
            "How to",
            "Checklist",
            "Principle",
            "Principles",
        ],
        headers_clean,
    ):
      issues.append(
          f"Playbook '{page}' is missing a Diagnostic"
          " Steps/Triage/Troubleshooting header."
      )
    if not _has_any_match(
        [
            "Mitigation",
            "Resolution",
            "Action",
            "Fix",
            "Workaround",
            "Common Issues",
            "Gotchas",
            "Failure Modes",
            "Mechanism",
            "Platform",
            "Deployment",
            "Command",
            "Access",
            "Run",
            "Procedure",
            "How to",
            "Checklist",
            "Check",
            "Identify",
            "Process",
            "Flow",
            "Analysis",
            "Issues",
            "Problems",
            "Failures",
            "Faults",
        ],
        headers_clean,
    ):
      issues.append(
          f"Playbook '{page}' is missing a Mitigations/Resolution/Gotchas"
          " header."
      )
    if not _has_any_match(["Connections", "Related"], headers_clean):
      issues.append(f"Playbook '{page}' is missing '## Connections' header.")


_LEAK_PATTERNS = (
    r"prototypes/projects/pathways_interruptions",
    r"workspace\.prototypes",
    r"workspace\.cloud",
    r"workspace\.cloud",
    r"internal[./]monorepo[./]path",
    r"/google/src/cloud/",
    r"/workspace/internal/",
)
# Compiled once at import, and combined into a single alternation so the common
# case (a clean file) costs one pass instead of one per pattern. The individual
# patterns are only re-tested on the rare file that actually matches, so the
# issue message still names the specific pattern.
_LEAK_ANY_RE = re.compile("|".join(f"(?:{p})" for p in _LEAK_PATTERNS))
_LEAK_RES = tuple((p, re.compile(p)) for p in _LEAK_PATTERNS)


def check_toplevel_doc_links(
    wiki_dir: str, all_pages: list[str], issues: list[str]
) -> None:
  """Validates [[wiki-links]] in exempt top-level docs such as TUTORIAL.md.

  Those files are excluded from the page walk because they are documentation
  *about* the wiki, not pages in it -- so frontmatter and orphan rules do not
  apply. The exclusion would otherwise exempt them from link integrity too, and
  a tutorial is mostly references: it is the document where a dead link does the
  most damage, because a newcomer cannot tell a stale pointer from their own
  mistake.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    all_pages: Wiki-relative paths of every page found by the main walk.
    issues: List to which issue messages are appended.
  """
  known = {os.path.splitext(os.path.basename(p))[0] for p in all_pages}
  srcs_dir = os.path.join(wiki_dir, "srcs")
  if os.path.isdir(srcs_dir):
    for entry in os.listdir(srcs_dir):
      entry_dir = os.path.join(srcs_dir, entry)
      if os.path.isdir(entry_dir):
        for f in os.listdir(entry_dir):
          if f.endswith(".md"):
            known.add(os.path.splitext(f)[0])
  for name in ("TUTORIAL.md",):
    full = os.path.join(wiki_dir, name)
    if not os.path.exists(full):
      continue
    try:
      content = _read_text(full)
    except _READ_ERRORS:
      continue
    for link in re.findall(r"\[\[([^\]]+)\]\]", content):
      target = (
          link.split("|")[0].split("#")[0].strip().lower().replace(" ", "-")
      )
      if target and target not in known:
        issues.append(f"Broken wiki-link in '{name}': [[{link}]]")


# Streaming/blocking flags. The glue runner captures a command's output to
# completion, so an action that streams indefinitely hangs the whole session.
_BLOCKING_PATTERNS = (
    (r"kubectl\b[^|;]*\s-w\b", "kubectl -w"),
    (r"kubectl\b[^|;]*\s--watch\b", "kubectl --watch"),
    (r"\blogs\b[^|;]*\s-f\b", "logs -f"),
    (r"--follow\b", "--follow"),
    (r"\btail\s+-f\b", "tail -f"),
)
# Flags that make an otherwise-streaming command terminate.
_BOUNDED_MARKERS = ("--request-timeout", "timeout ", "--timeout", "head -")


def check_blocking_commands(
    wiki_dir: str,
    issues: list[str],
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Flags command actions that stream forever and would hang the runner.

  `python_glue_runner.py` captures stdout to completion. An unbounded watch or
  follow therefore never returns, blocking the session with no recovery. Such
  actions must bound themselves (e.g. `--request-timeout=60s`).

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if not os.path.isdir(actions_dir):
      return
    actions_items = []
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      try:
        actions_items.append((f, _read_json(os.path.join(actions_dir, f))))
      except _JSON_FILE_ERRORS:
        continue
  else:
    actions_items = sorted(actions_cache.items())

  for f, d in actions_items:
    if not isinstance(d, dict) or d.get("execution_mode") != "command":
      continue
    instr = d.get("instruction", "") or ""
    if any(mark in instr for mark in _BOUNDED_MARKERS):
      continue
    if d.get("timeout_seconds"):
      continue
    for pattern, label in _BLOCKING_PATTERNS:
      if re.search(pattern, instr):
        issues.append(
            f"Blocking Action 'actions/{f}': instruction uses '{label}', which "
            "streams indefinitely and will hang the glue runner. Bound it "
            "(e.g. --request-timeout=<n>s) or set 'timeout_seconds'."
        )
        break


def _read_text(path: str, encoding: str = "utf-8") -> str:
  """Reads a file and closes it.

  `_read_text(path)` leaks the handle until the refcount drops, which shows up
  as ResourceWarning under `-W error` and is simply untidy at ~200 files a run.

  Args:
    path: Path of the file to read.
    encoding: Text encoding used to decode the file.

  Returns:
    The file's contents as a string.
  """
  with open(path, encoding=encoding) as f:
    return f.read()


def _read_json(path: str, encoding: str = "utf-8") -> Any:
  """Parses a JSON file and closes it."""
  with open(path, encoding=encoding) as f:
    return json.load(f)


CANONICAL_PAGE_METADATA_KEYS = (
    "title",
    "description",
    "type",
    "internal_only",
    "tags",
    "sources",
    "created",
    "updated",
    "contested",
    "contradictions",
)
CANONICAL_PAGE_KEYS = CANONICAL_PAGE_METADATA_KEYS

CANONICAL_SRC_METADATA_KEYS = (
    "sha256",
    "source_url",
    "title",
    "description",
    "type",
    "internal_only",
    "tags",
    "sources",
    "created",
    "updated",
    "contested",
    "contradictions",
)
CANONICAL_SRC_KEYS = CANONICAL_SRC_METADATA_KEYS

CANONICAL_WORKFLOW_METADATA_KEYS = (
    "title",
    "type",
    "description",
    "internal_only",
    "tags",
    "sources",
    "created",
    "updated",
    "entrypoints",
    "summary_fields",
    "steps",
    "internal_steps",
    "step_notes",
    "step_fallbacks",
    "internal_step_fallbacks",
)
CANONICAL_WORKFLOW_KEYS = CANONICAL_WORKFLOW_METADATA_KEYS

CANONICAL_ACTION_METADATA_KEYS = (
    "title",
    "type",
    "description",
    "internal_only",
    "tags",
    "sources",
    "created",
    "updated",
    "entrypoints",
    "execution_mode",
    "instruction",
    "prompt",
    "params",
    "outputs",
    "summary_fields",
    "requires_approval",
    "approval_exempt_params",
    "timeout_seconds",
    "deprecated",
    "superseded_by",
    "questionnaire",
    "fallback",
)
CANONICAL_ACTION_KEYS = CANONICAL_ACTION_METADATA_KEYS


class FlowSeq(list):
  """Marker class for YAML sequences dumped in flow style `[a, b]`."""


def _flow_seq_rep(dumper: yaml.SafeDumper, data: FlowSeq) -> yaml.SequenceNode:
  """Represents a FlowSeq as a YAML sequence in flow style."""
  return dumper.represent_sequence(
      "tag:yaml.org,2002:seq", data, flow_style=True
  )


class CleanYamlDumper(yaml.SafeDumper):
  """SafeDumper configured to render FlowSeq instances in flow style."""


CleanYamlDumper.add_representer(FlowSeq, _flow_seq_rep)


def _to_date_if_date_str(val: Any) -> Any:
  """Converts a `YYYY-MM-DD` string into a `datetime.date` when valid."""
  if isinstance(val, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
    try:
      return datetime.datetime.strptime(val, "%Y-%m-%d").date()
    except ValueError:
      return val
  return val


def _make_flow_lists(obj: Any) -> Any:
  """Recursively wraps flat lists of scalars in FlowSeq for YAML formatting."""
  if isinstance(obj, list):
    if all(not isinstance(x, (dict, list)) for x in obj):
      return FlowSeq([_make_flow_lists(x) for x in obj])
    return [_make_flow_lists(x) for x in obj]
  if isinstance(obj, dict):
    return {k: _make_flow_lists(v) for k, v in obj.items()}
  return obj


def dump_clean_yaml(data: dict[str, Any]) -> str:
  """Dumps a dict to YAML with inline lists and unquoted dates/booleans."""
  formatted = _make_flow_lists(data)
  out = yaml.dump(
      formatted,
      Dumper=CleanYamlDumper,
      sort_keys=False,
      default_flow_style=False,
  )
  return out.strip()


def get_canonical_key_order(
    keys: Collection[str], canonical_list: Sequence[str]
) -> list[str]:
  """Returns canonical keys in declared order, then extras alphabetically."""
  canonical_keys = [k for k in canonical_list if k in keys]
  additional_keys = sorted([k for k in keys if k not in canonical_list])
  return canonical_keys + additional_keys


def order_frontmatter_dict(
    data: dict[str, Any], canonical_keys: Sequence[str]
) -> dict[str, Any]:
  """Returns `data` with keys in canonical order and date strings as dates."""
  expected_keys = get_canonical_key_order(data.keys(), canonical_keys)
  ordered = {}
  for k in expected_keys:
    val = data[k]
    if k in ("created", "updated") or (
        isinstance(val, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", val)
    ):
      val = _to_date_if_date_str(val)
    ordered[k] = val
  return ordered


def sort_and_format_frontmatter(content: str, is_src: bool = False) -> str:
  """Sorts YAML frontmatter keys into canonical order and formats cleanly.

  Accepts either a full markdown string with frontmatter or a raw YAML block.
  Preserves body content exactly.

  Args:
    content: A markdown document with `---` frontmatter, or a bare YAML block.
    is_src: Whether to use the canonical key order for raw sources in `srcs/`
      rather than the one for wiki pages.

  Returns:
    The content with frontmatter keys reordered, or `content` unchanged if
    the keys are already in order or the YAML is not a parseable mapping.
  """
  canonical_keys = (
      CANONICAL_SRC_METADATA_KEYS if is_src else CANONICAL_PAGE_METADATA_KEYS
  )
  m = re.search(r"^---\s*\n(.*?)\n---([ \t]*\r?\n*)", content, re.DOTALL)
  if m:
    fm_text = m.group(1)
    trailing = m.group(2)
    body = content[m.end() :]
    try:
      d = yaml.safe_load(fm_text)
    except _YAML_ERRORS:
      return content
    if not isinstance(d, dict):
      return content

    expected_keys = get_canonical_key_order(d.keys(), canonical_keys)
    if list(d.keys()) == expected_keys:
      return content

    ordered = order_frontmatter_dict(d, canonical_keys)
    formatted_fm = dump_clean_yaml(ordered)
    if not trailing:
      trailing = "\n"
    return f"---\n{formatted_fm}\n---{trailing}{body}"
  else:
    try:
      d = yaml.safe_load(content)
    except _YAML_ERRORS:
      return content
    if not isinstance(d, dict):
      return content
    expected_keys = get_canonical_key_order(d.keys(), canonical_keys)
    if list(d.keys()) == expected_keys:
      return content
    ordered = order_frontmatter_dict(d, canonical_keys)
    return dump_clean_yaml(ordered) + "\n"


def sort_and_format_json(
    content: str | dict[str, Any], is_action: bool = False
) -> str | dict[str, Any]:
  """Sorts top-level JSON metadata keys into canonical order."""
  canonical_keys = (
      CANONICAL_ACTION_METADATA_KEYS
      if is_action
      else CANONICAL_WORKFLOW_METADATA_KEYS
  )
  is_str = isinstance(content, str)
  if is_str:
    try:
      d = json.loads(content)
    except json.JSONDecodeError:
      return content
  elif isinstance(content, dict):
    d = content
  else:
    return content

  if not isinstance(d, dict):
    return content

  expected_keys = get_canonical_key_order(d.keys(), canonical_keys)
  ordered = {k: d[k] for k in expected_keys}
  res = json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"
  if is_str and list(d.keys()) == expected_keys and content == res:
    return content
  return res


def check_and_sort_file(full_path: str, wiki_dir: str | None = None) -> bool:
  """Checks and sorts metadata in a single file in-place if out of order.

  Args:
    full_path: Absolute path of the file to check.
    wiki_dir: Unused; accepted for call-site compatibility.

  Returns:
    True if the file was modified, False otherwise.
  """
  del wiki_dir  # Unused.
  if not os.path.isfile(full_path):
    return False

  if full_path.endswith(".json"):
    parts = full_path.split(os.sep)
    is_action = "actions" in parts
    is_workflow = "workflows" in parts
    if not is_action and not is_workflow:
      return False
    try:
      content = _read_text(full_path)
      sorted_content = sort_and_format_json(content, is_action=is_action)
      if sorted_content != content:
        with open(full_path, "w", encoding="utf-8") as f:
          f.write(sorted_content)
        return True
    except _READ_ERRORS:
      return False

  elif full_path.endswith(".md"):
    parts = full_path.split(os.sep)
    is_src = "srcs" in parts
    try:
      content = _read_text(full_path)
      sorted_content = sort_and_format_frontmatter(content, is_src=is_src)
      if sorted_content != content:
        with open(full_path, "w", encoding="utf-8") as f:
          f.write(sorted_content)
        return True
    except _READ_ERRORS + _YAML_ERRORS:
      return False

  return False


def check_and_sort_metadata(wiki_dir: str) -> list[str]:
  """Sorts metadata in-place for pages, sources, actions, and workflows."""
  modified_files = []
  ignored_files = {
      "README.md",
      "SCHEMA.md",
      "LOG.md",
      "INDEX.md",
      "TUTORIAL.md",
      "ADVISOR_NOTES.md",
      "USER_LOG.md",
      "requirements.txt",
      "wiki_linter.py",
  }

  for root, dirs, files in os.walk(wiki_dir):
    prune_scan_dirs(root, dirs)
    for f in sorted(files):
      if f in ignored_files or _ROTATED_LOG_RE.fullmatch(f):
        continue
      if f.endswith((".md", ".json")):
        p = os.path.join(root, f)
        # tools/scripts/ holds script inputs, not wiki pages (see run_linter);
        # their frontmatter must not be rewritten.
        if os.path.relpath(p, wiki_dir).startswith("tools/scripts/"):
          continue
        if check_and_sort_file(p, wiki_dir=wiki_dir):
          modified_files.append(os.path.relpath(p, wiki_dir))

  srcs_dir = os.path.join(wiki_dir, "srcs")
  if os.path.isdir(srcs_dir):
    for root, dirs, files in os.walk(srcs_dir):
      for f in sorted(files):
        if f.endswith(".md"):
          p = os.path.join(root, f)
          if check_and_sort_file(p, wiki_dir=wiki_dir):
            modified_files.append(os.path.relpath(p, wiki_dir))

  return modified_files


@functools.cache
def _argparse_flags(script_path: str) -> set[str] | None:
  """Long-option names declared by a script's argparse calls, or None.

  Static parse only -- importing a diagnostic script would execute it.

  Args:
    script_path: Absolute path to the Python script.

  Returns:
    Set of normalized flag names, or None if the script cannot be parsed.
  """
  try:
    tree = ast.parse(_read_text(script_path))
  except (OSError, SyntaxError):
    return None
  flags = set()
  for node in ast.walk(tree):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ):
      for arg in node.args:
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("-")
        ):
          flags.add(arg.value.lstrip("-").replace("-", "_"))
  return flags


def check_action_script_flags(
    wiki_dir: str,
    issues: list[str],
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Flags actions passing options their target script does not accept.

  argparse exits 2 with "unrecognized arguments" -- so the action fails at
  runtime, mid-investigation, with a stack trace rather than a diagnosis. Two
  such defects reached production undetected because nothing checks the action's
  instruction against the script's actual interface.

  Instructions may pipe several scripts together, so the accepted set is the
  union over every script named in the instruction.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if not os.path.isdir(actions_dir):
      return
    actions_items = []
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      try:
        actions_items.append((f, _read_json(os.path.join(actions_dir, f))))
      except _JSON_FILE_ERRORS:
        continue
  else:
    actions_items = sorted(actions_cache.items())

  for f, d in actions_items:
    if not isinstance(d, dict) or d.get("execution_mode") != "command":
      continue
    instr = d.get("instruction", "") or ""
    scripts = re.findall(r"python3\s+([\w./-]+\.py)", instr)
    if not scripts:
      continue
    accepted = set()
    for rel in scripts:
      full = os.path.join(wiki_dir, rel)
      declared = _argparse_flags(full)
      if declared is None:
        accepted = None
        break
      accepted |= declared
    if accepted is None:
      continue
    used = {
        u.replace("-", "_") for u in re.findall(r"--([A-Za-z0-9_-]+)", instr)
    }
    unknown = sorted(used - accepted)
    if unknown:
      issues.append(
          f"Action 'actions/{f}' passes flags its script does not accept: "
          f"{unknown}. argparse will exit 2 with 'unrecognized arguments'."
      )


# Commands that change cluster or fleet state. Matched on the instruction rather
# than the action name: `mitigate-and-escalate` cordons a node and
# `enable-coredns-nodelocaldns` updates a cluster, and neither name suggests it.
_MUTATING_PATTERNS = (
    (
        (
            r"kubectl\s+(?:-\S+\s+)*"
            r"(?:patch|apply|delete|scale|taint|label|annotate|drain|cordon"
            r"|uncordon|edit|replace|create|set)\b"
        ),
        "kubectl mutation",
    ),
    (
        (
            r"gcloud\s+(?!logging\s+read)\S+[^|]*?"
            r"\b(?:delete|upgrade|update|resize|repair|reset|stop|start)\b"
        ),
        "gcloud mutation",
    ),
    # Verb-less mutations: the command name carries the effect, so the
    # generic gcloud/kubectl verb patterns above never match them.
    (
        r"gcloud\s+compute\s+instances\s+report-host-as-faulty\b",
        "host marked faulty",
    ),
    (r"\bxpk\s+workload\s+create\b", "xpk workload creation"),
)


def check_mutating_actions_require_approval(
    wiki_dir: str,
    issues: list[str],
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Flags state-changing actions that do not set `requires_approval`.

  `/troubleshoot` and the practitioner agent both promise to ask before running
  a mutating action, and both decide that from this flag. An action that changes
  state without setting it is executed silently: nine did, including
  `upgrade-gke-cluster` (`--quiet`, on the control plane) and `taint-node`.

  `gcloud logging read` is excluded explicitly -- audit filters legitimately
  contain method names like `compute.instances.delete`, which is a read.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if not os.path.isdir(actions_dir):
      return
    actions_items = []
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      try:
        actions_items.append((f, _read_json(os.path.join(actions_dir, f))))
      except _JSON_FILE_ERRORS:
        continue
  else:
    actions_items = sorted(actions_cache.items())

  for f, d in actions_items:
    if not isinstance(d, dict):
      continue
    # Only a genuine boolean true counts; a truthy non-bool does not.
    if isinstance(d.get("requires_approval"), bool) and d["requires_approval"]:
      continue
    # `ai` actions mutate too: their prompt instructs a subagent to run
    # `kubectl patch` or an isolate. Gating on
    # execution_mode == "command" leaves the whole `ai-` family
    # unchecked, so scan the prompt as well.
    if d.get("execution_mode") not in ("command", "ai"):
      continue
    instr = (
        (d.get("instruction", "") or "") + "\n" + (d.get("prompt", "") or "")
    )
    for pattern, label in _MUTATING_PATTERNS:
      m = re.search(pattern, instr)
      if m:
        issues.append(
            f"Mutating Action 'actions/{f}' does not set "
            f"'requires_approval': true -- instruction performs a {label} "
            f"({m.group(0)[:48]!r}). It would run without asking the user."
        )
        break


# Fields the glue runner can always synthesise itself, regardless of script.
_RUNNER_SYNTHESIZED = frozenset({
    "output_file",
    "execution_status",
    "affected_workload",
    "original_count",
    "deduplicated_count",
    "masking_ratio",
    "compression_ratio",
    "unmasked_entries",
    "extracted_log_count",
    "raw_lines_count",
    "combined_messages_count",
    "recommended_fallbacks",
    "_stdout_lines",
})


def check_summary_fields_are_emitted(
    wiki_dir: str,
    issues: list[str],
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Flags summary_fields the target script never produces.

  A declared field the script does not emit renders as `N/A`, and worse, the
  runner's name-based fallbacks may fill it with a line count that contradicts
  the script's own report. Ten actions carried fields copy-pasted from unrelated
  actions -- `trace-hbm-ecc-errors` declared `matched_symptoms_count`, which
  belongs to symptom triage and has nothing to do with ECC errors.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if not os.path.isdir(actions_dir):
      return
    actions_items = []
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      try:
        actions_items.append((f, _read_json(os.path.join(actions_dir, f))))
      except _JSON_FILE_ERRORS:
        continue
  else:
    actions_items = sorted(actions_cache.items())

  for f, d in actions_items:
    if not isinstance(d, dict) or d.get("execution_mode") != "command":
      continue
    fields = d.get("summary_fields") or []
    m = re.search(r"python3\s+([\w./-]+\.py)", d.get("instruction", "") or "")
    if not fields or not m:
      continue
    script = os.path.join(wiki_dir, m.group(1))
    if not os.path.exists(script):
      continue
    try:
      src = _read_text(script)
    except OSError:
      continue
    missing = [
        x for x in fields if x not in _RUNNER_SYNTHESIZED and x not in src
    ]
    if missing:
      issues.append(
          f"Action 'actions/{f}' declares summary_fields its script never "
          f"emits: {missing} (not found in {m.group(1)}). They will report N/A."
      )


# Directory -> the `type` a page in it must declare.
_DIR_TYPES = types.MappingProxyType({
    "concepts": "concept",
    "symptoms": "symptom",
    "playbooks": "playbook",
    "entities": "entity",
    "tools": "tool",
    "cases": "case",
})


def check_page_type_matches_directory(
    wiki_dir: str,
    issues: list[str],
    file_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
  """Flags pages whose `type` disagrees with the directory they live in.

  A page's directory is how both agents and the INDEX decide what it is, so a
  mismatch makes it findable as the wrong kind of thing.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    file_cache: Optional mapping of wiki-relative path to the per-file cache
      entry built by `run_linter`; when None, files are read from disk.
  """
  for sub, expected in sorted(_DIR_TYPES.items()):
    d = os.path.join(wiki_dir, sub)
    if not os.path.isdir(d):
      d = os.path.join(wiki_dir, "srcs", sub)
    if not os.path.isdir(d):
      continue
    rel_sub = os.path.relpath(d, wiki_dir)
    sub_prefix = f"{rel_sub}/"

    if file_cache is not None and any(
        p.startswith(sub_prefix) for p in file_cache
    ):
      matching_pages = sorted(
          p
          for p in file_cache.keys()
          if p.startswith(sub_prefix)
          and p.endswith(".md")
          and "/" not in p[len(sub_prefix) :]
      )
      for rel in matching_pages:
        entry = file_cache[rel]
        if not entry["has_fm"]:
          issues.append(f"Page '{rel}' has no YAML frontmatter.")
          continue
        if entry["frontmatter_error"]:
          issues.append(f"Page '{rel}' has unparseable frontmatter.")
          continue
        fm = entry["frontmatter"] or {}
        got = fm.get("type")
        if not got:
          issues.append(
              f"Page '{rel}' declares no 'type' (expected '{expected}')."
          )
        elif got != expected:
          issues.append(
              f"Page '{rel}' declares type '{got}' but lives in '{sub}/', "
              f"which requires '{expected}'."
          )
    else:
      for f in sorted(os.listdir(d)):
        if not f.endswith(".md"):
          continue
        rel = f"{rel_sub}/{f}"
        try:
          text = _read_text(os.path.join(d, f))
        except OSError:
          continue
        if not text.startswith("---"):
          issues.append(f"Page '{rel}' has no YAML frontmatter.")
          continue
        try:
          fm = yaml.safe_load(text.split("---", 2)[1]) or {}
        except _YAML_ERRORS + (IndexError,):
          issues.append(f"Page '{rel}' has unparseable frontmatter.")
          continue
        got = fm.get("type")
        if not got:
          issues.append(
              f"Page '{rel}' declares no 'type' (expected '{expected}')."
          )
        elif got != expected:
          issues.append(
              f"Page '{rel}' declares type '{got}' but lives in '{sub}/', "
              f"which requires '{expected}'."
          )


def check_unique_page_basenames(wiki_dir: str, issues: list[str]) -> None:
  """Flags two pages sharing a basename, which makes `[[name]]` ambiguous.

  Wikilinks resolve by basename, so `concepts/cloud-tpu.md` and
  `entities/cloud-tpu.md` made all 11 `[[cloud-tpu]]` references resolve
  non-deterministically to whichever the walk happened to reach first.

  Scoped to collisions between pages of the same representation. A Markdown
  playbook paired with a JSON workflow of the same name is deliberate
  architecture, not a collision -- SCHEMA.md requires exactly that split
  ("Files in `workflows/` must be structured JSON. Heuristic/mindset guides
  must be in `playbooks/` as Markdown"), and five such pairs exist by design.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
  """
  seen = {}
  subdirs = list(_DIR_TYPES.keys()) + ["actions", "workflows"]
  for sub in subdirs:
    d = os.path.join(wiki_dir, sub)
    if not os.path.isdir(d):
      d = os.path.join(wiki_dir, "srcs", sub)
    if not os.path.isdir(d):
      continue
    rel_dir = os.path.relpath(d, wiki_dir)
    for f in sorted(os.listdir(d)):
      base, ext = os.path.splitext(f)
      if ext not in (".md", ".json"):
        continue
      key = (base, ext)
      if key in seen:
        issues.append(
            f"Ambiguous wikilink target '[[{base}]]': two {ext} pages define "
            f"it -- '{seen[key]}' and '{rel_dir}/{f}'. Rename one."
        )
      else:
        seen[key] = f"{rel_dir}/{f}"


def check_logging_filter_syntax(
    wiki_dir: str,
    issues: list[str],
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Flags Cloud Logging filters whose regex literal is single-quoted.

  Cloud Logging requires `=~ "regex"`. Given `=~ 'regex'` it rejects the whole
  filter with `INVALID_ARGUMENT: Unparseable filter: regular expressions must
  begin and end with '"'`. The action still parses, still lints, and still runs
  -- it simply never returns a result, so the defect reads as "no matching
  events" rather than as a broken query. `trace-hbm-ecc-errors` shipped this way
  and had never produced output.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if not os.path.isdir(actions_dir):
      return
    actions_items = []
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      try:
        actions_items.append((f, _read_json(os.path.join(actions_dir, f))))
      except _JSON_FILE_ERRORS:
        continue
  else:
    actions_items = sorted(actions_cache.items())

  for f, d in actions_items:
    if not isinstance(d, dict):
      continue
    instr = d.get("instruction", "") or ""
    if re.search(r"=~\s*'", instr):
      issues.append(
          f"Invalid log filter in 'actions/{f}': a regex literal is "
          "single-quoted (=~ '...'). Cloud Logging requires double quotes "
          '(=~ "...") and rejects the entire filter otherwise, so the action '
          "returns nothing instead of failing loudly."
      )


def _normalized_command(instr: str | None) -> str:
  """Strips parameter names/values so duplicates are comparable."""
  s = re.sub(r"\$\{[^}]+\}", "\x00", instr or "")
  s = re.sub(r"<[a-z0-9_]+>", "\x00", s)
  s = re.sub(r"['\"]", "", s)
  return re.sub(r"\s+", " ", s).strip()


def check_duplicate_actions(
    wiki_dir: str,
    issues: list[str],
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Flags command actions whose instructions are identical once normalized.

  Two actions wrapping the same command are a selection hazard: an agent has no
  basis to choose between them. Mark one `deprecated` with `superseded_by`, or
  merge them.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if not os.path.isdir(actions_dir):
      return
    actions_items = []
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      try:
        actions_items.append((f, _read_json(os.path.join(actions_dir, f))))
      except _JSON_FILE_ERRORS:
        continue
  else:
    actions_items = sorted(actions_cache.items())

  by_cmd = {}
  for f, d in actions_items:
    if (
        not isinstance(d, dict)
        or d.get("execution_mode") != "command"
        or d.get("deprecated")
    ):
      continue
    norm = _normalized_command(d.get("instruction", ""))
    if not norm:
      continue
    by_cmd.setdefault(norm, []).append(f[:-5])
  for norm, names in sorted(by_cmd.items()):
    if len(names) > 1:
      issues.append(
          f"Duplicate Actions {sorted(names)} share an identical command "
          f"('{norm[:70]}'). Merge them, or mark one 'deprecated' with "
          "'superseded_by'."
      )


def check_test_fixture_references(wiki_dir: str, issues: list[str]) -> None:
  """Validates references inside tests/test_cases/*/solution.json.

  tests/ is excluded from the main page walk, which also exempted eval fixtures
  from reference checking. A fixture that names a renamed or deleted page can
  never be satisfied, so the corresponding test silently becomes unpassable
  rather than failing loudly.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
  """
  cases_dir = os.path.join(wiki_dir, "tests", "test_cases")
  if not os.path.isdir(cases_dir):
    return
  known = set()
  for sub in ("actions", "workflows"):
    d = os.path.join(wiki_dir, sub)
    if os.path.isdir(d):
      known |= {f[:-5] for f in os.listdir(d) if f.endswith(".json")}

  for case in sorted(os.listdir(cases_dir)):
    sol = os.path.join(cases_dir, case, "solution.json")
    if not os.path.exists(sol):
      continue
    rel = os.path.relpath(sol, wiki_dir)
    try:
      blob = _read_text(sol)
      json.loads(blob)
    except _JSON_FILE_ERRORS as e:
      issues.append(f"Test fixture {rel} is not valid JSON: {e}")
      continue
    for link in sorted(set(re.findall(r"\[\[([^\]]+)\]\]", blob))):
      if link.split("#")[0].strip() not in known:
        issues.append(
            f"Test fixture {rel} references non-existent page: [[{link}]]"
        )
    for path in sorted(
        set(
            re.findall(r'"((?:actions|workflows)/[A-Za-z0-9_.-]+\.json)"', blob)
        )
    ):
      if not os.path.exists(os.path.join(wiki_dir, path)):
        issues.append(
            f"Test fixture {rel} references non-existent file: {path}"
        )


ALLOWED_PAGE_METADATA_FIELDS = frozenset({
    "title",
    "description",
    "type",
    "tags",
    "sources",
    "created",
    "updated",
    "contested",
    "contradictions",
    "internal_only",
})

ALLOWED_SRC_METADATA_FIELDS = frozenset({
    "sha256",
    "source_url",
    "title",
    "description",
    "type",
    "internal_only",
    "tags",
    "sources",
    "created",
    "updated",
    "contested",
    "contradictions",
})

ALLOWED_CASE_METADATA_FIELDS = (
    ALLOWED_SRC_METADATA_FIELDS | ALLOWED_PAGE_METADATA_FIELDS
)

ALLOWED_ACTION_METADATA_FIELDS = frozenset({
    "title",
    "type",
    "description",
    "tags",
    "sources",
    "created",
    "updated",
    "internal_only",
    "execution_mode",
    "instruction",
    "parameters",
    "params",
    "outputs",
    "summary_fields",
    "prompt",
    "requires_approval",
    "approval_exempt_params",
    "timeout_seconds",
    "deprecated",
    "superseded_by",
    "entrypoints",
    "questionnaire",
    "fallback",
})

ALLOWED_WORKFLOW_METADATA_FIELDS = frozenset({
    "title",
    "type",
    "description",
    "tags",
    "sources",
    "created",
    "updated",
    "internal_only",
    "steps",
    "internal_steps",
    "step_notes",
    "step_fallbacks",
    "internal_step_fallbacks",
    "summary_fields",
    "entrypoints",
})

# Slash commands / skills an artifact can be reached from. `tags` describes the
# problem domain (tpu, hangs); `entrypoints` describes which operator-facing
# entry point can lead to executing the artifact. The two answer different
# questions, so neither substitutes for the other.
#
# `ingest` and `lint` are in the vocabulary but are expected to stay unused:
# they operate *on* the wiki rather than dispatching workflows. They are listed
ALLOWED_ENTRYPOINTS = frozenset({
    "troubleshoot",
    "handoff",
    "ingest",
    "lint",
    "dashboard",
})


def check_entrypoints_field(
    name: str, meta: dict[str, Any], issues: list[str]
) -> None:
  """Validates an artifact's `entrypoints` against the closed vocabulary."""
  if "entrypoints" not in meta:
    return
  vals = meta["entrypoints"]
  if not isinstance(vals, list) or not vals:
    issues.append(
        f"'entrypoints' in '{name}' must be a non-empty list; got"
        f" {vals!r}. Omit the field entirely rather than leaving it empty."
    )
    return
  for v in vals:
    if v not in ALLOWED_ENTRYPOINTS:
      issues.append(
          f"Unknown entrypoint '{v}' in '{name}'. Allowed:"
          f" {sorted(ALLOWED_ENTRYPOINTS)}"
      )
  if len(set(vals)) != len(vals):
    issues.append(f"Duplicate entries in 'entrypoints' of '{name}': {vals!r}")


def check_unknown_metadata_fields(
    wiki_dir: str,
    issues: list[str],
    file_cache: dict[str, dict[str, Any]] | None = None,
    actions_cache: dict[str, Any] | None = None,
) -> None:
  """Audits top-level metadata fields against closed-world allowlists.

  Checks:
  - Page frontmatter fields against ALLOWED_PAGE_METADATA_FIELDS.
  - Source frontmatter fields in srcs/ against ALLOWED_SRC_METADATA_FIELDS
    (or ALLOWED_CASE_METADATA_FIELDS for dual-frontmatter cases in srcs/cases/).
  - Action top-level JSON fields against ALLOWED_ACTION_METADATA_FIELDS.
  - Workflow top-level JSON fields against ALLOWED_WORKFLOW_METADATA_FIELDS.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    file_cache: Optional mapping of wiki-relative path to the per-file cache
      entry built by `run_linter`; when None, files are read from disk.
    actions_cache: Optional mapping of action filename to parsed action JSON;
      when None, `actions/` is read from disk.
  """
  ignored_files = {
      "README.md",
      "SCHEMA.md",
      "LOG.md",
      "INDEX.md",
      "TUTORIAL.md",
      "ADVISOR_NOTES.md",
      "USER_LOG.md",
      "requirements.txt",
      "wiki_linter.py",
  }

  # 1. Action top-level JSON fields
  if actions_cache is None:
    actions_dir = os.path.join(wiki_dir, "actions")
    if os.path.isdir(actions_dir):
      for f in sorted(os.listdir(actions_dir)):
        if f.endswith(".json"):
          try:
            d = _read_json(os.path.join(actions_dir, f))
            if isinstance(d, dict):
              for k in sorted(d.keys()):
                if k not in ALLOWED_ACTION_METADATA_FIELDS:
                  issues.append(
                      f"Unknown field '{k}' in action 'actions/{f}'. Allowed"
                      f" fields: {sorted(ALLOWED_ACTION_METADATA_FIELDS)}"
                  )
              check_entrypoints_field(f"actions/{f}", d, issues)
          except _JSON_FILE_ERRORS + (TypeError,):
            pass
  else:
    for f, d in sorted(actions_cache.items()):
      if isinstance(d, dict):
        action_path = f if f.startswith("actions/") else f"actions/{f}"
        for k in sorted(d.keys()):
          if k not in ALLOWED_ACTION_METADATA_FIELDS:
            issues.append(
                f"Unknown field '{k}' in action '{action_path}'. Allowed"
                f" fields: {sorted(ALLOWED_ACTION_METADATA_FIELDS)}"
            )
        check_entrypoints_field(action_path, d, issues)

  # 2. Workflow top-level JSON fields
  if file_cache is not None:
    for p, entry in sorted(file_cache.items()):
      if p.startswith("workflows/") and p.endswith(".json"):
        d = entry.get("json") or entry.get("frontmatter")
        if isinstance(d, dict):
          for k in sorted(d.keys()):
            if k not in ALLOWED_WORKFLOW_METADATA_FIELDS:
              issues.append(
                  f"Unknown field '{k}' in workflow '{p}'. Allowed fields:"
                  f" {sorted(ALLOWED_WORKFLOW_METADATA_FIELDS)}"
              )
          check_entrypoints_field(p, d, issues)
  else:
    workflows_dir = os.path.join(wiki_dir, "workflows")
    if os.path.isdir(workflows_dir):
      for f in sorted(os.listdir(workflows_dir)):
        if f.endswith(".json"):
          try:
            d = _read_json(os.path.join(workflows_dir, f))
            if isinstance(d, dict):
              for k in sorted(d.keys()):
                if k not in ALLOWED_WORKFLOW_METADATA_FIELDS:
                  issues.append(
                      f"Unknown field '{k}' in workflow 'workflows/{f}'."
                      " Allowed fields:"
                      f" {sorted(ALLOWED_WORKFLOW_METADATA_FIELDS)}"
                  )
              check_entrypoints_field(f"workflows/{f}", d, issues)
          except _JSON_FILE_ERRORS + (TypeError,):
            pass

  # 3. Page frontmatter fields. tools/scripts/ holds script inputs
  # (templates, schemas, fixtures), not wiki pages.
  non_page_prefixes = (
      "tests/",
      "scratch/",
      "archive/",
      ".agents/",
      "_agents/",
      "tools/scripts/",
  )
  if file_cache is not None:
    for p, entry in sorted(file_cache.items()):
      if not p.endswith(".md"):
        continue
      if p.startswith(non_page_prefixes):
        continue
      base = os.path.basename(p)
      if base in ignored_files or _ROTATED_LOG_RE.fullmatch(base):
        continue
      if entry.get("has_fm") and isinstance(entry.get("frontmatter"), dict):
        fm = entry["frontmatter"]
        for k in sorted(fm.keys()):
          if k not in ALLOWED_PAGE_METADATA_FIELDS:
            issues.append(
                f"Unknown frontmatter field '{k}' in page '{p}'. Allowed"
                f" fields: {sorted(ALLOWED_PAGE_METADATA_FIELDS)}"
            )
  else:
    for root, dirs, files in os.walk(wiki_dir):
      prune_scan_dirs(root, dirs)
      for f in sorted(files):
        if (
            not f.endswith(".md")
            or f in ignored_files
            or _ROTATED_LOG_RE.fullmatch(f)
        ):
          continue
        rel_path = os.path.relpath(os.path.join(root, f), wiki_dir)
        if rel_path.startswith(non_page_prefixes):
          continue
        try:
          c = _read_text(os.path.join(root, f))
          m = re.search(r"^---\s*\n(.*?)\n---\s*\n", c, re.DOTALL)
          if m:
            fm = parse_frontmatter(m.group(1))
            if isinstance(fm, dict):
              for k in sorted(fm.keys()):
                if k not in ALLOWED_PAGE_METADATA_FIELDS:
                  issues.append(
                      f"Unknown frontmatter field '{k}' in page '{rel_path}'."
                      f" Allowed fields: {sorted(ALLOWED_PAGE_METADATA_FIELDS)}"
                  )
        except _READ_ERRORS + _YAML_ERRORS:
          pass

  # 4. Source frontmatter fields in srcs/
  srcs_dir = os.path.join(wiki_dir, "srcs")
  if os.path.isdir(srcs_dir):
    for root, dirs, files in os.walk(srcs_dir):
      for f in sorted(files):
        if not f.endswith(".md"):
          continue
        rel_path = os.path.relpath(os.path.join(root, f), wiki_dir)
        try:
          c = _read_text(os.path.join(root, f))
          m = re.search(r"^---\s*\n(.*?)\n---\s*\n", c, re.DOTALL)
          if m:
            fm = parse_frontmatter(m.group(1))
            if isinstance(fm, dict):
              is_case = (
                  rel_path.startswith("srcs/cases/") or fm.get("type") == "case"
              )
              allowed = (
                  ALLOWED_CASE_METADATA_FIELDS
                  if is_case
                  else ALLOWED_SRC_METADATA_FIELDS
              )
              for k in sorted(fm.keys()):
                if k not in allowed:
                  issues.append(
                      f"Unknown frontmatter field '{k}' in source '{rel_path}'."
                      f" Allowed fields: {sorted(allowed)}"
                  )
        except _READ_ERRORS + _YAML_ERRORS:
          pass


def check_portability_leakage(
    wiki_dir: str,
    issues: list[str],
    file_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
  """Flags hardcoded Workspace workspace paths in wiki files.

  Scans .py, .sh, .json, .yaml and .md files, skipping the detectors
  themselves and a fixed set of exempt files, for the `_LEAK_RES` patterns.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
    file_cache: Optional mapping of wiki-relative path to the per-file cache
      entry built by `run_linter`; when None, files are read from disk.
  """
  if file_cache is not None:
    for rel_path, entry in sorted(file_cache.items()):
      f = os.path.basename(rel_path)
      if (
          f
          in [
              "wiki_linter.py",
              "internal_codename_mappings.json",
              "BUILD",
              "requirements.txt",
          ]
          or f.endswith(".blueprint")
          or f.endswith("_test.py")
      ):
        continue
      if rel_path in [
          "LOG.md",
          "playbooks/hardware-failure-modes.md",
          "tools/scripts/sync_external_wiki.py",
      ] or _ROTATED_LOG_RE.fullmatch(os.path.basename(rel_path)):
        continue
      if f.endswith((".py", ".sh", ".json", ".yaml", ".md")):
        content = entry["content"]
        if _LEAK_ANY_RE.search(content):
          for pattern, compiled in _LEAK_RES:
            if compiled.search(content):
              issues.append(
                  "Portability Error: Hardcoded Workspace path found in file"
                  f" '{rel_path}' matching pattern '{pattern}'"
              )
    return

  for root, dirs, files in os.walk(wiki_dir):
    prune_scan_dirs(root, dirs)
    for f in files:
      if (
          f
          in [
              "wiki_linter.py",
              "internal_codename_mappings.json",
              "BUILD",
              "requirements.txt",
          ]
          or f.endswith(".blueprint")
          or f.endswith("_test.py")
      ):
        continue
      full_path = os.path.join(root, f)
      rel_path = os.path.relpath(full_path, wiki_dir)
      if rel_path in [
          "LOG.md",
          "playbooks/hardware-failure-modes.md",
          "tools/scripts/sync_external_wiki.py",
      ] or _ROTATED_LOG_RE.fullmatch(f):
        continue
      if f.endswith((".py", ".sh", ".json", ".yaml", ".md")):
        try:
          with open(full_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
            if _LEAK_ANY_RE.search(content):
              for pattern, compiled in _LEAK_RES:
                if compiled.search(content):
                  issues.append(
                      "Portability Error: Hardcoded Workspace path found"
                      f" in file '{rel_path}' matching pattern"
                      f" '{pattern}'"
                  )
        except _READ_ERRORS:
          pass


_PII_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PII_EXEMPT_EMAIL_DOMAINS = frozenset({"example.com", "domain.com"})
_PII_USERNAME_RE = re.compile(r"\busername_placeholder\b", re.IGNORECASE)
_PII_MAC_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b")

_PII_CUSTOMER_PATTERNS = (
    (
        r"\[(?:CustomerA|CustomerB)\]",
        re.compile(r"\[(?:CustomerA|CustomerB)\]"),
    ),
    (
        r"\b(?:CustomerA|CustomerB)\b",
        re.compile(r"\b(?:CustomerA|CustomerB)\b"),
    ),
    (
        r"customer-context CustomerC",
        re.compile(
            r"(?i)(?:\b(?:customer|client|organization|org|account|tenant|"
            r"such as)\b[^.\n\r]{0,50}\bCustomerC\b|"
            r"\bCustomerC\b[^.\n\r]{0,50}"
            r"\b(?:customer|client|organization|org|account|tenant)\b)"
        ),
    ),
    (
        r"customer-org-\d+",
        re.compile(r"(?i)\b(?:customer|client)-org-[A-Za-z0-9_-]+\b"),
    ),
    (
        r"customer company/org pattern",
        re.compile(
            r"(?i)\b(?:(?:customer|client)\s+"
            r"(?:company|organization|org|name|identifier)"
            r"(?:\s+(?:name|identifier))?|"
            r"customer|company(?:\s+name)?|(?:organization|org)\s+name)"
            r"\s*[:=]\s*['\"]?"
            r"(?!(?:\[REDACTED|str\b|int\b|bool\b|None\b|dict\b|list\b|"
            r"Optional\b|Any\b))[A-Za-z0-9_.-]+"
        ),
    ),
    (
        r"customer company phrase",
        re.compile(
            r"(?i)\b(?:customer|client)\s+(?:company|organization|org)"
            r"\s+(?!\[REDACTED)[A-Za-z0-9_.-]+"
        ),
    ),
)

_PII_EXEMPT_FILES = frozenset({
    "wiki_linter.py",
})

# The PII sweep skips exactly the directories that are not wiki content.
_PII_EXEMPT_DIRS = EXCLUDED_SCAN_DIRS

_PUBLISHED_WIKI_DIRS = (
    "concepts/",
    "symptoms/",
    "playbooks/",
    "entities/",
    "tools/",
    "workflows/",
    "actions/",
)


def _scan_file_for_pii(rel_path: str, content: str, issues: list[str]) -> None:
  """Flags emails, usernames, customer names and MAC addresses in a file.

  Args:
    rel_path: Wiki-relative path of the file, used in messages.
    content: The file's text.
    issues: List to which issue messages are appended.
  """
  # 1. Unredacted email addresses
  emails = set(_PII_EMAIL_RE.findall(content))
  for email in sorted(emails):
    domain = email.split("@")[-1].lower()
    if (
        domain not in _PII_EXEMPT_EMAIL_DOMAINS
        and not domain.endswith(".example.com")
        and not domain.endswith(".domain.com")
    ):
      issues.append(
          f"PII Leakage: Found unredacted email in file '{rel_path}' matching"
          f" '{email}'"
      )

  # 2. Usernames / LDAP leakage
  m_user = _PII_USERNAME_RE.search(content)
  if m_user:
    issues.append(
        f"PII Leakage: Found unredacted username in file '{rel_path}' matching"
        f" '{m_user.group(0)}'"
    )

  # 3. Customer information / company names
  seen_cust = set()
  for _, pat in _PII_CUSTOMER_PATTERNS:
    m_cust = pat.search(content)
    if m_cust:
      val = m_cust.group(0)
      if val not in seen_cust:
        seen_cust.add(val)
        issues.append(
            "PII Leakage: Found unredacted customer company name in file"
            f" '{rel_path}' matching '{val}'"
        )

  # 4. MAC addresses
  macs = set(_PII_MAC_RE.findall(content))
  for mac in sorted(macs):
    issues.append(
        f"PII Leakage: Found unredacted MAC address in file '{rel_path}'"
        f" matching '{mac}'"
    )


def _is_published_target(rel_p: str) -> bool:
  """Returns True if `rel_p` is a published wiki file subject to PII checks."""
  norm = rel_p.replace("\\", "/")
  f = os.path.basename(norm)
  if f in _PII_EXEMPT_FILES or f == "LOG.md" or _ROTATED_LOG_RE.fullmatch(f):
    return False
  parts = norm.split("/")
  if any(p in _PII_EXEMPT_DIRS for p in parts):
    return False
  return (
      norm == "INDEX.md"
      or any(norm.startswith(d) for d in _PUBLISHED_WIKI_DIRS)
  ) and norm.endswith((".md", ".json"))


def check_pii_leakage(
    wiki_dir: str,
    issues: list[str],
    file_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
  """Checks published wiki files for PII and customer info leakage."""
  if file_cache is not None:
    for rel_path, entry in sorted(file_cache.items()):
      if not _is_published_target(rel_path):
        continue
      _scan_file_for_pii(rel_path.replace("\\", "/"), entry["content"], issues)
    return

  for root, dirs, files in os.walk(wiki_dir):
    prune_scan_dirs(root, dirs)
    for f in sorted(files):
      full_path = os.path.join(root, f)
      rel_path = os.path.relpath(full_path, wiki_dir)
      if not _is_published_target(rel_path):
        continue
      try:
        with open(
            full_path, "r", encoding="utf-8", errors="ignore"
        ) as file_obj:
          content = file_obj.read()
          _scan_file_for_pii(rel_path.replace("\\", "/"), content, issues)
      except _READ_ERRORS:
        pass


_DASHBOARD_EXEMPT_FILES = frozenset({
    "package-lock.json",
    "current_state.json",
    "clusters_config.json",
    "projects_cache.json",
    "organizations_cache.json",
    "BUILD",
    "METADATA",
    "OWNERS",
})
_DASHBOARD_EXEMPT_DIRS = frozenset({
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "logs",
    "scratch",
    "agent_logs",
    "test_results",
})
_DASHBOARD_INTERNAL_PATTERNS = (
    (
        "internal corp domain",
        re.compile(
            r"\b(?:[a-zA-Z0-9-]+\.)*(?:internal\.example\.com|internal\.example\.com)\b"
        ),
    ),
    (
        "hardcoded internal project/cluster/org identifier",
        re.compile(
            r"\b(?:example-internal-project-id|example-shared-project-id|"
            r"example-internal-cluster(?:-[a-zA-Z0-9_-]*)?|000000000000)\b"
        ),
    ),
    (
        "internal codename or tool",
        re.compile(r"(?i)\b(?:internal_codename_a|internal_tool_b(?:ing)?)\b"),
    ),
)


def check_dashboard_pii_and_internal_leaks(
    wiki_dir: str, issues: list[str]
) -> None:
  """Scans sibling _dashboard/ directory for PII and internal Google leaks."""
  dashboard_dir = os.path.join(os.path.dirname(wiki_dir), "_dashboard")
  if not os.path.isdir(dashboard_dir):
    return

  for root, dirs, files in os.walk(dashboard_dir):
    dirs[:] = [
        d
        for d in dirs
        if d not in _DASHBOARD_EXEMPT_DIRS and not d.startswith(".venv")
    ]
    for f in sorted(files):
      if f in _DASHBOARD_EXEMPT_FILES or not f.endswith(
          (".py", ".js", ".html", ".json", ".md", ".sh", ".css")
      ):
        continue
      full_path = os.path.join(root, f)
      rel_path = "../_dashboard/" + os.path.relpath(
          full_path, dashboard_dir
      ).replace("\\", "/")
      try:
        with open(
            full_path, "r", encoding="utf-8", errors="ignore"
        ) as file_obj:
          content = file_obj.read()
      except _READ_ERRORS:
        continue

      _scan_file_for_pii(rel_path, content, issues)
      for label, pat in _DASHBOARD_INTERNAL_PATTERNS:
        m = pat.search(content)
        if m:
          issues.append(
              f"Dashboard Leak ({label}): Found '{m.group(0)}' in file"
              f" '{rel_path}'"
          )


def _add_workflow_step_ref(raw: Any, out: set[str]) -> None:
  """Normalizes a workflow step reference and adds its key to out."""
  if not isinstance(raw, str):
    return
  step_clean = raw.strip()
  if step_clean.startswith("[[") and step_clean.endswith("]]"):
    step_clean = step_clean[2:-2].strip()
  step_core = (
      step_clean.split("|")[0].split("#")[0].strip().lower().replace(" ", "-")
  )
  if step_core:
    out.add(step_core)
    out.add(os.path.splitext(os.path.basename(step_core))[0])


def _collect_workflow_step_refs(
    workflow_data: dict[str, Any], out: set[str]
) -> None:
  """Collects referenced action/workflow names from steps and step_fallbacks."""
  for steps_key in ("steps", "internal_steps"):
    steps = workflow_data.get(steps_key, [])
    if isinstance(steps, list):
      for step in steps:
        _add_workflow_step_ref(step, out)
  for fb_key in ("step_fallbacks", "internal_step_fallbacks"):
    fb_map = workflow_data.get(fb_key, {})
    if isinstance(fb_map, dict):
      for step_key, branches in fb_map.items():
        _add_workflow_step_ref(step_key, out)
        if isinstance(branches, list):
          for branch in branches:
            if isinstance(branch, dict):
              for list_key in ("replace_with", "skip_steps"):
                refs = branch.get(list_key, [])
                if isinstance(refs, list):
                  for ref in refs:
                    _add_workflow_step_ref(ref, out)


def check_symptom_linkage(
    wiki_dir: str,
    warnings: list[str],
    file_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
  """Flags workflows and standalone actions lacking direct symptom linkage.

  Direct symptom linkage is encouraged for triage discoverability. Workflows
  and standalone actions not directly referenced by at least one symptom in
  symptoms/*.md receive an advisory warning (not an issue, so status does
  not fail).

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    warnings: List to which advisory warning messages are appended.
    file_cache: Optional mapping of wiki-relative path to the per-file cache
      entry built by `run_linter`; when None, files are read from disk.
  """
  symptoms_dir = os.path.join(wiki_dir, "symptoms")
  workflows_dir = os.path.join(wiki_dir, "workflows")
  actions_dir = os.path.join(wiki_dir, "actions")

  if file_cache is not None:
    symptom_links = set()
    for p, entry in file_cache.items():
      if p.startswith("symptoms/") and p.endswith(".md"):
        content = entry["content"]
        for link in re.findall(r"\[\[([^\]]+)\]\]", content):
          link_core = (
              link.split("|")[0].split("#")[0].strip().lower().replace(" ", "-")
          )
          if link_core:
            symptom_links.add(link_core)
            symptom_links.add(os.path.splitext(os.path.basename(link_core))[0])
        for link in re.findall(r"(?<!\])\]\(([^)]+)\)", content):
          if (
              not link.startswith("http")
              and not link.startswith("#")
              and not link.startswith("file://")
              and not link.startswith("mailto:")
          ):
            link_core = link.split("#")[0].strip().lower()
            if link_core:
              symptom_links.add(link_core)
              symptom_links.add(
                  os.path.splitext(os.path.basename(link_core))[0]
              )

    workflow_files = sorted([
        os.path.basename(p)
        for p in file_cache
        if p.startswith("workflows/") and p.endswith(".json")
    ])
    actions_in_workflows = set()
    for wf in workflow_files:
      d = file_cache.get(f"workflows/{wf}", {}).get("json")
      if isinstance(d, dict):
        _collect_workflow_step_refs(d, actions_in_workflows)

    for wf in workflow_files:
      wf_name = os.path.splitext(wf)[0]
      if (
          wf_name not in symptom_links
          and f"workflows/{wf_name}" not in symptom_links
          and f"workflows/{wf}" not in symptom_links
      ):
        warnings.append(
            f"Symptom linkage warning: workflow '{wf_name}' is not directly"
            " referenced by any symptom in symptoms/ (encouraged for triaging"
            " discoverability)"
        )

    action_files = sorted([
        os.path.basename(p)
        for p in file_cache
        if p.startswith("actions/") and p.endswith(".json")
    ])
    for f in action_files:
      action_name = os.path.splitext(f)[0]
      if (
          action_name in actions_in_workflows
          or f"actions/{action_name}" in actions_in_workflows
          or f in actions_in_workflows
      ):
        continue
      if (
          action_name not in symptom_links
          and f"actions/{action_name}" not in symptom_links
          and f not in symptom_links
      ):
        warnings.append(
            f"Symptom linkage warning: standalone action '{action_name}' is"
            " neither part of a workflow nor referenced by any symptom in"
            " symptoms/"
        )
    return

  # Disk fallback
  symptom_links = set()
  if os.path.isdir(symptoms_dir):
    for f in sorted(os.listdir(symptoms_dir)):
      if not f.endswith(".md"):
        continue
      symptom_path = os.path.join(symptoms_dir, f)
      try:
        content = _read_text(symptom_path)
      except _READ_ERRORS:
        continue

      # Collect [[wiki-links]]
      for link in re.findall(r"\[\[([^\]]+)\]\]", content):
        link_core = (
            link.split("|")[0].split("#")[0].strip().lower().replace(" ", "-")
        )
        if link_core:
          symptom_links.add(link_core)
          symptom_links.add(os.path.splitext(os.path.basename(link_core))[0])

      # Collect markdown links [text](target)
      for link in re.findall(r"(?<!\])\]\(([^)]+)\)", content):
        if (
            not link.startswith("http")
            and not link.startswith("#")
            and not link.startswith("file://")
            and not link.startswith("mailto:")
        ):
          link_core = link.split("#")[0].strip().lower()
          if link_core:
            symptom_links.add(link_core)
            symptom_links.add(os.path.splitext(os.path.basename(link_core))[0])

  # 2. Collect actions that are part of workflow steps.
  workflow_files = []
  if os.path.isdir(workflows_dir):
    workflow_files = sorted(
        [f for f in os.listdir(workflows_dir) if f.endswith(".json")]
    )

  actions_in_workflows = set()
  for wf in workflow_files:
    wf_path = os.path.join(workflows_dir, wf)
    try:
      d = _read_json(wf_path)
    except _JSON_FILE_ERRORS:
      continue
    if isinstance(d, dict):
      _collect_workflow_step_refs(d, actions_in_workflows)

  # 3. Check workflows: warn if not directly referenced by any symptom in
  # symptoms/
  for wf in workflow_files:
    wf_name = os.path.splitext(wf)[0]
    if (
        wf_name not in symptom_links
        and f"workflows/{wf_name}" not in symptom_links
        and f"workflows/{wf}" not in symptom_links
    ):
      warnings.append(
          f"Symptom linkage warning: workflow '{wf_name}' is not directly"
          " referenced by any symptom in symptoms/ (encouraged for triaging"
          " discoverability)"
      )

  # 4. Check standalone actions: warn if neither in workflow steps nor in
  # symptoms
  if os.path.isdir(actions_dir):
    for f in sorted(os.listdir(actions_dir)):
      if not f.endswith(".json"):
        continue
      action_name = os.path.splitext(f)[0]
      if (
          action_name in actions_in_workflows
          or f"actions/{action_name}" in actions_in_workflows
          or f in actions_in_workflows
      ):
        continue

      if (
          action_name not in symptom_links
          and f"actions/{action_name}" not in symptom_links
          and f not in symptom_links
      ):
        warnings.append(
            f"Symptom linkage warning: standalone action '{action_name}' is"
            " neither part of a workflow nor referenced by any symptom in"
            " symptoms/"
        )


_EXEMPT_UTILITY_SCRIPTS = frozenset({
    "__init__.py",
    "audit_wiki.py",
    "audit_dupes.py",
    "reorder_index.py",
    "categorize_connections.py",
    "pull_and_format_logs.py",
    "python_glue_runner.py",
    "agent_dispatch.py",
})


def check_orphan_scripts(wiki_dir: str, issues: list[str]) -> None:
  """Flags top-level scripts in tools/scripts/ not referenced by any artifact.

  Scans `tools/scripts/*.py` (excluding private `_*`, `*_test.py`, and exempt
  repo/runner utility scripts) and verifies each script is referenced by at
  least one action in `actions/*.json`, workflow in `workflows/*.json`, tool
  page in `tools/*.md`, playbook in `playbooks/*.md`, or imported/called by
  another script in `tools/scripts/`.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
  """
  scripts_dir = os.path.join(wiki_dir, "tools", "scripts")
  if not os.path.isdir(scripts_dir):
    return

  candidate_scripts = []
  for f in sorted(os.listdir(scripts_dir)):
    full = os.path.join(scripts_dir, f)
    if not os.path.isfile(full) or not f.endswith(".py"):
      continue
    if (
        f.startswith("_")
        or f.endswith("_test.py")
        or f in _EXEMPT_UTILITY_SCRIPTS
    ):
      continue
    candidate_scripts.append(f)

  if not candidate_scripts:
    return

  corpus_parts = []
  for sub in ("actions", "workflows", "tools", "playbooks", "concepts"):
    d = os.path.join(wiki_dir, sub)
    if not os.path.isdir(d):
      continue
    for f in sorted(os.listdir(d)):
      p = os.path.join(d, f)
      if os.path.isfile(p) and f.endswith((".json", ".md")):
        try:
          corpus_parts.append(_read_text(p))
        except _READ_ERRORS:
          pass

  other_scripts_text = {}
  for root, _, files in os.walk(scripts_dir):
    for f in sorted(files):
      if not f.endswith(".py") or f.endswith("_test.py"):
        continue
      p = os.path.join(root, f)
      rel = os.path.relpath(p, scripts_dir)
      try:
        other_scripts_text[rel] = _read_text(p)
      except _READ_ERRORS:
        pass

  combined_corpus = "\n".join(corpus_parts)

  for script_file in candidate_scripts:
    stem = script_file[:-3]
    if script_file in combined_corpus:
      continue
    referenced_by_script = False
    import_re = re.compile(
        rf"(?:\bfrom\s+(?:tools\.scripts\.)?{re.escape(stem)}\s+import\b|"
        rf"\bimport\s+(?:tools\.scripts\.)?{re.escape(stem)}\b|"
        rf"\b{re.escape(script_file)}\b)"
    )
    for rel, text in other_scripts_text.items():
      if rel == script_file:
        continue
      if import_re.search(text):
        referenced_by_script = True
        break
    if not referenced_by_script:
      issues.append(
          "Orphan script (not referenced by any action or script):"
          f" tools/scripts/{script_file}"
      )


_MANGLED_PLACEHOLDER_PATTERNS = (
    re.compile(r"\{\s+\{[A-Z0-9_]+\}\s+\}"),
    re.compile(r"(?<!\{)\{[A-Z0-9_]+\}\}"),
    re.compile(r"\{\{[A-Z0-9_]+\}(?!\})"),
)
_VALID_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")


def check_template_placeholders(wiki_dir: str, issues: list[str]) -> None:
  """Checks handoff template files for mangled placeholders.

  Verifies that markdown files under `tools/scripts/handoff/` do not contain
  mangled Mustache placeholders (such as `{ {KEY} }` or `{KEY}}`) and that any
  template in `tools/scripts/handoff/templates/*.md` containing `{{KEY}}`
  placeholders includes `<!-- mdformat global-off -->`.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
  """
  handoff_dir = os.path.join(wiki_dir, "tools", "scripts", "handoff")
  if not os.path.isdir(handoff_dir):
    return

  for root, _, files in os.walk(handoff_dir):
    for f in sorted(files):
      if not f.endswith(".md"):
        continue
      full = os.path.join(root, f)
      rel = os.path.relpath(full, wiki_dir).replace("\\", "/")
      try:
        content = _read_text(full)
      except _READ_ERRORS:
        continue

      for pat in _MANGLED_PLACEHOLDER_PATTERNS:
        m = pat.search(content)
        if m:
          issues.append(
              f"Mangled template placeholder '{m.group(0)}' in '{rel}'."
              " Expected '{{KEY}}'."
          )
          break

      if rel.startswith("tools/scripts/handoff/templates/"):
        if (
            _VALID_PLACEHOLDER_RE.search(content)
            and "<!-- mdformat global-off -->" not in content
        ):
          issues.append(
              f"Template '{rel}' contains '{{{{KEY}}}}' placeholders but is"
              " missing '<!-- mdformat global-off -->'."
          )


_CONNECTIONS_SECTION_RE = re.compile(
    r"^##\s+Connections[ \t]*\n(.*?)(?=\n##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CONNECTIONS_CATEGORY_RE = re.compile(
    r"^[-*][ \t]+(\*\*(?:Workflows|Actions|Playbooks|Symptoms|Concepts|"
    r"Entities|Tools|Cases)\*\*:)[ \t]*$"
)


def check_connections_indentation(wiki_dir: str, issues: list[str]) -> None:
  """Verifies that sub-bullets under Connections headers are indented."""
  for sub in ("concepts", "symptoms", "playbooks", "entities", "tools"):
    sub_dir = os.path.join(wiki_dir, sub)
    if not os.path.isdir(sub_dir):
      continue
    for f in sorted(os.listdir(sub_dir)):
      if not f.endswith(".md"):
        continue
      full = os.path.join(sub_dir, f)
      if not os.path.isfile(full):
        continue
      rel = f"{sub}/{f}"
      try:
        content = _read_text(full)
      except _READ_ERRORS:
        continue
      m_sec = _CONNECTIONS_SECTION_RE.search(content)
      if not m_sec:
        continue
      current_category: str | None = None
      sub_bullet_count = 0
      for raw_line in m_sec.group(1).splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("<!--"):
          continue
        m_cat = _CONNECTIONS_CATEGORY_RE.match(raw_line)
        if m_cat:
          if current_category is not None and sub_bullet_count == 0:
            issues.append(
                f"Empty Connections category in '{rel}':"
                f" '{current_category}' has no sub-bullets."
            )
          current_category = m_cat.group(1)
          sub_bullet_count = 0
          continue
        if current_category is not None:
          if re.search(r"^[ \t]{2,}[-*][ \t]+", raw_line):
            sub_bullet_count += 1
          elif re.search(r"^[-*][ \t]+(?:\[\[|\[)", raw_line):
            issues.append(
                f"Connections indentation error in '{rel}': sub-bullet"
                f" '{raw_line.strip()}' under '{current_category}' must be"
                " indented (e.g. 4 spaces)."
            )
            sub_bullet_count += 1
      if current_category is not None and sub_bullet_count == 0:
        issues.append(
            f"Empty Connections category in '{rel}': '{current_category}' has"
            " no sub-bullets."
        )


_RAW_BUG_CL_CHAT_RE = re.compile(
    r"\b(?:b|cl)/(\d{5,})\b|https?://chat\.google\.com/room/[^\s)\>\]\"\x27,]+"
)
_EXEMPT_DUMMY_TICKET_FILES = frozenset()
_EXEMPT_DUMMY_TICKET_IDS = frozenset({"12345678", "99999999"})


def check_raw_bug_cl_chat_mentions(wiki_dir: str, issues: list[str]) -> None:
  """Flags raw b/..., cl/..., or chat room links in wiki pages and JSONs.

  All issue-tracker tickets, changelists, and chat threads must be ingested
  into `_wiki/srcs/known_bugs/` or `_wiki/srcs/on_duty_logs/` and referenced via
  `^[srcs/...]` provenance citations and `"sources"` metadata rather than raw
  inline IDs.

  Args:
    wiki_dir: Absolute path to the `_wiki/` directory.
    issues: List to which issue messages are appended.
  """
  for sub in (
      "concepts",
      "symptoms",
      "playbooks",
      "entities",
      "tools",
      "actions",
      "workflows",
  ):
    sub_dir = os.path.join(wiki_dir, sub)
    if not os.path.isdir(sub_dir):
      continue
    for f in sorted(os.listdir(sub_dir)):
      if not f.endswith((".md", ".json")):
        continue
      full = os.path.join(sub_dir, f)
      if not os.path.isfile(full):
        continue
      rel = f"{sub}/{f}"
      try:
        content = _read_text(full)
      except _READ_ERRORS:
        continue
      for m in _RAW_BUG_CL_CHAT_RE.finditer(content):
        matched_id = m.group(1)
        if (
            rel in _EXEMPT_DUMMY_TICKET_FILES
            and matched_id in _EXEMPT_DUMMY_TICKET_IDS
        ):
          continue
        issues.append(
            f"Raw bug/CL/chat reference '{m.group(0)}' in '{rel}': ingest into"
            " 'srcs/known_bugs/' or 'srcs/on_duty_logs/' and cite via"
            " '^[srcs/...]' or 'sources' metadata instead."
        )
        break


@functools.lru_cache(maxsize=4096)
def _normalize_link(link: str) -> str:
  """Normalizes a `[[wiki-link]]` target to its lowercase slug."""
  return link.split("|")[0].split("#")[0].strip().lower().replace(" ", "-")


@functools.lru_cache(maxsize=4096)
def _norm_path_join(base_dir: str, path: str) -> str:
  """Joins and normalizes a relative path against `base_dir`."""
  return os.path.normpath(os.path.join(base_dir, path))


@functools.lru_cache(maxsize=1024)
def _file_exists(wiki_dir: str, rel_path: str) -> bool:
  """Returns True if `rel_path` exists under `wiki_dir`."""
  return os.path.exists(os.path.join(wiki_dir, rel_path))


def run_linter() -> None:
  """Runs every wiki check, records the run in LOG.md and prints a report.

  The report is a JSON object on stdout with `status` ("PASS" or "FAIL"),
  `issues`, `warnings` and `checked_pages`.
  """
  _file_exists.cache_clear()
  wiki_dir = os.path.dirname(os.path.abspath(__file__))
  issues = []
  warnings = []

  sorted_files = check_and_sort_metadata(wiki_dir)
  for sf in sorted_files:
    warnings.append(
        f"Auto-resolved: Sorted metadata keys to canonical order in {sf}"
    )

  current_date = datetime.date.today()
  current_date_str = current_date.strftime("%Y-%m-%d")

  allowed_types = [
      "concept",
      "symptom",
      "playbook",
      "workflow",
      "action",
      "tool",
      "entity",
  ]
  allowed_tags = [
      "tpu",
      "gpu",
      "training",
      "inference",
      "eval",
      "benchmark",
      "out_of_memory",
      "timeout",
      "hardware_fault",
      "hangs",
      "methodology",
      "triage",
      "metrics",
      "goodput",
      "analysis",
      "ironwood",
      "scheduling",
      "pathways",
      "capacity",
      "datacenter",
      "mxla",
      "collective_communication",
      "gke",
      "kubernetes",
      "kueue",
      "networking",
      "dns",
      "storage",
      "auth",
      "gcp",
      "gcloud",
      "observability",
      "logging",
      "monitoring",
      "workflow",
      "approval",
  ]

  allowed_srcs_subdirs = {
      "playbooks",
      "user_guides",
      "on_duty_logs",
      "training_notes",
      "known_bugs",
      "research_papers",
      "cases",
  }

  ignored_files = [
      "SCHEMA.md",
      "INDEX.md",
      "LOG.md",
      # Top-level meta-documentation, like the entries above: describes the
      # wiki rather than being a page in it. There is no `tutorial` page type
      # and forcing it under playbooks/ would pollute the diagnostic taxonomy.
      # Its links are still checked, by check_toplevel_doc_links below.
      "TUTORIAL.md",
      "wiki_linter.py",
      "requirements.txt",
      # Forward-looking write targets: created by agents at runtime, not
      # present in a clean tree. Removing them would make the linter flag
      # the file the moment the agent writes it.
      "ADVISOR_NOTES.md",
      "USER_LOG.md",
  ]

  file_cache = {}
  all_pages = []
  page_to_links = {}
  inbound_links = {}

  for root, dirs, files in os.walk(wiki_dir):
    prune_scan_dirs(root, dirs)
    for f in files:
      rel_path = os.path.relpath(os.path.join(root, f), wiki_dir)
      full_path = os.path.join(root, f)

      if f.endswith((".py", ".sh", ".json", ".yaml", ".md")):
        try:
          content = _read_text(full_path)
          file_size = os.path.getsize(full_path)
        except _READ_ERRORS:
          continue

        entry = {
            "content": content,
            "size": file_size,
            "frontmatter": None,
            "frontmatter_error": None,
            "has_fm": False,
            "body": "",
            "json": None,
            "json_error": None,
        }
        if f.endswith(".md"):
          entry["body"] = content
          fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
          if fm_match:
            entry["has_fm"] = True
            entry["body"] = content[fm_match.end() :]
            try:
              entry["frontmatter"] = parse_frontmatter(fm_match.group(1))
            except _YAML_ERRORS as e:
              entry["frontmatter_error"] = e
        elif f.endswith(".json"):
          try:
            entry["json"] = json.loads(content)
            entry["frontmatter"] = entry["json"]
          except json.JSONDecodeError as e:
            entry["json_error"] = e
            entry["frontmatter_error"] = e

        file_cache[rel_path] = entry

      if f in ignored_files or _ROTATED_LOG_RE.fullmatch(f):
        continue
      # Files under tools/scripts/ are script inputs (templates, schemas,
      # fixtures), not wiki pages: they stay in file_cache for the portability
      # and command checks but get no page frontmatter or INDEX entry.
      if rel_path.startswith("tools/scripts/"):
        continue
      if f.endswith(".md") or f.endswith(".json"):
        if rel_path.startswith("workflows/") and not rel_path.endswith(".json"):
          issues.append(f"Workflow file {rel_path} must be a JSON file.")
        if rel_path.startswith("playbooks/") and not rel_path.endswith(".md"):
          issues.append(
              f"Playbook file {rel_path} must be a Markdown (.md) file."
          )
        all_pages.append(rel_path)
        inbound_links[rel_path] = 0

  actions_cache = {
      os.path.basename(p): entry["json"]
      for p, entry in file_cache.items()
      if p.startswith("actions/")
      and p.endswith(".json")
      and entry["json"] is not None
  }

  check_toplevel_doc_links(wiki_dir, all_pages, issues)
  check_blocking_commands(wiki_dir, issues, actions_cache=actions_cache)
  check_logging_filter_syntax(wiki_dir, issues, actions_cache=actions_cache)
  check_action_script_flags(wiki_dir, issues, actions_cache=actions_cache)
  check_mutating_actions_require_approval(
      wiki_dir, issues, actions_cache=actions_cache
  )
  check_summary_fields_are_emitted(
      wiki_dir, issues, actions_cache=actions_cache
  )
  check_page_type_matches_directory(wiki_dir, issues, file_cache=file_cache)
  check_unique_page_basenames(wiki_dir, issues)
  check_duplicate_actions(wiki_dir, issues, actions_cache=actions_cache)
  check_test_fixture_references(wiki_dir, issues)
  check_unknown_metadata_fields(
      wiki_dir, issues, file_cache=file_cache, actions_cache=actions_cache
  )
  check_orphan_scripts(wiki_dir, issues)
  check_template_placeholders(wiki_dir, issues)
  check_connections_indentation(wiki_dir, issues)
  check_raw_bug_cl_chat_mentions(wiki_dir, issues)
  check_symptom_linkage(wiki_dir, warnings, file_cache=file_cache)

  index_path = os.path.join(wiki_dir, "INDEX.md")
  indexed_links = []
  if os.path.exists(index_path):
    index_content = _read_text(index_path)
    for match in re.findall(r"(?<!\])\]\(([^)]+)\)", index_content):
      link_core = match.split("#")[0]
      if not link_core.startswith("http") and not link_core.startswith(
          "file://"
      ):
        indexed_links.append(os.path.normpath(link_core))

  for il in indexed_links:
    if il in ignored_files and os.path.exists(os.path.join(wiki_dir, il)):
      continue
    if il.startswith("srcs/") and os.path.exists(os.path.join(wiki_dir, il)):
      continue
    if il not in inbound_links:
      issues.append(f"INDEX.md references non-existent page: '{il}'")

  page_frontmatters = {}

  page_by_basename = {}
  all_pages_by_basename: dict[str, list[str]] = {}
  for p in all_pages:
    b = os.path.splitext(os.path.basename(p))[0]
    if b not in page_by_basename:
      page_by_basename[b] = p
    all_pages_by_basename.setdefault(b, []).append(p)

  srcs_by_basename = {}
  srcs_dir = os.path.join(wiki_dir, "srcs")
  if os.path.isdir(srcs_dir):
    for entry in os.listdir(srcs_dir):
      entry_dir = os.path.join(srcs_dir, entry)
      if os.path.isdir(entry_dir):
        for f in os.listdir(entry_dir):
          if f.endswith(".md"):
            b = os.path.splitext(f)[0]
            if b not in srcs_by_basename:
              srcs_by_basename[b] = os.path.join("srcs", entry, f)

  skills_dir = os.path.join(wiki_dir, "..", "_agents", "skills")
  if not os.path.isdir(skills_dir):
    skills_dir = os.path.join(wiki_dir, "_agents", "skills")
  if os.path.isdir(skills_dir):
    for entry in os.listdir(skills_dir):
      entry_dir = os.path.join(skills_dir, entry)
      if os.path.isdir(entry_dir) and os.path.exists(
          os.path.join(entry_dir, "SKILL.md")
      ):
        b = entry.lower()
        if b not in page_by_basename:
          page_by_basename[b] = os.path.join(skills_dir, entry, "SKILL.md")

  commands_dir = os.path.join(wiki_dir, "..", ".claude", "commands")
  if not os.path.isdir(commands_dir):
    commands_dir = os.path.join(wiki_dir, ".claude", "commands")
  if os.path.isdir(commands_dir):
    for f in os.listdir(commands_dir):
      if f.endswith(".md"):
        b = os.path.splitext(f)[0].lower()
        if b not in page_by_basename:
          page_by_basename[b] = os.path.join(commands_dir, f)

  for page in all_pages:
    entry = file_cache.get(page)
    if entry is not None:
      content = entry["content"]
      file_size = entry["size"]
      frontmatter = {}
      body = entry["body"]
      if page.endswith(".md"):
        if entry["has_fm"]:
          if entry["frontmatter_error"]:
            issues.append(
                f"Frontmatter parse error in {page}:"
                f" {entry['frontmatter_error']}"
            )
          else:
            frontmatter = entry["frontmatter"] or {}
        else:
          issues.append(f"Missing Frontmatter in markdown page {page}")
      elif page.endswith(".json"):
        if entry["json_error"]:
          issues.append(f"JSON parse error in {page}: {entry['json_error']}")
        else:
          frontmatter = entry["frontmatter"] or {}
    else:
      full_path = os.path.join(wiki_dir, page)
      content = _read_text(full_path)
      file_size = os.path.getsize(full_path)
      frontmatter = {}
      body = content
      if page.endswith(".md"):
        fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if fm_match:
          try:
            frontmatter = parse_frontmatter(fm_match.group(1))
            body = content[fm_match.end() :]
          except _YAML_ERRORS as e:
            issues.append(f"Frontmatter parse error in {page}: {e}")
        else:
          issues.append(f"Missing Frontmatter in markdown page {page}")
      elif page.endswith(".json"):
        try:
          frontmatter = json.loads(content)
          body = ""
        except json.JSONDecodeError as e:
          issues.append(f"JSON parse error in {page}: {e}")

    if file_size > 10 * 1024:
      warnings.append(
          f"Page size limit warning in {page}: File size ({file_size} bytes)"
          " exceeds 10KB"
      )

    if page.endswith(".md"):
      if frontmatter:
        check_internal_only_boolean(page, frontmatter, issues)
    else:
      check_internal_only_boolean(page, frontmatter, issues)

    if frontmatter and isinstance(frontmatter, dict):
      page_frontmatters[page] = frontmatter
      if "title" not in frontmatter:
        issues.append(f"Missing 'title' in frontmatter of {page}")
      if "type" not in frontmatter or frontmatter["type"] not in allowed_types:
        issues.append(
            f"Invalid or missing 'type' ({frontmatter.get('type')}) in {page}"
        )
      if "tags" in frontmatter:
        tags_list = frontmatter["tags"]
        if isinstance(tags_list, list):
          seen_tags = set()
          for t in tags_list:
            if t in seen_tags:
              issues.append(f"Duplicated tag '{t}' in {page}")
            seen_tags.add(t)
            if t not in allowed_tags:
              issues.append(f"Invalid tag '{t}' in {page}")
      p_type = frontmatter.get("type")
      if "sources" not in frontmatter:
        if p_type == "symptom" and os.path.isdir(
            os.path.join(wiki_dir, "srcs")
        ):
          issues.append(f"Missing 'sources' in frontmatter of {page}")
      else:
        if not isinstance(frontmatter["sources"], list):
          issues.append(
              f"Field 'sources' in frontmatter of {page} must be a list"
          )
        else:
          if len(frontmatter["sources"]) != len(set(frontmatter["sources"])):
            duplicates = sorted(
                list({
                    src
                    for src in frontmatter["sources"]
                    if frontmatter["sources"].count(src) > 1
                })
            )
            issues.append(f"Duplicated sources found in '{page}': {duplicates}")
          for src in frontmatter["sources"]:
            src_path = os.path.join(wiki_dir, src)
            if not os.path.exists(src_path):
              issues.append(
                  f"Referenced source file '{src}' in {page} frontmatter does"
                  " not exist"
              )

      updated_str = frontmatter.get("updated")
      if not updated_str:
        issues.append(f"Missing 'updated' date in frontmatter of {page}")
      else:
        try:
          updated_date = datetime.datetime.strptime(
              str(updated_str), "%Y-%m-%d"
          ).date()
          days_diff = (current_date - updated_date).days
          if days_diff > 30:
            warnings.append(
                f"Stale content warning in {page}: Last updated on"
                f" {updated_str} ({days_diff} days ago)"
            )
        except ValueError:
          issues.append(
              f"Invalid date format for 'updated' in {page}: {updated_str}"
          )

      if page.endswith(".md"):
        validate_markdown_format(body, page, frontmatter.get("type"), issues)
      elif page.endswith(".json") and p_type == "action":
        validate_action_json(frontmatter, page, wiki_dir, issues)
      elif page.endswith(".json") and p_type == "workflow":
        validate_workflow_json(frontmatter, page, wiki_dir, issues)

    links = []
    for link in re.findall(r"\[\[([^\]]+)\]\]", content):
      link_normalized = _normalize_link(link)
      matching_targets = all_pages_by_basename.get(link_normalized, [])
      found_target = page_by_basename.get(
          link_normalized
      ) or srcs_by_basename.get(link_normalized)
      if found_target:
        links.append(found_target)
        for tgt in matching_targets:
          if tgt != page and tgt in inbound_links:
            inbound_links[tgt] += 1
      else:
        issues.append(f"Broken wiki-link [[{link}]] in {page}")

    page_dirname = os.path.dirname(page)
    for link in re.findall(r"(?<!\])\]\(([^)]+)\)", body):
      if (
          not link.startswith("http")
          and not link.startswith("#")
          and not link.startswith("file://")
      ):
        link_core = link.split("#")[0]
        norm_link = _norm_path_join(page_dirname, link_core)
        if (
            norm_link.endswith(".py")
            or norm_link.endswith(".sh")
            or norm_link.startswith("srcs/")
        ):
          if not _file_exists(wiki_dir, norm_link):
            issues.append(f"Broken file path link {link} in {page}")
        elif norm_link in ignored_files:
          if not _file_exists(wiki_dir, norm_link):
            issues.append(f"Broken file path link {link} in {page}")
        elif norm_link not in all_pages:
          issues.append(f"Broken file path link {link} in {page}")
        else:
          if norm_link != page and norm_link in inbound_links:
            inbound_links[norm_link] += 1

    page_to_links[page] = links

  def _credit_non_page_links(doc_text: str, doc_dir: str) -> None:
    """Credits inbound links from top-level docs, agent configs, and srcs/."""
    for link in re.findall(r"\[\[([^\]]+)\]\]", doc_text):
      link_normalized = _normalize_link(link)
      for tgt in all_pages_by_basename.get(link_normalized, []):
        if tgt in inbound_links:
          inbound_links[tgt] += 1
    for token in re.findall(r"`([a-z0-9][a-z0-9_-]+)`", doc_text):
      for tgt in all_pages_by_basename.get(token, []):
        if tgt in inbound_links:
          inbound_links[tgt] += 1
    for link in re.findall(r"(?<!\])\]\(([^)]+)\)", doc_text):
      if (
          not link.startswith("http")
          and not link.startswith("#")
          and not link.startswith("file://")
      ):
        link_core = link.split("#")[0]
        norm_link = os.path.normpath(os.path.join(doc_dir, link_core))
        if norm_link in inbound_links:
          inbound_links[norm_link] += 1

  for doc_rel in ("TUTORIAL.md", "README.md"):
    doc_full = os.path.join(wiki_dir, doc_rel)
    if os.path.isfile(doc_full):
      try:
        _credit_non_page_links(_read_text(doc_full), "")
      except _READ_ERRORS:
        pass

  repo_root = os.path.dirname(wiki_dir)
  for root_doc in ("README.md", "CLAUDE.md", "prompt.md"):
    doc_full = os.path.join(repo_root, root_doc)
    if os.path.isfile(doc_full):
      try:
        _credit_non_page_links(_read_text(doc_full), "..")
      except _READ_ERRORS:
        pass

  for agent_sub in ("_agents", ".claude"):
    agent_dir = os.path.join(repo_root, agent_sub)
    if os.path.isdir(agent_dir):
      for root, _, files in os.walk(agent_dir):
        for f in sorted(files):
          if f.endswith((".md", ".json")):
            full_p = os.path.join(root, f)
            try:
              _credit_non_page_links(
                  _read_text(full_p), os.path.relpath(root, wiki_dir)
              )
            except _READ_ERRORS:
              pass

  srcs_dir = os.path.join(wiki_dir, "srcs")
  if os.path.exists(srcs_dir):
    for root, dirs, files in os.walk(srcs_dir):
      for f in files:
        if f.endswith(".md"):
          rel_root = os.path.relpath(root, srcs_dir)
          if rel_root == "." or rel_root not in allowed_srcs_subdirs:
            issues.append(
                f"Invalid source location for '{f}': located in"
                f" 'srcs/{rel_root}'"
                " which is not an approved subdirectory."
            )
          full_path = os.path.join(root, f)
          content = _read_text(full_path)
          fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
          actual_body = content
          if fm_match:
            actual_body = content[fm_match.end() :]
            try:
              fm = parse_frontmatter(fm_match.group(1))
              claimed_sha = fm.get("sha256")
              actual_sha = hashlib.sha256(
                  actual_body.encode("utf-8")
              ).hexdigest()
              if claimed_sha != actual_sha:
                warnings.append(
                    f"Source drift detected in source file '{f}': hash changed"
                    f" from '{claimed_sha}' to '{actual_sha}'"
                )
                rel_src_path = os.path.relpath(full_path, wiki_dir)
                for p_name in all_pages:
                  p_fm = page_frontmatters.get(p_name)
                  if p_fm and isinstance(p_fm, dict):
                    p_srcs = p_fm.get("sources")
                    if isinstance(p_srcs, list) and rel_src_path in p_srcs:
                      warnings.append(
                          f"Page '{p_name}' references drifted source '{f}' and"
                          " may be out of date."
                      )
                if claimed_sha:
                  new_content = content.replace(str(claimed_sha), actual_sha, 1)
                else:
                  new_content = content.replace(
                      "---\n", f"---\nsha256: {actual_sha}\n", 1
                  )
                with open(full_path, "w") as fw:
                  fw.write(new_content)
            except (OSError, AttributeError) + _YAML_ERRORS as e:
              issues.append(f"Error parsing source frontmatter in {f}: {e}")
          else:
            issues.append(
                f"Missing Frontmatter in source file '{rel_root}/{f}'"
            )

          rel_src_path = os.path.relpath(full_path, wiki_dir)
          for link in re.findall(r"\[\[([^\]]+)\]\]", content):
            link_normalized = _normalize_link(link)
            for tgt in all_pages_by_basename.get(link_normalized, []):
              if tgt in inbound_links:
                inbound_links[tgt] += 1
          for link in re.findall(r"(?<!\])\]\(([^)]+)\)", actual_body):
            if (
                not link.startswith("http")
                and not link.startswith("#")
                and not link.startswith("file://")
            ):
              link_core = link.split("#")[0]
              norm_link = os.path.normpath(
                  os.path.join(os.path.dirname(rel_src_path), link_core)
              )
              if not os.path.exists(os.path.join(wiki_dir, norm_link)):
                issues.append(
                    f"Broken link '{link}' in source file '{rel_src_path}'"
                )
              else:
                if norm_link in inbound_links:
                  inbound_links[norm_link] += 1

  missing_index_pages = []
  indexed_links_set = set(indexed_links)
  indexed_bases_set = {os.path.splitext(il)[0] for il in indexed_links}
  for page in all_pages:
    if inbound_links[page] == 0:
      if not page.startswith("entities/"):
        issues.append(f"Orphan page (no inbound links): {page}")

    page_base = os.path.splitext(page)[0]
    if page not in indexed_links_set and page_base not in indexed_bases_set:
      missing_index_pages.append(page)

  if missing_index_pages:
    update_index_file(wiki_dir, missing_index_pages, current_date_str)
    for page in missing_index_pages:
      warnings.append(f"Auto-resolved: Added missing page {page} to INDEX.md")

  # Explicitly check that no scratch files are tracked by version control
  scratch_dir = os.path.join(wiki_dir, "scratch")
  if os.path.isdir(scratch_dir) and os.listdir(scratch_dir):
    try:
      res = subprocess.run(
          ["hg", "files", "--pager=never", "-y", "scratch/"],
          cwd=wiki_dir,
          capture_output=True,
          text=True,
          timeout=30,
          check=False,
      )
      if res.returncode == 0 and res.stdout.strip():
        for tracked_file in res.stdout.strip().splitlines():
          issues.append(
              f"File '{tracked_file}' is tracked by version control. Files from"
              " the scratch/ directory must not be uploaded to the CL."
          )
    except subprocess.TimeoutExpired:
      # A slow VCS/filesystem is an environment problem, not a wiki defect.
      warnings.append(
          "Skipped tracked-scratch-files check: VCS file listing timed out"
          " after 30 seconds."
      )
    except FileNotFoundError:
      pass
    except (OSError, subprocess.SubprocessError, ValueError) as e:
      issues.append(
          f"Failed to check tracked scratch files via VCS: {e}"
      )

  check_portability_leakage(wiki_dir, issues, file_cache=file_cache)
  check_pii_leakage(wiki_dir, issues, file_cache=file_cache)
  check_dashboard_pii_and_internal_leaks(wiki_dir, warnings)
  rotate_log_if_needed(wiki_dir, current_date_str)
  append_lint_entry(wiki_dir, current_date_str, len(issues))

  report = {
      "status": "PASS" if not issues else "FAIL",
      "issues": issues,
      "warnings": warnings,
      "checked_pages": all_pages,
  }
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  if len(sys.argv) > 1:
    print(
        f"wiki_linter.py takes no arguments (got {sys.argv[1:]}).\n"
        "The former --operational flag was a no-op and has been removed; "
        "there is a single check set.",
        file=sys.stderr,
    )
    sys.exit(2)
  run_linter()

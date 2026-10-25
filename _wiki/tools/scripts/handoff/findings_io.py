#!/usr/bin/env python3
"""findings_io.py - Loader and accessors for troubleshoot_findings.json.

troubleshoot_findings.json is the primary data source for every artifact
rendered by the handoff suite (manifests, READMEs, failure tables, partner
issue body). It is extracted from a Debug Wiki /troubleshoot report or
converted from triage state JSON.

Rules enforced here:
  * Missing, null or empty values render as NOT_PROVIDED in markdown and as
    None (JSON null) in machine-readable output.
  * Destination buckets have no default: they resolve from --bucket,
    findings['identity']['bucket'] or $HANDOFF_BUCKET, and anything that
    writes to a bucket fails when none of those is set (require_bucket).
  * GCS paths are keyed by identity.storage_key (the support case folder),
    never by the partner issue id.
  * Partner issue references use identity.issue_id or NOT_YET_FILED.

CLI:
  python3 findings_io.py validate --findings=<path>
  python3 findings_io.py get --findings=<path> --key=identity.storage_key
"""

import argparse
from collections.abc import Sequence
import importlib
import json
import os
import sys
import types
import unicodedata

# _wiki/scratch/, resolved from this file so it is right whatever the cwd.
WIKI_SCRATCH_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scratch"
    )
)

NOT_PROVIDED = "NOT PROVIDED"
NOT_YET_FILED = "NOT YET FILED"
NOT_COLLECTED = "not collected in troubleshoot run"
PARTNER_ISSUETRACKER_URL = os.environ.get(
    "PARTNER_ISSUETRACKER_URL", "https://issuetracker.google.com/issues/"
)
PARTNER_COMPONENT = os.environ.get("PARTNER_COMPONENT", "")
PARTNER_TEMPLATE = os.environ.get("PARTNER_TEMPLATE", "")

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "templates",
    "troubleshoot_findings.schema.json",
)

CATEGORY_MAP = types.MappingProxyType({
    "step_time": "category_1_step_time_latency",
    "category_1": "category_1_step_time_latency",
    "category_1_step_time_latency": "category_1_step_time_latency",
    "memory": "category_2_memory_oom",
    "category_2": "category_2_memory_oom",
    "category_2_memory_oom": "category_2_memory_oom",
    "orchestration": "category_3_orchestration_thrash",
    "category_3": "category_3_orchestration_thrash",
    "category_3_orchestration_thrash": "category_3_orchestration_thrash",
    "hardware": "category_4_hardware_optical_torus",
    "category_4": "category_4_hardware_optical_torus",
    "category_4_hardware_optical_torus": "category_4_hardware_optical_torus",
    "sdc": "category_5_silent_data_corruption",
    "category_5": "category_5_silent_data_corruption",
    "category_5_silent_data_corruption": "category_5_silent_data_corruption",
})

_ASCII_MAP = types.MappingProxyType({
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00d7": "x",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u00a0": " ",
    "\u2022": "*",
    "\u00b5": "u",
    "\u03bc": "u",
    "\u2248": "~",
})


class FindingsError(Exception):
  """Raised when the findings file is missing or unreadable."""


def ascii_text(value: object) -> str:
  """Returns value as 7-bit ASCII text (typographic normalization only)."""
  text = str(value)
  for source_char, ascii_replacement in _ASCII_MAP.items():
    text = text.replace(source_char, ascii_replacement)
  text = unicodedata.normalize("NFKD", text)
  return text.encode("ascii", "replace").decode("ascii")


def load_findings(path: str) -> dict[str, object]:
  """Loads a findings file.

  Args:
    path: Path to troubleshoot_findings.json. A leading `_wiki/` is added or
      stripped when that makes the path resolve from the current directory.

  Returns:
    The parsed findings object.

  Raises:
    FindingsError: The path is empty, absent, not valid JSON, or does not hold
      a JSON object.
  """
  if not path:
    raise FindingsError(
        "No findings file given. Pass --findings=<troubleshoot_findings.json>."
    )
  if not os.path.isfile(path):
    if path.startswith("_wiki/") and os.path.isfile(path[len("_wiki/") :]):
      path = path[len("_wiki/") :]
    elif os.path.isfile(os.path.join("_wiki", path)):
      path = os.path.join("_wiki", path)
  if not os.path.isfile(path):
    raise FindingsError(f"Findings file not found: {path}")
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, ValueError) as exc:
    raise FindingsError(
        f"Findings file is not valid JSON: {path}: {exc}"
    ) from exc
  if not isinstance(data, dict):
    raise FindingsError(f"Findings file must contain a JSON object: {path}")
  return data


def is_missing(value: object) -> bool:
  """Returns True when value is None, blank, or an empty container."""
  if value is None:
    return True
  if isinstance(value, str) and not value.strip():
    return True
  if isinstance(value, (list, tuple, dict)) and not value:
    return True
  return False


def get_raw(findings: dict[str, object], dotted: str) -> object | None:
  """Returns the value at a dotted path, or None if missing/null/empty."""
  current = findings
  for part in dotted.split("."):
    if isinstance(current, dict) and part in current:
      current = current[part]
    else:
      return None
  return None if is_missing(current) else current


def get(
    findings: dict[str, object], dotted: str, missing: str = NOT_PROVIDED
) -> str:
  """Returns the value at a dotted path as ASCII text, or 'NOT PROVIDED'."""
  raw_value = get_raw(findings, dotted)
  if raw_value is None:
    return missing
  if isinstance(raw_value, (list, tuple)):
    return ascii_text(", ".join(str(item) for item in raw_value))
  if isinstance(raw_value, dict):
    return ascii_text(json.dumps(raw_value, sort_keys=True))
  return ascii_text(raw_value)


def get_list(findings: dict[str, object], dotted: str) -> list[object]:
  """Returns the value at a dotted path as a list, or [] when missing."""
  raw_value = get_raw(findings, dotted)
  if raw_value is None:
    return []
  if isinstance(raw_value, (list, tuple)):
    return list(raw_value)
  return [raw_value]


def get_int(findings: dict[str, object], dotted: str) -> int | None:
  """Returns the int at a dotted path, or None when absent or non-integer."""
  raw_value = get_raw(findings, dotted)
  if isinstance(raw_value, bool):
    return None
  if isinstance(raw_value, int):
    return raw_value
  if isinstance(raw_value, str) and raw_value.strip().isdigit():
    return int(raw_value.strip())
  return None


def json_value(value: object) -> object | None:
  """Normalizes value for JSON output: missing -> None, numbers kept."""
  if is_missing(value):
    return None
  if isinstance(value, str):
    return ascii_text(value)
  return value


def md_cell(value: object, missing: str = NOT_PROVIDED) -> str:
  """Formats a value for a markdown table cell (pipes escaped, no newlines)."""
  if is_missing(value):
    return missing
  if not isinstance(value, (list, tuple)):
    formatted_value = value
  else:
    formatted_value = ", ".join(map(str, value))
  text = ascii_text(formatted_value)
  return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def canonical_category(findings: dict[str, object]) -> str | None:
  """Returns the canonical category name, the raw value if unmapped, or None."""
  raw_value = get_raw(findings, "category")
  if raw_value is None:
    return None
  return CATEGORY_MAP.get(str(raw_value).strip(), ascii_text(raw_value))


def gcs_bucket(
    findings: dict[str, object] | None = None,
    override_bucket: str | None = None,
) -> str | None:
  """Resolves the GCS bucket dynamically, or None when it is not specified.

  Precedence:
  1. override_bucket CLI argument
  2. findings['identity']['bucket']
  3. $HANDOFF_BUCKET environment variable

  Args:
    findings: Optional findings object to read identity.bucket from.
    override_bucket: Optional bucket name or gs:// URI given on the CLI.

  Returns:
    The bucket name without gs:// prefix or trailing slash, or None.
  """
  if override_bucket and override_bucket.strip():
    bucket_name = override_bucket.strip()
    if bucket_name.startswith("gs://"):
      bucket_name = bucket_name[5:]
    return bucket_name.rstrip("/")

  if findings:
    raw_value = get_raw(findings, "identity.bucket")
    if raw_value and str(raw_value).strip():
      bucket_name = str(raw_value).strip()
      if bucket_name.startswith("gs://"):
        bucket_name = bucket_name[5:]
      return bucket_name.rstrip("/")

  env_bucket = os.getenv("HANDOFF_BUCKET")
  if env_bucket and env_bucket.strip():
    bucket_name = env_bucket.strip()
    if bucket_name.startswith("gs://"):
      bucket_name = bucket_name[5:]
    return bucket_name.rstrip("/")

  return None


def require_bucket(
    findings: dict[str, object] | None = None,
    override_bucket: str | None = None,
) -> str:
  """Like gcs_bucket, but raises FindingsError when no bucket is specified.

  Destination buckets depend on the customer project, so there is no default:
  the operator must be asked for one.

  Args:
    findings: Optional findings object to read identity.bucket from.
    override_bucket: Optional bucket name or gs:// URI given on the CLI.

  Returns:
    The resolved bucket name.

  Raises:
    FindingsError: No bucket is specified.
  """
  bucket_name = gcs_bucket(findings, override_bucket)
  if not bucket_name:
    raise FindingsError(
        "No destination GCS bucket specified. Pass --bucket=<name>, set"
        " identity.bucket in troubleshoot_findings.json, or export"
        " HANDOFF_BUCKET. Ask the operator for the customer project's bucket."
    )
  return bucket_name


def storage_key(findings: dict[str, object]) -> str | None:
  """Returns identity.storage_key as ASCII text, or None when missing."""
  raw_value = get_raw(findings, "identity.storage_key")
  return ascii_text(raw_value) if raw_value is not None else None


def storage_root(
    findings: dict[str, object], bucket: str | None = None
) -> str | None:
  """gs://<bucket>/<storage_key>/ or None when bucket or key is missing."""
  key = storage_key(findings)
  bucket_name = gcs_bucket(findings, bucket)
  return f"gs://{bucket_name}/{key}/" if key and bucket_name else None


def gcs_path(
    findings: dict[str, object], relative_path: str, bucket: str | None = None
) -> str | None:
  """Returns gs://<bucket>/<storage_key>/<relative_path>, or None."""
  root = storage_root(findings, bucket)
  return f"{root}{relative_path.lstrip('/')}" if root else None


def console_url(
    findings: dict[str, object], bucket: str | None = None
) -> str | None:
  """Returns the Cloud Console browser URL for the incident folder, or None."""
  key = storage_key(findings)
  bucket_name = gcs_bucket(findings, bucket)
  if not key or not bucket_name:
    return None
  return (
      f"https://console.cloud.google.com/storage/browser/{bucket_name}/{key}/"
  )


def issue_id(findings: dict[str, object]) -> str | None:
  """Returns identity.issue_id as ASCII text, or None when missing."""
  raw_value = get_raw(findings, "identity.issue_id")
  return ascii_text(raw_value) if raw_value is not None else None


def issue_ref(findings: dict[str, object]) -> str:
  """Partner issue reference (b/<issue_id>) or NOT YET FILED."""
  resolved_issue_id = issue_id(findings)
  return f"b/{resolved_issue_id}" if resolved_issue_id else NOT_YET_FILED


def issue_url(findings: dict[str, object]) -> str:
  """Returns the Partner IssueTracker URL for the issue or template creation."""
  resolved_issue_id = issue_id(findings)
  if resolved_issue_id:
    return f"{PARTNER_ISSUETRACKER_URL}{resolved_issue_id}"
  component = get_raw(findings, "identity.component_id") or PARTNER_COMPONENT
  template = get_raw(findings, "identity.template_id") or PARTNER_TEMPLATE
  if component and template:
    return (
        f"{PARTNER_ISSUETRACKER_URL}new?component={component}"
        f"&template={template} ({NOT_YET_FILED})"
    )
  if component:
    return (
        f"{PARTNER_ISSUETRACKER_URL}new?component={component} ({NOT_YET_FILED})"
    )
  return f"{PARTNER_ISSUETRACKER_URL}new ({NOT_YET_FILED})"


def artifacts(findings: dict[str, object]) -> list[dict[str, object]]:
  """Artifacts actually uploaded: list of {'path', 'kind'} with clean paths."""
  out = []
  for item in get_list(findings, "artifacts"):
    if isinstance(item, dict) and not is_missing(item.get("path")):
      out.append({
          "path": ascii_text(item["path"]).lstrip("/"),
          "kind": item.get("kind"),
      })
    elif isinstance(item, str) and item.strip():
      out.append({"path": ascii_text(item).lstrip("/"), "kind": None})
  return out


def artifact_paths(findings: dict[str, object]) -> list[str]:
  """Returns the list of clean relative paths for all uploaded artifacts."""
  return [artifact["path"] for artifact in artifacts(findings)]


def artifacts_of_kind(findings: dict[str, object], kind: str) -> list[str]:
  """Returns relative paths of uploaded artifacts matching the given kind."""
  return [
      artifact["path"]
      for artifact in artifacts(findings)
      if artifact.get("kind") == kind
  ]


def filter_to_artifacts(
    findings: dict[str, object], refs: Sequence[object] | None
) -> list[str]:
  """Keeps only refs that name an uploaded artifact (exact relative path)."""
  present = set(artifact_paths(findings))
  return [
      ref.lstrip("/")
      for ref in refs or []
      if isinstance(ref, str) and ref.lstrip("/") in present
  ]


def fmt_step(value: object) -> str:
  """Formats a step time for markdown: '3.4' -> '3.4s'."""
  if is_missing(value):
    return NOT_PROVIDED
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return f"{value}s"
  return ascii_text(value)


def window_text(findings: dict[str, object], name: str) -> str:
  """Formats windows.<name>.start and end as '<start> to <end>'."""
  start = get(findings, f"windows.{name}.start")
  end = get(findings, f"windows.{name}.end")
  if start == NOT_PROVIDED and end == NOT_PROVIDED:
    return NOT_PROVIDED
  return f"{start} to {end}"


def bullet_list(
    items: Sequence[object],
    missing: str = NOT_PROVIDED,
) -> str:
  """Renders a sequence of items as a markdown bullet list."""
  clean_items = [ascii_text(item) for item in items if not is_missing(item)]
  if not clean_items:
    return f"- {missing}"
  return "\n".join(f"- {item}" for item in clean_items)


def render_template(text: str, values: dict[str, object]) -> str:
  """Replaces {{KEY}} placeholders. Values are ASCII-normalized."""
  for key, raw_value in values.items():
    replacement = ascii_text(
        raw_value if raw_value is not None else NOT_PROVIDED
    )
    text = text.replace("{{" + key + "}}", replacement)
  return text


def proves_md(findings: dict[str, object]) -> str:
  """Renders the P-list (proves) items as a markdown bullet list."""
  items = [
      prove_item
      for prove_item in get_list(findings, "proves")
      if isinstance(prove_item, dict)
  ]
  if not items:
    return f"- {NOT_PROVIDED}"
  lines = []
  for prove_item in items:
    refs = prove_item.get("evidence_refs") or []
    suffix = f" (evidence: {', '.join(map(str, refs))})" if refs else ""
    prove_id = ascii_text(prove_item.get("id", "?"))
    prove_text = ascii_text(prove_item.get("text", NOT_PROVIDED))
    lines.append(f"- {prove_id}: {prove_text}{ascii_text(suffix)}")
  return "\n".join(lines)


def cannot_prove_md(findings: dict[str, object]) -> str:
  """Renders the C-list (cannot_prove) items as a markdown bullet list."""
  items = [
      cannot_prove_item
      for cannot_prove_item in get_list(findings, "cannot_prove")
      if isinstance(cannot_prove_item, dict)
  ]
  if not items:
    return f"- {NOT_PROVIDED}"
  return "\n".join(
      f"- {ascii_text(cannot_prove_item.get('id', '?'))}:"
      f" {ascii_text(cannot_prove_item.get('text', NOT_PROVIDED))}"
      for cannot_prove_item in items
  )


def next_steps_md(findings: dict[str, object]) -> str:
  """Bullet list of customer_next_steps with owner and C-item reference."""
  items = [
      step_item
      for step_item in get_list(findings, "customer_next_steps")
      if isinstance(step_item, dict)
  ]
  if not items:
    return f"- {NOT_PROVIDED}"
  lines = []
  for step_item in items:
    cannot_prove_ref = ""
    if not is_missing(step_item.get("c_ref")):
      cannot_prove_ref = f" (addresses {ascii_text(step_item['c_ref'])})"
    owner = NOT_PROVIDED
    if not is_missing(step_item.get("owner")):
      owner = ascii_text(step_item.get("owner"))
    step = ascii_text(step_item.get("step", NOT_PROVIDED))
    lines.append(f"- [Owner: {owner}] {step}{cannot_prove_ref}")
  return "\n".join(lines)


def questions_md(findings: dict[str, object]) -> str:
  """Bullet list of questions_for_google with their if-yes/if-no branches."""
  items = [
      question_item
      for question_item in get_list(findings, "questions_for_google")
      if isinstance(question_item, dict)
  ]
  if not items:
    return f"- {NOT_PROVIDED}"
  lines = []
  for question_item in items:
    question_id = (
        ascii_text(question_item.get("id"))
        if not is_missing(question_item.get("id"))
        else "Q?"
    )
    owner_str = (
        ascii_text(question_item.get("owner"))
        if not is_missing(question_item.get("owner"))
        else NOT_PROVIDED
    )
    c_ref_text = ascii_text(question_item.get("c_ref", NOT_PROVIDED))
    question_text = ascii_text(question_item.get("question", NOT_PROVIDED))
    lines.append(
        f"- {question_id} (cites {c_ref_text}; owner: {owner_str}):"
        f" {question_text}"
    )
    if_yes = (
        ascii_text(question_item.get("if_yes"))
        if not is_missing(question_item.get("if_yes"))
        else NOT_PROVIDED
    )
    if_no = (
        ascii_text(question_item.get("if_no"))
        if not is_missing(question_item.get("if_no"))
        else NOT_PROVIDED
    )
    lines.append(f"  - If yes: {if_yes}")
    lines.append(f"  - If no: {if_no}")
  return "\n".join(lines)


def discrepancies_md(findings: dict[str, object]) -> str:
  """Bullet list of recorded report-vs-telemetry discrepancies."""
  items = [
      discrepancy
      for discrepancy in get_list(findings, "discrepancies")
      if isinstance(discrepancy, dict)
  ]
  if not items:
    return "- None recorded"
  lines = []
  for discrepancy in items:
    report_value = NOT_PROVIDED
    if not is_missing(discrepancy.get("report_value")):
      report_value = ascii_text(discrepancy.get("report_value"))
    note = ""
    if not is_missing(discrepancy.get("note")):
      note = ascii_text(discrepancy.get("note"))
    field = ascii_text(discrepancy.get("field", "?"))
    lines.append(
        f'- `{field}`: report says "{report_value}" -- {note}'.rstrip(" -")
    )
  return "\n".join(lines)


def artifacts_md(
    findings: dict[str, object],
    prefix: str = "",
    bucket: str | None = None,
) -> str:
  """Bullet list of uploaded artifacts as gs:// links."""
  paths = [
      artifact_path
      for artifact_path in artifact_paths(findings)
      if artifact_path.startswith(prefix)
  ]
  if not paths:
    return "- None uploaded"
  return "\n".join(
      f"- `{gcs_path(findings, artifact_path, bucket=bucket) or artifact_path}`"
      for artifact_path in paths
  )


def validate(findings: dict[str, object]) -> list[str]:
  """Returns a list of problems. Uses jsonschema when available."""
  problems = []
  for key in ("source", "identity"):
    if not isinstance(findings.get(key), dict):
      problems.append(f"missing required object '{key}'")
  for key in ("identity.case_id", "identity.storage_key"):
    if get_raw(findings, key) is None:
      problems.append(f"missing required field '{key}'")
  for index, prove_item in enumerate(get_list(findings, "proves")):
    if (
        not isinstance(prove_item, dict)
        or is_missing(prove_item.get("id"))
        or is_missing(prove_item.get("text"))
    ):
      problems.append(f"proves[{index}] needs 'id' and 'text'")
  for index, cannot_prove_item in enumerate(get_list(findings, "cannot_prove")):
    if (
        not isinstance(cannot_prove_item, dict)
        or is_missing(cannot_prove_item.get("id"))
        or is_missing(cannot_prove_item.get("text"))
    ):
      problems.append(f"cannot_prove[{index}] needs 'id' and 'text'")
  c_ids = {
      cannot_prove_item.get("id")
      for cannot_prove_item in get_list(findings, "cannot_prove")
      if isinstance(cannot_prove_item, dict)
  }
  for index, question_item in enumerate(
      get_list(findings, "questions_for_google")
  ):
    if isinstance(question_item, dict) and question_item.get("c_ref") not in (
        c_ids
    ):
      problems.append(
          f"questions_for_google[{index}] cites unknown C item"
          f" '{question_item.get('c_ref')}'"
      )
  # jsonschema is optional: without it the schema check is skipped.
  try:
    jsonschema = importlib.import_module("jsonschema")
  except ImportError:
    jsonschema = None
  if jsonschema is not None and os.path.isfile(SCHEMA_PATH):
    try:
      with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
      for validation_error in jsonschema.Draft202012Validator(
          schema
      ).iter_errors(findings):
        location = (
            ".".join(
                str(path_part) for path_part in validation_error.absolute_path
            )
            or "<root>"
        )
        problems.append(f"schema: {location}: {validation_error.message}")
    except (
        OSError,
        ValueError,
        AttributeError,
        jsonschema.SchemaError,
    ) as exc:
      problems.append(f"schema validation skipped: {exc}")
  return problems


def load_for_handoff(
    path: str, override_bucket: str | None = None
) -> tuple[dict[str, object], str]:
  """Loads findings for a handoff generator; raises FindingsError if unusable.

  Validation problems are fatal rather than warnings, and a destination bucket
  must resolve, so no generator renders a package from incomplete findings.

  Args:
    path: Path to troubleshoot_findings.json.
    override_bucket: Optional bucket name or gs:// URI given on the CLI.

  Returns:
    A (findings, bucket) tuple of the validated findings and resolved bucket.

  Raises:
    FindingsError: The file is unreadable or invalid, or no bucket resolves.
  """
  findings = load_findings(path)
  problems = validate(findings)
  if problems:
    raise FindingsError(
        f"{path} fails findings validation:\n  - " + "\n  - ".join(problems)
    )
  return findings, require_bucket(findings, override_bucket)


def workload_dir(findings: dict[str, object]) -> str | None:
  """identity.workload as a single path segment, or None when missing."""
  workload = get_raw(findings, "identity.workload")
  if workload is None:
    return None
  return ascii_text(workload).replace("/", "_")


def required_package_files(workload: str | None) -> list[str]:
  """Files every published handoff package must contain, relative to its root.

  Customer-authored files (findings/customer/_customer_latest.md) and raw
  telemetry under data/ are deliberately excluded: they exist only when there
  is something to put in them.

  Args:
    workload: identity.workload as a path segment, or None for an
      incident-only package.

  Returns:
    Relative paths of the required files.
  """
  files = [
      "BUCKET_STRUCTURE.md",
      "INCIDENT_MANIFEST.json",
      "INCIDENT_OVERVIEW.md",
      "README.md",
      "infrastructure/findings/_infra_latest.md",
  ]
  if workload:
    files += [
        f"jobs/{workload}/JOB_MANIFEST.json",
        f"jobs/{workload}/JOB_README.md",
        f"jobs/{workload}/README.md",
        f"jobs/{workload}/findings/google/_google_latest.md",
    ]
  return files


# Content that carries no information: empty, whitespace, or a bare marker.
PLACEHOLDER_TEXTS = frozenset({"", NOT_PROVIDED, "NOT_PROVIDED", "{}", "[]"})


def content_problem(relative_path: str, data: bytes) -> str | None:
  """Returns why a package file's content is unusable, or None if it is fine.

  Catches 0-byte files, placeholder-only files and unparseable JSON. Folder
  markers (.keep) are expected to be empty and are always accepted.

  Args:
    relative_path: Path of the file relative to the package root.
    data: Raw file content.

  Returns:
    A human-readable problem description, or None.
  """
  if os.path.basename(relative_path) == ".keep":
    return None
  if not data:
    return "file is empty (0 bytes)"
  try:
    text = data.decode("utf-8")
  except UnicodeDecodeError:
    return None  # Binary payloads (e.g. profiles) are not inspected further.
  if text.strip() in PLACEHOLDER_TEXTS:
    return f"file holds only a placeholder ({text.strip()!r})"
  if relative_path.endswith(".json"):
    try:
      json.loads(text)
    except json.JSONDecodeError as exc:
      return f"invalid JSON: {exc}"
  return None


def main() -> None:
  """CLI entry point for validating or querying troubleshoot_findings.json."""
  parser = argparse.ArgumentParser(
      description="troubleshoot_findings.json helper"
  )
  parser.add_argument("command", choices=["validate", "get", "bucket"])
  parser.add_argument(
      "--findings", required=True, help="Path to troubleshoot_findings.json"
  )
  parser.add_argument("--key", default="", help="Dotted key for 'get'")
  parser.add_argument(
      "--bucket", default="", help="Optional GCS bucket override"
  )
  args = parser.parse_args()
  try:
    findings = load_findings(args.findings)
  except FindingsError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)
  if args.command == "get":
    print(get(findings, args.key))
    return
  if args.command == "bucket":
    print(gcs_bucket(findings, args.bucket) or NOT_PROVIDED)
    return
  problems = validate(findings)
  if problems:
    for problem in problems:
      print(f"[WARN] {problem}")
    sys.exit(2)
  print(f"[OK] {args.findings} passes findings validation")


if __name__ == "__main__":
  main()

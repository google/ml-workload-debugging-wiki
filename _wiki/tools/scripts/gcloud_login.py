#!/usr/bin/env python3
"""GCloud Authentication and Application Default Credentials (ADC) refresher.

Executes 'gcloud auth login --update-adc --no-launch-browser' and verifies
resulting tokens.
"""

import argparse
import json
import subprocess
import sys
from typing import Any


def _str_to_bool(val: Any) -> bool:
  """Parses a boolean or string representation of a boolean."""
  if isinstance(val, bool):
    return val
  string_value = str(val).strip().lower()
  if string_value in ("true", "1", "yes", "y", "on"):
    return True
  if string_value in ("false", "0", "no", "n", "off", ""):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {val!r}")


def get_active_account() -> str:
  """Returns the active gcloud account, or '' if unset or unavailable."""
  try:
    res = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if res.returncode == 0:
      acc = res.stdout.strip()
      if acc and acc != "(unset)":
        return acc
  except (OSError, subprocess.SubprocessError):
    pass
  return ""


def verify_tokens() -> dict[str, Any]:
  """Checks if gcloud and ADC access tokens are healthy.

  Returns:
    A dictionary containing boolean validity flags and optional error strings
    for gcloud and ADC tokens.
  """
  status: dict[str, Any] = {
      "gcloud": False,
      "adc": False,
      "gcloud_error": None,
      "adc_error": None,
  }

  # 1. Verify gcloud user token
  try:
    g_tok = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    if g_tok.returncode == 0 and g_tok.stdout.strip().startswith("ya29."):
      status["gcloud"] = True
    else:
      err_output = (g_tok.stderr or "") + (g_tok.stdout or "")
      status["gcloud_error"] = _classify_auth_error(err_output)
  except (OSError, subprocess.SubprocessError) as e:
    status["gcloud_error"] = str(e)

  # 2. Verify ADC token
  try:
    a_tok = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    if a_tok.returncode == 0 and a_tok.stdout.strip().startswith("ya29."):
      status["adc"] = True
    else:
      err_output = (a_tok.stderr or "") + (a_tok.stdout or "")
      status["adc_error"] = _classify_auth_error(err_output)
  except (OSError, subprocess.SubprocessError) as e:
    status["adc_error"] = str(e)

  return status


def _classify_auth_error(output: str) -> str:
  """Diagnoses permission denial or token expiration in command output."""
  lower = output.lower()
  if "access_denied" in lower or "permission denied" in lower or "403" in lower:
    return (
        "Access was denied during token retrieval. Verify IAM permissions "
        "or re-authenticate with `gcloud auth login --update-adc`."
    )
  if (
      "refreshing your current auth tokens" in lower
      or "token has expired" in lower
      or "invalid_grant" in lower
      or "reauthentication" in lower
  ):
    return (
        "Auth tokens have expired or are missing. Re-authentication required."
    )
  if output.strip():
    return output.strip()
  return "Unknown authentication failure"


def main() -> None:
  """Runs the CLI to verify or refresh gcloud and ADC authentication."""
  parser = argparse.ArgumentParser(
      description="Run gcloud auth login --update-adc and verify tokens."
  )
  parser.add_argument(
      "--account",
      default="",
      help="Specific user account to authenticate (optional).",
  )
  parser.add_argument(
      "--no-browser",
      dest="no_browser",
      nargs="?",
      const=True,
      default=False,
      type=_str_to_bool,
      help="Authorize without launching a local web browser (default: False).",
  )
  parser.add_argument(
      "--no-launch-browser",
      dest="no_browser",
      nargs="?",
      const=True,
      default=False,
      type=_str_to_bool,
      help="Alias for --no-browser.",
  )
  parser.add_argument(
      "--check-only",
      dest="check_only",
      nargs="?",
      const=True,
      default=False,
      type=_str_to_bool,
      help="Only check token validity without triggering login.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Force re-authentication even if tokens are already valid.",
  )

  args = parser.parse_args()

  initial_tokens = verify_tokens()
  if initial_tokens["gcloud"] and initial_tokens["adc"] and not args.force:
    active_acc = get_active_account()
    result = {
        "status": "ALREADY_AUTHENTICATED",
        "active_account": active_acc or "unknown",
        "gcloud_token_valid": True,
        "adc_token_valid": True,
    }
    print(json.dumps(result, indent=2))
    print("\n--- SUMMARY REPORT (summary_fields) ---")
    print(f"  status: {result['status']}")
    print(f"  active_account: {result['active_account']}")
    print(f"  gcloud_token_valid: {result['gcloud_token_valid']}")
    print(f"  adc_token_valid: {result['adc_token_valid']}")
    print("---------------------------------------")
    return

  if not args.check_only:
    cmd = ["gcloud", "auth", "login", "--update-adc"]
    if args.no_browser:
      cmd.append("--no-launch-browser")
    if args.account:
      cmd.append(args.account)

    try:
      subprocess.run(cmd, check=False)
    except (OSError, subprocess.SubprocessError) as e:
      print(f"Error launching gcloud auth login: {e}", file=sys.stderr)

  tokens = verify_tokens()
  active_acc = get_active_account()

  success = tokens["gcloud"] and tokens["adc"]
  if success:
    status_str = "SUCCESS"
  elif args.check_only:
    status_str = "UNAUTHENTICATED"
  else:
    status_str = "INCOMPLETE"

  result = {
      "status": status_str,
      "active_account": active_acc or "unknown",
      "gcloud_token_valid": tokens["gcloud"],
      "adc_token_valid": tokens["adc"],
  }
  if tokens.get("gcloud_error"):
    result["gcloud_error"] = tokens["gcloud_error"]
  if tokens.get("adc_error"):
    result["adc_error"] = tokens["adc_error"]

  print(json.dumps(result, indent=2))
  print("\n--- SUMMARY REPORT (summary_fields) ---")
  print(f"  status: {result['status']}")
  print(f"  active_account: {result['active_account']}")
  print(f"  gcloud_token_valid: {result['gcloud_token_valid']}")
  print(f"  adc_token_valid: {result['adc_token_valid']}")
  print("---------------------------------------")

  if not success:
    sys.exit(1)


if __name__ == "__main__":
  main()

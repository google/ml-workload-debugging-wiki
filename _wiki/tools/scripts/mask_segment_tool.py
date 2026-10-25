#!/usr/bin/env python3
"""Tool to convert raw log strings into MaskedSegment equivalent format.

Supports word-level tokenization preserving exact original punctuation layout.
"""

import argparse
import functools
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Token Types
TOKEN_TYPE_UNKNOWN = "UNKNOWN"
TOKEN_TYPE_WORD = "WORD"
TOKEN_TYPE_ALPHABET = "ALPHABET"
TOKEN_TYPE_NUMBER = "NUMBER"
TOKEN_TYPE_NUMBER_ALPHABET = "NUMBER_ALPHABET"
TOKEN_TYPE_FILEPATH = "FILEPATH"
TOKEN_TYPE_WHITESPACE = "WHITESPACE"
TOKEN_TYPE_IPV4_ADDRESS = "IPV4_ADDRESS"
TOKEN_TYPE_IPV6_ADDRESS = "IPV6_ADDRESS"
TOKEN_TYPE_TIMESTAMP = "TIMESTAMP"
TOKEN_TYPE_CAPITALIZED_ALPHABET_WITH_UNDERSCORES = "CAPITALIZED_ALPHABET_WITH_UNDERSCORES"
TOKEN_TYPE_LOWERCASE_ALPHABET_WITH_UNDERSCORES = "LOWERCASE_ALPHABET_WITH_UNDERSCORES"
TOKEN_TYPE_BOOLEAN = "BOOLEAN"
TOKEN_TYPE_TPU_TOPOLOGY = "TPU_TOPOLOGY"
TOKEN_TYPE_TIME_UNIT = "TIME_UNIT"
TOKEN_TYPE_HTTP_URL = "HTTP_URL"
TOKEN_TYPE_HOST_NAME = "HOST_NAME"
TOKEN_TYPE_PCIE_ADDRESS = "PCIE_ADDRESS"

# Regexes
WORD_REGEX = re.compile(r"^\w+")
ALPHABET_REGEX = re.compile(r"^[A-Za-z]+$")
NUMBER_REGEX = re.compile(r"^(-)?\d+(\.\d+)?$")
NUMBER_BEFORE_ALPHABET_REGEX = re.compile(r"^\d+[A-Za-z]+[A-Za-z0-9]*$")
NUMBER_AFTER_ALPHABET_WITH_UNDERSCORES_REGEX = re.compile(r"^[A-Za-z]+_\d+$")
NUMBER_AFTER_ALPHABET_WITH_MORE_ALPHABETS_OR_NUMBERS_REGEX = re.compile(
    r"^[A-Za-z]+\d+[A-Za-z0-9]*$"
)
HEX_NUMBER_REGEX = re.compile(r"^0x[0-9a-fA-F]+$")
LINUX_FILEPATH_REGEX = re.compile(r"^(/(/?[\w.=_-]+))+$")
WINDOWS_FILEPATH_REGEX = re.compile(
    r"^[a-zA-Z]:\\([\w\-\_\.]+\\)*[\w\-\_\.]+$"
)
IPV4_ADDRESS_REGEX = re.compile(
    r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$"
)
IPV6_ADDRESS_REGEX = re.compile(
    r"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|"
    r"([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}"
    r"(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"
    r"([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}"
    r"(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:"
    r"((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|"
    r"::(ffff(:0{1,4}){0,1}:){0,1}\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"([0-9a-fA-F]{1,4}:){1,4}:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))(:\d+)?$"
)
FULL_TIMESTAMP_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)
RFC3339_TIMESTAMP_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
DATE_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})$")
TIME_REGEX = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d{3})?$")
TIME_UNIT_REGEX = re.compile(r"^\d+(\.\d+)?(ns|us|ms|s|m|h)$")
CAPITALIZED_ALPHABET_WITH_UNDERSCORES_REGEX = re.compile(
    r"^[A-Z]+(?:_[A-Z]+)+$"
)
LOWERCASE_ALPHABET_WITH_UNDERSCORES_REGEX = re.compile(r"^[a-z]+(?:_[a-z]+)+$")
BOOLEAN_REGEX = re.compile(r"^(true|false|True|False)$")
TPU_TOPOLOGY_REGEX = re.compile(r"^\d+x\d+x\d+_TC\d+$")
HTTP_URL_REGEX1 = re.compile(
    r"""(https?://([^\s<>]|-)+?)[\\\(\)\.:;,"']*(?:$|[\s<>,])"""
)
HTTP_URL_REGEX2 = re.compile(
    r"(http|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))"
    r"([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
)
PCIE_ADDRESS_REGEX = re.compile(
    r"^([0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}(\.[0-9a-fA-F]+)?$"
)
# Matches standard Fully Qualified Domain Name (FQDN) hostnames
HOSTNAME_REGEX = re.compile(r"^[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\.[a-zA-Z]{2,}$")

# Matches standard worker tasks hostnames (e.g. worker-host-1, worker-host-2)
HOST_REGEX = re.compile(r"^[a-z]{4,7}[0-9]{1,3}$")

# Matches composite cell/location hostnames from cluster index table
COMPOSITE_HOST_REGEX = re.compile(r"^[a-z]{5,8}-[a-z0-9]{2,5}$")

# Matches dynamic GKE VM nodes (e.g. gke-tpu-node-pool-xxxx)
GKE_NODE_REGEX = re.compile(r"^gke-[a-z0-9-]+-[a-f0-9]{8}-[a-z0-9]{4}$")

# Matches Kubernetes/GKE Pod names (e.g. j-930219-v-0-s-1-2-x9p76)
POD_NAME_REGEX = re.compile(r"^[a-z0-9-]{5,80}-[0-9]+-[a-z0-9]{5}$")

# Constants
JSON_SYNTAX_CHARACTERS = ['"', "'", "{", "}", "[", "]", ","]
TOKEN_SEPARATOR = ""  # Empty string separator to join tokens
SPAN_SEPARATOR = "\n"

# Default Masked Types (from loganalyzer.py)
DEFAULT_MASKED_TYPES = [
    TOKEN_TYPE_NUMBER,
    TOKEN_TYPE_NUMBER_ALPHABET,
    TOKEN_TYPE_FILEPATH,
    TOKEN_TYPE_IPV4_ADDRESS,
    TOKEN_TYPE_IPV6_ADDRESS,
    TOKEN_TYPE_TIMESTAMP,
    TOKEN_TYPE_CAPITALIZED_ALPHABET_WITH_UNDERSCORES,
    TOKEN_TYPE_LOWERCASE_ALPHABET_WITH_UNDERSCORES,
    TOKEN_TYPE_BOOLEAN,
    TOKEN_TYPE_TPU_TOPOLOGY,
    TOKEN_TYPE_TIME_UNIT,
    TOKEN_TYPE_HTTP_URL,
    TOKEN_TYPE_HOST_NAME,
    TOKEN_TYPE_PCIE_ADDRESS,
]

# Mask Patterns
MASK_PATTERNS = {
    TOKEN_TYPE_WORD: "[WORD_%d]",
    TOKEN_TYPE_ALPHABET: "[ALPHABET_%d]",
    TOKEN_TYPE_NUMBER: "[NUMBER_%d]",
    TOKEN_TYPE_NUMBER_ALPHABET: "[NUMBER_ALPHABET_%d]",
    TOKEN_TYPE_FILEPATH: "[FILEPATH_%d]",
    TOKEN_TYPE_UNKNOWN: "[UNKNOWN_%d]",
    TOKEN_TYPE_IPV4_ADDRESS: "[IPV4_ADDRESS_%d]",
    TOKEN_TYPE_IPV6_ADDRESS: "[IPV6_ADDRESS_%d]",
    TOKEN_TYPE_TIMESTAMP: "[TIMESTAMP_%d]",
    TOKEN_TYPE_TIME_UNIT: "[TIME_UNIT_%d]",
    TOKEN_TYPE_CAPITALIZED_ALPHABET_WITH_UNDERSCORES: (
        "[CAPITALIZED_ALPHABET_WITH_UNDERSCORES_%d]"
    ),
    TOKEN_TYPE_LOWERCASE_ALPHABET_WITH_UNDERSCORES: (
        "[LOWERCASE_ALPHABET_WITH_UNDERSCORES_%d]"
    ),
    TOKEN_TYPE_BOOLEAN: "[BOOLEAN_%d]",
    TOKEN_TYPE_TPU_TOPOLOGY: "[TPU_TOPOLOGY_%d]",
    TOKEN_TYPE_HTTP_URL: "[HTTP_URL_%d]",
    TOKEN_TYPE_HOST_NAME: "[HOST_NAME_%d]",
    TOKEN_TYPE_PCIE_ADDRESS: "[PCIE_ADDRESS_%d]",
}

SPLIT_WHITESPACE_REGEX = re.compile(r"(\s+)")
TOKEN_BOUNDARY_REGEX = re.compile(r"^([^a-zA-Z0-9/]*)(.*?)([^a-zA-Z0-9]*)$")


@functools.lru_cache(maxsize=16384)
def _classify_core_token(token_string: str) -> str:
  """Classifies a token string into a token type based on regex matching."""
  if (
      token_string.startswith("gke-")
      or HOSTNAME_REGEX.search(token_string)
      or GKE_NODE_REGEX.search(token_string)
      or HOST_REGEX.search(token_string)
      or COMPOSITE_HOST_REGEX.search(token_string)
      or POD_NAME_REGEX.search(token_string)
  ):
    return TOKEN_TYPE_HOST_NAME
  if TIME_UNIT_REGEX.search(token_string):
    return TOKEN_TYPE_TIME_UNIT
  if LINUX_FILEPATH_REGEX.search(token_string) or WINDOWS_FILEPATH_REGEX.search(
      token_string
  ):
    return TOKEN_TYPE_FILEPATH
  if HTTP_URL_REGEX1.search(token_string) or HTTP_URL_REGEX2.search(
      token_string
  ):
    return TOKEN_TYPE_HTTP_URL
  if IPV4_ADDRESS_REGEX.search(token_string):
    return TOKEN_TYPE_IPV4_ADDRESS
  if IPV6_ADDRESS_REGEX.search(token_string):
    return TOKEN_TYPE_IPV6_ADDRESS
  if PCIE_ADDRESS_REGEX.search(token_string):
    return TOKEN_TYPE_PCIE_ADDRESS
  if (
      FULL_TIMESTAMP_REGEX.search(token_string)
      or DATE_REGEX.search(token_string)
      or TIME_REGEX.search(token_string)
      or RFC3339_TIMESTAMP_REGEX.search(token_string)
  ):
    return TOKEN_TYPE_TIMESTAMP
  if TPU_TOPOLOGY_REGEX.search(token_string):
    return TOKEN_TYPE_TPU_TOPOLOGY
  if CAPITALIZED_ALPHABET_WITH_UNDERSCORES_REGEX.search(token_string):
    return TOKEN_TYPE_CAPITALIZED_ALPHABET_WITH_UNDERSCORES
  if LOWERCASE_ALPHABET_WITH_UNDERSCORES_REGEX.search(token_string):
    return TOKEN_TYPE_LOWERCASE_ALPHABET_WITH_UNDERSCORES
  if BOOLEAN_REGEX.search(token_string):
    return TOKEN_TYPE_BOOLEAN
  if HEX_NUMBER_REGEX.search(token_string) or NUMBER_REGEX.search(token_string):
    return TOKEN_TYPE_NUMBER
  if (
      NUMBER_BEFORE_ALPHABET_REGEX.search(token_string)
      or NUMBER_AFTER_ALPHABET_WITH_MORE_ALPHABETS_OR_NUMBERS_REGEX.search(
          token_string
      )
      or NUMBER_AFTER_ALPHABET_WITH_UNDERSCORES_REGEX.search(token_string)
  ):
    return TOKEN_TYPE_NUMBER_ALPHABET
  if WORD_REGEX.search(token_string):
    return TOKEN_TYPE_WORD

  return TOKEN_TYPE_UNKNOWN


def get_token_type(
    token_string: str, known_hostnames: Optional[List[str]] = None
) -> str:
  """Determines the token type, checking known hostnames first."""
  if known_hostnames and token_string in known_hostnames:
    return TOKEN_TYPE_HOST_NAME
  return _classify_core_token(token_string)


def sanitize_json_syntax(s: str) -> str:
  """Replaces JSON syntax characters in string with spaces."""
  for char in JSON_SYNTAX_CHARACTERS:
    s = s.replace(char, " ")
  return s


def tokenize_line(
    line: str, known_hostnames: Optional[List[str]] = None
) -> List[Dict[str, str]]:
  """Tokenizes a line of text into structured tokens with boundary affixes."""
  if known_hostnames is None:
    known_hostnames = []

  parts = SPLIT_WHITESPACE_REGEX.split(line)
  tokens: List[Dict[str, str]] = []

  for part in parts:
    if not part:
      continue
    if part.isspace():
      tokens.append({"type": TOKEN_TYPE_WHITESPACE, "value": part})
    else:
      # Extract non-alphanumeric boundary prefixes and suffixes (keeping
      # dot/slash/dash in core)
      match = TOKEN_BOUNDARY_REGEX.match(part)
      if not match:
        tokens.append({
            "type": TOKEN_TYPE_WORD,
            "value": part,
            "prefix": "",
            "suffix": "",
        })
        continue

      prefix = match.group(1)
      core = match.group(2)
      suffix = match.group(3)

      if not core:
        tokens.append(
            {"type": TOKEN_TYPE_WORD, "value": part, "prefix": "", "suffix": ""}
        )
        continue

      # Classify the core token
      token_type = get_token_type(core, known_hostnames)

      tokens.append({
          "type": token_type,
          "value": core,
          "prefix": prefix,
          "suffix": suffix,
      })

  return tokens


@functools.lru_cache(maxsize=16384)
def _get_mask_substitution(token_type: str, count: int) -> str:
  """Generates a mask placeholder substitution string for a token type."""
  pattern = MASK_PATTERNS.get(token_type, "[TYPE_%d]")
  return pattern % count


def mask_segment(
    spans: List[List[Dict[str, str]]],
    mask_token_types: List[str],
    mask_counters: Optional[Dict[str, int]] = None,
) -> Tuple[str, Dict[str, Dict[str, str]]]:
  """Masks specified token types in tokenized spans, returning substitutions."""
  if mask_counters is None:
    mask_counters = {}
  masked_spans: List[str] = []
  masked_tokens_map: Dict[str, Dict[str, str]] = {}
  mask_token_types_set = (
      set(mask_token_types)
      if not isinstance(mask_token_types, set)
      else mask_token_types
  )

  for span in spans:
    masked_tokens = []
    for token in span:
      token_type = token.get("type", "")
      token_value = token.get("value", "")
      prefix = token.get("prefix", "")
      suffix = token.get("suffix", "")

      if token_type == TOKEN_TYPE_WHITESPACE:
        masked_tokens.append(token_value)
        continue

      if token_type in mask_token_types_set:
        count = mask_counters.get(token_type, 0) + 1
        mask_counters[token_type] = count
        masked_core = _get_mask_substitution(token_type, count)
        masked_tokens.append(f"{prefix}{masked_core}{suffix}")
        masked_tokens_map[masked_core] = {
            "type": token_type,
            "value": token_value,
        }
      else:
        masked_tokens.append(f"{prefix}{token_value}{suffix}")

    masked_spans.append("".join(masked_tokens))

  return SPAN_SEPARATOR.join(masked_spans), masked_tokens_map


def main() -> None:
  """Parses arguments and runs log segment masking."""
  parser = argparse.ArgumentParser(description="Mask raw log strings.")
  parser.add_argument(
      "infile",
      nargs="?",
      type=str,
      default=None,
      help="Input file (default: stdin)",
  )
  parser.add_argument(
      "--known-hostnames", help="Comma-separated list of known hostnames."
  )
  parser.add_argument(
      "--format",
      choices=["text", "json", "auto"],
      default="auto",
      help="Input format (default: auto)",
  )
  parser.add_argument(
      "--deduplicate",
      action="store_true",
      help="Deduplicate masked segments by content",
  )
  parser.add_argument(
      "--output-format",
      choices=["text", "json"],
      default="json",
      help="Output format (default: json)",
  )

  args = parser.parse_args()

  mask_types = DEFAULT_MASKED_TYPES
  known_hostnames = []
  if args.known_hostnames:
    known_hostnames = [h.strip() for h in args.known_hostnames.split(",")]

  if args.infile and args.infile != "-":
    with open(args.infile, "r", encoding="utf-8") as f:
      content = f.read()
  else:
    content = sys.stdin.read()

  segments: List[Any] = []
  input_format = args.format
  if input_format == "auto":
    try:
      segments = json.loads(content)
      if isinstance(segments, list):
        input_format = "json"
      else:
        input_format = "text"
    except json.JSONDecodeError:
      input_format = "text"

  if input_format == "json":
    if args.format != "auto":
      try:
        segments = json.loads(content)
      except json.JSONDecodeError as e:
        print(f"Error parsing JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(segments, list):
      print("JSON input must be a list of strings.", file=sys.stderr)
      sys.exit(1)
    for s in segments:
      if not isinstance(s, str):
        print("JSON input elements must be strings.", file=sys.stderr)
        sys.exit(1)
  else:
    segments = [content]

  results = []
  seen_masked = set()

  for segment in segments:
    spans = []
    for line in segment.splitlines():
      tokens = tokenize_line(line, known_hostnames)
      if tokens:
        spans.append(tokens)
    if spans:
      masked_str, tokens_map = mask_segment(spans, mask_types)

      if args.deduplicate:
        template = re.sub(r"_\d+\]", "]", masked_str)
        if template in seen_masked:
          continue
        seen_masked.add(template)

      results.append(
          {"masked_string": masked_str, "masked_tokens_map": tokens_map}
      )
    else:
      if not args.deduplicate or "" not in seen_masked:
        if args.deduplicate:
          seen_masked.add("")
        results.append({"masked_string": "", "masked_tokens_map": {}})

  if args.output_format == "json":
    print(json.dumps(results, indent=2))
  else:
    for r in results:
      print(r["masked_string"])

if __name__ == "__main__":
  main()

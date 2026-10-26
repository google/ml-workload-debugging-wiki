#!/usr/bin/env python3
"""Unmasks log files using mappings generated during masking."""

import argparse
import json
import os
import sys


def main():
  parser = argparse.ArgumentParser(
      description="Unmask masked log files using mappings.")
  parser.add_argument("--input_file", required=True,
                      help="Path to the masked input file")
  parser.add_argument("--mappings_file", required=True,
                      help="Path to the mappings JSON file")
  parser.add_argument("--output_file", required=True,
                      help="Path to write the unmasked output file")

  args = parser.parse_args()

  if not os.path.exists(args.input_file):
    print(f"Error: Input file {args.input_file} does not exist.",
          file=sys.stderr)
    sys.exit(1)

  if not os.path.exists(args.mappings_file):
    print(f"Error: Mappings file {args.mappings_file} does not exist.",
          file=sys.stderr)
    sys.exit(1)

  with open(args.mappings_file, "r") as f:
    try:
      mappings = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
      print(f"Error loading mappings JSON: {e}", file=sys.stderr)
      sys.exit(1)

  with open(args.input_file, "r") as f:
    content = f.read()

  # Sort placeholder keys by length descending to prevent partial replacements
  # e.g. [HOST_NAME_12] before [HOST_NAME_1]
  sorted_tokens = sorted(mappings.keys(), key=len, reverse=True)

  unmasked_content = content
  for token in sorted_tokens:
    info = mappings[token]
    orig_value = info.get("value", "")
    if orig_value:
      unmasked_content = unmasked_content.replace(token, orig_value)

  with open(args.output_file, "w") as f:
    f.write(unmasked_content)

  print(f"Successfully unmasked {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
  main()

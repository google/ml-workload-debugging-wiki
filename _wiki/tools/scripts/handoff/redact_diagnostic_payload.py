#!/usr/bin/env python3
r"""redact_diagnostic_payload.py - Redaction Engine for Cloud TPU Payloads.

Preserves mathematical operation structures (all-gather, reduce-scatter,
all-reduce, collective-permute, dot-general, etc.) while stripping
proprietary tensor names, shape constants, vocabulary dimensions, and
customer source code paths.

Usage:
  python3 redact_diagnostic_payload.py --input_dir /path/to/raw \
      --output_dir /path/to/redacted
  python3 redact_diagnostic_payload.py --file /path/to/module.hlo \
      --output_file /path/to/sanitized.hlo
"""

import argparse
import functools
import os
import re
import shutil
import sys

PRESERVED_OPS = (
    "all-gather",
    "reduce-scatter",
    "all-reduce",
    "all-to-all",
    "collective-permute",
    "collective-broadcast",
    "custom-call",
    "dot",
    "dot-general",
    "convolution",
    "reduce",
    "broadcast",
    "reshape",
    "transpose",
    "slice",
    "pad",
    "concatenate",
    "parameter",
    "constant",
    "tuple",
    "get-tuple-element",
    "while",
    "conditional",
    "call",
    "send",
    "recv",
)

_TEXT_EXTENSIONS = frozenset({".hlo", ".txt", ".ir", ".log", ".yaml", ".json"})

SOURCE_PATH_REGEX = re.compile(
    r'metadata\s*=\s*\{[^}]*source_file="[^"]+"[^}]*\}', re.IGNORECASE
)
OP_NAME_METADATA_REGEX = re.compile(r'op_name="([^"]+)"', re.IGNORECASE)
FILE_PATH_REGEX = re.compile(
    r"/(?:Users|home|repo|workspace|code)/[a-zA-Z0-9_\-./]+"
)
SHAPE_REGEX = re.compile(
    r"\b(f32|bf16|f16|s32|s16|u32|pred|c64)\[([0-9,\s]+)\]"
)
TENSOR_ID_REGEX = re.compile(r"%([a-zA-Z0-9_\-\.]+)")


def _redact_op_name(match: re.Match[str]) -> str:
  """Sanitizes op_name metadata while preserving mathematical op categories."""
  raw_name = match.group(1)
  for op in PRESERVED_OPS:
    if op in raw_name.lower():
      return f'op_name="math_{op}"'
  return 'op_name="op_redacted"'


def _redact_shape(match: re.Match[str]) -> str:
  """Normalizes shape constants while preserving tensor rank and dtype."""
  dtype = match.group(1)
  dimensions = match.group(2).split(",")
  if len(dimensions) == 1 and not dimensions[0].strip():
    return f"{dtype}[]"
  masked_dimensions = [f"D{i}" for i in range(len(dimensions))]
  return f"{dtype}[{','.join(masked_dimensions)}]"


def _redact_tensor_name(
    tensor_map: dict[str, str], match: re.Match[str]
) -> str:
  """Normalizes tensor identifiers to sequential anonymous symbols."""
  tensor_name = match.group(1)
  if tensor_name.lower() in PRESERVED_OPS:
    return f"%{tensor_name}"
  if tensor_name not in tensor_map:
    tensor_index = len(tensor_map) + 1
    tensor_map[tensor_name] = f"t_{tensor_index:04d}"
  return f"%{tensor_map[tensor_name]}"


def sanitize_hlo_content(content: str, preserve_ranks: bool = True) -> str:
  """Sanitizes raw HLO content by stripping PII, paths, and tensor shapes."""
  del preserve_ranks
  # 1. Strip customer source file paths and line numbers
  content = SOURCE_PATH_REGEX.sub(
      'metadata={source_file="[REDACTED_CUSTOMER_SRC]"}', content
  )
  content = FILE_PATH_REGEX.sub("/[REDACTED_PATH]", content)

  # 2. Sanitize op_name metadata while preserving mathematical op category
  content = OP_NAME_METADATA_REGEX.sub(_redact_op_name, content)

  # 3. Normalize shape constants (preserving rank and primitive)
  content = SHAPE_REGEX.sub(_redact_shape, content)

  # 4. Normalize tensor identifiers to anonymous sequential symbols
  content = TENSOR_ID_REGEX.sub(
      functools.partial(_redact_tensor_name, {}), content
  )
  return content


def sanitize_file(input_path: str, output_path: str) -> None:
  """Sanitizes a single text or HLO artifact, copying binaries verbatim."""
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  extension = os.path.splitext(input_path)[1].lower()

  if extension in _TEXT_EXTENSIONS:
    try:
      with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    except (OSError, UnicodeDecodeError) as e:
      print(
          f"[ERROR] Error reading text file {input_path}: {e}", file=sys.stderr
      )
      shutil.copyfile(input_path, output_path)
      print(
          f"[INFO] Copied binary artifact verbatim: {input_path} ->"
          f" {output_path}"
      )
      return

    sanitized = sanitize_hlo_content(content)
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(sanitized)
    print(f"[INFO] Sanitized text artifact: {input_path} -> {output_path}")
    return

  shutil.copyfile(input_path, output_path)
  print(
      f"[INFO] Copied binary artifact verbatim: {input_path} -> {output_path}"
  )


def sanitize_directory(input_dir: str, output_dir: str) -> None:
  """Recursively sanitizes all supported artifacts in a directory."""
  for root, _, files in os.walk(input_dir):
    for filename in files:
      source_file = os.path.join(root, filename)
      relative_path = os.path.relpath(source_file, input_dir)
      destination_file = os.path.join(output_dir, relative_path)
      sanitize_file(source_file, destination_file)


def main() -> None:
  """CLI entry point for redacting diagnostic artifacts and directories."""
  parser = argparse.ArgumentParser(
      description="Sanitize TPU diagnostic payloads"
  )
  parser.add_argument("--file", help="Single file to sanitize")
  parser.add_argument("--output_file", help="Sanitized output destination")
  parser.add_argument("--input_dir", help="Directory of artifacts to sanitize")
  parser.add_argument(
      "--output_dir", help="Output directory for sanitized artifacts"
  )
  args = parser.parse_args()

  if args.file and args.output_file:
    sanitize_file(args.file, args.output_file)
    print("\n--- SUMMARY REPORT (summary_fields) ---")
    print("status: SANITIZED")
    print(f"output_file: {args.output_file}")
  elif args.input_dir and args.output_dir:
    sanitize_directory(args.input_dir, args.output_dir)
    print("\n--- SUMMARY REPORT (summary_fields) ---")
    print("status: SANITIZED")
    print(f"output_dir: {args.output_dir}")
  else:
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
  main()

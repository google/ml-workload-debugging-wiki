#!/usr/bin/env python3
"""Builds a log graph using log-clustering pipeline's log graph building algorithm."""

import argparse
import json
import re
import sys
from typing import Any, Dict, List

# Regex to identify the whitespace prefix level
whitespace_prefix_regex = re.compile(r"^\s*")


class Node:
  """Represents a node in the hierarchical log graph."""

  level: int
  text: str
  children: List["Node"]

  def __init__(self, level: int, text: str) -> None:
    self.level = level
    self.text = text
    self.children = []

  def to_dict(self) -> Dict[str, Any]:
    """Serializes the Node and its children to a dict."""
    return {
        "level": self.level,
        "text": self.text,
        "children": [child.to_dict() for child in self.children],
    }


def _get_node_level_map(lines: List[str]) -> Dict[str, int]:
  """Sorts unique indentation levels and maps them to an index."""
  distinct_whitespaces = set()
  for line in lines:
    match = whitespace_prefix_regex.match(line)
    whitespaces = match.group(0) if match else ""
    distinct_whitespaces.add(whitespaces)

  sorted_whitespaces = sorted(list(distinct_whitespaces))
  node_levels_map = {ws: idx for idx, ws in enumerate(sorted_whitespaces)}
  return node_levels_map


def _build_graph(lines: List[str]) -> Node:
  """Builds a hierarchical tree of Node objects using a stack."""
  node_levels_map = _get_node_level_map(lines)

  root_node = Node(level=-1, text="")
  stack: List[Node] = [root_node]

  for line in lines:
    if not line or not line.strip():
      # Empty lines reset stack to root level
      while len(stack) > 1:
        stack.pop()
      continue

    match = whitespace_prefix_regex.match(line)
    whitespaces = match.group(0) if match else ""
    current_level = node_levels_map[whitespaces]

    current_node = Node(level=current_level, text=line)

    parent_node = stack[-1]
    if current_level > parent_node.level:
      parent_node.children.append(current_node)
      stack.append(current_node)
    else:
      while len(stack) > 0 and stack[-1].level >= current_level:
        stack.pop()
      parent_node = stack[-1]
      parent_node.children.append(current_node)
      stack.append(current_node)

  return root_node


def _render_text_tree(node: Node, depth: int = 0) -> str:
  """Recursively constructs and returns the indented text layout tree."""
  lines: List[str] = []
  if node.level != -1:
    indent = "  " * (depth - 1)
    lines.append(f"{indent}{node.text.lstrip()}")
  for child in node.children:
    child_str = _render_text_tree(child, depth + 1)
    if child_str:
      lines.append(child_str)
  return "\n".join(lines)


def build_and_render_tree(lines: List[str]) -> List[str]:
  """Converts a list of raw log strings, builds a log graph and renders formatting."""
  root = _build_graph(lines)

  log_messages = []
  for node in root.children:
    log_string = _render_text_tree(node, depth=node.level + 1)
    log_messages.append(log_string)
  return log_messages


def main() -> None:
  """Parses command-line arguments and runs log graph construction."""
  parser = argparse.ArgumentParser(
      description="Build Log Graph from Log messages"
  )
  parser.add_argument(
      "--log-file",
      required=True,
      help="Path to local text file to read logs from directly",
  )
  parser.add_argument(
      "--format",
      default="text",
      help="Output format ('text' or 'json')",
  )

  args = parser.parse_args()

  log_file = args.log_file.strip() if args.log_file else ""
  output_format = args.format.strip().lower() if args.format else "text"
  if output_format not in ("text", "json"):
    output_format = "text"

  log_lines: List[str] = []

  try:
    with open(log_file, "r", encoding="utf-8") as f:
      for line in f:
        line_clean = line.rstrip("\r\n")
        log_lines.append(line_clean)
  except OSError as e:
    print(f"Error reading log file {log_file}: {e}", file=sys.stderr)
    sys.exit(1)

  if not log_lines:
    print("No log lines found to build graph.", file=sys.stderr)
    sys.exit(0)

  results = build_and_render_tree(log_lines)

  if output_format == "json":
    print(json.dumps(results, indent=2))
  else:
    for line in results:
      print(line)


if __name__ == "__main__":
  main()

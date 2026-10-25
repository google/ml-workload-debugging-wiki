---
title: Mask Segment Tool
type: tool
internal_only: false
tags: [analysis, methodology]
created: 2026-07-05
updated: 2026-07-05
---

# Mask Segment Tool

## Overview

The Mask Segment Tool is a utility designed to convert raw log strings into
`MaskedSegment.MaskedString` equivalent format. It re-implements tokenization
and masking logic from core log-clustering pipeline libraries (such as `log-clustering pipeline`,
`tokenizer.go`, and `token.go`) to simulate how logs are processed and
obfuscated for analysis. This tool is useful for verifying how specific log
messages will be masked or for preparing data in a format compatible with
log-clustering pipeline's downstream pipelines.

## Location

`tools/scripts/mask_segment_tool.py`

## Usage & Commands

### Basic Usage

The tool reads from `stdin` or an input file and outputs the masked text to
`stdout`.

```bash
python3 tools/scripts/mask_segment_tool.py < input.log
```

or via pipe:

```bash
echo "Error in file /path/to/log.txt at 2026-01-01T12:00:00Z" | python3 tools/scripts/mask_segment_tool.py
```

### Options

*   `--known-hostnames`: Comma-separated list of known hostnames to identify and
    mask as `HOST_NAME`.
*   `--format`: Input format (`text`, `json`, `auto`). Defaults to `auto`. If
    `json`, the input should be a JSON array of strings.
*   `--output-format`: Output format (`text`, `json`). Defaults to `json`.

## Output Analysis

### JSON Format (Default)

The output is a JSON array of objects. Each object contains:

-   `masked_string`: The masked text.
-   `masked_tokens_map`: A mapping from placeholder (e.g., `[NUMBER_1]`) to
    original token information `{"type": "...", "value": "..."}`.

For example, given the input: `User 123 failed to connect to [REDACTED_IP]`

The default JSON output might look like:

```json
[
  {
    "masked_string": "User [NUMBER_1] failed to connect to [IPV4_ADDRESS_1]",
    "masked_tokens_map": {
      "[NUMBER_1]": {
        "type": "NUMBER",
        "value": "123"
      },
      "[IPV4_ADDRESS_1]": {
        "type": "IPV4_ADDRESS",
        "value": "[REDACTED_IP]"
      }
    }
  }
]
```

### Text Format

Output consists of ONLY the plain masked string (dropping the map) to maintain
Unix pipe compatibility.

## Troubleshooting Scenarios

*   **Token Misidentification:** If a token is not masked as expected, check the
    regex definitions in the script or verify if the token type is included in
    the `DEFAULT_MASKED_TYPES` list in the script.
*   **JSON Parsing Errors:** Ensure that JSON input is a valid array of strings
    if using `--format json`.

## Connections

-   **Actions**:
    -   [[mask-segment]]
    -   [[build-and-mask-log-graph]]
    -   [[unmask-logs]]

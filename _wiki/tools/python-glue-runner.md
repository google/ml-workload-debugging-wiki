---
title: Python Glue Runner
type: tool
internal_only: false
tags: [triage, methodology]
created: 2026-07-20
updated: 2026-07-20
---

# Python Glue Runner

## Overview

The Python Glue Runner (`python_glue_runner.py`) is an execution wrapper
designed to dynamically load wiki action JSON schemas, parse their parameters,
interpolate variables, and run command-line sequences. It serves as a glue layer
to execute actions inside the wiki environment using standard Python
subprocesses.

## Location

`tools/scripts/python_glue_runner.py`

## Usage & Commands

### Basic Usage

The glue runner takes an action name (e.g. `triage-workload-signals`) and a
parameters JSON string:

```bash
python3 tools/scripts/python_glue_runner.py <action_name> '<parameters_json>'
```

For example:

```bash
python3 tools/scripts/python_glue_runner.py triage-workload-signals '{"workload_prefix": "my-workload", "project": "my-project"}'
```

### Interpolation Behavior

The script reads the action's `run_command` string from
`actions/<action_name>.json`. It finds all variable placeholders matching
`${variable}` and replaces them with values supplied in the parameters JSON
string before executing the command in a bash shell.

## Troubleshooting / Common Issues

If variable interpolation fails, make sure that the parameter keys in the input
JSON string match the placeholders defined in the Action Schema.

## Connections

-   **Actions**:
    -   [[triage-workload-signals]]

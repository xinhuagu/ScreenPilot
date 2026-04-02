#!/bin/bash
# PostToolUse hook: auto-run ruff format + check on .py files
f=$(jq -r '.tool_input.file_path')
if [[ "$f" == *.py ]] && [[ -f "$f" ]]; then
  ruff format "$f" 2>/dev/null
  ruff check --fix "$f" 2>/dev/null
fi
exit 0

#!/usr/bin/env bash
# Hook: PreToolUse (Edit|Write)
# Blocks edits to protected files. Exit 2 = block the action.

PROTECTED_PATTERNS=(
  ".env"
  ".env.*"
  "*.pem"
  "*.key"
  "*.p12"
  "*.pfx"
  "*.crt"
  "*.cer"
  "*credentials*"
  "*credential*"
  "*secret*"
  "token.json"
  "auth.json"
  "service-account*.json"
)

# jq is required to safely parse tool input
if ! command -v jq &>/dev/null; then
  echo "Warning: jq not found — protect-files.sh cannot verify file path, allowing by default" >&2
  exit 0
fi

# Read the file path from tool input
FILE_PATH=$(jq -r '.tool_input.file_path // .tool_input.command // ""' 2>/dev/null)

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$(basename "$FILE_PATH")" == $pattern ]]; then
    echo "Blocked: refusing to modify protected file '$FILE_PATH'" >&2
    exit 2
  fi
done

exit 0

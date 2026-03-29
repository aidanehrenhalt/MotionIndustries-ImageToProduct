#!/usr/bin/env bash
# Hook: PostToolUse (Write|Edit)
# Detects documentation markdown files with instructions that lack a Quick Start section.
# Outputs additionalContext nudging Claude to run /quickstart — once per file per session.
#
# Cost model:
#   - Hook process spawn: free (local shell), but avoided entirely for non-.md
#     files via the "if" regex in settings.json
#   - additionalContext output: injected into Claude's conversation as input
#     tokens on the next API call — this is the only cost
#   - Once-per-session guard ensures that cost is paid at most once per file

# jq is required to parse tool input
if ! command -v jq &>/dev/null; then
  exit 0
fi

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""')

# ── Scope exclusions (fastest exits first) ──────────────────────────
# Skip non-documentation markdown: changelogs, memory, config, plans, licenses
BASENAME=$(basename "$FILE_PATH")
case "$BASENAME" in
  CHANGELOG*|changelog*|HISTORY*|history*|LICENSE*|license*|MEMORY*|memory*|CLAUDE.md) exit 0 ;;
esac
case "$FILE_PATH" in
  */.claude/skills/*|*/.claude/plans/*|*/.claude/memory/*|*/.claude/agents/*|*/.claude/rules/*) exit 0 ;;
  */node_modules/*|*/.git/*) exit 0 ;;
esac

# ── Once-per-session-per-file guard ─────────────────────────────────
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "default"')
NUDGE_DIR="/tmp/claude-quickstart-nudges"
mkdir -p "$NUDGE_DIR"
FILE_HASH=$(echo "$FILE_PATH" | md5 -q 2>/dev/null || echo "$FILE_PATH" | md5sum 2>/dev/null | cut -d' ' -f1)
NUDGE_FLAG="$NUDGE_DIR/${SESSION_ID}-${FILE_HASH}"
if [ -f "$NUDGE_FLAG" ]; then
  exit 0
fi

# ── Read content ────────────────────────────────────────────────────
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""')
if [ "$TOOL_NAME" = "Write" ]; then
  CONTENT=$(echo "$INPUT" | jq -r '.tool_input.content // ""')
elif [ "$TOOL_NAME" = "Edit" ]; then
  [ -f "$FILE_PATH" ] && CONTENT=$(cat "$FILE_PATH") || exit 0
else
  exit 0
fi

# Skip short content (not real documentation)
if [ ${#CONTENT} -lt 100 ]; then
  exit 0
fi

# ── Already has a Quick Start / Getting Started section ─────────────
if echo "$CONTENT" | grep -qiE '^#+\s+(quick\s*start|getting\s+started)'; then
  exit 0
fi

# ── Detect instructional content ────────────────────────────────────
# Must match at least one pattern to be considered documentation with instructions
echo "$CONTENT" | grep -qE '^[0-9]+\.\s' \
  || echo "$CONTENT" | grep -qiE '##\s*(install|setup|usage|how to|configuration|prerequisites|requirements|steps)' \
  || echo "$CONTENT" | grep -qE '```(bash|sh|shell|zsh)' \
  || exit 0

# ── Fire the nudge (once) ──────────────────────────────────────────
touch "$NUDGE_FLAG"

jq -n --arg name "$BASENAME" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: ($name + " has instructions but no Quick Start. Consider: /quickstart <topic> --write <file>")
  }
}'

exit 0

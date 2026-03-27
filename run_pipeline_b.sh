#!/usr/bin/env bash
# Pipeline B: Demo Manufacturer Site
# Assumes .venv is already set up with requirements installed.
# Usage:
#   ./run_pipeline_b.sh           — skip rebuild, run pipeline, serve
#   ./run_pipeline_b.sh --rebuild — rebuild demo site artifacts first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
OUTPUT_JSON="$REPO_ROOT/src/demo_mfr_site/pipeline_output/json"
SITE_DIR="$REPO_ROOT/src/demo_mfr_site/site"
PORT=8000

cd "$REPO_ROOT"

# --- Step 1: Optional rebuild ---
if [[ "${1:-}" == "--rebuild" ]]; then
    echo "==> Rebuilding demo site artifacts..."
    "$VENV_PYTHON" src/demo_mfr_site/build_demo_site.py
    echo "==> Rebuild complete."
fi

# --- Step 2: Run demo scrape + classify pipeline ---
echo "==> Running demo pipeline (scrape + classify)..."
"$VENV_PYTHON" src/demo_mfr_site/run_demo_pipeline.py
echo "==> Pipeline complete."

# --- Step 3: Verify classification output ---
echo "==> Checking for predicted_class fields in output..."
grep -R "predicted_class" "$OUTPUT_JSON" || echo "(no predicted_class entries found)"

# --- Step 4: Serve demo site ---
echo ""
echo "==> Serving demo site at http://localhost:$PORT/"
echo "    Useful pages:"
echo "      http://localhost:$PORT/"
echo "      http://localhost:$PORT/products/uct305.html"
echo "    Press Ctrl+C to stop."
echo ""
cd "$SITE_DIR"
python3 -m http.server "$PORT"

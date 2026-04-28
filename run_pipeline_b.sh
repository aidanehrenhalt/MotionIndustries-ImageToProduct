#!/usr/bin/env bash
# Pipeline B: Demo Manufacturer Site + React UI
#
# Assumes .venv is already set up with requirements installed and the React app
# has been built at least once:
#   cd Demo_MFR_Review/Demo_MFR_Review/client && npm install && npm run build
#
# Usage:
#   ./run_pipeline_b.sh           — build site, run pipeline, start all services
#   ./run_pipeline_b.sh --no-rebuild — skip build_demo_site.py (faster re-runs)
#
# Services started:
#   :8000  — Static demo site  (AMI Bearings catalog pages)
#   :5050  — API server        (pipeline backend)
#   :3000  — React UI          (Input UI + Output UI + Review History)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
OUTPUT_JSON="$REPO_ROOT/src/demo_mfr_site/pipeline_output/json"
SITE_DIR="$REPO_ROOT/src/demo_mfr_site/site"
REACT_BUILD="$REPO_ROOT/Demo_MFR_Review/Demo_MFR_Review/client/build"
DEMO_CSV="$REPO_ROOT/src/web_scraping/Demo_Site_Products.csv"
DEMO_JOB_ID="demo-pipeline"
SITE_PORT=8000
API_PORT=5050
REACT_PORT=3000

cd "$REPO_ROOT"

# ── Helpers ────────────────────────────────────────────────────────────────────

kill_on_exit() {
    trap 'echo ""; echo "==> Shutting down..."; kill $(jobs -p) 2>/dev/null; wait' EXIT INT TERM
}

wait_for_port() {
    local port="$1" label="$2" timeout=20 elapsed=0
    while ! nc -z 127.0.0.1 "$port" 2>/dev/null; do
        sleep 0.5
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$((timeout * 2))" ]; then
            echo "WARN: $label did not respond on :$port within ${timeout}s"
            return 0
        fi
    done
    echo "    $label ready on :$port"
}

# ── Step 1: Build demo site artifacts ─────────────────────────────────────────

if [[ "${1:-}" != "--no-rebuild" ]]; then
    echo "==> Building demo site artifacts..."
    "$VENV_PYTHON" src/demo_mfr_site/build_demo_site.py
    echo "==> Site artifacts ready."
fi

# ── Step 2: Run demo scrape + classify pipeline ────────────────────────────────

echo "==> Running demo pipeline (scrape → classify → rank)..."
"$VENV_PYTHON" src/demo_mfr_site/run_demo_pipeline.py
echo "==> Pipeline complete."

echo "==> Checking for predicted_class fields in output..."
grep -R "predicted_class" "$OUTPUT_JSON" || echo "    (no predicted_class entries found)"

# ── Step 3: Start static demo site ────────────────────────────────────────────

echo "==> Starting static demo site on :$SITE_PORT (background)..."
(cd "$SITE_DIR" && python3 -m http.server "$SITE_PORT" --bind 127.0.0.1 2>/dev/null) &
wait_for_port "$SITE_PORT" "Demo site"

# ── Step 4: Start API server ───────────────────────────────────────────────────

echo "==> Starting API server on :$API_PORT (background)..."
API_PORT="$API_PORT" "$VENV_PYTHON" src/api/server.py &
wait_for_port "$API_PORT" "API server"

# ── Step 5: Serve React UI ─────────────────────────────────────────────────────

if [ ! -d "$REACT_BUILD" ]; then
    echo ""
    echo "WARN: React build not found at $REACT_BUILD"
    echo "      Run the following once to build it:"
    echo "        cd Demo_MFR_Review/Demo_MFR_Review/client"
    echo "        npm install && npm run build"
    echo ""
else
    echo "==> Serving React UI on :$REACT_PORT (background)..."
    (cd "$REACT_BUILD" && python3 -m http.server "$REACT_PORT" --bind 127.0.0.1 2>/dev/null) &
    wait_for_port "$REACT_PORT" "React UI"
fi

# ── Demo instructions ──────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " DEMO READY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " Static demo site (AMI Bearings catalog):"
echo "   http://localhost:$SITE_PORT/"
echo "   http://localhost:$SITE_PORT/products/uct305.html"
echo "   http://localhost:$SITE_PORT/review.html   ← static review page"
echo ""
echo " React UI (full pipeline interface):"
echo "   http://localhost:$REACT_PORT/"
echo ""
echo " ── Demo walkthrough ──────────────────────────────────────────"
echo ""
echo " 1. INPUT UI  →  Upload this CSV to see the 4 demo products:"
echo "    $DEMO_CSV"
echo ""
echo " 2. OUTPUT UI →  Paste this URL to load the review queue:"
echo "    http://localhost:$API_PORT/api/review-queue/$DEMO_JOB_ID"
echo ""
echo " 3. Review each product's ranked candidate images and approve,"
echo "    reject, or skip."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Press Ctrl+C to stop all services."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

kill_on_exit
wait

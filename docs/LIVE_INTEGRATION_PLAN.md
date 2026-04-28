# Live Integration Plan: Image-to-Product Pipeline + React Frontend

**Date:** 2026-04-08  
**Branch:** Demo_MFR_Site  
**Status:** In progress — backend API server added, frontend updated

---

## 1. Current-State Architecture Summary

### Pipeline Stages

```
[CSV / Elasticsearch]
        │
        ▼
 web_scraper.py           ← Scrapes Wikimedia Commons + OpenVerse APIs
        │                   Writes JSON records to output/json/
        │                   Downloads images to output/images/
        │                   Optionally indexes to Elasticsearch + MinIO
        ▼
 classify_json_images.py  ← CNN (8-class, 500×500) reads each JSON record
        │                   Loads image bytes (local or MinIO)
        │                   Writes predicted_class + classifier_confidence back to JSON
        │                   Optionally syncs to Elasticsearch
        ▼
 image_search_ranker.py   ← Fuses CNN confidence + text similarity + heuristics
        │                   final_score = 0.30×ai_conf + 0.20×class_match
        │                   + 0.30×text_sim + 0.20×prelim_score
        │                   Writes final_rank to JSON + exports rankings.csv
        ▼
 review_queue.json        ← Structured output for the React review UI
        │
        ▼
 React Review UI          ← Loads review_queue.json, shows product + ranked images
                            User approves / rejects / skips (session-only history)
```

### Demo Pipeline (Self-Contained)

`src/demo_mfr_site/run_demo_pipeline.py` is a fully self-contained variant:
- Reads from hardcoded `site/assets/data/products.json` (4 AMI Bearings products)
- Starts an ephemeral HTTP server on 127.0.0.1 to serve the demo HTML pages
- Scrapes those local pages, classifies, ranks, and exports `site/assets/data/review_queue.json`
- The React app's "Output UI" tab can load this file directly from `assets/data/review_queue.json`

### React Frontend (3 tabs)

| Tab | Name | Current behaviour |
|---|---|---|
| Input | Input UI | Upload .xlsx/.xls/.csv/.json; preview table; no backend connection |
| Review | Output UI | Load review_queue.json by URL; Approve/Reject/Skip each product |
| History | Review History | Session-only decision log |

### Storage

| Layer | Dev | Prod target |
|---|---|---|
| Image binaries | Local `output/images/` or MinIO (docker) | GCP Cloud Storage |
| Metadata | Elasticsearch (docker) or JSON files | Managed Elasticsearch / GCP |
| Pipeline state | In-memory (no persistence across restarts) | Persistent DB / Cloud Run |

### Known Hardcoding

| Location | Hardcoded value | Impact |
|---|---|---|
| `web_scraper.py:43-48` | `OUTPUT_DIR = Path("output")` (relative to cwd) | Must run from project root |
| `classify_json_images.py:547-549` | Default json-dir based on `__file__` | Self-corrects; pass `--json-dir` explicitly |
| `image_search_ranker.py:992-997` | Same pattern | Same — pass `--json-dir` and `--output` |
| `run_demo_pipeline.py:27-35` | `PRODUCT_DATA` = demo products.json | Demo-only |
| `mb2-10.json` image_url | `http://127.0.0.1:46557/...` | Only valid during demo server run |
| `pipelineFolderParser.js:1` | `DEFAULT_REVIEW_QUEUE_URL = "assets/data/review_queue.json"` | Only works for static/demo site |
| `img_classifier.py:64` | `/home/hice1/rlopez76/scratch/motion_dataset` | Training script only — never used in prod |

---

## 2. Gap Analysis for Live Deployment

| Gap | Severity | Current state | Required state |
|---|---|---|---|
| No backend API server | **Critical** | None — pipeline is CLI-only | Flask server accepting uploads and triggering pipeline |
| File upload not connected to pipeline | **Critical** | Browser parses file, holds in React state only | File POSTed to backend, parsed server-side, written as input.csv |
| Review queue served as static file | **Critical** | React fetches `assets/data/review_queue.json` (demo path) | React fetches `http://localhost:5050/api/review-queue/<jobId>` |
| No pipeline progress visibility | High | Run manually, no UI feedback | API polls `/api/pipeline/status/<jobId>` |
| Output images not accessible to browser | High | Images on local filesystem at absolute paths | Need HTTP-accessible URLs or base64 encoding |
| Review decisions not persisted | Medium | Session-only (lost on page reload) | POST decisions to backend or export to file |
| Text Analysis stage incomplete | Medium | `ranker_score` is text similarity (token overlap), not semantic | See INTEGRATION_PLAN.md §1 |
| Concurrent job isolation | Medium | All jobs write to same `output/` dir | Each job needs its own output directory (planned) |
| Column name normalisation | Medium | Scraper expects specific CSV headers | API server normalises on upload |
| No ES/MinIO for images in review UI | Low | `file://` URLs in review_queue.json won't load in browser | Must serve images via HTTP or embed as data URIs |
| Review decisions not fed back to pipeline | Low | Approve/reject stays in browser | Need writeback mechanism for production |

---

## 3. Recommended Backend/Frontend Integration Design

### 3.1 Architecture

```
React App (localhost:3000)
    │
    │  POST /api/upload          multipart file
    │  POST /api/pipeline/run    { jobId, limit? }
    │  GET  /api/pipeline/status/<jobId>  (poll every 2.5s)
    │  GET  /api/review-queue/<jobId>    (after done)
    ▼
Flask API Server (localhost:5050)   ← NEW: src/api/server.py
    │
    │  subprocess: .venv/bin/python src/web_scraping/web_scraper.py --csv --classify
    │  subprocess: .venv/bin/python src/web_scraping/image_search_ranker.py --json-dir --output
    │  function:   _build_review_queue(json_dir, rankings_csv, output_path)
    ▼
Python Pipeline (ROOT)
    ├── output/json/*.json         per-product records
    ├── output/images/**           downloaded images
    ├── output/rankings.csv        ranked candidates
    └── uploads/jobs/<jobId>/
        ├── input.csv              normalised upload
        └── review_queue.json      per-job snapshot for the React UI
```

### 3.2 API Endpoints (Implemented)

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/upload` | multipart `file` field | `{ jobId, fileName, rowCount, headers, missingRequired }` |
| POST | `/api/pipeline/run` | `{ jobId, limit?, es?, minio? }` | `{ jobId, status }` |
| GET | `/api/pipeline/status/<jobId>` | — | `{ status, log[], rowCount, startedAt, finishedAt, error }` |
| GET | `/api/review-queue` | — | Latest review_queue.json |
| GET | `/api/review-queue/<jobId>` | — | Job-specific review_queue.json |
| GET | `/api/jobs` | — | All jobs (debug) |
| GET | `/health` | — | `{ ok, model_exists, scraper_exists }` |

### 3.3 Image URL Problem

The pipeline stores images at local filesystem paths (e.g. `output/images/s10807860/bearing.jpg`).
These are `file://` URLs and won't load in a browser. Three options:

**Option A (Simplest — recommended for dev):** Add a static file route to the Flask server:
```python
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(ROOT / "output" / "images", filename)
```
Then update `_build_review_queue` to generate `/images/...` URLs instead of `file://` paths.

**Option B:** Embed images as base64 data URIs in review_queue.json (increases file size significantly).

**Option C (Production):** Upload images to MinIO/GCP and use HTTPS URLs.

---

## 4. I/O Connector Design

### Upload Ingestion Flow

```
Browser file input (.xlsx/.xls/.csv/.json)
    │
    │  1. validateFileType() — check extension
    │  2. FormData.append("file", file)
    │  3. POST /api/upload
    ▼
Flask /api/upload
    │
    │  4. Validate extension server-side (second gate)
    │  5. _parse_upload():
    │       .csv  → pd.read_csv(dtype=str)
    │       .xlsx → pd.read_excel(dtype=str)
    │       .xls  → pd.read_excel(dtype=str)
    │       .json → pd.DataFrame(data) or data["products"]
    │  6. _normalise_columns() — COLUMN_MAP lookup (case-insensitive)
    │  7. _validate_df() — check for REQUIRED_OUTPUT_COLUMNS
    │  8. df.to_csv(uploads/jobs/<jobId>/input.csv)
    │
    ▼ returns: { jobId, rowCount, headers, missingRequired }
```

### Required Column Names (post-normalisation)

| Pipeline field | Common upload variations accepted |
|---|---|
| `motion_product_id` | "Product ID", "motion product id", "id" |
| `mfr_name` | "Manufacturer Name", "Manufacturer", "Brand" |
| `mfr_part_number` | "Manufacturer Part Number", "Part Number", "MPN" |
| `web_desc` | "Description", "Web Product Description", "Product Description" |
| `item_number` | "Item Number", "Item No" (optional) |
| `enterprise_name` | "Enterprise Name", "Enterprise" (optional) |
| `pgc` | "PGC", "Product Group Code" (optional, used by classifier) |
| `category` | "PGC Description", "Category" (optional) |
| `internal_description` | "Internal Description", "Motion Internal Desc" (optional) |

### Pipeline Handoff

After upload, `input.csv` is passed to `web_scraper.py`:
```bash
.venv/bin/python src/web_scraping/web_scraper.py \
    --csv uploads/jobs/<jobId>/input.csv \
    --classify \
    [--limit N]
```

The `--classify` flag runs CNN classification inline (no separate subprocess needed).
The ranker is then called:
```bash
.venv/bin/python src/web_scraping/image_search_ranker.py \
    --json-dir output/json \
    --output output/rankings.csv
```

Then `_build_review_queue()` assembles `review_queue.json` from the JSON records + rankings CSV.

---

## 5. Step-by-Step Implementation Plan

### Phase 1 — Backend API (DONE)
- [x] Install Flask + flask-cors into `.venv`
- [x] Create `src/api/server.py` with upload, run, status, review-queue endpoints
- [x] Column normalisation map
- [x] Pipeline subprocess orchestration (scrape → rank → export)
- [x] `_build_review_queue()` assembles review_queue.json from JSON + CSV

### Phase 2 — Frontend Upload Integration (DONE)
- [x] Update `InputPage.jsx` to POST file to `/api/upload`
- [x] Add "Run Pipeline" button with limit input
- [x] Poll `/api/pipeline/status/<jobId>` every 2.5s during run
- [x] Show pipeline log output
- [x] Display review queue URL when done

### Phase 3 — Image Serving (NEEDED)
Add to `src/api/server.py`:
```python
from flask import send_from_directory

@app.route("/images/<path:filename>")
def serve_image(filename):
    images_root = ROOT / "output" / "images"
    return send_from_directory(str(images_root), filename)
```
Update `_build_review_queue()` to generate `/images/...` URLs instead of `file://` paths.

### Phase 4 — Review UI Pointing to Live API (NEEDED)
The ReviewPage.jsx currently loads from a user-entered URL. Users must manually paste the
review queue URL from the Input tab. This is workable for testing but should be automated:
- After pipeline completes, auto-navigate to Review tab with the URL pre-filled, OR
- Store `jobId` in App state and pass to ReviewPage so it can auto-load.

### Phase 5 — Concurrent Job Isolation (Future)
Currently all jobs write to `ROOT/output/`. Each job should write to its own directory.
This requires passing an output directory flag to web_scraper.py (not yet supported).
Solution: add `--output-dir` flag to web_scraper.py and update the API server.

### Phase 6 — Persistent Decision Storage (Future)
Review decisions are currently stored in browser session only. Add:
```
POST /api/decisions  { jobId, productId, decision, imageId, feedback }
GET  /api/decisions/<jobId>  → list
```

---

## 6. End-to-End Testing Plan

### Prerequisites

| Service | How to start | Verify |
|---|---|---|
| Python venv | `.venv/bin/python --version` | Shows Python 3.12.x |
| Flask dependencies | `.venv/bin/pip show flask` | Shows flask 3.x |
| Pipeline dependencies | `.venv/bin/pip show torch` | Shows torch (or skip for scrape-only) |
| trained_model.pth | `ls src/Image_Classifier/trained_model.pth` | File exists (3 MB) |
| API server | `.venv/bin/python src/api/server.py` | "Starting API server on http://localhost:5050" |
| React dev server | `cd Demo_MFR_Review/Demo_MFR_Review/client && npm start` | Browser opens localhost:3000 |

Docker (Elasticsearch + MinIO) is **optional** for basic testing. Omit `--es` and `--minio` flags.

### Test T1 — API server health

```bash
curl http://localhost:5050/health
```
**Expected:**
```json
{"ok": true, "model_exists": true, "scraper_exists": true, ...}
```
**Failure:** server not running → start with `.venv/bin/python src/api/server.py`

### Test T2 — File upload endpoint

```bash
curl -X POST http://localhost:5050/api/upload \
  -F "file=@src/web_scraping/test_products_sample.csv"
```
**Expected:**
```json
{
  "jobId": "<uuid>",
  "fileName": "test_products_sample.csv",
  "rowCount": 20,
  "headers": ["motion_product_id", "primary_image_filename", "item_number", ...],
  "missingRequired": []
}
```
**Success criteria:** `rowCount > 0`, `missingRequired` is empty or only has non-critical fields.

### Test T3 — Pipeline trigger + status polling

```bash
# Store job ID
JOB=$(curl -s -X POST http://localhost:5050/api/upload \
  -F "file=@src/web_scraping/test_products_sample.csv" | python3 -c "import sys,json; print(json.load(sys.stdin)['jobId'])")

# Start pipeline with limit 2
curl -s -X POST http://localhost:5050/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d "{\"jobId\":\"$JOB\",\"limit\":2}"

# Poll status
curl -s "http://localhost:5050/api/pipeline/status/$JOB" | python3 -m json.tool
```
**Expected progression:** status: uploaded → running → done  
**Polling:** run the status command every 10s until `"status": "done"`  
**Verify:** `"log"` array shows scraper, ranker, and review_queue build messages

### Test T4 — Verify JSON output (scraper wrote correctly)

```bash
ls output/json/
cat output/json/*.json | python3 -c "import json,sys; [print(r['product']['motion_product_id'], len(r.get('candidate_images',[])), 'images') for r in [json.loads(open(f).read()) for f in __import__('glob').glob('output/json/*.json')]]"
```
**Expected:** One .json file per product, each showing image count (may be 0 if APIs return nothing).

### Test T5 — Verify classification ran

```bash
python3 -c "
import json, glob
for f in glob.glob('output/json/*.json'):
    d = json.load(open(f))
    for img in d.get('candidate_images', []):
        if img.get('downloaded'):
            print(d['product']['motion_product_id'], img.get('predicted_class'), img.get('classifier_confidence'))
"
```
**Expected:** Lines showing product ID, class index (0-7), and confidence (0-1 float).  
**Failure:** `predicted_class` missing → web_scraper.py `--classify` flag not used, or model not found.

### Test T6 — Verify rankings CSV

```bash
cat output/rankings.csv | head -5
```
**Expected:** CSV with columns including `motion_product_id`, `image_rank`, `final_score`, `ai_confidence`.

### Test T7 — Verify review_queue.json

```bash
JOB=<your-job-id>
curl -s "http://localhost:5050/api/review-queue/$JOB" | python3 -c "
import json, sys
q = json.load(sys.stdin)
print('Products:', len(q['products']))
for p in q['products'][:2]:
    print(' ', p['productId'], '-', len(p.get('candidateImages',[])), 'images')
"
```
**Expected:** Shows 1-2 products with their candidate image counts.

### Test T8 — React upload via browser

1. Navigate to http://localhost:3000
2. Click "Input UI" tab
3. Drag or click to upload `src/web_scraping/test_products_sample.csv`
4. Verify: table appears with 20 rows, headers match CSV columns
5. Set limit to `2`
6. Click "Run Pipeline"
7. Verify: "Uploading..." → "Pipeline running..." → progress log appears
8. Verify: "Done — copy this URL..." message appears with review queue URL

### Test T9 — React review UI

1. After T8 completes, copy the review queue URL shown in the Input tab
2. Click "Output UI" tab
3. Paste the URL into the "Review Queue URL" field
4. Click "Load Review Queue"
5. Verify: product card and candidate images appear
6. Click Approve / Reject / Skip
7. Verify: product removed from queue, count decrements
8. Click "Review History" tab
9. Verify: decision appears in history table

### Test T10 — Demo pipeline (no backend needed)

To verify the existing demo works independently:
```bash
.venv/bin/python src/demo_mfr_site/run_demo_pipeline.py
```
Then open the React app and in the "Output UI" tab enter:
```
http://localhost:3000/assets/data/review_queue.json
```
(This only works when React is served with the static site; see Demo Runbook.)

---

## 7. Risks, Blockers, Edge Cases, and Production Concerns

### Blockers (must fix before end-to-end test passes)

| # | Issue | Fix |
|---|---|---|
| B1 | Images stored as `file://` URLs — won't load in browser | Add `/images/<path>` route to Flask server (Phase 3 above) |
| B2 | `pipelineFolderParser.js` resolves relative image URLs against the JSON's base URL — `file://` paths break this | Fix after B1 |
| B3 | `web_scraper.py` returns 0 images for many products (Wikimedia/OpenVerse API rate limits or no results) | Use `--limit 2` initially; ensure internet connectivity |
| B4 | `--classify` in web_scraper.py requires PyTorch; if not installed the whole run fails | Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Wikimedia/OpenVerse API returns 0 results | Medium | Many industrial part numbers are not in these public image databases. The demo pipeline bypasses this by using manufacturer HTML pages directly. |
| Concurrent pipeline runs overwrite each other | Medium | For dev use, run one job at a time. Fix: add `--output-dir` flag to web_scraper.py. |
| PyTorch not installed in venv | Medium | Run: `.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| CORS errors from React → Flask | Low | flask-cors is installed and configured for all `/api/*` routes. |
| API server process killed mid-run | Low | Job state is in-memory; restart loses status. Persist to a JSON file for recovery. |
| React hot-reload loses jobId state | Low | Expected during development. Re-upload the file after a reload. |

### Edge Cases

- **File with no matching images:** Pipeline completes; `review_queue.json` has products with empty `candidateImages`. Review UI shows "No candidate images" and offers a Skip button.
- **CSV with non-standard columns:** `missingRequired` in upload response warns which required columns were not mapped. Pipeline still runs but may produce low-quality results.
- **JSON upload format:** Supported: array of objects `[{...}, ...]` or `{"products": [...]}`. Other shapes return a 422 error.
- **Very large files:** No upload size limit set. Flask default is 16MB. Increase with `app.config["MAX_CONTENT_LENGTH"]` if needed.
- **Re-running pipeline:** Each upload creates a new `jobId`. Old results persist in `uploads/jobs/<oldJobId>/`. Disk space accumulates.

### Production Concerns

- **Authentication:** The API server has no auth. All endpoints are open. Add API key or session auth before any shared deployment.
- **Image storage:** `file://` URLs are a dev-only mechanism. Production must use MinIO or GCP Cloud Storage with HTTP-accessible URLs.
- **Elasticsearch/MinIO:** Docker-based services are dev-only. Production uses GCP Cloud Storage + managed Elasticsearch (see INTEGRATION_PLAN.md §GCP Migration).
- **Pipeline execution time:** Each product scrapes 2 API sources with 1.5s delay between requests. 20 products ≈ 60s. Use `--limit` during testing.
- **Review decision persistence:** Browser session only — lost on page reload. Production requires a backend decisions store.
- **Text analysis stage:** `ranker_score` currently uses token-overlap similarity, not semantic embeddings. See INTEGRATION_PLAN.md §1 for the planned improvement.

---

## 8. Next Steps to Test as a User

This is the complete sequential test procedure a human can execute directly.

### Prerequisites

Open two terminal windows in the project root.

**Terminal 1 — Check dependencies:**
```bash
# Verify Python venv exists
ls .venv/bin/python

# Verify Flask is installed
.venv/bin/python -c "import flask; print('Flask', flask.__version__)"

# Verify pandas is installed
.venv/bin/python -c "import pandas; print('pandas', pandas.__version__)"

# Verify PyTorch is installed (required for classification)
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
# If this fails, install it:
# .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify model file exists
ls -lh src/Image_Classifier/trained_model.pth
```

**If any check fails:**
- Flask missing: `.venv/bin/pip install flask flask-cors`
- Torch missing: `.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

### Step 1 — Start the API server

**Terminal 1:**
```bash
cd /home/aceaid/MotionIndustries-ImageToProduct
.venv/bin/python src/api/server.py
```

**Expected output:**
```
INFO - Pipeline API server starting on http://localhost:5050
INFO - ROOT: /home/aceaid/MotionIndustries-ImageToProduct
INFO - Scraper: .../web_scraper.py (exists=True)
INFO - Model:   .../trained_model.pth (exists=True)
```

**Verify (Terminal 2):**
```bash
curl http://localhost:5050/health
```
Expected: `{"ok": true, "model_exists": true, "scraper_exists": true, ...}`

---

### Step 2 — Start the React development server

**Terminal 2:**
```bash
cd /home/aceaid/MotionIndustries-ImageToProduct/Demo_MFR_Review/Demo_MFR_Review/client
npm install   # only needed first time
npm start
```

**Expected:** Browser opens at http://localhost:3000 showing the "Input UI" tab.

---

### Step 3 — Upload a CSV file through the React UI

1. In the browser at http://localhost:3000, you should see the **Input UI** tab.
2. Click the upload area (or drag a file onto it).
3. Select the test file: `src/web_scraping/test_products_sample.csv`
4. **Expected:** A table appears with 20 rows. You should see columns like  
   `motion_product_id`, `mfr_name`, `mfr_part_number`, `web_desc`, etc.
5. **Failure:** Error message appears → check file format and column names.

---

### Step 4 — Verify parsing and pipeline submission

1. In the "Limit rows" input, type `2` (to limit to 2 products for a fast test run).
2. Click **Run Pipeline**.
3. **Expected:** Button changes to "Uploading..." then "Pipeline running..."
4. A log panel appears below showing stage progress:
   - `Stage 1/3: Scraping images...`
   - `Stage 2/3: Ranking candidates...`
   - `Stage 3/3: Building review_queue.json...`
5. After 30-120 seconds, the message changes to:
   ```
   Pipeline complete. Go to the Output UI tab and paste the review queue URL above.
   ```
6. A URL appears: `http://localhost:5050/api/review-queue/<jobId>`

**Verify DB writes (Terminal 2):**
```bash
# Check JSON files were written
ls output/json/
cat output/json/*.json | python3 -c "
import json, sys, glob
for f in glob.glob('output/json/*.json'):
    d = json.load(open(f))
    imgs = d.get('candidate_images', [])
    print(d['product']['motion_product_id'], '→', len(imgs), 'images')
"
```
**Expected:** 2 JSON files, each showing image count (0 or more).

```bash
# Check classifications were written
python3 -c "
import json, glob
for f in glob.glob('output/json/*.json'):
    d = json.load(open(f))
    for img in d.get('candidate_images', []):
        if img.get('downloaded'):
            print('classified:', img.get('predicted_class'), 'confidence:', img.get('classifier_confidence'))
"
```
**Expected:** Lines with `predicted_class` integer (0-7) and `classifier_confidence` float.  
**If no output:** APIs returned 0 images for these products (common for industrial parts). Try products with more common descriptions.

```bash
# Check rankings CSV
ls output/rankings.csv && head -3 output/rankings.csv
```

---

### Step 5 — Verify image classification and ranking

```bash
# Check that predicted_class is set and confidence > 0
python3 -c "
import json, glob
count = 0
for f in glob.glob('output/json/*.json'):
    d = json.load(open(f))
    for img in d.get('candidate_images', []):
        if img.get('predicted_class') is not None:
            count += 1
            print(f\"  {d['product']['motion_product_id']}: class={img['predicted_class']} conf={img.get('classifier_confidence', 0):.3f}\")
print(f'Total classified: {count}')
"
```

```bash
# Check final_score in rankings
python3 -c "
import csv
with open('output/rankings.csv') as f:
    for row in list(csv.DictReader(f))[:5]:
        print(row.get('motion_product_id'), '| rank', row.get('image_rank'), '| final_score', row.get('final_score'))
"
```

---

### Step 6 — Verify the review UI/UX flow

1. Copy the review queue URL displayed in the Input tab.  
   It looks like: `http://localhost:5050/api/review-queue/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

2. Click the **Output UI** tab in the React app.

3. The "Load Pipeline Review Queue" card appears. Paste your URL into the "Review Queue URL" field.

4. Click **Load Review Queue**.

5. **Expected:**
   - Product name, manufacturer, and part number appear in the left panel
   - Candidate images appear in the right panel (or "No candidate images" if scraper returned 0)
   - Confidence scores appear in the table below
   - Approve / Reject / Skip buttons are active

6. Click **Approve** on a product.  
   **Expected:** Product disappears from the queue; progress bar updates.

7. Click **Review History** tab.  
   **Expected:** Decision entry appears showing product name, decision, and timestamp.

---

### Step 7 — Known gaps and what to do if the flow doesn't complete

| Symptom | Root cause | Fix |
|---|---|---|
| Step 4 hangs indefinitely | Pipeline subprocess failed silently | Check Terminal 1 for error output; run scraper manually: `.venv/bin/python src/web_scraping/web_scraper.py --csv src/web_scraping/test_products_sample.csv --limit 2` |
| "Pipeline error" in log | Scraper exception (common: torch not installed, or output dir issue) | Install torch, then retry |
| 0 images for all products | Wikimedia/OpenVerse APIs return no results for industrial part numbers | Use demo pipeline instead (Step 8 below) |
| Images don't load in review UI | `file://` URLs not accessible from browser | Phase 3 not implemented yet — add Flask `/images/` route |
| "No completed pipeline runs yet" from API | Job ID doesn't match or server restarted | Re-upload file; don't restart server during a test run |
| React 3000 → Flask 5050 CORS error | CORS header missing | Verify flask-cors is installed: `.venv/bin/pip show flask-cors` |
| `review_queue.json` shows products but no images | Scraper found 0 images (common for Wikimedia/OpenVerse with industrial part numbers) | Expected behaviour — review UI handles empty image sets |

---

### Step 8 — Alternative: Test the demo pipeline (guaranteed to work)

If the live scrape returns 0 images (common), use the self-contained demo:

**Terminal 1:**
```bash
.venv/bin/python src/demo_mfr_site/run_demo_pipeline.py
```

**Expected output:**
```json
{
  "products_processed": 4,
  "review_queue_json": "src/demo_mfr_site/site/assets/data/review_queue.json"
}
```

Then in the React "Output UI" tab, click "Load Review Queue" with the default URL  
(`assets/data/review_queue.json`). This only works when React is served from the demo  
site directory. For the npm dev server, serve the static demo site separately:

```bash
cd src/demo_mfr_site/site
python3 -m http.server 8080
```

Then in the React "Output UI" tab, enter:
```
http://localhost:8080/assets/data/review_queue.json
```

This will show 4 AMI Bearings products with classified, ranked images you can Approve/Reject/Skip.

---

### Minimum Changes Required for Full Upload → Review Flow

The following items are **not yet implemented** and will prevent a complete live-upload test:

1. **Image serving route in Flask** — without this, images in the review UI show as broken.
   Add to `src/api/server.py`:
   ```python
   from flask import send_from_directory

   @app.route("/images/<path:filename>")
   def serve_image(filename):
       return send_from_directory(str(ROOT / "output" / "images"), filename)
   ```
   And update `_build_review_queue()` to use `/images/...` URLs instead of `file://...`.

2. **Auto-navigate to Review tab after pipeline completes** — currently the user must manually copy and paste the URL. This is functional but manual.

3. **Wikimedia/OpenVerse returning results** — industrial part numbers often return 0 images from public APIs. The real live pipeline needs manufacturer-site scrapers (`--mfr-scraping`) or a different image source.

The demo pipeline (`run_demo_pipeline.py`) bypasses items 1 and 3 and is the recommended path for UI/UX testing until manufacturer scrapers are integrated.

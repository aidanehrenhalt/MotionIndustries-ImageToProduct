# Pipeline Guide — Motion Industries Image-to-Product

> **This is the canonical start-to-finish guide.** Follow it in order. Every section references actual commands pulled from the repository.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Prerequisites](#2-prerequisites)
3. [Installation and Setup](#3-installation-and-setup)
4. [Configuration Reference](#4-configuration-reference)
5. [Pipeline A — Full Web Scrape Pipeline (CSV Input)](#5-pipeline-a--full-web-scrape-pipeline-csv-input)
6. [Pipeline B — Demo Site Pipeline + React UI (Recommended First Run)](#6-pipeline-b--demo-site-pipeline--react-ui-recommended-first-run)
7. [Pipeline C — Elasticsearch-backed Pipeline](#7-pipeline-c--elasticsearch-backed-pipeline)
8. [React UI — Input and Review Interface](#8-react-ui--input-and-review-interface)
9. [Verifying Each Stage](#9-verifying-each-stage)
10. [Resetting and Cleaning Up](#10-resetting-and-cleaning-up)
11. [Common Failures and Fixes](#11-common-failures-and-fixes)

---

## 1. What This Project Does

This pipeline finds and ranks candidate product images for an industrial parts catalog (Motion Industries). Given a product CSV file with manufacturer names, part numbers, and descriptions, it:

1. **Scrapes** candidate images from Wikimedia Commons, OpenVerse, and (optionally) manufacturer websites.
2. **Classifies** each image with a pre-trained 8-class CNN (bearing, seal, power transmission, etc.).
3. **Ranks** candidates by fusing AI confidence + class match + text similarity + heuristic score into a single `final_score`.
4. **Presents** a review queue in a React UI where a human reviewer can approve, reject, or skip each product's best candidate image.

### System Components

| Component | Directory | Port | Purpose |
|-----------|-----------|------|---------|
| Web Scraper | `src/web_scraping/web_scraper.py` | — | Core orchestrator: scrape → download → produce JSON |
| Image Classifier | `src/Image_Classifier/classify_json_images.py` | — | CNN classification pass on downloaded images |
| Text Ranker | `src/web_scraping/image_search_ranker.py` | — | Text-similarity scoring + final score fusion |
| API Server | `src/api/server.py` | 5050 | Flask backend for file upload + pipeline trigger + review queue |
| React UI | `Demo_MFR_Review/…/client/` | 3000 | Input UI (upload CSV) + Output UI (review images) |
| Demo Site | `src/demo_mfr_site/site/` | 8000 | Static AMI Bearings catalog used as a scrape target |
| Elasticsearch | Docker | 9200 | Product catalog + candidate image metadata storage |
| MinIO | Docker | 9000 | S3-compatible image object storage (optional) |

---

## 2. Prerequisites

### Required
- Python 3.10 or higher
- Docker and Docker Compose
- Node.js 18+ and npm (only needed to rebuild the React app — a pre-built `build/` directory is included)

### Optional
- Chromium (for Playwright-based Tier 2 manufacturer scraping)
- ChromeDriver (for Selenium-based Tier 1 manufacturer scraping)

### Verify your environment

```bash
python3 --version     # must be 3.10+
docker --version
docker compose version
node --version         # only needed for React rebuild
npm --version
```

---

## 3. Installation and Setup

### 3.1 Clone and enter the repo

```bash
git clone <repo-url>
cd MotionIndustries-ImageToProduct
```

### 3.2 Create the virtual environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3.3 Install PyTorch (required for classification)

The `requirements.txt` includes `torch>=2.0` and `torchvision>=0.15`, but the correct variant depends on your hardware. Install **after** the requirements step above:

```bash
# CPU-only (works everywhere, slower):
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (if you have an NVIDIA GPU):
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** NumPy is pinned to `<2.0` in `requirements.txt`. Do not upgrade it; the current PyTorch/torchvision build used here fails classification transforms with NumPy 2.x.

### 3.4 Install Playwright (optional — only for Tier 2 scraping)

```bash
.venv/bin/playwright install chromium
```

### 3.5 Start Docker services

```bash
docker compose up -d
```

Services started:

| Service | URL |
|---------|-----|
| Elasticsearch | http://localhost:9200 |
| Kibana (ES admin UI) | http://localhost:5601 |
| MinIO S3 API | http://localhost:9000 |
| MinIO Console | http://localhost:9001 |

Default MinIO credentials: `minioadmin` / `minioadmin`

### 3.6 Initialize Elasticsearch indices (one-time)

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py
```

This creates the `mi_products` and `mi_candidate_images` indices. Use `--recreate` only if you need to drop and rebuild them:

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

### 3.7 Build the React UI (only needed if the build is stale or missing)

The `build/` directory is already committed. Skip this unless you changed the frontend source.

```bash
cd Demo_MFR_Review/Demo_MFR_Review/client
npm install
npm run build
cd ../../..
```

---

## 4. Configuration Reference

### Environment Variables

| Variable | Default | Used By | Purpose |
|----------|---------|---------|---------|
| `MINIO_ENDPOINT` | `http://localhost:9000` | web_scraper.py, classify_json_images.py | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | web_scraper.py, classify_json_images.py | MinIO credentials |
| `MINIO_SECRET_KEY` | `minioadmin` | web_scraper.py, classify_json_images.py | MinIO credentials |
| `MINIO_BUCKET` | `mi-images` | web_scraper.py, classify_json_images.py | Storage bucket name |
| `API_PORT` | `5050` | server.py | Flask API listen port |
| `UPLOAD_TRACE` | (unset) | server.py | Set to `1` to log upload parsing details |

All MinIO variables default to dev values. Do not use defaults in production.

### Input CSV Column Names

The API server and web scraper accept multiple column name variants and map them automatically. The canonical names are:

| Canonical | Also accepted |
|-----------|--------------|
| `motion_product_id` | `product_id`, `id`, `[<ID>]` |
| `mfr_name` | `manufacturer_name`, `manufacturer`, `brand` |
| `mfr_part_number` | `manufacturer_part_number`, `part_number`, `mpn` |
| `web_desc` | `web_product_description`, `description` |
| `category` | `pgc_description`, `product_category` |
| `item_number` | `item_number`, `item no` |
| `enterprise_name` | `enterprise_number`, `enterprise` |
| `pgc` | `pgc_code`, `product_group_code` |

> **Important:** `web_scraper.py` does NOT use the same column normalisation as the API server. When running the scraper directly with `--csv`, use the original column names from the CSV — see the `load_product_catalog()` function header for the exact expected column names.

### Required Columns for Pipeline (Recommended Minimum)

- `motion_product_id` — unique identifier
- `mfr_name` — manufacturer name (used in search queries)
- `mfr_part_number` — part number (primary search key)
- `web_desc` — product description (used for text ranking)

### Output Locations

| Artifact | Path |
|----------|------|
| Per-product JSON records | `output/json/<product_id>.json` |
| Downloaded images (local) | `output/images/` |
| Rankings CSV | `output/rankings.csv` |
| API job directories | `uploads/jobs/<job_id>/` |
| Demo pipeline output | `src/demo_mfr_site/pipeline_output/` |
| Demo review queue (static) | `src/demo_mfr_site/site/assets/data/review_queue.json` |

---

## 5. Pipeline A — Full Web Scrape Pipeline (CSV Input)

Use this to run the full pipeline against real external APIs (Wikimedia Commons + OpenVerse) from a CSV file.

### 5.1 Prepare your CSV

The demo CSV at `src/web_scraping/Demo_Site_Products.csv` shows the correct format:

```
motion_product_id,mfr_name,mfr_part_number,web_desc,category
MB2-10,AMI Bearings Inc.,MB2-10,"Stainless Steel Set Screw Locking Bearing Insert, MB200 Series","Spherical OD Set Screw Stainless Steel Inserts"
```

For large catalogs, use `ImageToProduct-Missing_Product_Images.csv` (included at repo root).

### 5.2 Run the scraper

```bash
# Minimal: scrape + classify, local storage only
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/Demo_Site_Products.csv \
  --classify

# With Elasticsearch + MinIO + limit for testing
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/Demo_Site_Products.csv \
  --limit 5 \
  --es \
  --minio \
  --classify

# Include manufacturer website scraping (Tier 1, no browser required)
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/Demo_Site_Products.csv \
  --mfr-scraping \
  --classify

# Full flags
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv <your-file.csv>
  --limit 10           # number of products (omit for all)
  --es                 # index into Elasticsearch
  --minio              # store images in MinIO
  --classify           # run CNN + text ranker + final score
  --mfr-scraping       # Tier 1 manufacturer sites (requests+BS4)
  --mfr-tier2          # Tier 2 og:image fallback (requires Playwright)
```

### 5.3 Verify scraper output

```bash
# Check JSON records were written
ls output/json/

# Verify classification ran
grep -R "predicted_class" output/json/

# Spot-check one record
cat output/json/<product_id>.json | python3 -m json.tool | head -60
```

### 5.4 Export rankings CSV (if not already done by --classify)

```bash
.venv/bin/python src/web_scraping/image_search_ranker.py \
  --json-dir output/json \
  --output output/rankings.csv
```

### 5.5 Start the API server and review UI

See [Section 8](#8-react-ui--input-and-review-interface) for the full React UI flow.

---

## 6. Pipeline B — Demo Site Pipeline + React UI (Recommended First Run)

This is the fastest way to see the full end-to-end flow. It uses a local static AMI Bearings demo site as the scrape source — no external API calls required.

### Prerequisites

- Steps 3.1–3.3 complete (venv + PyTorch)
- `src/Image_Classifier/trained_model.pth` must exist

### Run everything with one script

```bash
./run_pipeline_b.sh
```

This script:

1. Builds the static demo site HTML (4 AMI Bearings products)
2. Scrapes those products from the local site — downloads images and runs the full pipeline (classify → rank → export)
3. Starts the static demo site at **http://localhost:8000**
4. Starts the API server at **http://localhost:5050**
5. Serves the React UI at **http://localhost:3000**
6. Prints the demo review queue URL

To skip rebuilding the demo site on subsequent runs (faster):

```bash
./run_pipeline_b.sh --no-rebuild
```

### Demo walkthrough

After the script prints "DEMO READY":

**Step 1 — Input UI**
1. Open http://localhost:3000/
2. Go to the **Input UI** tab
3. Click **Choose File** and upload `src/web_scraping/Demo_Site_Products.csv`
4. The 4 AMI Bearings products will appear in the table

**Step 2 — Output UI (Review Queue)**
1. Click the **Output UI** tab
2. In the **Review Queue URL** field, paste:
   ```
   http://localhost:5050/api/review-queue/demo-pipeline
   ```
3. Click **Load Review Queue**
4. The 4 products appear with their ranked candidate images

**Step 3 — Review**
- Click each product to see candidate images with AI confidence, text score, and final score
- Use **Approve**, **Reject**, or **Skip** to record decisions
- Review history is tracked in the **Review History** tab

> **What you're seeing:** The images come from the local demo site at port 8000. The pipeline already ran when you executed `run_pipeline_b.sh`. The API server pre-loaded the completed job at startup from `uploads/jobs/demo-pipeline/`.

### Useful demo site pages

```
http://localhost:8000/                       # Catalog index
http://localhost:8000/products/uct305.html   # Product page example
http://localhost:8000/review.html            # Static review page (non-React)
```

### Stop all services

Press `Ctrl+C` in the terminal running `run_pipeline_b.sh`. All three background services are cleaned up automatically.

---

## 7. Pipeline C — Elasticsearch-backed Pipeline

Use this when you have ingested a large product catalog into Elasticsearch and want to scrape by manufacturer or enterprise filter.

### 7.1 Ingest catalog into Elasticsearch

```bash
.venv/bin/python src/web_scraping/ingest_catalog.py \
  --csv ImageToProduct-Missing_Product_Images.csv
```

Verify ingestion:

```bash
curl http://localhost:9200/mi_products/_count
```

### 7.2 Run from Elasticsearch

```bash
# All products for one manufacturer
.venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --mfr-filter "SKF" \
  --limit 10 \
  --es --minio --classify

# All products for an enterprise
.venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --enterprise-filter "SCHAEFFLER GROUP" \
  --es --minio --classify
```

### 7.3 Query and validate ES results

```bash
.venv/bin/python src/web_scraping/query_elasticsearch.py --stats
.venv/bin/python src/web_scraping/minio_es_match.py
```

---

## 8. React UI — Input and Review Interface

The React app provides two main views connected by the API server.

### Start the API server

```bash
.venv/bin/python src/api/server.py
# Listening on http://localhost:5050
```

The API server pre-loads any completed demo jobs from `uploads/jobs/` at startup. It also serves downloaded images at `http://localhost:5050/images/<filename>`.

### Serve the React build

```bash
cd Demo_MFR_Review/Demo_MFR_Review/client/build
python3 -m http.server 3000
```

Open http://localhost:3000/

### Input UI flow

1. Upload a CSV, Excel (.xlsx/.xls), or JSON file
2. The table shows the parsed rows with column headers
3. Set an optional row limit, then click **Run Pipeline**
4. The pipeline runs in the background — a log panel shows progress
5. When done, the review queue URL is displayed — copy it

Supported encodings for CSV: UTF-8, UTF-8-BOM, CP1252, Latin-1, UTF-16.

### Output UI flow

1. Paste the review queue URL (e.g. `http://localhost:5050/api/review-queue/<job_id>`)
2. Click **Load Review Queue**
3. For each product:
   - Left panel: product metadata, manufacturer info
   - Right panel: candidate images with scores
   - Bottom: Confidence Table — all candidates ranked by final score
4. Use arrow keys or click to select different candidate images
5. Submit with **Approve**, **Reject**, or **Skip**

### API endpoints reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/upload` | Upload product file; returns `jobId` |
| POST | `/api/pipeline/run` | Start pipeline for a jobId |
| GET | `/api/pipeline/status/<job_id>` | Poll pipeline progress |
| GET | `/api/review-queue/<job_id>` | Fetch completed review queue |
| GET | `/api/review-queue` | Fetch most recent completed job |
| GET | `/api/jobs` | List all jobs (debug) |
| GET | `/images/<path>` | Serve a downloaded image |
| GET | `/health` | Server health check |

---

## 9. Verifying Each Stage

### After setup

```bash
# ES running
curl http://localhost:9200/_cluster/health
# MinIO running
curl http://localhost:9000/minio/health/live
# API server running
curl http://localhost:5050/health
```

### After scraping

```bash
# JSON records exist
ls output/json/
# Images downloaded
ls output/images/
# Spot check a record
python3 -c "import json; d=json.load(open('output/json/<id>.json')); print(len(d['candidate_images']), 'candidates')"
```

### After classification

```bash
# predicted_class field present
grep -l "predicted_class" output/json/*.json | wc -l
# Full field check
python3 -c "
import json, glob
for p in glob.glob('output/json/*.json'):
    d = json.load(open(p))
    for img in d.get('candidate_images', []):
        assert 'predicted_class' in img, f'Missing in {p}'
print('All records have predicted_class')
"
```

### After ranking

```bash
# Rankings CSV exists and has rows
wc -l output/rankings.csv
# View top candidates
python3 -c "
import csv
rows = list(csv.DictReader(open('output/rankings.csv')))
for r in rows[:5]:
    print(r['motion_product_id'], r['image_rank'], r['final_score_pct'])
"
```

### Running tests

```bash
.venv/bin/pytest tests/ -v
```

The test suite covers:
- CSV/Excel/JSON upload parsing with multiple encodings
- Binary file rejection
- Pipeline boundary (scraper → classifier)
- MinIO-backed classification (mocked)
- ES sync (mocked)

---

## 10. Resetting and Cleaning Up

### Clear pipeline outputs

```bash
rm -rf output/json/* output/images/* output/rankings.csv
```

### Clear API job state

The API server is stateless — job state is in-memory only. To reset:

```bash
# Restart the server
# OR clear persisted demo job:
rm -rf uploads/jobs/demo-pipeline/
```

### Reset demo pipeline output

```bash
rm -rf src/demo_mfr_site/pipeline_output/
rm -f src/demo_mfr_site/site/assets/data/review_queue.json
rm -f src/demo_mfr_site/site/assets/data/review_rankings.csv
rm -rf src/demo_mfr_site/site/assets/review-images/
```

### Reset Elasticsearch indices

```bash
# WARNING: deletes all product and image metadata
.venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

### Stop Docker services

```bash
docker compose down
# To also delete all stored data (volumes):
docker compose down -v
```

---

## 11. Common Failures and Fixes

### `trained_model.pth not found`

**Symptom:** Classification fails with a path error.

**Fix:** The model weights file must exist at `src/Image_Classifier/trained_model.pth`. This file is not committed to the repo (too large for git). Obtain it from the project team or re-train using `src/Image_Classifier/train.py`.

---

### `Could not connect to Elasticsearch`

**Symptom:** `--es` or `--from-es` flag causes connection error.

**Fix:**
```bash
docker compose up -d
# Wait 30–60 seconds for ES to be ready, then:
curl http://localhost:9200/_cluster/health
```

---

### `Could not connect to MinIO`

**Symptom:** `--minio` flag causes connection error.

**Fix:**
```bash
docker compose up -d
curl http://localhost:9000/minio/health/live
```

---

### React UI shows `Failed to load review queue`

**Cause:** API server is not running, or wrong URL.

**Fix:**
1. Confirm API server is running: `curl http://localhost:5050/health`
2. Confirm the job ID is correct: `curl http://localhost:5050/api/jobs`
3. For the demo job specifically: `curl http://localhost:5050/api/review-queue/demo-pipeline`

---

### `No images appear in the review page`

**Cause:** Images are served from `http://localhost:5050/images/` but the demo pipeline images weren't copied there, or the wrong URL is in review_queue.json.

**Fix:**
```bash
# Re-run the demo pipeline to regenerate the API job with correct URLs
./run_pipeline_b.sh --no-rebuild
```

---

### CSV upload fails with encoding error

**Symptom:** API server returns `422 Parse error: Unable to decode CSV upload`.

**Fix:** The API server handles UTF-8, UTF-8-BOM, CP1252, Latin-1, and UTF-16. If it still fails, check the file for embedded null bytes or truly binary content. Enable `UPLOAD_TRACE=1` to see per-encoding failure details:

```bash
UPLOAD_TRACE=1 .venv/bin/python src/api/server.py
```

---

### `web_scraper.py --csv` fails on non-ASCII CSV

**Symptom:** Scraper crashes or shows garbled text for accented characters.

**Cause:** `load_product_catalog()` in `web_scraper.py` only attempts `utf-8-sig` encoding. This is a known limitation — the API server's multi-encoding fallback is not used for CLI runs.

**Workaround:** Convert your CSV to UTF-8 before passing it to the scraper:
```bash
iconv -f cp1252 -t utf-8 input.csv > input_utf8.csv
```

---

### `run_pipeline_b.sh` hangs waiting for a port

**Symptom:** `wait_for_port` loops indefinitely or times out.

**Cause:** `nc` (netcat) is not installed.

**Fix:** Install netcat or manually verify the services started:
```bash
# Ubuntu/Debian
sudo apt-get install -y netcat-openbsd
# macOS
brew install netcat
```

---

### `No candidate images found` for a product

**Cause:** Wikimedia Commons and OpenVerse returned no results for the search query, or all downloads failed.

**Check:** Look at the `scrape_summary.sources_queried` field in the product's JSON. If it's empty, the keywords produced no results.

**Fix:** Enrich the `web_desc` column in your CSV. The scraper generates queries from the first 4–5 words of the description. Vague descriptions (e.g., "PART ABC-123") produce poor results.

---

### Pipeline runs but images are blank or missing in the review UI

**Cause:** The image URL in `review_queue.json` points to a path that is no longer valid (e.g., a temporary server port from a prior demo pipeline run).

**Fix:** Re-run the demo pipeline. Each run regenerates `review_queue.json` with fresh image URLs:
```bash
./run_pipeline_b.sh --no-rebuild
```

---

*Last updated: 2026-04-09. Source-of-truth for all pipeline operations.*

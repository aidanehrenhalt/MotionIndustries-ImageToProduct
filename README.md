# Motion Industries — Image-to-Product Pipeline

End-to-end system for automatically sourcing, classifying, ranking, and human-reviewing candidate product images for the Motion Industries catalog. Given a product record (manufacturer name, part number, description), the pipeline scrapes candidate images, classifies them with a trained CNN, ranks them by fused confidence score, and presents results in a React-based review interface for human approval.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [System Architecture](#2-system-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation and Setup](#4-installation-and-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
   - [Pipeline A — Full Web Scrape (CSV Input)](#pipeline-a--full-web-scrape-csv-input)
   - [Pipeline B — Demo Site (Recommended First Run)](#pipeline-b--demo-site-recommended-first-run)
   - [Pipeline C — Elasticsearch-Backed](#pipeline-c--elasticsearch-backed)
6. [React Review UI](#6-react-review-ui)
7. [Configuration Reference](#7-configuration-reference)
8. [Verifying Outputs](#8-verifying-outputs)
9. [Resetting the Environment](#9-resetting-the-environment)
10. [Troubleshooting](#10-troubleshooting)
11. [Documentation Map](#11-documentation-map)
12. [Project Backlog and Status](#12-project-backlog-and-status)
13. [Team and Contacts](#13-team-and-contacts)

---

## 1. What This Project Does

Motion Industries sells millions of industrial products, many of which lack product images on their website. Missing images reduce customer confidence and increase manual sourcing burden on the content team.

This pipeline automates that process:

1. **Scrapes** candidate images from Wikimedia Commons, OpenVerse, and (optionally) manufacturer catalog pages.
2. **Classifies** each image with a pre-trained 8-class CNN (bearings, seals, power transmission, etc.).
3. **Ranks** candidates by fusing CNN confidence, class-category match, text similarity, and heuristic score into a single `final_score`.
4. **Presents** results in a React review UI where a Motion content team member can approve, reject, or skip each candidate image.

### Pipeline Flow

```
[CSV / Elasticsearch]
        │
        ▼
 web_scraper.py           — Scrapes Wikimedia Commons + OpenVerse APIs
        │                   Writes output/json/<product_id>.json
        │                   Downloads images to output/images/
        │                   Optionally indexes to Elasticsearch + MinIO
        ▼
 classify_json_images.py  — CNN (8-class, 500×500 input)
        │                   Writes predicted_class + classifier_confidence to JSON
        │                   Optionally syncs to Elasticsearch
        ▼
 image_search_ranker.py   — Fuses signals into final_score
        │                   final_score = 0.30×ai_conf + 0.20×class_match
        │                                + 0.30×text_sim + 0.20×prelim_score
        │                   Exports output/rankings.csv
        ▼
 review_queue.json        — Structured output for the React UI
        │
        ▼
 React Review UI          — Approve / Reject / Skip per product
```

---

## 2. System Architecture

### Components

| Component | Location | Port | Purpose |
|-----------|----------|------|---------|
| Web Scraper | `src/web_scraping/web_scraper.py` | — | Core orchestrator: scrape → download → JSON |
| Image Classifier | `src/Image_Classifier/classify_json_images.py` | — | CNN classification pass |
| Text Ranker | `src/web_scraping/image_search_ranker.py` | — | Text similarity + final score fusion |
| Flask API Server | `src/api/server.py` | 5050 | File upload + pipeline trigger + review queue |
| React UI | `Demo_MFR_Review/…/client/` | 3000 | Input UI + Output UI (review) |
| Demo Site | `src/demo_mfr_site/site/` | 8000 | Local AMI Bearings catalog (scrape target) |
| Elasticsearch | Docker | 9200 | Product and image metadata store |
| MinIO | Docker | 9000 | S3-compatible image object storage |

### Repository Layout

The repository is organized by pipeline stage, with each top-level `src/` module owning one stage and supporting infrastructure at the root.

```
.
├── src/
│   ├── web_scraping/           Scraper, ranker, Elasticsearch & MinIO helpers
│   ├── Image_Classifier/       CNN model, inference, training entrypoint
│   ├── api/                    Flask API bridging the React UI and the pipeline
│   └── demo_mfr_site/          Self-contained AMI Bearings demo catalog + pipeline
├── Demo_MFR_Review/            React.js human review/triage UI
├── Model_Development/          Training notebooks, checkpoints, dataset helpers
├── data/                       Raw and processed datasets + training manifest
├── models/                     Trained model artifacts
├── notebooks/                  Exploratory Jupyter notebooks
├── tests/                      Pytest integration and unit tests
├── docs/                       Runbooks and detailed design docs
├── output/                     Pipeline JSON + image outputs (generated at runtime)
├── uploads/                    Catalog uploads posted through the API
├── docker-compose.yml          Elasticsearch, Kibana, and MinIO for local dev
├── requirements.txt            Python dependencies
└── run_pipeline_b.sh           Wrapper that runs the demo MFR pipeline
```

### Storage Architecture

The pipeline uses a two-layer storage model:

**Elasticsearch** — metadata only. Two indices:
- `mi_products` — one document per Motion catalog product
- `mi_candidate_images` — one document per candidate image, keyed by `SHA1("{product_id}:{image_url}")`

**MinIO** — binary image storage. Object key layout:
```
images/{product_id}/{product_id}_{index}_{url_hash}.{ext}
```

**Local filesystem** — development fallback:
```
output/
  json/           per-product JSON records
  images/         downloaded image files
  rankings.csv    ranked candidate export
```

For full Elasticsearch and MinIO reference, see [`docs/ELASTICSEARCH.md`](docs/ELASTICSEARCH.md) and [`docs/MINIO.md`](docs/MINIO.md).

### CNN Classifier — 8-Class Reference

| `predicted_class` | PGC1 | Description |
|---|---|---|
| 0 | 1 | BEARINGS |
| 1 | 2 | SEALS AND ACCESSORIES |
| 2 | 3 | POWER TRANSMISSION |
| 3 | 4 | ELECTRICAL & MAT'L HAND'G |
| 4 | 5 | HOSE AND FITTINGS |
| 5 | 6 | FLUID POWER |
| 6 | 7 | PROCESS PUMPS AND EQUIPMENT |
| 7 | 8 | INDUSTRIAL SUPPLIES |

---

## 3. Prerequisites

### Required
- Python 3.10+
- Docker and Docker Compose
- Node.js 18+ and npm (only needed to rebuild the React app — a pre-built `build/` is included)

### Optional
- Chromium + Playwright (Tier 2 manufacturer scraping)

### Verify

```bash
python3 --version      # must be 3.10+
docker --version
docker compose version
node --version         # only if rebuilding React
npm --version
```

---

## 4. Installation and Setup

### 4.1 Clone and enter the repo

```bash
git clone <repo-url>
cd MotionIndustries-ImageToProduct
```

### 4.2 Create the virtual environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 4.3 Install PyTorch

The correct variant depends on your hardware. Run **after** the requirements step:

```bash
# CPU-only (works everywhere, slower):
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (NVIDIA GPU):
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** NumPy is pinned to `<2.0` in `requirements.txt`. **DO NOT UPGRADE IT** — The current PyTorch/torchvision build fails classification transforms with NumPy 2.x.

### 4.4 Start Docker services

```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| Elasticsearch | http://localhost:9200 |
| Kibana | http://localhost:5601 |
| MinIO API | http://localhost:9000 |
| MinIO Console | http://localhost:9001 |

Default MinIO credentials: `minioadmin` / `minioadmin`

### 4.5 Initialize Elasticsearch indices (one-time)

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py
```

Use `--recreate` only when mapping changes require dropping and rebuilding the indices:

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

### 4.6 Install Playwright (optional — Tier 2 scraping only)

```bash
.venv/bin/playwright install chromium
```

---

## 5. Running the Pipeline

### Pipeline A — Full Web Scrape (CSV Input)

Scrapes real external APIs (Wikimedia Commons + OpenVerse) from a product CSV.

**Prepare your CSV.** The demo CSV at `src/web_scraping/Demo_Site_Products.csv` shows the exepcted format:

```
motion_product_id,mfr_name,mfr_part_number,web_desc,category
MB2-10,AMI Bearings Inc.,MB2-10,"Stainless Steel Set Screw Locking Bearing Insert, MB200 Series","Spherical OD Set Screw Stainless Steel Inserts"
```

**Run:**

```bash
# Minimal: scrape + classify, local storage only
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/Demo_Site_Products.csv \
  --classify

# With Elasticsearch + MinIO, limit for testing
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/Demo_Site_Products.csv \
  --limit 5 --es --minio --classify

# Include Tier 1 manufacturer site scraping
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/Demo_Site_Products.csv \
  --mfr-scraping --classify
```

> **Note on live manufacturer sites:** Many high-value manufacturer portals (Grainger, MSC, and others) prohibit automated access via their Terms of Service or enforce bot-protection middleware. The pipeline does not attempt to circumvent these restrictions. See [Scraper Status](#scraper-status) below.

**Export rankings CSV (if not already done by `--classify`):**

```bash
.venv/bin/python src/web_scraping/image_search_ranker.py \
  --json-dir output/json \
  --output output/rankings.csv
```

---

### Pipeline B — Demo Site (Recommended First Run)

The fastest way to see the full end-to-end flow. Uses a local static AMI Bearings demo site as the scrape source — no external API calls required.

**Prerequisites:** Steps 4.1–4.3 complete, and `src/Image_Classifier/trained_model.pth` must exist.

**Run everything with one script:**

```bash
./run_pipeline_b.sh
```

This script:
1. Builds the static demo site (4 AMI Bearings products)
2. Scrapes those products locally — downloads images, classifies, ranks, and exports
3. Starts the demo site at **http://localhost:8000**
4. Starts the API server at **http://localhost:5050**
5. Serves the React UI at **http://localhost:3000**

To skip rebuilding on subsequent runs:

```bash
./run_pipeline_b.sh --no-rebuild
```

**Demo walkthrough:**

1. Open http://localhost:3000 → **Input UI** tab → upload `src/web_scraping/Demo_Site_Products.csv`
2. Click **Output UI** tab → paste `http://localhost:5050/api/review-queue/demo-pipeline` → click **Load Review Queue**
3. Approve / Reject / Skip each product's candidate images

**Stop all services:** `Ctrl+C` in the terminal running `run_pipeline_b.sh`.

---

### Pipeline C — Elasticsearch-Backed

For large catalog runs where products are already ingested into Elasticsearch.

**Ingest catalog:**

```bash
.venv/bin/python src/web_scraping/ingest_catalog.py \
  --csv ImageToProduct-Missing_Product_Images.csv
```

**Run from Elasticsearch:**

```bash
# By manufacturer
.venv/bin/python src/web_scraping/web_scraper.py \
  --from-es --mfr-filter "SKF" --limit 10 --es --minio --classify

# By enterprise
.venv/bin/python src/web_scraping/web_scraper.py \
  --from-es --enterprise-filter "SCHAEFFLER GROUP" --es --minio --classify
```

---

### Scraper Status

The web scraper is fully functional for:
- Open API sources (Wikimedia Commons, OpenVerse)
- The local demo pipeline (AMI Bearings demo site)
- Tier 1 manufacturer scrapers for approved domains (`--mfr-scraping`)

Live manufacturer portal scraping is currently blocked for the following reason: many high-value industrial distributor sites (including Grainger, MSC, and Motion's own web properties) prohibit automated access in their Terms of Service and/or enforce Incapsula/Imperva bot-protection middleware. The team made a deliberate decision not to use headless browser automation to circumvent these restrictions, in compliance with 18 U.S.C. § 1030 (CFAA) and 17 U.S.C. § 1201 (DMCA). Authorized data access arrangements with Motion Industries' supplier contacts are the planned path to unblocking this source tier.

---

## 6. React Review UI

### Start the API server

```bash
.venv/bin/python src/api/server.py
# Listening on http://localhost:5050
```

### Serve the React build

```bash
cd Demo_MFR_Review/Demo_MFR_Review/client/build
python3 -m http.server 3000
```

Open http://localhost:3000/

### Input UI

1. Upload a CSV, Excel (.xlsx/.xls), or JSON file
2. Set an optional row limit, then click **Run Pipeline**
3. A log panel shows pipeline progress
4. When done, the review queue URL is displayed

### Output UI (Review)

1. Paste the review queue URL → click **Load Review Queue**
2. Left panel: product metadata and manufacturer info
3. Right panel: candidate images with `ai_confidence`, `text_score`, `final_score`
4. Bottom: Confidence Table — all candidates ranked by `final_score`
5. Submit decisions with **Approve**, **Reject**, or **Skip**

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/upload` | Upload product file; returns `jobId` |
| POST | `/api/pipeline/run` | Start pipeline for a jobId |
| GET | `/api/pipeline/status/<job_id>` | Poll pipeline progress |
| GET | `/api/review-queue/<job_id>` | Fetch completed review queue |
| GET | `/api/review-queue` | Fetch most recent completed job |
| GET | `/images/<path>` | Serve a downloaded image |
| GET | `/health` | Server health check |

---

## 7. Configuration Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO credentials |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO credentials |
| `MINIO_BUCKET` | `mi-images` | Storage bucket name |
| `API_PORT` | `5050` | Flask API listen port |
| `UPLOAD_TRACE` | (unset) | Set to `1` to log upload parsing details |

### Input CSV — Required Columns

| Canonical field | Also accepted |
|-----------------|--------------|
| `motion_product_id` | `product_id`, `id` |
| `mfr_name` | `manufacturer_name`, `manufacturer`, `brand` |
| `mfr_part_number` | `manufacturer_part_number`, `part_number`, `mpn` |
| `web_desc` | `web_product_description`, `description` |
| `category` | `pgc_description`, `product_category` |

### Output Locations

| Artifact | Path |
|----------|------|
| Per-product JSON records | `output/json/<product_id>.json` |
| Downloaded images (local) | `output/images/` |
| Rankings CSV | `output/rankings.csv` |
| API job directories | `uploads/jobs/<job_id>/` |
| Demo pipeline output | `src/demo_mfr_site/pipeline_output/` |

---

## 8. Verifying Outputs

### After scraping

```bash
ls output/json/
python3 -c "import json; d=json.load(open('output/json/<id>.json')); print(len(d['candidate_images']), 'candidates')"
```

### After classification

```bash
grep -l "predicted_class" output/json/*.json | wc -l
```

### After ranking

```bash
wc -l output/rankings.csv
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

The test suite covers: CSV/Excel/JSON upload parsing, binary file rejection, scraper↔classifier boundary, MinIO-backed classification (mocked), and ES sync (mocked).

---

## 9. Resetting the Environment

```bash
# Clear pipeline outputs
rm -rf output/json/* output/images/* output/rankings.csv

# Reset demo pipeline output
rm -rf src/demo_mfr_site/pipeline_output/
rm -f src/demo_mfr_site/site/assets/data/review_queue.json

# Reset Elasticsearch indices (WARNING: deletes all metadata)
.venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate

# Stop Docker services
docker compose down

# Stop Docker and delete all stored data
docker compose down -v
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `trained_model.pth not found` | Model weights not committed (too large for git) | Obtain from project team or retrain via `src/Image_Classifier/train.py` |
| `Could not connect to Elasticsearch` | Docker not running | `docker compose up -d`, wait 30–60s, then `curl http://localhost:9200/_cluster/health` |
| `Could not connect to MinIO` | Docker not running | `docker compose up -d`, then `curl http://localhost:9000/minio/health/live` |
| `Failed to load review queue` | API server not running or wrong URL | `curl http://localhost:5050/health`, then `curl http://localhost:5050/api/jobs` |
| Images don't load in review UI | `file://` URLs not browser-accessible | Add Flask `/images/<path>` route (Phase 3 of live integration plan) |
| CSV upload fails with encoding error | Non-UTF-8 encoding | Enable `UPLOAD_TRACE=1` to see per-encoding failure details |
| `No candidate images found` | Wikimedia/OpenVerse returned no results | Enrich the `web_desc` column — the scraper uses the first 4–5 words as search terms |
| `run_pipeline_b.sh` hangs on port check | `nc` (netcat) not installed | `sudo apt-get install -y netcat-openbsd` or `brew install netcat` |

---

## 11. Documentation Map

| File | Contents |
|------|----------|
| `README.md` | This file — quick start and full pipeline reference |
| `CLAUDE.md` | AI assistant instructions and code conventions |
| `docs/ELASTICSEARCH.md` | Index mappings, query examples, write path, validation commands |
| `docs/MINIO.md` | Object key layout, health checks, environment variables |
| `docs/INTEGRATION_PLAN.md` | Implementation status, remaining work, GCP migration plan |
| `docs/LIVE_INTEGRATION_PLAN.md` | Backend API + React frontend integration design and test plan |
| `docs/backlog.md` | Sprint-by-sprint commit history and open task list |
| `docs/SourceProductImages.md` | Source priority list, scraping recipes, legal constraints |
| `docs/architecture.md` | High-level architecture notes (stub — expand as needed) |
| `docs/conventions.md` | Code style and testing conventions |
| `FlowChartImage-To-Text.md` | Text translation of the pipeline flowchart |
| `Demo_MFR_Review-PipelineIntegration.md` | Checklist for merging the review frontend into the demo site |
| `src/web_scraping/README.md` | Web scraper CLI reference |
| `src/Image_Classifier/README.md` | CNN classifier architecture and usage |
| `src/demo_mfr_site/README.md` | Demo site rebuild and serve instructions |
| `tests/README.md` | Test suite reference |

---

## 12. Project Backlog and Status

See [`docs/backlog.md`](docs/backlog.md) for the full sprint-by-sprint commit history.

### Current Status (as of April 2026)

| Stage | Status |
|-------|--------|
| Web Scraper (open APIs) | ✅ Functional |
| Web Scraper (demo site) | ✅ Functional |
| Web Scraper (live manufacturer portals) | ⚠️ Blocked — see [Scraper Status](#scraper-status) |
| CNN Image Classifier | ✅ Functional (88–98% test accuracy) |
| Text Similarity / Ranker | ✅ Functional (token-overlap baseline) |
| Final Score + Ranking | ✅ Functional |
| Flask API Server | ✅ Functional |
| React Review UI | ✅ Functional |
| Image serving in review UI (`/images/` route) | 🔲 Pending (Phase 3) |
| Review decision persistence (backend) | 🔲 Pending (Phase 6) |
| GCP Cloud Storage migration | 🔲 Pending — contact established with Motion Industries |

### GCP Production Migration

For production deployment, the following transitions are planned:

| Dev component | Production equivalent |
|---|---|
| MinIO (Docker) | GCP Cloud Storage |
| Elasticsearch (Docker) | Managed Elasticsearch or GCP-equivalent |
| Flask API (local) | Cloud Run |
| Docker images | Artifact Registry |
| CNN + text models | Vertex AI |

Contact: anu.shrestha@motion.com, george.baldwin@motion.com

---

## 13. Team and Contacts

| Name | Role | Email |
|------|------|-------|
| Ace Ehrenhalt | Feedback Pipeline, Web Scraper, Database | aehrenhalt3@gatech.edu |
| Rodrigo Gaeta-Lopez | ML Model Development | rlopez76@gatech.edu |
| Nia Simon | Image Discovery, Expo Coordinator | nsimon33@gatech.edu |
| Faris Unal | UI / Webmaster | funal7@gatech.edu |
| Prof. Patricio Vela | Faculty Advisor | pvela@gatech.edu |

Motion Industries shared dataset:
[GT Capstone OneDrive](https://genparts-my.sharepoint.com/:f:/r/personal/michael_flack_corp_motion-ind_com/Documents/GT%20Capstone?csf=1&web=1&e=s92NcQ)

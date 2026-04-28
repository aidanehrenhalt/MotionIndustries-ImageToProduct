# Motion Industries Image-to-Product

Repository for validating the full image-to-product pipeline:

- Web scraper
- Image classifier
- Text analysis / metadata ranker
- Final fused ranking

The main end-to-end entrypoint is [`src/web_scraping/web_scraper.py`](src/web_scraping/web_scraper.py). When you run it with `--classify`, it performs:

1. Scraping
2. CNN image classification
3. Text-based ranking
4. Final score + final rank assignment

## Pipeline Architecture

```
CSV / Elasticsearch catalog
        │
        ▼
  Web Scraper  ──► Wikimedia / OpenVerse / (optional) manufacturer sites
        │
        ▼
  Image store (local filesystem or MinIO)
  Metadata store (JSON files, optional Elasticsearch index)
        │
        ▼
  CNN Image Classifier  (predicted_class, classifier_confidence)
        │
        ▼
  Text / Metadata Ranker (ranker_score, score_breakdown)
        │
        ▼
  Final Fused Ranking (final_score, final_rank)
        │
        ▼
  review_queue.json  ──►  Demo MFR Review UI (React)
```

See [`FlowChartImage-To-Text.md`](FlowChartImage-To-Text.md) and [`docs/PIPELINE_RUNBOOK.md`](docs/PIPELINE_RUNBOOK.md) for deeper detail.

## Prerequisites

- Python 3.10+ with `venv`
- Docker + Docker Compose (for Elasticsearch, Kibana, MinIO)
- Node.js 18+ and `npm` (only for the React review UI)
- ~2 GB RAM free for Elasticsearch; additional for PyTorch inference
- Google Chrome/Chromium available on `PATH` (only for manufacturer scraping via Selenium/Playwright)

## Repository Layout

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
├── run_pipeline_b.sh           Wrapper that runs the demo MFR pipeline
└── README.md                   This file
```

## Quick Start

Use this section to run the full local pipeline from scratch.

### 1. Create the virtual environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If PyTorch was not installed from `requirements.txt`, install it explicitly:

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Optional, only if you plan to use Tier 2 manufacturer scraping:

```bash
.venv/bin/playwright install chromium
```

### 2. Start local services

```bash
docker-compose up -d
```

Services:

- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

Default MinIO credentials:

- Username: `minioadmin`
- Password: `minioadmin`

### 3. Initialize Elasticsearch mappings

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py
```

Use `--recreate` only if you intentionally want to drop and rebuild the local indices.

### 4. Run the full pipeline on a small sample

This command validates scraper + classifier + text analysis + final ranker in one run:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 2 \
  --es \
  --minio \
  --classify
```

What this does:

- Loads products from the sample CSV
- Scrapes candidate images
- Stores metadata in Elasticsearch
- Uploads images to MinIO
- Runs CNN classification
- Runs text/metadata ranking
- Writes `final_score` and `final_rank` into the JSON output

### 5. Validate outputs

Check that JSON records were written:

```bash
ls output/json
```

Check that classifier fields were written:

```bash
grep -R "predicted_class" output/json
grep -R "classifier_confidence" output/json
```

Check that ranker fields were written:

```bash
grep -R "ranker_score" output/json
grep -R "score_breakdown" output/json
grep -R "final_rank" output/json
```

Check Elasticsearch summaries:

```bash
.venv/bin/python src/web_scraping/query_elasticsearch.py --stats
.venv/bin/python src/web_scraping/query_elasticsearch.py --images
```

Check MinIO and Elasticsearch consistency:

```bash
.venv/bin/python src/web_scraping/minio_es_match.py --verify
```

### 6. If you want a local-files-only run

This skips Elasticsearch and MinIO and writes images to `output/images`:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 2 \
  --classify
```

### 7. If you want to rerun only the ranker on existing JSON files

```bash
.venv/bin/python src/web_scraping/image_search_ranker.py \
  --json-dir output/json \
  --text-only
```

## Straightforward Pipeline Run Instructions

All commands below assume you are in the repository root.

### Pipeline A: CSV input -> scrape -> classify -> text analysis -> final rank

1. Start services:

```bash
docker-compose up -d
```

2. Initialize Elasticsearch:

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py
```

3. Run the full pipeline:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 5 \
  --es \
  --minio \
  --classify
```

4. Inspect output files:

- JSON: `output/json/`
- Local images: `output/images/` when `--minio` is not used
- MinIO objects: bucket `mi-images` when `--minio` is used

5. Confirm the final pipeline fields exist in JSON:

- `predicted_class`
- `classifier_confidence`
- `ranker_score`
- `score_breakdown`
- `final_score`
- `final_rank`

### Pipeline B: Elasticsearch product catalog -> scrape -> classify -> rank

1. Ingest the catalog into `mi_products`:

```bash
.venv/bin/python src/web_scraping/ingest_catalog.py \
  --csv ImageToProduct-Missing_Product_Images.csv
```

2. Run the pipeline from Elasticsearch:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --mfr-filter SKF \
  --limit 10 \
  --es \
  --minio \
  --classify
```

3. Validate results:

```bash
grep -R "final_rank" output/json
.venv/bin/python src/web_scraping/query_elasticsearch.py --stats
```

## Environment Variables

The pipeline reads the following environment variables (all optional; defaults match `docker-compose.yml`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ES_HOST` | `localhost` | Elasticsearch host |
| `ES_PORT` | `9200` | Elasticsearch port |
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO S3-compatible endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `mi-images` | Bucket for candidate images |

Override any of these on the CLI (for example `--es-host`, `--minio-endpoint`) or via your shell environment.

## Pipeline API Server

A Flask API in [`src/api/server.py`](src/api/server.py) exposes the full pipeline to the React review UI.

```bash
.venv/bin/python src/api/server.py
```

Endpoints:

| Method / Path | Purpose |
|---------------|---------|
| `POST /api/upload` | Upload and parse a product catalog (CSV/Excel) |
| `POST /api/pipeline/run` | Kick off scrape → classify → rank for an uploaded file |
| `GET  /api/pipeline/status/<job_id>` | Poll pipeline progress |
| `GET  /api/review-queue` | Serve the latest `review_queue.json` |
| `GET  /api/review-queue/<job_id>` | Serve `review_queue.json` for a specific job |
| `GET  /api/jobs` | List all jobs (debug) |
| `GET  /health` | Liveness probe |

All jobs run with the project root as the working directory, so they share `output/`. This is intentional for local/dev use — not production-safe for concurrent jobs.

## Demo Manufacturer Pipeline

The repository also includes a self-contained demo pipeline under `src/demo_mfr_site/`.

Run it with:

```bash
./run_pipeline_b.sh
```

Or rebuild the demo artifacts first:

```bash
./run_pipeline_b.sh --rebuild
```

The demo pipeline writes outputs to:

- `src/demo_mfr_site/pipeline_output/json/`
- `src/demo_mfr_site/pipeline_output/images/`
- `src/demo_mfr_site/pipeline_output/run_summary.json`

It also runs the same three processing stages:

- CNN classifier
- Text/metadata ranker
- Final fused ranking

## Quick Start: Demo MFR Review Site (React.js)

### What is it?
Think of it like a triage screen for a doctor — but instead of patients, you're reviewing candidate product images ranked by the pipeline. It's a browser-based app (built with React) that lets a human reviewer look at each product, compare the images the pipeline found, and stamp each one as approved, rejected, or skipped.

### Why would you use it?
The pipeline produces ranked candidate images automatically, but someone still needs to make the final call on whether a match is good. This site replaces that step from a spreadsheet or raw JSON file with a structured, click-through review interface.

### Get started

1. **Install dependencies** — from the repo root, run:

   ```bash
   cd Demo_MFR_Review/Demo_MFR_Review/client
   npm install
   ```

2. **Place the pipeline output file** — copy `review_queue.json` (produced by the demo pipeline) into `public/assets/data/review_queue.json` so the app can load it. The demo pipeline writes this file to `src/demo_mfr_site/pipeline_output/`; the default URL the app fetches is `assets/data/review_queue.json`.

3. **Start the development server:**

   ```bash
   npm start
   ```

   The app opens at `http://localhost:3000` in your browser.

4. **Use the three tabs in the app:**
   - **Input UI** — upload the product Excel/CSV to browse the product database.
   - **Output UI** — enter the URL to `review_queue.json`, click "Load Review Queue", then approve or reject each product's best candidate image. Use arrow keys to flip between candidate images.
   - **Review History** — see every decision made during the current browser session.

To build a static copy instead of running the dev server:

```bash
npm run build
```

Then serve the `build/` folder with any static file server (e.g. `npx serve build`). A pre-built copy already lives at `Demo_MFR_Review/Demo_MFR_Review/client/build/`.

### What to expect
After loading the review queue, you will see one product at a time with its ranked candidate images and confidence scores; each decision moves the product out of the queue and into the history tab.

## What This Repository Does

- Ingests product catalog rows from CSV or Elasticsearch
- Scrapes candidate images from Wikimedia, OpenVerse, and optional manufacturer sources
- Stores image binaries locally or in MinIO
- Stores product and candidate metadata in Elasticsearch
- Runs the CNN classifier on downloaded images
- Runs a text/metadata ranker on scraped candidates
- Combines both signals into a final ranking

## Documentation Map

Top-level:

- [`README.md`](README.md): quick start and end-to-end run instructions (this file)
- [`CLAUDE.md`](CLAUDE.md): agent/contributor coding conventions
- [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md): extended end-to-end pipeline guide
- [`FlowChartImage-To-Text.md`](FlowChartImage-To-Text.md): visual flow chart of the stages
- [`Demo_MFR_Review-PipelineIntegration.md`](Demo_MFR_Review-PipelineIntegration.md): how the React UI integrates with the pipeline

`docs/`:

- [`docs/PIPELINE_RUNBOOK.md`](docs/PIPELINE_RUNBOOK.md): canonical runbook
- [`docs/PIPELINE_TEST_STEPS.md`](docs/PIPELINE_TEST_STEPS.md): step-by-step validation procedures
- [`docs/ELASTICSEARCH.md`](docs/ELASTICSEARCH.md): Elasticsearch mappings and query examples
- [`docs/MINIO.md`](docs/MINIO.md): MinIO object layout and verification
- [`docs/INTEGRATION_PLAN.md`](docs/INTEGRATION_PLAN.md): integration backlog and remaining work
- [`docs/LIVE_INTEGRATION_PLAN.md`](docs/LIVE_INTEGRATION_PLAN.md): live integration plan
- [`docs/SourceProductImages.md`](docs/SourceProductImages.md): image source inventory
- [`docs/backlog.md`](docs/backlog.md): outstanding backlog
- [`docs/architecture.md`](docs/architecture.md): architecture notes
- [`docs/conventions.md`](docs/conventions.md): language/framework conventions

Subsystem READMEs:

- [`src/web_scraping/README.md`](src/web_scraping/README.md): scraper CLI reference
- [`src/Image_Classifier/README.md`](src/Image_Classifier/README.md): CNN architecture and inference
- [`src/demo_mfr_site/README.md`](src/demo_mfr_site/README.md): AMI Bearings demo catalog
- [`tests/README.md`](tests/README.md): test suite coverage

## Notes

- The scraper currently generates search terms via `simple_search_keywords()`.
- `--classify` on [`web_scraper.py`](/home/aceaid/MotionIndustries-ImageToProduct/src/web_scraping/web_scraper.py) runs more than just the CNN. It also runs the text ranker and applies the final ranking pass.
- Standalone classification via [`classify_json_images.py`](/home/aceaid/MotionIndustries-ImageToProduct/src/Image_Classifier/classify_json_images.py) only performs the classifier pass unless you separately run the ranker.

## Model Training

Training artifacts and scripts live under `Model_Development/` and `src/Image_Classifier/`:

- [`Model_Development/training.py`](Model_Development/training.py) and [`Model_Development/training_notebook.ipynb`](Model_Development/training_notebook.ipynb) — training loop, loss/accuracy tracking, checkpointing
- [`Model_Development/filtering_images.py`](Model_Development/filtering_images.py), [`Model_Development/class_analysis.py`](Model_Development/class_analysis.py) — dataset preparation and class distribution analysis
- [`src/Image_Classifier/train.py`](src/Image_Classifier/train.py) — standalone trainer mirroring the notebook
- [`src/Image_Classifier/trained_model.pth`](src/Image_Classifier/trained_model.pth) — pretrained weights consumed at inference time
- [`data/training_manifest.csv`](data/training_manifest.csv) — manifest mapping product rows to image files

Inference at pipeline time is handled by [`src/Image_Classifier/img_classifier.py`](src/Image_Classifier/img_classifier.py) and [`src/Image_Classifier/classify_json_images.py`](src/Image_Classifier/classify_json_images.py). See [`src/Image_Classifier/README.md`](src/Image_Classifier/README.md) for the full model architecture table.

## Testing

```bash
# Full test suite
.venv/bin/python -m pytest -q

# Scraper/classifier/ranker integration
.venv/bin/python -m pytest -q tests/test_scraper_classifier_pipeline.py

# API upload parsing
.venv/bin/python -m pytest -q tests/test_api_upload_parsing.py
```

See [`tests/README.md`](tests/README.md) for the full list of covered scenarios.

## Useful Validation Test

There is an integration-style test covering the scraper/classifier boundary and ranker output fields:

```bash
.venv/bin/python -m pytest -q tests/test_scraper_classifier_pipeline.py
```

## Motion Shared Folder

[Image Dataset, Image Mapping, Images](https://genparts-my.sharepoint.com/:f:/r/personal/michael_flack_corp_motion-ind_com/Documents/GT%20Capstone?csf=1&web=1&e=s92NcQ)

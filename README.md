# Motion Industries Image-to-Product

Repository for scraping candidate product images, storing image binaries, indexing metadata, and classifying images with the current CNN model.

## What This Repository Does

- Ingests product catalog rows from CSV or Elasticsearch.
- Scrapes candidate images from Wikimedia/OpenVerse and optional manufacturer sources.
- Stores image binaries locally or in MinIO.
- Stores product and candidate metadata in Elasticsearch.
- Runs the CNN classifier and writes `predicted_class` back to JSON (and Elasticsearch when enabled).
- Includes a local AMI manufacturer demo site pipeline under `src/demo_mfr_site/`.

## Data Storage Model

- Elasticsearch stores metadata:
  - `mi_products`
  - `mi_candidate_images`
- MinIO (or local filesystem) stores image binaries.

Detailed database docs:

- `docs/ELASTICSEARCH.md`
- `docs/MINIO.md`
- `docs/INTEGRATION_PLAN.md`

## Documentation Map

- `README.md` (this file): project overview and new-user setup.
- `docs/PIPELINE_RUNBOOK.md`: canonical run instructions for scraper to classifier and demo usage.
- `docs/ELASTICSEARCH.md`: mappings, write path, and query examples.
- `docs/MINIO.md`: object layout, consistency checks, and retrieval utilities.
- `docs/INTEGRATION_PLAN.md`: remaining integration work.
- `docs/SCRAPER_ML_PIPELINE_REVIEW.md`: validation history and findings archive.
- `docs/GCPAccess.md`: cloud access planning notes.

## Prerequisites

- Python 3.12+
- Docker with Compose
- Git
- For classification: `torch` and `torchvision`
- Optional for manufacturer Tier 2 scraping: Playwright Chromium

## Environment Setup

Create and populate a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install PyTorch if your environment did not resolve it from `requirements.txt`:

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Optional extras:

```bash
.venv/bin/playwright install chromium
```

## Start Local Services

```bash
docker-compose up -d
```

Service endpoints:

- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

Default MinIO credentials:

- Username: `minioadmin`
- Password: `minioadmin`

## Initialize Elasticsearch

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py
```

Use `--recreate` only when mapping changes require dropping local index data.

## First Validation Run

Run a minimal scrape from CSV:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 2
```

Then validate outputs:

- JSON records in `output/json/`
- Images in `output/images/` (local mode) or MinIO (if `--minio` was used)

## Canonical Execution Guide

Use `docs/PIPELINE_RUNBOOK.md` for:

- Scraper to classifier execution paths.
- Elasticsearch + MinIO mode.
- Demo manufacturer site workflow.
- Output reading and verification steps.

## Current Implementation Notes

- Search query generation in scraper still defaults to `simple_search_keywords()`.
- `text_based_search.py` and `product_to_vector()` exist, but full `vector_to_query(...)` cutover is still pending.
- Elasticsearch mappings are strict; new indexed fields require mapping updates and index recreation.

## Troubleshooting Quick Checks

- Elasticsearch health: `http://localhost:9200`
- MinIO health: `http://localhost:9000/minio/health/live`
- Missing classifier dependencies: install `torch` and `torchvision`
- Playwright errors for Tier 2 paths: run `.venv/bin/playwright install chromium`

## Motion Shared Folder

[Image Dataset, Image Mapping, Images](https://genparts-my.sharepoint.com/:f:/r/personal/michael_flack_corp_motion-ind_com/Documents/GT%20Capstone?csf=1&web=1&e=s92NcQ)

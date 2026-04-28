# Pipeline Runbook

Canonical run instructions for:

- Scraper -> classifier execution
- Elasticsearch + MinIO mode
- Demo manufacturer site execution
- Output inspection and validation

All commands below assume you are in the repository root.

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
docker-compose up -d
.venv/bin/python src/web_scraping/setup_elasticsearch.py
```

## Pipeline A: Scraper to Classifier

### 1. Run scraper on CSV input

Small local-only run:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 5
```

CSV with Elasticsearch + MinIO + integrated classification:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 5 \
  --es \
  --minio \
  --classify
```

### 2. Run scraper from Elasticsearch product index

First ingest a catalog if needed:

```bash
.venv/bin/python src/web_scraping/ingest_catalog.py \
  --csv ImageToProduct-Missing_Product_Images.csv
```

Then run from `mi_products`:

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --mfr-filter SKF \
  --limit 10 \
  --es \
  --minio \
  --classify
```

### 3. Optional standalone classification pass

Use this for local-file JSON artifacts already on disk:

```bash
.venv/bin/python src/Image_Classifier/classify_json_images.py \
  --json-dir output/json \
  --model src/Image_Classifier/trained_model.pth \
  --es
```

Note:

- `web_scraper.py --classify` uses scoped file lists from the active scrape run.
- `classify_json_images.py` directory mode scans `output/json` and only classifies local-file entries.

### 4. Read outputs

Primary output locations:

- `output/json/`
- `output/images/` (local storage mode)
- MinIO bucket `mi-images` (if `--minio`)

Quick checks:

```bash
# Check JSON predictions
grep -R "predicted_class" output/json || true

# Elasticsearch summaries
.venv/bin/python src/web_scraping/query_elasticsearch.py --stats
.venv/bin/python src/web_scraping/query_elasticsearch.py --images

# MinIO <-> ES consistency
.venv/bin/python src/web_scraping/minio_es_match.py --verify
```

## Pipeline B: Demo Manufacturer Site

This flow uses AMI demo pages and writes demo outputs under `src/demo_mfr_site/pipeline_output`.

### 1. Optional: rebuild demo site artifacts

From existing scraped artifacts:

```bash
.venv/bin/python src/demo_mfr_site/build_demo_site.py
```

### 2. Run demo scrape + classify pipeline

```bash
.venv/bin/python src/demo_mfr_site/run_demo_pipeline.py
```

### 3. Read demo outputs

- Summary: `src/demo_mfr_site/pipeline_output/run_summary.json`
- Classified records: `src/demo_mfr_site/pipeline_output/json/*.json`
- Downloaded images: `src/demo_mfr_site/pipeline_output/images/`

Verify classification fields:

```bash
grep -R "predicted_class" src/demo_mfr_site/pipeline_output/json || true
```

### 4. Serve and inspect the demo site locally

```bash
cd src/demo_mfr_site/site
python3 -m http.server 8000
```

Open:

- `http://localhost:8000/`
- `http://localhost:8000/products/uct305.html`

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Classifier import/runtime error | Torch packages not installed | Install `torch` and `torchvision` in `.venv` |
| No Elasticsearch writes | Service not running | `docker-compose up -d` and rerun setup script |
| No MinIO uploads | Endpoint/credentials mismatch | Confirm `MINIO_*` env vars and MinIO health |
| No demo predictions | Missing model file | Ensure `src/Image_Classifier/trained_model.pth` exists |
| No scrape candidates returned | Source query mismatch for chosen products | Try alternate `--mfr-filter`, broader sample, or manufacturer flags |

# Pipeline Test Run Steps

Quick reference for running the full scrape → classify pipeline end-to-end in a test environment.
For full documentation see [docs/PIPELINE_RUNBOOK.md](docs/PIPELINE_RUNBOOK.md).

---

## Prerequisites

- `.venv` created and activated: `python3 -m venv .venv && source .venv/bin/activate`
- Dependencies installed: `pip install -r requirements.txt`
- PyTorch installed per your platform: https://pytorch.org/get-started/locally/
- Docker running (required for Elasticsearch + MinIO):
  ```bash
  docker-compose up -d
  ```
- Elasticsearch index exists (run once, or after mapping changes):
  ```bash
  .venv/bin/python src/web_scraping/setup_elasticsearch.py
  ```
- Trained model present at `src/Image_Classifier/trained_model.pth`

---

## Step 1 — Run Unit Tests

Verify nothing is broken before running the live pipeline.

```bash
.venv/bin/python -m pytest -q src/web_scraping/test_manufacturer_scrapers.py
```

Expected: `57 passed` with no failures.

---

## Step 2 — Scrape Images (API sources only)

Quickest smoke test — no external manufacturer sites, no infrastructure required.

In your root directory containing .venv/ and src/

```bash
.venv/bin/python src/web_scraping/web_scraper.py \
    --csv src/web_scraping/test_products_sample.csv \
    --limit 3
```

**Check:** `output/json/` contains `.json` files. At least one should have `candidate_images`
with `"downloaded": true`.

---

## Step 3 — Scrape with Tier 1 Manufacturer Sources

Use the tier 1 dataset (AMI Bearings + NTN approved scrapers) alongside standard API sources.

```bash
cd src/web_scraping
../.venv/bin/python web_scraper.py \
    --csv test_products_tier1_classifier.csv \
    --mfr-scraping \
    --limit 5
```

**Check:** JSON files for AMI and NTN products should have candidate images with
`"source_name": "Manufacturer Site"` (or similar). Verify at least one tier 1 image appears
alongside Wikimedia/OpenVerse results.

---

## Step 4 — Full Pipeline: Scrape + Classify in One Shot

Runs the scraper and immediately classifies downloaded images via the CNN.

```bash
cd src/web_scraping
../.venv/bin/python web_scraper.py \
    --csv test_products_tier1_classifier.csv \
    --mfr-scraping \
    --limit 5 \
    --classify
```

**Check:** Open any JSON in `output/json/`. Each downloaded image should have a
`"predicted_class"` field (integer 0–7).

---

## Step 5 — Full Pipeline with Elasticsearch + MinIO

Requires Docker services running from Step 0.

```bash
cd src/web_scraping
../.venv/bin/python web_scraper.py \
    --csv test_products_tier1_classifier.csv \
    --mfr-scraping \
    --limit 5 \
    --es \
    --minio \
    --classify
```

**Check infrastructure:**
```bash
# Products indexed
curl -s 'http://localhost:9200/mi_products/_count' | python3 -m json.tool

# Candidate images indexed
curl -s 'http://localhost:9200/mi_candidate_images/_count' | python3 -m json.tool

# Images uploaded to MinIO (via Kibana at http://localhost:5601 or mc CLI)
```

---

## Step 6 — Standalone Classification Pass (Post-Scrape)

If you scraped without `--classify`, run the classifier separately:

```bash
.venv/bin/python src/Image_Classifier/classify_json_images.py \
    --json-dir src/web_scraping/output/json \
    --model src/Image_Classifier/trained_model.pth
```

To also push predictions to Elasticsearch:

```bash
.venv/bin/python src/Image_Classifier/classify_json_images.py \
    --json-dir src/web_scraping/output/json \
    --model src/Image_Classifier/trained_model.pth \
    --es
```

---

## Quick Sanity Checks

```bash
# Count JSON files written this run
ls src/web_scraping/output/json/*.json | wc -l

# Show products that got at least one downloaded image
grep -l '"downloaded": true' src/web_scraping/output/json/*.json

# Show products that were classified
grep -l '"predicted_class"' src/web_scraping/output/json/*.json

# ES: check a specific product
curl -s 'http://localhost:9200/mi_products/_doc/z00000001?pretty'

# ES: list classified images for a product
curl -s -X GET 'http://localhost:9200/mi_candidate_images/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{"query": {"term": {"motion_product_id": "z00000001"}}, "size": 5}'
```

---

## PGC1 Class Reference (Classifier Output)

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

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| 0 tier 1 images found | `--mfr-scraping` flag missing | Re-run with `--mfr-scraping` |
| AMI/NTN returns 0 images | Sitemap unreachable or DDG throttled | Run without `--mfr-only`; API sources still produce results |
| `predicted_class` missing | `--classify` flag omitted or model not found | Add `--classify` or verify `trained_model.pth` exists |
| ES push fails | Elasticsearch not running | `docker-compose up -d elasticsearch` |
| MinIO upload fails | MinIO not running or bucket missing | `docker-compose up -d minio` |
| `57 passed` → fewer tests pass | Code drift vs. registry | Re-run test suite; check `MANUFACTURER_REGISTRY` in `manufacturer_scrapers.py` |

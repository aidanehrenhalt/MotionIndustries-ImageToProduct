# Motion Industries Image-to-Product

Repository for validating the full image-to-product pipeline:

- Web scraper
- Image classifier
- Text analysis / metadata ranker
- Final fused ranking

The main end-to-end entrypoint is [`src/web_scraping/web_scraper.py`](/home/aceaid/MotionIndustries-ImageToProduct/src/web_scraping/web_scraper.py). When you run it with `--classify`, it performs:

1. Scraping
2. CNN image classification
3. Text-based ranking
4. Final score + final rank assignment

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

## What This Repository Does

- Ingests product catalog rows from CSV or Elasticsearch
- Scrapes candidate images from Wikimedia, OpenVerse, and optional manufacturer sources
- Stores image binaries locally or in MinIO
- Stores product and candidate metadata in Elasticsearch
- Runs the CNN classifier on downloaded images
- Runs a text/metadata ranker on scraped candidates
- Combines both signals into a final ranking

## Documentation Map

- [`README.md`](/home/aceaid/MotionIndustries-ImageToProduct/README.md): quick start and end-to-end run instructions
- [`docs/PIPELINE_RUNBOOK.md`](/home/aceaid/MotionIndustries-ImageToProduct/docs/PIPELINE_RUNBOOK.md): canonical runbook
- [`docs/ELASTICSEARCH.md`](/home/aceaid/MotionIndustries-ImageToProduct/docs/ELASTICSEARCH.md): Elasticsearch mappings and query examples
- [`docs/MINIO.md`](/home/aceaid/MotionIndustries-ImageToProduct/docs/MINIO.md): MinIO object layout and verification
- [`docs/INTEGRATION_PLAN.md`](/home/aceaid/MotionIndustries-ImageToProduct/docs/INTEGRATION_PLAN.md): integration backlog and remaining work
- [`docs/architecture.md`](/home/aceaid/MotionIndustries-ImageToProduct/docs/architecture.md): architecture notes

## Notes

- The scraper currently generates search terms via `simple_search_keywords()`.
- `--classify` on [`web_scraper.py`](/home/aceaid/MotionIndustries-ImageToProduct/src/web_scraping/web_scraper.py) runs more than just the CNN. It also runs the text ranker and applies the final ranking pass.
- Standalone classification via [`classify_json_images.py`](/home/aceaid/MotionIndustries-ImageToProduct/src/Image_Classifier/classify_json_images.py) only performs the classifier pass unless you separately run the ranker.

## Useful Validation Test

There is an integration-style test covering the scraper/classifier boundary and ranker output fields:

```bash
.venv/bin/python -m pytest -q tests/test_scraper_classifier_pipeline.py
```

## Motion Shared Folder

[Image Dataset, Image Mapping, Images](https://genparts-my.sharepoint.com/:f:/r/personal/michael_flack_corp_motion-ind_com/Documents/GT%20Capstone?csf=1&web=1&e=s92NcQ)

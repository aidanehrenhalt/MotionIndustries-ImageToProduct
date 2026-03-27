# Image-to-Product Pipeline — Manual Runbook

End-to-end manual pipeline for scraping product images, assembling training data,
training the classifier, and running inference.

## Prerequisites

- Python 3.12+ with venv
- Dependencies: `pip install -r requirements.txt`
- PyTorch: install from https://pytorch.org/get-started/locally/ (CPU or CUDA)
- Playwright (for Tier 2 scraping): `playwright install chromium`
- Docker (optional, for Elasticsearch + MinIO): `docker-compose up -d`

## Pipeline Steps

### Step 1: Scrape Product Images

```bash
cd src/web_scraping

# Small test run (local storage, no ES/MinIO)
python web_scraper.py --csv test_products_sample.csv --limit 5

# From CSV with ES + MinIO
python web_scraper.py --csv test_products_sample.csv --limit 10 --es --minio

# From Elasticsearch with manufacturer filter
python web_scraper.py --from-es --mfr-filter SKF --limit 10 --es --minio
```

**Outputs:**
- `output/json/{product_id}_{timestamp}.json` — one per product
- `output/images/` — downloaded images (local mode)
- Elasticsearch `mi_products` + `mi_candidate_images` (if `--es`)
- MinIO `mi-images` bucket (if `--minio`)

**Verify:** Check that JSON files exist and some have `candidate_images` with `downloaded: true`.

---

### Step 2: Assemble Training Dataset

Builds a manifest CSV from scraper JSON outputs, resolving labels from the product catalog.

```bash
python src/Image_Classifier/assemble_dataset.py \
    --json-dir output/json \
    --csv ImageToProduct-Missing_Product_Images.csv \
    --output data/training_manifest.csv
```

**Optional flags:**
- `--excel Model_Development/cleaned_product_list.xlsx` — add Excel ground-truth labels (requires `openpyxl`)
- `--seed 42` — random seed for train/test split
- `--train-ratio 0.8` — fraction of data for training

**Outputs:** `data/training_manifest.csv` with columns:
`image_path, motion_product_id, pgc1, class_index, label_source, split`

**Verify:**
```bash
head -5 data/training_manifest.csv
wc -l data/training_manifest.csv
```
All `class_index` values should be 0–7. All `image_path` files should exist on disk.

**Note:** Images stored in MinIO (`storage_type == "minio"`) are skipped — only locally-stored images are included. Products not found in the CSV/Excel are also skipped.

---

### Step 3: Train the Model

```bash
# Smoke test (2 epochs)
python src/Image_Classifier/train.py \
    --manifest data/training_manifest.csv \
    --epochs 2

# Full training
python src/Image_Classifier/train.py \
    --manifest data/training_manifest.csv \
    --epochs 20

# Resume from checkpoint
python src/Image_Classifier/train.py \
    --manifest data/training_manifest.csv \
    --epochs 10 \
    --resume
```

**Optional flags:**
- `--model-out path/to/model.pth` — output path (default: `src/Image_Classifier/trained_model.pth`)
- `--checkpoint path/to/checkpoint.tar` — checkpoint path (default: `src/Image_Classifier/checkpoint.tar`)
- `--batch-size 32` — batch size
- `--lr 0.001` — learning rate

**Outputs:**
- `src/Image_Classifier/trained_model.pth` — model weights
- `src/Image_Classifier/checkpoint.tar` — full checkpoint (model + optimizer + epoch)

**Verify:** Training loss should decrease across epochs. Check that `trained_model.pth` has non-zero file size.

---

### Step 4: Classify Scraped Images (Inference)

```bash
# Classify and write predictions to JSON files
python src/Image_Classifier/classify_json_images.py \
    --json-dir output/json \
    --model src/Image_Classifier/trained_model.pth

# Also push predictions to Elasticsearch
python src/Image_Classifier/classify_json_images.py \
    --json-dir output/json \
    --model src/Image_Classifier/trained_model.pth \
    --es
```

**Optional flags:**
- `--es-host localhost` — Elasticsearch host
- `--es-port 9200` — Elasticsearch port

**Outputs:** Each candidate image in the JSON files gets a `predicted_class` field (0–7).

**Verify:** Open a JSON file and confirm `predicted_class` is present on candidate images.

---

### Step 5: Inspect Results

```bash
# Query Elasticsearch for classified images
cd src/web_scraping
python query_elasticsearch.py --images --stats

# Check specific product
python query_elasticsearch.py --product s10807860

# Verify MinIO ↔ ES consistency
python minio_es_match.py --verify
```

Or use **Kibana** at http://localhost:5601 to browse the `mi_candidate_images` index.

---

## One-Shot Pipeline (Scrape + Classify)

```bash
cd src/web_scraping
python web_scraper.py --csv test_products_sample.csv --limit 5 --classify
```

This scrapes images and immediately classifies them using the existing `trained_model.pth`. Add `--es` to also index results to Elasticsearch.

---

## PGC1 Class Reference

| Class Index | PGC1 | Description |
|-------------|------|-------------|
| 0 | 1 | BEARINGS |
| 1 | 2 | SEALS AND ACCESSORIES |
| 2 | 3 | POWER TRANSMISSION |
| 3 | 4 | ELECTRICAL & MAT'L HAND'G |
| 4 | 5 | HOSE AND FITTINGS |
| 5 | 6 | FLUID POWER |
| 6 | 7 | PROCESS PUMPS AND EQUIPMENT |
| 7 | 8 | INDUSTRIAL SUPPLIES |

PGC1=9 (MISCELLANEOUS) is excluded from the model.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| 0 samples assembled | No scraped images have labels in CSV | Ensure product IDs in JSON match the catalog CSV |
| 0 samples assembled | All images in MinIO | Re-scrape without `--minio` for local storage |
| Model won't load | Architecture mismatch | Ensure model was trained with the same `build_model()` architecture |
| `openpyxl not installed` warning | Missing optional dependency | `pip install openpyxl` (only needed for Excel label source) |
| ES push fails | Elasticsearch not running | `docker-compose up -d elasticsearch` |
| Training loss explodes | Learning rate too high | Try `--lr 0.0001` |

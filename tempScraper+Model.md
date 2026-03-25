## Quick-Start: Testing the Scraper → Model Pipeline

Run these steps from the project root to verify the full scrape-to-classify flow.

### 1. Scrape a few products (local image storage, no Docker required)

```bash
cd src/web_scraping
python web_scraper.py --csv test_products_sample.csv --limit 5
cd ../..
```

Confirm: `output/json/` has new JSON files and `output/images/` has downloaded images.

### 2. Assemble a training manifest

```bash
python src/Image_Classifier/assemble_dataset.py \
    --json-dir output/json \
    --csv src/web_scraping/test_products_sample.csv \
    --output data/training_manifest.csv
```

Confirm: `data/training_manifest.csv` exists with rows. Each row has `class_index` 0–7
and an `image_path` that points to an existing file.

> **Tip:** Use `--csv ImageToProduct-Missing_Product_Images.csv` instead if your scraped
> products come from the full catalog rather than the test sample.

### 3. Train the model (smoke test — 2 epochs)

```bash
python src/Image_Classifier/train.py \
    --manifest data/training_manifest.csv \
    --epochs 2
```

Confirm: prints train/test loss+accuracy each epoch; writes `src/Image_Classifier/trained_model.pth`.

### 4. Classify scraped images with the trained model

```bash
python src/Image_Classifier/classify_json_images.py \
    --json-dir output/json \
    --model src/Image_Classifier/trained_model.pth
```

Confirm: output says "classified N images" (N > 0). Open a JSON file in `output/json/` and
verify `predicted_class` (0–7) is present on candidate images.

### 5. (Optional) Push predictions to Elasticsearch

Requires Docker services running (`docker-compose up -d`).

```bash
python src/web_scraping/setup_elasticsearch.py
python src/Image_Classifier/classify_json_images.py \
    --json-dir output/json \
    --model src/Image_Classifier/trained_model.pth \
    --es
```

### One-shot shortcut (scrape + classify in one command)

```bash
cd src/web_scraping
python web_scraper.py --csv test_products_sample.csv --limit 5 --classify
```

---

## Plan: Integrate Training with WebScraper Branch

Build an MVP manual pipeline that connects scraping outputs to both model retraining and inference. Start by bringing WebScraper branch code into the current branch, define a stable scraped-data contract, then add dataset-build and training/inference entry points that use hybrid labels (Excel first, scraper/category fallback).

**Steps**
1. Phase 1 — Baseline and branch reconciliation
   1.1 Confirm branch topology and bring scraper source into current branch (merge/cherry-pick from WebScraper).
   1.2 Inventory concrete scraper entry points and outputs once imported (expected: web_scraper.py, setup_elasticsearch.py, minio_es_match.py).
   1.3 Pin MVP execution path as manual scripts/notebooks only (no scheduler).

2. Phase 2 — Define integration data contract (blocks 3 and 5)
   2.1 Freeze a canonical schema for product + candidate images consumed by ML from scraper artifacts (JSON and/or Elasticsearch + MinIO object keys).
   2.2 Define label resolution policy: Excel labels first; scraper-derived category mapping as fallback.
   2.3 Add data quality gates for training eligibility (image exists, loadable, minimum dimensions, accepted file format, resolved label).

3. Phase 3 — Build training dataset assembly from scraping outputs (depends on 2)
   3.1 Add a dataset assembly script/module that reads scraped metadata and materializes a training manifest (image_path/object_key, product_id, label_source, label_id, split).
   3.2 Join logic: motion_product_id key between scraped product docs and Excel mapping.
   3.3 Implement fallback labeling map (scraper category/PGC mapping) and mark provenance per row.
   3.4 Output reproducible artifacts for ML: merged_manifest and cleaned_manifest (train-ready rows only).

4. Phase 4 — Refactor training into reusable, runnable module (parallel with 5 after 2)
   4.1 Consolidate notebook/script divergence into one canonical training entry point.
   4.2 Fix current training inconsistencies (loader variable naming, label column normalization, checkpoint save/load signatures, architecture input-size assumptions).
   4.3 Parameterize paths and hyperparameters via CLI/config; keep defaults for current local environment.
   4.4 Emit consistent artifacts: model weights, optional checkpoint, metrics summary.

5. Phase 5 — Add inference pass for scraped candidates (depends on 2, can run parallel with 4 after model interface fixed)
   5.1 Create inference script that loads trained model and scores scraped candidate images.
   5.2 Persist predictions back to artifact store (JSON and/or Elasticsearch fields) with confidence and timestamp.
   5.3 Add simple ranking policy combining scraper preliminary score and model confidence for review prioritization.

6. Phase 6 — MVP orchestration and handoff (depends on 3,4,5)
   6.1 Document manual run sequence: scrape -> assemble dataset -> train -> infer -> inspect outputs.
   6.2 Add minimal validation commands and expected outputs per step.
   6.3 Record failure handling (missing files, no label fallback, empty scrape results) and safe skips.

**Relevant files**
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/Model_Development/training.py — Current training script; refactor into canonical training entry point.
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/Model_Development/training_notebook.ipynb — Reference for checkpoint/device flow and current model architecture.
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/Model_Development/filtering_images.py — Existing dataset cleaning pattern to reuse for manifest filtering.
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/Model_Development/class_analysis.py — Existing label-space analysis pattern for post-merge QA.
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/output/json/s10807860_20260318_125551.json — Example scraped artifact schema.
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/README.md — Scraper architecture and runbook currently documented.
- /home/aceaid/ECE4013/MotionIndustries-ImageToProduct/src/web_scraping/ (from WebScraper branch) — Scraper implementation expected to be merged before integration coding.

**Verification**
1. Branch import verification: scraper source files from WebScraper are present and runnable in current branch.
2. Contract verification: run dataset assembly on a small sample (3-10 products), confirm joined manifest row counts and label provenance distribution.
3. Training verification: run 1-2 epoch smoke training on assembled sample; ensure model and checkpoint artifacts are written.
4. Inference verification: score scraped candidate images for the same sample; confirm prediction fields are persisted and non-empty.
5. End-to-end MVP verification: execute documented manual sequence and validate expected outputs at each stage.

**Decisions**
- Include: both retraining from scraped data and inference on scraped candidates.
- Label policy: hybrid with Excel-preferred labels and scraper/category fallback.
- Execution model: manual MVP scripts/notebooks only.
- Exclude for now: production scheduler, UI expansion, large architecture redesign, distributed training.

**Further Considerations**
1. Prefer file-based MVP first (JSON/manifests) even if Elasticsearch/MinIO are enabled, to keep debugging deterministic.
2. Keep model architecture unchanged for first integration milestone; optimize only after pipeline stability is proven.
3. Treat zero-image scrape products as valid pipeline records but exclude from train-ready manifest.
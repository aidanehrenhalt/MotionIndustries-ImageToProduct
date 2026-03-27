# Scraper to ML Classification Pipeline Review

This document is a validation and findings record.

For canonical run instructions, use `docs/PIPELINE_RUNBOOK.md`.

## Scope

This review focuses on the handoff between:

- scraper ingestion in `src/web_scraping/`
- JSON artifact generation in `output/json/`
- Elasticsearch indexing in `mi_products` and `mi_candidate_images`
- CNN classification in `src/Image_Classifier/`

## What I validated

- Python syntax compilation for `src/web_scraping`, `src/Image_Classifier`, and `Model_Development`
- Runtime import availability for key dependencies in the current environment
- Static contract review between scraper output fields and classifier input expectations
- Static contract review between classifier output and Elasticsearch mappings
- Existing repository artifacts under `output/json/` and `src/web_scraping/output/json/`

## Environment limits during this review

- `torch` and `torchvision` are not installed in the current environment, so the classifier could not be executed here
- `pytest` is not installed in the current environment, so repository tests were not executed here
- Docker services and live network scraping were not exercised in this pass

## Summary

The repository structure is generally coherent, and the JSON schema, classifier output shape, and Elasticsearch mapping for `predicted_class` are aligned. The main integration problem is that the documented "full pipeline" path does not actually work when images are stored in MinIO: the classifier only reads local files and explicitly skips MinIO-backed images.

That means the current scraper and classifier only work end-to-end if images are stored locally, not when `--minio` is used.

## Findings

### 1. Critical: `--minio --classify` is currently non-functional

Files:

- `src/web_scraping/web_scraper.py:460`
- `src/web_scraping/web_scraper.py:936`
- `src/Image_Classifier/classify_json_images.py:111`
- `README.md:628`

Details:

- When the scraper runs with `--minio`, `download_image()` stores the image as a MinIO object key and sets `storage_type` to `"minio"`.
- The classifier explicitly skips any image whose `storage_type == "minio"` because it only supports local filesystem paths.
- The scraper still advertises `--from-es --mfr-filter SKF --es --minio --classify` as a "Full pipeline with classification", and the README examples strongly imply MinIO plus full-pipeline support.

Impact:

- Classification silently does no useful work for MinIO-backed scrape runs.
- `predicted_class` will never be added for those images unless they are first downloaded back to local disk or classification is changed to read from MinIO.

Artifact evidence from this repository:

- `output/json/` currently contains 33 candidate images: 29 are `minio`, 4 are `null`, 0 are `local`
- `src/web_scraping/output/json/` currently contains 82 candidate images: 50 are `minio`, 32 are `null`, 0 are `local`
- No `predicted_class` fields were found in either JSON artifact set

Recommendation:

- Either remove/support-limit the `--minio --classify` path in docs and CLI help, or implement one of these:
- classify directly from MinIO bytes
- download MinIO objects to a temp directory before classification
- store both a local cache path and the MinIO object key

### 2. High: post-scrape classification is not scoped to the current run

Files:

- `src/web_scraping/web_scraper.py:903`
- `src/web_scraping/web_scraper.py:956`
- `src/web_scraping/web_scraper.py:959`

Details:

- The scraper tracks `saved_files`, but never uses it.
- After scraping, `classify_json_dir(JSON_DIR, _model_path)` processes every JSON file in `output/json`, not only the files created in the current run.
- `push_predictions_to_es(JSON_DIR, es)` does the same for Elasticsearch updates.

Impact:

- Re-running a small test can unexpectedly classify stale artifacts from previous runs.
- Elasticsearch updates can include historical JSON files unrelated to the current scrape invocation.
- This makes validation noisy and can hide regressions.

Recommendation:

- Restrict classification and ES sync to the `saved_files` list from the current run, or write run-specific subdirectories.

### 3. Medium: Elasticsearch update failures for predictions are hidden

File:

- `src/web_scraping/web_scraper.py:770`

Details:

- `push_predictions_to_es()` catches all exceptions and drops them silently.
- The comment mentions 404 or mapping issues, but the code suppresses everything, including connection errors and schema drift.

Impact:

- A broken classification-to-ES sync can appear successful.
- This is especially risky if an older Elasticsearch index was created before `predicted_class` was added to the mapping.

Recommendation:

- Log skipped updates with the document id and exception class.
- Treat mapping errors and connection failures as warnings at minimum.

### 4. Medium: CSV ingestion likely loses full-catalog fields needed for good scraping

Files:

- `src/web_scraping/web_scraper.py:159`
- `src/web_scraping/web_scraper.py:173`
- `src/web_scraping/web_scraper.py:180`
- `src/web_scraping/web_scraper.py:182`

Details:

- The loader comment says it handles the full dataset with uppercase column names.
- In practice, uppercase handling is only implemented for some fields.
- `mfr_part_number`, `item_number`, `internal_description`, and `primary_image_filename` are only read from title-case keys.

Impact:

- If the real catalog uses uppercase names for those fields, `mfr_part_number` may be blank after ingestion.
- That directly degrades search keyword generation and any manufacturer-specific scraping keyed on part number.
- The classifier itself is not harmed by this directly, but the upstream candidate-image quality drops.

Recommendation:

- Normalize all expected CSV aliases consistently, especially `MFR_PART_NUMBER`, `ITEM_NUMBER`, `MOTION_INTERNAL_DESCRIPTION`, and `PRIMARYIMAGEFILENAME` if those exist in the real export.

### 5. Medium: the repository does not include a runnable automated test for the scraper-to-classifier handoff

Files:

- `src/web_scraping/test_manufacturer_scrapers.py`
- `requirements.txt`

Details:

- Existing automated tests are focused on manufacturer scraping utilities, not the scraper-to-JSON-to-classifier pipeline.
- `pytest` is not listed in `requirements.txt`.

Impact:

- The most important integration boundary in this review is currently unguarded.
- Regressions like the MinIO/classifier mismatch can persist unnoticed.

Recommendation:

- Add one local-storage integration test that:
- writes a minimal JSON record with one local image path
- runs classification
- verifies `predicted_class` is written
- optionally verifies ES sync against a test index

## Positive notes

- The classifier’s local-path resolution is sensible for root-relative `output/images/...` paths
- `predicted_class` is already present in the Elasticsearch mapping, so the schema is prepared for classification output
- Candidate image Elasticsearch document IDs are deterministic via `SHA1(product_id:image_url)`, which is a good choice for upserts and later synchronization
- The code separates scraping, persistence, and classification concerns reasonably well, which should make fixes contained

## Recommended priority order

1. Fix the MinIO/classifier mismatch or explicitly document local-only classification.
2. Scope post-scrape classification to the current run only.
3. Stop swallowing Elasticsearch prediction update errors.
4. Harden CSV field alias handling for the real catalog format.
5. Add one integration test covering scraper artifact to classifier output.

## Start-to-finish test instructions

These steps reflect the current repository behavior accurately.

### A. Local storage path: the only current end-to-end route for classification

1. Create and activate a virtual environment.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install pytest
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. Start infrastructure services.

```bash
docker-compose up -d
```

3. Create Elasticsearch indices.

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

4. Run a small scrape with local image storage and classification enabled.

Important:

- Do not use `--minio` for this test.
- This is the current code path that can actually classify images end-to-end.

```bash
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 2 \
  --es \
  --classify
```

5. Verify that JSON output now contains `predicted_class`.

```bash
rg -n "predicted_class" output/json
```

6. Verify Elasticsearch received the predictions.

```bash
curl -X GET 'http://localhost:9200/mi_candidate_images/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "exists": { "field": "predicted_class" }
    },
    "size": 10
  }'
```

7. Run existing scraper unit tests once `pytest` is installed.

```bash
venv/bin/python -m pytest -q src/web_scraping/test_manufacturer_scrapers.py
```

### B. MinIO path: valid for storage/indexing, not valid for classification in the current code

1. Run the scraper with MinIO enabled.

```bash
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 2 \
  --es \
  --minio
```

2. Verify storage and metadata consistency.

```bash
venv/bin/python src/web_scraping/minio_es_match.py --verify
```

3. If you want to classify these MinIO-backed images with the repository as it exists today, first download them back to local disk.

```bash
venv/bin/python src/web_scraping/minio_es_match.py --download s10807860
```

Important:

- This download command does not automatically rewrite the JSON `local_path` fields to point at the downloaded files.
- Because of that, classification still will not work end-to-end for MinIO runs without either:
- a code change
- or manual JSON path rewriting

## Suggested acceptance criteria for this pipeline

- Scraper writes candidate images with either a valid local path or a resolvable MinIO reference
- Classifier can process every downloaded image for the selected run
- `predicted_class` is written into the JSON artifact for each successfully classified image
- `predicted_class` is synchronized into `mi_candidate_images`
- Running the pipeline twice does not reprocess unrelated historical artifacts unless explicitly requested

## Bottom line

The codebase is close to a workable pipeline, but the current scraper-to-ML integration is only truly functional for local image storage. The advertised MinIO-backed classification flow is not implemented.

---

## Revalidation Results

This section records the follow-up validation after the repository was patched.

### Status

- The scraper-to-ML pipeline segment is now functioning for MinIO-backed images
- The new integration tests for the scraper/classifier boundary pass
- Docker-backed infrastructure validation succeeded
- One environment compatibility issue was found and corrected during validation: `numpy` had to be pinned below `2.0` for the current `torch` / `torchvision` stack
- Some manufacturer-scraper tests still fail, but those failures are outside the scraper-to-ML classification handoff

### What I ran

Infrastructure:

- Verified Docker containers were running and healthy:
- `mi_elasticsearch`
- `mi_kibana`
- `mi_minio`

Environment setup:

- Installed `torch` and `torchvision` into the project `venv`
- Installed a PyTorch-compatible NumPy version:

```bash
venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install 'numpy<2'
```

Schema and ingest:

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
venv/bin/python src/web_scraping/ingest_catalog.py --csv src/web_scraping/test_products_sample.csv
```

Automated tests:

```bash
venv/bin/python -m pytest -q tests/test_scraper_classifier_pipeline.py
venv/bin/python -m pytest -q src/web_scraping/test_manufacturer_scrapers.py
```

Live pipeline run:

```bash
venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --limit 2 \
  --es \
  --minio \
  --classify
```

Positive-path MinIO-backed classification validation:

- Performed a focused validation using an existing JSON artifact and existing MinIO object already present in the repository environment
- Re-indexed that candidate image into `mi_candidate_images`
- Ran `classify_json_files(...)` against the MinIO-backed JSON
- Ran `push_predictions_to_es(...)`
- Verified `predicted_class` existed both in the JSON artifact and in Elasticsearch

### Results

#### 1. New scraper/classifier integration tests passed

Command:

```bash
venv/bin/python -m pytest -q tests/test_scraper_classifier_pipeline.py
```

Result:

- `5 passed`

This confirms:

- local-file classification works
- MinIO-backed classification works
- ES sync writes `predicted_class`
- classification is scoped to the supplied file list
- missing MinIO objects warn and do not crash the run

#### 2. Docker-backed infrastructure validation passed

Observed state:

- Elasticsearch was reachable on `http://localhost:9200`
- MinIO was reachable on `http://localhost:9000`
- Kibana was running on `http://localhost:5601`

#### 3. Elasticsearch schema recreation passed

Command:

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

Result:

- `mi_products` recreated successfully
- `mi_candidate_images` recreated successfully

#### 4. Catalog ingestion succeeded

Command:

```bash
venv/bin/python src/web_scraping/ingest_catalog.py --csv src/web_scraping/test_products_sample.csv
```

Observed behavior:

- Bulk ingest reported success for 20 products
- The script’s immediate final count log reported `0`, but a direct Elasticsearch count right after returned `20`

Interpretation:

- Ingestion itself succeeded
- The final count line appears to be affected by Elasticsearch refresh timing, not by failed writes

#### 5. Live scraper run completed cleanly, but the selected products returned zero candidates

Command:

```bash
venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --limit 2 \
  --es \
  --minio \
  --classify
```

Observed behavior:

- the scraper ran successfully
- JSON files were written
- Elasticsearch updates completed
- classification ran without crashing
- those two products returned `0` candidate images from Wikimedia/OpenVerse, so `0` images were classified in that specific live run

Interpretation:

- This run validated the no-result path and the patched control flow
- It did not by itself prove the positive classification path because the upstream image sources returned no candidates for those two products during this run

#### 6. Positive-path MinIO-backed classification was validated successfully

Using an existing MinIO-backed JSON/image pair from the repository’s stored artifacts:

- `classify_json_files(...)` classified the image successfully
- `predicted_class` was written to the JSON artifact
- `push_predictions_to_es(...)` updated the matching `mi_candidate_images` document

Observed result:

- `classified_count = 1`
- `predicted_class = 0` in JSON
- `predicted_class = 0` in Elasticsearch
- Elasticsearch document retained:
- `storage_type = "minio"`
- `local_path = images/s10807860/s10807860_01_b2ed562c.jpg`

This is the key confirmation that the patched MinIO-to-classifier-to-Elasticsearch handoff now works.

### Additional fix applied during validation

File updated:

- `requirements.txt`

Change:

- added `numpy<2`

Reason:

- With the installed `torch` / `torchvision` stack, `numpy 2.x` caused classification failures during test execution:
- `Could not classify ...: Numpy is not available`

This is now documented in dependencies so a clean setup reproduces the working environment.

### Remaining issues

#### Manufacturer scraper tests still have 3 failures

Command:

```bash
venv/bin/python -m pytest -q src/web_scraping/test_manufacturer_scrapers.py
```

Result:

- `3 failed, 54 passed`

Failure scope:

- these failures are in the manufacturer scraping module, not in the scraper-to-ML classification handoff
- the tests still expect `renderer == "requests"` for AMI and NTN
- the current implementation uses `renderer == "ddg_item"`
- one AMI test also mocks the wrong fetch path for the current implementation strategy

Interpretation:

- the failing tests reflect implementation/test drift in the manufacturer scraper subsystem
- they do not invalidate the ML classification pipeline changes validated above

#### MinIO ↔ Elasticsearch consistency currently shows historical orphaned MinIO objects

Command:

```bash
venv/bin/python src/web_scraping/minio_es_match.py --verify
```

Observed behavior:

- one actively indexed validation object matched between ES and MinIO
- many older MinIO objects exist without corresponding current ES documents

Interpretation:

- this is expected after recreating Elasticsearch indices while leaving MinIO bucket contents intact
- it is not a failure of the current classifier patch

### Current conclusion

The scraper ingestion to ML classification pipeline segment has been successfully revalidated after patching.

Working now:

- MinIO-backed image classification
- scoped classification to current-run file lists
- scoped ES sync for `predicted_class`
- JSON artifact updates with `predicted_class`
- Elasticsearch updates with `predicted_class`
- pytest coverage for the integration boundary

Still outstanding, but separate from this pipeline segment:

- manufacturer scraper test drift
- README/docs should still be checked for full alignment with the implemented behavior

## Next steps to patch the repository

These are the concrete changes needed so the full scraper -> storage -> classification -> Elasticsearch path works end-to-end.

### 1. Make classification work for MinIO-backed images

Required code changes:

- Update `src/Image_Classifier/classify_json_images.py` so it can classify from either:
- a local filesystem path
- or a MinIO object key by fetching image bytes through `boto3`

Recommended implementation:

- Add a helper like `load_image_bytes_from_minio(object_key: str, s3_client) -> PIL.Image`
- Initialize an S3 client inside the classifier using the same `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, and `MINIO_BUCKET` environment variables already used by the scraper
- When `storage_type == "minio"`, fetch the object from MinIO and classify it directly from memory instead of skipping it

Alternative acceptable implementation:

- Download MinIO objects to a temp directory before classification
- Classify the temp files
- Remove temp files after processing

### 2. Scope classification to only the current scrape run

Required code changes:

- Update `src/web_scraping/web_scraper.py`
- Use the existing `saved_files` list
- Replace `classify_json_dir(JSON_DIR, _model_path)` with a function that classifies only those newly created JSON files
- Replace `push_predictions_to_es(JSON_DIR, es)` with a function that syncs predictions only from those same files

Recommended implementation:

- Refactor `classify_json_images.py` to expose `classify_json_files(json_files: list[Path], model_path: Path) -> int`
- Refactor `push_predictions_to_es()` to accept `json_files: list[Path]`

### 3. Stop hiding Elasticsearch sync failures

Required code changes:

- Update `src/web_scraping/web_scraper.py`
- Narrow the `except Exception` block in `push_predictions_to_es()`
- Log at least:
- document id
- image url
- exception type
- exception message

Recommended behavior:

- Ignore only 404 if that is intentional
- Warn on mapping conflicts
- Error on connection failures

### 4. Harden CSV ingestion for the real catalog format

Required code changes:

- Update `src/web_scraping/web_scraper.py`
- Normalize all supported input column aliases in `load_product_catalog()`

At minimum, support both title-case and uppercase variants for:

- `item_number`
- `mfr_part_number`
- `internal_description`
- `primary_image_filename`

This is needed because upstream search quality depends heavily on part number and description quality.

### 5. Add automated tests for the integration boundary

Required code changes:

- Add pytest-based tests for the scraper/classifier handoff
- Add at least one test file, for example:
- `tests/test_scraper_classifier_pipeline.py`

Recommended test coverage:

- local image path classification writes `predicted_class`
- MinIO-backed classification writes `predicted_class`
- ES sync updates `mi_candidate_images`
- classification only processes current-run JSON files
- missing image objects produce warnings but do not crash the run

Implementation guidance:

- Use `tmp_path` for temporary JSON and image fixtures
- Mock `boto3` MinIO reads where practical
- Mock ES client updates for unit-level tests
- Keep one optional Docker-backed integration check separate from the default test suite

### 6. Add pytest to the project’s developer setup

Required code changes:

- Add `pytest` to `requirements.txt` or create a separate `requirements-dev.txt`

Recommended minimum:

```text
pytest>=8.0
```

### 7. Make PyTorch installation explicit and reproducible

Current state:

- `requirements.txt` lists `torch` and `torchvision`
- actual installation often depends on CPU vs CUDA wheels and platform compatibility

Recommended patch:

- Keep `torch` and `torchvision` documented separately from generic requirements
- Add a short setup section in the README for CPU-only local validation

Recommended commands:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install pytest
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If the team uses Apple Silicon, Linux CUDA, or another target, document the exact supported install commands per platform.

### 8. Validate the Docker-backed full pipeline segment

The repository should support this full sequence reliably:

1. Docker launches Elasticsearch, Kibana, and MinIO
2. catalog ingestion loads products into `mi_products`
3. scraper fetches candidate images
4. images are stored in MinIO
5. classifier reads those MinIO-backed images
6. `predicted_class` is written to JSON
7. `predicted_class` is synced into `mi_candidate_images`

Required validation steps after patching:

```bash
docker-compose up -d
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
venv/bin/python src/web_scraping/ingest_catalog.py --csv src/web_scraping/test_products_sample.csv
venv/bin/python src/web_scraping/web_scraper.py \
  --from-es \
  --limit 2 \
  --es \
  --minio \
  --classify
```

Expected results:

- JSON files appear in `output/json/`
- candidate images are present in MinIO
- `predicted_class` appears in the new JSON files
- `predicted_class` appears in `mi_candidate_images`

Verification commands:

```bash
rg -n "predicted_class" output/json
venv/bin/python src/web_scraping/minio_es_match.py --verify
curl -X GET 'http://localhost:9200/mi_candidate_images/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "exists": { "field": "predicted_class" }
    },
    "size": 10
  }'
```

### 9. Add one repeatable “full pipeline” test section to the README

The README should explicitly distinguish:

- local-storage classification flow
- MinIO-backed classification flow

It should also stop implying that `--minio --classify` works until the MinIO classification patch is in place.

## Proposed patch sequence

Implement in this order:

1. Patch classifier support for MinIO-backed images.
2. Scope classification and ES sync to current-run files only.
3. Improve ES error handling and logging.
4. Fix catalog field alias handling.
5. Add pytest-based tests.
6. Update README with exact PyTorch, pytest, Docker, and full-pipeline commands.

## Definition of done

This pipeline segment should be considered fixed when all of the following are true:

- `docker-compose up -d` brings up Elasticsearch, Kibana, and MinIO successfully
- `setup_elasticsearch.py --recreate` succeeds
- `ingest_catalog.py` loads the sample catalog without schema errors
- `web_scraper.py --from-es --es --minio --classify --limit 2` completes successfully
- the new JSON output files contain `predicted_class`
- `mi_candidate_images` contains `predicted_class`
- `pytest` passes for both unit tests and the scraper/classifier integration tests
- the README instructions reproduce the flow on a clean machine

---

## Files Changed

### `src/Image_Classifier/classify_json_images.py`

- Added `import io`, `import os`
- Added `_get_s3_client()` — builds a boto3 S3 client from the same `MINIO_*` env vars used by `web_scraper.py`
- Added `classify_image_from_bytes(model, img_bytes, preprocess)` — classifies from raw bytes in-memory (no temp files needed)
- Added `classify_json_files(json_files, model_path, s3_client=None)` — new scoped public API; accepts a specific list of `Path` objects and handles both local-filesystem and MinIO-backed images
- Updated `classify_json_dir()` docstring to note it is the legacy standalone path, skips MinIO, and that `classify_json_files()` is preferred for programmatic use

### `src/web_scraping/web_scraper.py`

- Added `NotFoundError` and `ConnectionError as ESConnectionError` to the `from elasticsearch import ...` line
- `push_predictions_to_es(json_dir, es)` refactored to `push_predictions_to_es(json_files, es)` — parameter changed from a directory `Path` to a `list` of specific file paths to prevent reprocessing historical artifacts
- Post-scrape classification call changed from `classify_json_dir(JSON_DIR, _model_path)` to `classify_json_files(saved_files, _model_path, s3_client=s3)` — scoped to current-run files and MinIO-aware
- Post-scrape ES sync call changed from `push_predictions_to_es(JSON_DIR, es)` to `push_predictions_to_es(saved_files, es)`
- Replaced silent `except Exception: pass` in `push_predictions_to_es` with three-tier logging: `debug` for 404 (stale docs), `error` for connection failures, `warning` for all other exceptions (includes doc id, product id, image url, exception type and message)
- Added uppercase column name fallbacks in `load_product_catalog()` for `item_number` (`ITEM_NUMBER`), `mfr_part_number` (`MFR_PART_NUMBER`), `internal_description` (`MOTION_INTERNAL_DESCRIPTION`), and `primary_image_filename` (`PRIMARYIMAGEFILENAME`)

### `tests/test_scraper_classifier_pipeline.py` _(new file)_

Five pytest tests covering the scraper-to-classifier integration boundary:

1. Local image classification writes `predicted_class`
2. MinIO-backed classification writes `predicted_class` (boto3 mocked)
3. ES sync calls `es.update` with the correct SHA1 doc id and `predicted_class` value (ES mocked)
4. Classification is scoped to `saved_files` only — a file not in the list is not modified
5. A missing MinIO object (`NoSuchKey`) logs a warning and does not crash the run

### `requirements.txt`

- Added `pytest>=8.0` under a new `# Development / Testing` section

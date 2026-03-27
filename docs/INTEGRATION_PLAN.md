# Integration Plan

Single reference for all planned and in-progress integration work on the Image-to-Product pipeline.

---

## Architecture Overview

- **Elasticsearch** is the metadata system of record for products and candidate-image metadata.
- **Object storage** is the binary-image store:
  - MinIO for local/dev runs
  - GCP Cloud Storage for production (see [GCP Migration](#gcp-migration))
- `mi_products` and `mi_candidate_images` hold searchable metadata and classifier outputs.
- Image bytes are stored as object payloads; object keys are referenced in metadata (`local_path` for MinIO-backed records).

---

## Already In Place

- Elasticsearch and MinIO are wired into the scraper.
- `mi_products` and `mi_candidate_images` mappings include `predicted_class`.
- `web_scraper.py --classify` classifies the current scrape run and syncs predictions to Elasticsearch.
- MinIO-backed classification is supported via `classify_json_files()`.
- `src/web_scraping/text_based_search.py` is present.
- `product_to_vector()` exists in `src/web_scraping/web_scraper.py`.
- 5 integration tests pass for the scraper ↔ classifier boundary (`tests/test_scraper_classifier_pipeline.py`).

---

## Remaining Work

### 1. Replace the Temporary Search Query Builder

Current code still uses `simple_search_keywords()` in:

- CSV load: `load_product_catalog()`
- Elasticsearch load: `load_products_from_es()`

Change both paths to `vector_to_query(product_to_vector(product))`. Keep `search_keywords` as a list so the downstream scraper flow does not change. The bridge function and query builder are already present — this is a call-site swap only.

Reference field mapping:

| Scraper dict key | `ProductVector` field |
|---|---|
| `motion_product_id` | `id_number` |
| `item_number` | `item_number` |
| `enterprise_name` | `enterprise_number` |
| `mfr_name` | `manufacture_name` |
| `mfr_part_number` | `manufacture_part_number` |
| `web_desc` | `web_product_description` |
| `internal_description` | `motion_internal_desc` |
| `pgc` | `pgc` |
| `category` | `pgc_description` |

---

### 2. Image Ranking

After classification, each candidate image has `predicted_class` (0–7 or -1) stored in JSON and ES, but no re-ranking occurs. Images remain sorted by `preliminary_score`, a heuristic computed at scrape time that does not use classifier output.

#### Goal

Produce a `final_score` and `final_rank` for each candidate image combining:

- Classifier confidence (softmax probability of the predicted class)
- Whether the predicted class matches the product's expected category
- The existing preliminary heuristic score

#### Phase 1 — Expose classifier confidence

**File:** `src/Image_Classifier/classify_json_images.py` → `classify_image()`

Change the return type from `int` to `dict`:

```python
probs = torch.nn.functional.softmax(output, dim=1)
confidence, predicted = torch.max(probs, 1)
return {
    "predicted_class": int(predicted.item()),
    "classifier_confidence": round(float(confidence.item()), 4),
}
```

The failure path returns `{"predicted_class": -1, "classifier_confidence": 0.0}`. Update all call sites that currently expect an `int`.

#### Phase 2 — Map products to expected classes

**File:** `src/Image_Classifier/pgc_mapping.py` (exists, 53 lines)

Add `get_expected_classes(product: dict) -> set[int]` — returns the set of class indices that would be correct for the product's PGC code. Returns an empty set if the PGC is unmapped (unknown categories must not be penalized in scoring).

#### Phase 3 — Compute final score

Add `compute_final_score(candidate, expected_classes, w_confidence=0.4, w_class_match=0.3, w_preliminary=0.3) -> float` to `classify_json_images.py`.

Formula:

```
final_score = 0.4 * classifier_confidence
            + 0.3 * class_match        # 1.0 if predicted_class ∈ expected_classes, else 0.0
            + 0.3 * preliminary_score
```

When `expected_classes` is empty (unknown PGC), redistribute the 0.3 class-match weight equally between the other two signals so the score remains on a 0–1 scale.

Weights are a starting point — see open questions below before using on production data.

#### Phase 4 — Re-rank after classification

Add `rank_candidates(product_data: dict) -> dict` to `classify_json_images.py`. Call it inside `classify_json_files()` after the classification loop and before writing JSON back to disk. The function assigns `final_score` and `final_rank` (1-based integer) to each candidate, then sorts the list by `final_score` descending.

#### Phase 5 — Schema updates

New fields added to each item in the `candidate_images` array:

| Field | Type | Set by | Null before ranking |
|---|---|---|---|
| `classifier_confidence` | float 0–1 | Phase 1 | yes |
| `final_score` | float 0–1 | Phase 3 | yes |
| `final_rank` | int ≥ 1 | Phase 4 | yes |

**`setup_elasticsearch.py`** — add to `IMAGES_MAPPINGS["properties"]`:

```python
"classifier_confidence": {"type": "float"},
"final_score":           {"type": "float"},
"final_rank":            {"type": "integer"},
```

Existing deployments must run `python setup_elasticsearch.py --recreate`.

#### Phase 6 — Extend ES sync

**`web_scraper.py`** → `push_predictions_to_es()`: sync `classifier_confidence`, `final_score`, and `final_rank` alongside `predicted_class`.

#### Phase 7 — Demo pipeline

**`src/demo_mfr_site/run_demo_pipeline.py`**: ranking is implicit once Phase 4 is inside `classify_json_files()`. Add a log line confirming final ranks were assigned.

#### Rollout order

Phases 1–4 can be developed and tested against local JSON files without touching Elasticsearch. Phase 5 (schema change) should only run once Phases 1–4 are stable.

#### Open questions

- **PGC mapping coverage**: before weighting class-match heavily, verify what fraction of products in the test CSV have a PGC that maps to a known class. If coverage is below 50%, reduce `w_class_match`.
- **Weight calibration**: 0.4 / 0.3 / 0.3 is a starting guess. Spot-check `final_rank == 1` images from demo pipeline runs against the expected product image. Adjust before using on production data.
- **Model calibration**: if the CNN produces high confidence on wrong classes, reduce `w_confidence` and treat it as a tiebreaker until the model is re-evaluated on held-out data.
- **Rodrigo collaboration**: confidence scoring and weight calibration were flagged as a task to coordinate with Rodrigo. Loop him in before finalizing.

---

### 3. Validate End-To-End Behavior

After ranking integration:

1. Run a small CSV scrape: `web_scraper.py --csv ... --es --classify`
2. Run a `--from-es` scrape: `web_scraper.py --from-es --es --minio --classify`
3. Confirm JSON output, Elasticsearch documents, and MinIO objects are consistent
4. Verify `final_rank == 1` images look correct for a sample of products
5. Compare ranking output against the old heuristic ordering

---

## Tweaks

Minor issues that do not block the pipeline but should be addressed.

### Manufacturer scraper test drift

**File:** `src/web_scraping/test_manufacturer_scrapers.py`

Three tests fail because they assert `renderer == "requests"` for AMI and NTN scrapers, but the current implementation uses `renderer == "ddg_item"`. One AMI test also mocks the wrong fetch path for the current strategy. These are test/implementation drift issues and do not affect the ML classification pipeline.

Fix: update the three affected test cases to match the current `ddg_item` renderer strategy.

### `ingest_catalog.py` final count log

The script logs `0 products ingested` on its final line after a successful bulk ingest because it reads the count before Elasticsearch has refreshed. This is cosmetic — the data is actually written.

Fix: add `?refresh=true` to the bulk request, or remove the final count log line.

### PyTorch install documentation

`requirements.txt` now pins `numpy<2`. Document the full CPU-install sequence explicitly in the README setup section to avoid platform-specific failures:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'numpy<2'
```

Note that CUDA or Apple Silicon targets need different install commands.

---

## GCP Migration

When local development is ready for production deployment, the following GCP resources and permissions are needed.

Contact: anu.shrestha@motion.com, george.baldwin@motion.com

### Services

| Service | Purpose |
|---|---|
| Cloud Storage | Replaces MinIO for image/object storage |
| Secret Manager | API keys, ES credentials, scraper credentials |
| Cloud Run | Containerized pipeline services and APIs |
| Artifact Registry | Docker image storage |
| Cloud Logging | Debugging and audit trails |
| Pub/Sub | Event-based triggers (e.g. request images for a product without manual scraper invocation) |
| Vertex AI | Hosting the image classifier and text models |
| Cloud Scheduler | Scheduled re-scrapes for aging or low-quality images |

### IAM — User

- Cloud Run Developer
- Artifact Registry Reader/Writer
- Storage Object Admin (project-specific bucket)
- Secret Manager Secret Accessor
- Logs Viewer
- Service Account User
- Vertex AI User

### IAM — Pipeline Service Account

- Read/Write to image storage bucket
- Read access to secrets
- Write logs
- Call Vertex AI endpoints
- Publish to Pub/Sub

### Migration checklist

- [ ] Email Motion contacts to create dedicated GCP project and grant access
- [ ] Provision service account with above IAM roles
- [ ] Replace `MINIO_*` env vars with Cloud Storage equivalents in scraper and classifier
- [ ] Confirm storage client global variable refactor (commit `9a56fe9`) covers the classifier path
- [ ] Decide whether Elasticsearch stays self-hosted or moves to a managed equivalent

---

## Out Of Scope

- New storage backends beyond MinIO / GCP Cloud Storage
- UI work
- Model retraining

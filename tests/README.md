# Tests

Integration and unit tests for the scraper → classifier pipeline and the API upload parsing layer.

## Running Tests

```bash
# Run all tests
pytest -q

# Run a specific file
pytest -q tests/test_scraper_classifier_pipeline.py
pytest -q tests/test_api_upload_parsing.py
```

## Dependencies

```bash
pip install pytest pillow torch torchvision botocore flask
```

---

## `test_scraper_classifier_pipeline.py`

Integration tests for the boundary between the web scraper and the CNN classifier. All external services (MinIO, Elasticsearch, PyTorch model) are mocked.

### What's Tested

| # | Test | Description |
|---|------|-------------|
| 1 | `test_local_classification_writes_predicted_class` | Local image is classified and `predicted_class` is written back to the JSON record |
| 2 | `test_minio_classification_writes_predicted_class` | MinIO-backed image is fetched via mocked boto3 and `predicted_class` is written correctly |
| 3 | `test_es_sync_updates_candidate_images` | `push_predictions_to_es` calls ES update with the correct SHA1 document ID and `predicted_class` value |
| 4 | `test_classification_scoped_to_saved_files_only` | Only files passed to `classify_json_files` are modified; other JSON files on disk are untouched |
| 5 | `test_missing_minio_object_warns_and_continues` | A missing MinIO key (`NoSuchKey`) logs a warning and does not crash the run; no `predicted_class` is written |
| 6 | `test_classifier_confidence_is_written` | `classifier_confidence` (float in `[0.0, 1.0]`) is written alongside `predicted_class` |
| 7 | `test_rank_json_files_writes_ranker_fields` | `rank_json_files` writes `ranker_score`, `score_pct`, and `score_breakdown` to every candidate |
| 8 | `test_apply_final_ranking_creates_deterministic_order` | `apply_final_ranking` assigns 1-based `final_rank` with rank 1 = highest `final_score` |
| 9 | `test_compute_final_score_pgc_known_vs_unknown` | PGC-known path applies `class_match` weight correctly; PGC-unknown redistributes weight and stays in `[0, 1]` |
| 10 | `test_rank_json_files_synthesizes_missing_index` | Candidates missing an `index` field get a synthesized one; ranker fields still land on the correct candidate and a warning is logged |

### Helpers

`_make_jpeg_bytes()` — generates a minimal valid 500×500 JPEG in memory.

`_write_local_record()` — writes a local image file and a matching JSON record to a temp directory.

`_mock_model()` — returns a `MagicMock` that mimics a trained PyTorch model, with a configurable winning class index.

### Final Score Formula (Test 9)

When PGC is known:

```
final_score = 0.3 × classifier_confidence
            + 0.2 × class_match          (1.0 if predicted_class in expected_classes, else 0.0)
            + 0.3 × ranker_score
            + 0.2 × preliminary_score
```

When PGC is unknown, the `class_match` weight is redistributed across the remaining signals.

---

## `test_api_upload_parsing.py`

Unit and route-level tests for `server._parse_upload()` and the `/api/upload` endpoint. Covers encoding edge cases and error handling.

### What's Tested

| # | Test | Description |
|---|------|-------------|
| 1 | `test_parse_upload_accepts_utf8_csv` | Standard UTF-8 CSV is parsed into a DataFrame with correct headers and values |
| 2 | `test_parse_upload_accepts_utf8_bom_csv` | UTF-8 CSV with a BOM prefix (`\ufeff`) is handled cleanly — BOM is stripped from column names |
| 3 | `test_parse_upload_accepts_cp1252_csv` | CP-1252 encoded CSV (e.g. `Café`) is decoded correctly |
| 4 | `test_parse_upload_rejects_binary_csv_with_clear_error` | Binary data (e.g. a PNG) passed as a CSV raises a `ValueError` with a descriptive message |
| 5 | `test_parse_upload_accepts_cp1252_json` | CP-1252 encoded JSON payload is parsed into a DataFrame correctly |
| 6 | `test_parse_upload_rejects_invalid_json_with_clear_error` | Malformed JSON raises a `ValueError` matching `"Invalid JSON upload"` |
| 7 | `test_upload_route_traces_and_quarantines_failed_upload` | With `UPLOAD_TRACE=1`, a failed upload returns HTTP 422, saves the raw bytes to the debug directory, and logs the parser branch and failure |

### Environment Variables (Test 7)

| Variable | Effect |
|----------|--------|
| `UPLOAD_TRACE=1` | Enables request tracing and quarantine logging on the `/api/upload` route |

### Supported Upload Formats

| Format | Encodings Handled |
|--------|-------------------|
| CSV | UTF-8, UTF-8 BOM, CP-1252 |
| JSON | UTF-8, CP-1252 |

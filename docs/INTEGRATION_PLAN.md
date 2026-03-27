# Integration Plan

This file now tracks only the remaining integration work.

## Database Responsibilities

- Elasticsearch is the metadata system of record for products and candidate-image metadata.
- Object storage is the binary-image store:
	- MinIO for local/dev runs
	- Cloud storage equivalent in production

Current intended split:

- `mi_products` and `mi_candidate_images` hold searchable metadata and classifier outputs.
- Image bytes are stored as object payloads; object keys are referenced in metadata (`local_path` for MinIO-backed records).

## Already In Place

- Elasticsearch and MinIO are wired into the scraper.
- `mi_products` and `mi_candidate_images` mappings include `predicted_class`.
- `web_scraper.py --classify` classifies the current scrape run and syncs predictions to Elasticsearch.
- MinIO-backed classification is supported in the integrated `classify_json_files()` path.
- `src/web_scraping/text_based_search.py` is present.
- `product_to_vector()` exists in `src/web_scraping/web_scraper.py`.

## Remaining Work

### 1. Replace The Temporary Search Query Builder

Current code still uses `simple_search_keywords()` in these paths:

- CSV load in `load_product_catalog()`
- Elasticsearch load in `load_products_from_es()`

Next change:

- switch both paths to `vector_to_query(product_to_vector(product))`
- keep `search_keywords` as a list so the downstream scraper flow does not change

Why this is still pending:

- the bridge function exists, but the actual scraper query builder has not been cut over yet

### 2. Integrate Ranking

Target:

- replace or augment `compute_confidence_hints()` with ranking logic derived from `image_search_ranker (3).py`
- rank candidates against the product vector, not just title/license/resolution heuristics

Required implementation tasks:

1. Bring the ranking module into `src/web_scraping/`
2. Define a stable interface between scraper candidate metadata and the ranking input
3. Run ranking after candidate collection and before final sort
4. Decide which ranking outputs are persisted
5. If new ranking fields are stored in Elasticsearch, update `setup_elasticsearch.py` and recreate indices

Recommended stored fields if ranking is added:

- `ranking_score`
- `ranking_reasons`
- `ranking_version`

### 3. Reconcile Heuristic Score Vs Ranking Score

Current sort key:

- `confidence_hints.preliminary_score`

Decision needed:

- replace the heuristic score entirely, or
- keep the heuristic score and add ranking as a second score

Recommended approach:

- keep `confidence_hints.preliminary_score` for backwards compatibility
- add ranking fields separately
- sort primarily by ranking score once validated

### 4. Validate End-To-End Behavior

After ranking integration:

1. run a small CSV scrape
2. run a small `--from-es` scrape
3. run with `--es --minio --classify`
4. confirm JSON output, Elasticsearch documents, and MinIO objects remain consistent
5. compare ranking output against the current heuristic ordering

## Out Of Scope For This Plan

- new storage backends beyond MinIO
- UI work
- model retraining changes
- GCP deployment work

# Elasticsearch

Reference for how the current pipeline uses Elasticsearch.

Setup and run commands live in [README.md](../README.md).

## Role In The Pipeline

Elasticsearch stores metadata, not binary image files.

- `mi_products`: one document per Motion product
- `mi_candidate_images`: one document per scraped candidate image

Image files are stored separately in MinIO or on local disk. When MinIO is used, the object key is stored in `local_path`.

## Indices

### `mi_products`

Primary document ID: `motion_product_id`

Important fields:

- `motion_product_id`
- `item_number`
- `enterprise_name`
- `mfr_name`
- `mfr_name_text`
- `mfr_part_number`
- `mfr_part_number_text`
- `description`
- `internal_description`
- `pgc`
- `category`
- `search_keywords`
- `catalog_loaded_at`
- `schema_version`
- `scrape_summary`

### `mi_candidate_images`

Primary document ID: `SHA1("{motion_product_id}:{image_url}")`

Important fields:

- `motion_product_id`
- `mfr_name`
- `mfr_part_number`
- `candidate_index`
- `scraped_at`
- `schema_version`
- `image_url`
- `thumbnail_url`
- `source_page`
- `source_name`
- `title`
- `license`
- `attribution`
- `tags`
- `mime_type`
- `api_width`
- `api_height`
- `downloaded`
- `storage_type`
- `local_path`
- `file_size_bytes`
- `actual_width`
- `actual_height`
- `actual_format`
- `download_error`
- `confidence_hints`
- `predicted_class`

## Mapping Rules

- Both indices use `dynamic: "strict"`.
- Any new indexed field must be added in `src/web_scraping/setup_elasticsearch.py`.
- If mappings change, recreate the indices:

```bash
.venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

## Write Path

`src/web_scraping/web_scraper.py` writes to Elasticsearch through `index_to_elasticsearch()`.

For each scraped product it:

1. Upserts the product into `mi_products`
2. Bulk-indexes candidate images into `mi_candidate_images`
3. Updates `scrape_summary` on the product record

Prediction sync:

- `web_scraper.py --classify` writes `predicted_class` into the saved JSON files
- `push_predictions_to_es()` then updates `mi_candidate_images`

Catalog-only load path:

- `src/web_scraping/ingest_catalog.py` bulk-loads product records into `mi_products`

## Deterministic IDs

- Product `_id`: `motion_product_id`
- Candidate image `_id`: `SHA1("{motion_product_id}:{image_url}")`

This makes re-scrapes idempotent for unchanged image URLs.

## Query Examples

All products:

```json
GET /mi_products/_search
{
  "query": { "match_all": {} },
  "size": 50
}
```

Single product:

```json
GET /mi_products/_doc/s10807860
```

Candidate images for one product:

```json
GET /mi_candidate_images/_search
{
  "query": { "term": { "motion_product_id": "s10807860" } },
  "sort": [
    { "confidence_hints.preliminary_score": { "order": "desc" } }
  ]
}
```

Products with scraped images:

```json
GET /mi_products/_search
{
  "query": {
    "range": { "scrape_summary.total_images_found": { "gt": 0 } }
  }
}
```

## Common Local Checks

```bash
.venv/bin/python src/web_scraping/query_elasticsearch.py
.venv/bin/python src/web_scraping/query_elasticsearch.py --product s10807860
.venv/bin/python src/web_scraping/query_elasticsearch.py --images
.venv/bin/python src/web_scraping/query_elasticsearch.py --stats
```

Kibana Dev Tools is available at `http://localhost:5601`.

## Notes That Matter

- API image dimensions are stored as `api_width` and `api_height`.
- Verified downloaded dimensions are stored as `actual_width` and `actual_height`.
- `predicted_class` is mapped and ready for classifier output.
- MinIO-backed runs store the MinIO object key in `local_path`, not a local filesystem path.

# Elasticsearch — Motion Industries Image-to-Product

This document covers how Elasticsearch is used in the Image-to-Product pipeline:
how data is structured, how the scraper writes to it, and how to query it.

For setup instructions (Docker, venv, running the scraper), see the main [README](README.md).

---

## Overview

Elasticsearch serves as the metadata store for the pipeline. It holds two indices:

| Index | Purpose | Doc ID |
|---|---|---|
| `mi_products` | One document per product from the CSV catalog | `motion_product_id` (e.g., `s10807860`) |
| `mi_candidate_images` | One document per scraped image candidate (many-to-one with products) | `SHA1("{motion_product_id}:{image_url}")` |

Image **files** are stored separately in MinIO (see [README > MinIO](README.md#minio--image-storage)).
Elasticsearch only stores metadata, confidence scores, and a reference to the MinIO object key.

---

## Index Schemas

### `mi_products`

| Field | Type | Notes |
|---|---|---|
| `motion_product_id` | keyword | Primary key |
| `item_number` | keyword | Motion Industries internal number |
| `enterprise_name` | keyword | Parent company (e.g., "SKF") |
| `mfr_name` | keyword | Manufacturer name — exact filter/aggregation |
| `mfr_name_text` | text | Manufacturer name — full-text search |
| `mfr_part_number` | keyword | Part number — exact filter |
| `mfr_part_number_text` | text | Part number — full-text search (custom analyzer) |
| `description` | text | Web product description (English analyzer) |
| `internal_description` | text | Motion internal description |
| `pgc` | keyword | Product Group Code |
| `category` | text | PGC description (English analyzer) |
| `search_keywords` | keyword[] | Generated search query strings |
| `catalog_loaded_at` | date | When the product was indexed |
| `schema_version` | keyword | Index schema version |
| `scrape_summary` | object | Aggregate stats, updated after each scrape run |

`scrape_summary` sub-fields:

| Field | Type |
|---|---|
| `total_images_found` | integer |
| `images_downloaded` | integer |
| `sources_queried` | keyword[] |
| `avg_preliminary_score` | float |
| `last_scraped_at` | date |

### `mi_candidate_images`

| Field | Type | Notes |
|---|---|---|
| `motion_product_id` | keyword | Foreign key back to `mi_products` |
| `mfr_name` | keyword | Denormalized for filtering without joining |
| `mfr_part_number` | keyword | Denormalized for filtering without joining |
| `candidate_index` | integer | Position in the scraper's result list |
| `scraped_at` | date | UTC timestamp of the scrape run |
| `schema_version` | keyword | |
| `image_url` | keyword | Full-size image URL (stored, not indexed) |
| `thumbnail_url` | keyword | Thumbnail URL (stored, not indexed) |
| `source_page` | keyword | Attribution page URL (stored, not indexed) |
| `source_name` | keyword | e.g., "Wikimedia Commons", "OpenVerse / flickr" |
| `title` | text | Image title from API |
| `license` | keyword | e.g., "cc-by-sa-3.0", "CC0" |
| `attribution` | keyword | Creator / author |
| `tags` | keyword[] | Image tags from API |
| `mime_type` | keyword | e.g., "image/jpeg" |
| `api_width` / `api_height` | integer | Dimensions reported by the source API |
| `downloaded` | boolean | Whether the image file was successfully saved |
| `storage_type` | keyword | `"minio"` or `"local"` |
| `local_path` | keyword | MinIO object key or filesystem path (stored, not indexed) |
| `file_size_bytes` | integer | |
| `actual_width` / `actual_height` | integer | Post-download Pillow-verified dimensions |
| `actual_format` | keyword | e.g., "JPEG", "PNG", "GIF" |
| `download_error` | text | Error message if download failed (stored, not indexed) |
| `confidence_hints` | object | See below |

`confidence_hints` sub-fields:

| Field | Type |
|---|---|
| `has_product_keywords_in_title` | boolean |
| `is_permissively_licensed` | boolean |
| `meets_minimum_resolution` | boolean |
| `source_reliability` | keyword |
| `preliminary_score` | float |

---

## How the Scraper Writes to Elasticsearch

The function `index_to_elasticsearch()` in `web_scraper.py` performs a three-step write
after each product is scraped:

1. **Index the product** into `mi_products` (upsert by `motion_product_id`)
2. **Bulk-index all candidate images** into `mi_candidate_images` (upsert by SHA1 doc ID)
3. **Update `scrape_summary`** on the product document with aggregate stats

### Deterministic Document IDs

- Product docs use `motion_product_id` as their `_id`.
- Image docs use `SHA1("{motion_product_id}:{image_url}")` as their `_id`.

This means **re-scraping is safe** — running the scraper on the same product overwrites
existing documents rather than creating duplicates.

### Denormalized Fields

`mfr_name` and `mfr_part_number` are copied onto each `mi_candidate_images` document.
This lets downstream consumers (React UI, query scripts) filter and display image results
without joining back to `mi_products`.

---

## Querying with Kibana Dev Tools

Kibana Dev Tools provides an interactive console for running Elasticsearch queries
without `curl`. This is the easiest way to explore your data.

### Getting Started

1. Open http://localhost:5601 in your browser
2. Click the **hamburger menu** (☰) in the top-left corner
3. Scroll down to **Management** and click **Dev Tools**
4. You'll see a split-pane editor: **left** for requests, **right** for responses
5. Type a query in the left pane and click the **green play button** (▶) or press **Ctrl+Enter** to execute

### Kibana Query Syntax

In Dev Tools, you omit `curl`, the host, headers, and `-d` flags. Just write the
HTTP method, path, and JSON body directly:

```
GET /mi_products/_search
{
  "query": { "match_all": {} }
}
```

Multiple queries can be stacked in the editor — place your cursor on the one you
want to run and press **Ctrl+Enter**.

### Common Queries (Kibana format)

**View all products:**

```
GET /mi_products/_search
{
  "query": { "match_all": {} },
  "size": 50
}
```

**Look up a specific product by ID (e.g., `s10807860`):**

```
GET /mi_products/_doc/s10807860
```

This returns the single document directly by its `_id`. Use this when you know the
exact `motion_product_id`.

**Search for a product ID with a query (useful for partial matches):**

```
GET /mi_products/_search
{
  "query": { "term": { "motion_product_id": "s10807860" } }
}
```

**View all candidate images for a specific product:**

```
GET /mi_candidate_images/_search
{
  "query": { "term": { "motion_product_id": "s10807860" } },
  "sort": [{ "confidence_hints.preliminary_score": { "order": "desc" } }]
}
```

**Products that have at least one image found:**

```
GET /mi_products/_search
{
  "query": {
    "range": { "scrape_summary.total_images_found": { "gt": 0 } }
  }
}
```

**Products with no successfully downloaded images:**

```
GET /mi_products/_search
{
  "query": { "term": { "scrape_summary.images_downloaded": 0 } }
}
```

**Full-text search across product catalog:**

```
GET /mi_products/_search
{
  "query": {
    "multi_match": {
      "query": "spherical roller bearing",
      "fields": ["description", "mfr_name_text", "category"]
    }
  }
}
```

**Filter by manufacturer:**

```
GET /mi_products/_search
{
  "query": { "term": { "mfr_name": "SKF" } }
}
```

**License distribution (aggregation):**

```
GET /mi_candidate_images/_search
{
  "size": 0,
  "aggs": { "by_license": { "terms": { "field": "license" } } }
}
```

**Check cluster health:**

```
GET /_cluster/health
```

**List all indices:**

```
GET /_cat/indices?v
```

---

## Useful Queries (curl)

The same queries can be run from the terminal with `curl`. Replace `localhost:9200`
with your Elasticsearch host if it differs.

**Check cluster health:**

```bash
curl 'http://localhost:9200/_cluster/health?pretty'
```

**List all indices:**

```bash
curl 'http://localhost:9200/_cat/indices?v'
```

**Look up a specific product by ID:**

```bash
curl 'http://localhost:9200/mi_products/_doc/s10807860?pretty'
```

**Get all images for a product, sorted by confidence score:**

```bash
curl -X GET 'http://localhost:9200/mi_candidate_images/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": { "term": { "motion_product_id": "s10807860" } },
    "sort": [{ "confidence_hints.preliminary_score": { "order": "desc" } }]
  }'
```

**Products with no successfully downloaded images:**

```bash
curl -X GET 'http://localhost:9200/mi_products/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{ "query": { "term": { "scrape_summary.images_downloaded": 0 } } }'
```

**Full-text search across product catalog:**

```bash
curl -X GET 'http://localhost:9200/mi_products/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "multi_match": {
        "query": "spherical roller bearing",
        "fields": ["description", "mfr_name_text", "category"]
      }
    }
  }'
```

**License distribution (aggregation):**

```bash
curl -X GET 'http://localhost:9200/mi_candidate_images/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": { "by_license": { "terms": { "field": "license" } } }
  }'
```

---

## Data Flow

```
CSV catalog (test_products_sample.csv)
    |
    v
web_scraper.py ──scrape──> Wikimedia / OpenVerse APIs
    |                           |
    |  (product metadata)       |  (candidate images + download metadata)
    |                           |
    v                           v
mi_products (ES)  1───*  mi_candidate_images (ES)
                                |
                                | (image files)
                                v
                         MinIO (mi-images bucket)
                                |
                                v
                     React Review UI (future)
```

---

## Index Management

**Create indices** (skips if they already exist):

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py
```

**Recreate indices** (drops and rebuilds — destroys all data):

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

> You must recreate indices if the mapping in `setup_elasticsearch.py` has changed since
> the indices were originally created. Elasticsearch does not allow changing field types
> on an existing index.

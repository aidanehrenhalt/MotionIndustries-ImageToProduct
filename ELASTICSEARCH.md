# Elasticsearch Setup — Motion Industries Image-to-Product

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.12+ with the project `venv` activated

---

## 1. Start the Stack

From the **project root**:

```bash
docker-compose up -d
```

This starts two containers:

| Container        | URL                        | Purpose                          |
|-----------------|----------------------------|----------------------------------|
| `mi_elasticsearch` | http://localhost:9200   | Elasticsearch database           |
| `mi_kibana`        | http://localhost:5601   | Visual index browser (optional)  |

Data is persisted in a Docker volume (`esdata`) — stopping the containers does **not** delete your data.

To stop:

```bash
docker-compose down
```

To stop **and delete all data**:

```bash
docker-compose down -v
```

---

## 2. Install Dependencies

> **Important:** Python virtual environments are **not portable** between machines or operating
> systems. Every team member must recreate the `venv` locally after cloning the repo.
> Never copy a `venv/` folder from another machine.

**macOS / Linux / WSL:**

```bash
python3 -m venv venv                        # only needed once (or after cloning)
venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell / Command Prompt):**

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

---

## 3. Create the Elasticsearch Indices

**macOS / Linux / WSL:**

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py
```

**Windows (PowerShell / Command Prompt):**

```powershell
venv\Scripts\python src\web_scraping\setup_elasticsearch.py
```

Expected output:

```
Connected to Elasticsearch 8.17.0 at localhost:9200

Setting up indices...
  Created index: mi_products
  Created index: mi_candidate_images

Done. Indices:
  mi_products                    0 documents
  mi_candidate_images            0 documents
```

### Options

| Flag          | Effect                                        |
|--------------|-----------------------------------------------|
| *(none)*     | Create indices; skip if they already exist    |
| `--recreate` | Drop and rebuild both indices from scratch    |
| `--host`     | Elasticsearch host (default: `localhost`)     |
| `--port`     | Elasticsearch port (default: `9200`)          |

**Recreate example (macOS / Linux / WSL):**

```bash
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

**Recreate example (Windows):**

```powershell
venv\Scripts\python src\web_scraping\setup_elasticsearch.py --recreate
```

---

## 4. Index Structure

### `mi_products`

One document per product loaded from the CSV catalog.

**Document ID:** `motion_product_id` (e.g., `s10807860`)

| Field                  | Type      | Notes                                      |
|-----------------------|-----------|--------------------------------------------|
| `motion_product_id`   | keyword   | Primary key                                |
| `item_number`         | keyword   | Motion Industries internal number          |
| `enterprise_name`     | keyword   | Parent company (e.g., "SKF")               |
| `mfr_name`            | keyword   | Manufacturer name — exact filter           |
| `mfr_name_text`       | text      | Manufacturer name — full-text search       |
| `mfr_part_number`     | keyword   | Part number — exact filter                 |
| `mfr_part_number_text`| text      | Part number — full-text search             |
| `description`         | text      | Web product description (English analyzer) |
| `internal_description`| text      | Motion internal description                |
| `pgc`                 | keyword   | Product Group Code                         |
| `category`            | text      | PGC description (English analyzer)         |
| `search_keywords`     | keyword[] | Generated search query strings             |
| `catalog_loaded_at`   | date      | When the CSV row was loaded                |
| `schema_version`      | keyword   | Index schema version                       |
| `scrape_summary`      | object    | Updated after each scrape run (see below)  |

`scrape_summary` sub-fields:

| Field                   | Type    |
|------------------------|---------|
| `total_images_found`   | integer |
| `images_downloaded`    | integer |
| `sources_queried`      | keyword |
| `avg_preliminary_score`| float   |
| `last_scraped_at`      | date    |

---

### `mi_candidate_images`

One document per scraped image candidate. Many-to-one with `mi_products`.

**Document ID:** `SHA1("{motion_product_id}:{image_url}")` — deterministic, so re-scraping the same URL upserts rather than duplicates.

```python
import hashlib
doc_id = hashlib.sha1(f"{motion_product_id}:{image_url}".encode()).hexdigest()
```

| Field                           | Type      | Notes                                       |
|--------------------------------|-----------|---------------------------------------------|
| `motion_product_id`            | keyword   | Foreign key back to `mi_products`           |
| `mfr_name`                     | keyword   | Denormalized for efficient filtering        |
| `mfr_part_number`              | keyword   | Denormalized for efficient filtering        |
| `candidate_index`              | integer   | Position in the scraper's result list       |
| `scraped_at`                   | date      | UTC timestamp of the scrape run             |
| `schema_version`               | keyword   |                                             |
| `image_url`                    | keyword   | Full-size image URL (stored, not indexed)   |
| `thumbnail_url`                | keyword   | Thumbnail URL (stored, not indexed)         |
| `source_page`                  | keyword   | Attribution page URL (stored, not indexed)  |
| `source_name`                  | keyword   | e.g., "Wikimedia Commons", "OpenVerse / flickr" |
| `title`                        | text      | Image title from API                        |
| `license`                      | keyword   | e.g., "CC BY-SA 4.0", "CC0"                |
| `attribution`                  | keyword   | Creator / author                            |
| `tags`                         | keyword[] | Image tags from API                         |
| `mime_type`                    | keyword   | e.g., "image/jpeg"                          |
| `api_width` / `api_height`     | integer   | Dimensions reported by the API              |
| `downloaded`                   | boolean   | Whether the image file was saved to disk    |
| `local_path`                   | keyword   | Filesystem path (stored, not indexed)       |
| `file_size_bytes`              | integer   |                                             |
| `actual_width` / `actual_height` | integer | Post-download Pillow-verified dimensions    |
| `actual_format`                | keyword   | e.g., "JPEG", "PNG"                         |
| `download_error`               | text      | Error message if download failed            |
| `confidence_hints`             | object    | See below                                   |

`confidence_hints` sub-fields:

| Field                          | Type    |
|-------------------------------|---------|
| `has_product_keywords_in_title`| boolean |
| `is_permissively_licensed`    | boolean |
| `meets_minimum_resolution`    | boolean |
| `source_reliability`          | keyword |
| `preliminary_score`           | float   |

---

## 5. Useful Queries

> **macOS / Linux / WSL:** use the `curl` commands below directly in your terminal.
>
> **Windows PowerShell:** `curl` is an alias for `Invoke-WebRequest` and has different
> syntax. The easiest alternative is **Kibana Dev Tools** — open http://localhost:5601,
> go to **Management → Dev Tools**, and paste the query body directly (no `curl` needed).
>
> **Windows with Git Bash or WSL:** the `curl` commands below work as-is.

**Check cluster health:**

```bash
curl http://localhost:9200/_cluster/health?pretty
```

**List all indices:**

```bash
curl http://localhost:9200/_cat/indices?v
```

**Get all images for a product, sorted by confidence score:**

```bash
curl -X GET http://localhost:9200/mi_candidate_images/_search?pretty \
  -H 'Content-Type: application/json' \
  -d '{
    "query": { "term": { "motion_product_id": "s10807860" } },
    "sort": [{ "confidence_hints.preliminary_score": { "order": "desc" } }]
  }'
```

**Products with no successfully downloaded images:**

```bash
curl -X GET http://localhost:9200/mi_products/_search?pretty \
  -H 'Content-Type: application/json' \
  -d '{ "query": { "term": { "scrape_summary.images_downloaded": 0 } } }'
```

**Full-text search across product catalog:**

```bash
curl -X GET http://localhost:9200/mi_products/_search?pretty \
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

**Images with permissive license only, sorted by score:**

```bash
curl -X GET http://localhost:9200/mi_candidate_images/_search?pretty \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "filter": [
          { "term": { "confidence_hints.is_permissively_licensed": true } },
          { "term": { "downloaded": true } }
        ]
      }
    },
    "sort": [{ "confidence_hints.preliminary_score": { "order": "desc" } }]
  }'
```

**License distribution across all images (aggregation):**

```bash
curl -X GET http://localhost:9200/mi_candidate_images/_search?pretty \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": { "by_license": { "terms": { "field": "license" } } }
  }'
```

---

## 6. Architecture Overview

### Data Flow

```
CSV catalog
    |
    v
web_scraper.py ──scrape──> Wikimedia / OpenVerse APIs
    |                           |
    |  (product metadata)       |  (candidate images + download metadata)
    v                           v
┌──────────────┐       ┌────────────────────┐
│ mi_products  │ 1───* │ mi_candidate_images│
│  (ES index)  │       │    (ES index)      │
└──────┬───────┘       └────────┬───────────┘
       │                        │
       └────────────┬───────────┘
                    │
                    v
            React Review UI
         (queries ES directly)
```

### Current vs. Target State

| Aspect | Before (JSON files) | After (Elasticsearch) |
|---|---|---|
| Storage | One `.json` file per product per run in `output/json/` | Two ES indices (`mi_products`, `mi_candidate_images`) |
| Querying | Manual — open files, `grep`, or write scripts | Full-text search, filters, aggregations via ES query DSL |
| Deduplication | None — re-scraping creates new files | Deterministic doc IDs — re-scraping upserts in place |
| React UI access | Would need a file-serving API | Queries ES directly (or via thin API proxy) |
| Download metadata | Lost — `download_image()` returns it but `save_record()` doesn't persist all fields | Stored per-image: `local_path`, `file_size_bytes`, `actual_width/height`, `actual_format`, `download_error` |

### Key Design Decisions

1. **Deterministic document IDs** — Product docs use `motion_product_id`; image docs use `SHA1("{motion_product_id}:{image_url}")`. This means every write is an upsert. Re-scraping the same product safely overwrites stale data without creating duplicates.

2. **Denormalized fields on images** — `mfr_name` and `mfr_part_number` are copied onto each `mi_candidate_images` doc so the React UI can filter/display image results without joining back to `mi_products`.

3. **`scrape_summary` lives on the product** — After each scrape run, the product doc's `scrape_summary` is updated with aggregate stats (`total_images_found`, `images_downloaded`, `avg_preliminary_score`, `last_scraped_at`). This lets the UI quickly show which products need attention without scanning all images.

4. **Images on disk are supplementary** — Downloaded files still go to `output/images/` for local inspection. The `local_path` field in ES points to them, but the source of truth for metadata is ES, not the filesystem.

---

## 7. Integrating the Scraper with Elasticsearch

The scraper's `save_record()` currently writes a single JSON file. The ES integration
replaces (or supplements) that with three indexed writes per product:

### Write Pattern

```python
import hashlib
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch("http://localhost:9200")

# Step 1 — Index the product (upsert by motion_product_id)
es.index(index="mi_products", id=product["motion_product_id"], document={
    "motion_product_id": ...,
    "mfr_name": ...,
    "mfr_part_number": ...,
    "description": ...,
    "category": ...,
    "search_keywords": [...],
    "schema_version": "1.0",
    "catalog_loaded_at": "<ISO timestamp>",
    # ... remaining CSV fields
})

# Step 2 — Bulk-index all candidate images (upsert by SHA1 doc ID)
actions = []
for img in enriched_images:
    doc_id = hashlib.sha1(
        f"{motion_product_id}:{img['image_url']}".encode()
    ).hexdigest()
    actions.append({
        "_op_type": "index",
        "_index":   "mi_candidate_images",
        "_id":      doc_id,
        "_source":  {
            "motion_product_id": motion_product_id,
            "mfr_name":          ...,
            "mfr_part_number":   ...,
            "candidate_index":   img["index"],
            "scraped_at":        ...,
            "image_url":         ...,
            "thumbnail_url":     ...,
            "source_page":       ...,
            "source_name":       ...,
            "title":             ...,
            "license":           ...,
            "attribution":       ...,
            "tags":              [...],
            "mime_type":         ...,
            "api_width":         ...,
            "api_height":        ...,
            "downloaded":        True/False,
            "local_path":        ...,        # from download_image()
            "file_size_bytes":   ...,        # from download_image()
            "actual_width":      ...,        # from download_image()
            "actual_height":     ...,        # from download_image()
            "actual_format":     ...,        # from download_image()
            "download_error":    ...,        # from download_image()
            "confidence_hints":  { ... },
            "schema_version":    "1.0",
        },
    })
bulk(es, actions)

# Step 3 — Update the product's scrape_summary
es.update(index="mi_products", id=motion_product_id, doc={
    "scrape_summary": {
        "total_images_found":    ...,
        "images_downloaded":     ...,
        "sources_queried":       [...],
        "avg_preliminary_score": ...,
        "last_scraped_at":       "<ISO timestamp>",
    }
})
```

### What Changes in `web_scraper.py`

1. **Thread download metadata into enriched dicts** — `download_image()` already returns `local_path`, `file_size_bytes`, `actual_width`, `actual_height`, `actual_format`, and `download_error`, but the enriched dict built in `scrape_product_images()` only copies `downloaded`. All six fields need to be included so they can be indexed.

2. **Add `index_to_elasticsearch()` function** — Takes the same `record` dict that `save_record()` uses and performs the three-step write shown above.

3. **Wire into `__main__`** — Create an `Elasticsearch` client once, add an `--es` flag to enable ES writes, call `index_to_elasticsearch(record, es)` in the product loop.

> **Re-scraping is safe.** Because document IDs are deterministic, re-running the scraper
> on the same product upserts existing documents rather than creating duplicates.

---

## 8. React Review UI Integration (Future)

The React review UI will read from the same two ES indices the scraper writes to.

### Query Patterns the UI Will Use

| UI Feature | ES Query | Index |
|---|---|---|
| Product list / search | `multi_match` on `description`, `mfr_name_text`, `category` | `mi_products` |
| Products needing review | `term: scrape_summary.images_downloaded > 0` | `mi_products` |
| Products with no images | `term: scrape_summary.images_downloaded = 0` | `mi_products` |
| Image candidates for a product | `term: motion_product_id` + sort by `confidence_hints.preliminary_score` desc | `mi_candidate_images` |
| Filter by license | `bool > filter > term: confidence_hints.is_permissively_licensed = true` | `mi_candidate_images` |
| Stats dashboard | Aggregations: `by_license`, `by_source_name`, avg score | `mi_candidate_images` |

### Access Options

- **Direct from browser** — ES accepts REST requests; the React app can `fetch()` against `http://localhost:9200/mi_candidate_images/_search` during development.
- **Thin API proxy** — For production or to add auth, a lightweight Express/FastAPI server can sit between React and ES, forwarding queries and enforcing access control.

### Example: Fetch Top Candidates for a Product (React)

```javascript
const response = await fetch("http://localhost:9200/mi_candidate_images/_search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: { term: { motion_product_id: productId } },
    sort: [{ "confidence_hints.preliminary_score": { order: "desc" } }],
    size: 20,
  }),
});
const data = await response.json();
const images = data.hits.hits.map(hit => hit._source);
```

# ECE 4013 - Senior Design
## Image-to-Product Pipeline for Motion Industries

Automated pipeline that scrapes product images from open-source APIs, scores them with
confidence heuristics, stores images in MinIO (S3-compatible object storage), and indexes
metadata into Elasticsearch for search and review.

## Link to Motion Industry Shared Documents
[Image Dataset, Image Mapping, Images](https://genparts-my.sharepoint.com/:f:/r/personal/michael_flack_corp_motion-ind_com/Documents/GT%20Capstone?csf=1&web=1&e=s92NcQ)

### Tech Stack

| Layer | Technology |
|---|---|
| Scraper | Python 3.12+, Requests, Pillow, BeautifulSoup |
| Image Sources | Wikimedia Commons API, OpenVerse API |
| Image Storage | MinIO (local dev), S3-compatible (production) |
| Metadata Store | Elasticsearch 8.17.0 |
| Visualization | Kibana 8.17.0 |
| Containers | Docker Compose |
| Review UI | React (planned) |

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Docker Services](#docker-services)
4. [Python Environment](#python-environment)
5. [Elasticsearch](#elasticsearch)
   - [Setup](#setup)
   - [Index Schemas](#index-schemas)
   - [How the Scraper Writes to ES](#how-the-scraper-writes-to-es)
   - [Data Flow](#data-flow)
   - [Querying with Kibana Dev Tools](#querying-with-kibana-dev-tools)
   - [Querying with curl](#querying-with-curl)
   - [Index Management](#index-management)
6. [MinIO — Image Storage](#minio--image-storage)
7. [Running the Scraper](#running-the-scraper)
8. [File Descriptions](#file-descriptions)
9. [Project Structure](#project-structure)
10. [Integration Details](#integration-details)
11. [Text-Based Search Integration](#text-based-search-integration)
12. [Production Deployment (GCP)](#production-deployment-gcp)

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.12+
- Git

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd MotionIndustries-ImageToProduct

# 2. Start Docker services (Elasticsearch, Kibana, MinIO)
docker-compose up -d

# 3. Create Python virtual environment and install dependencies
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 4. Create Elasticsearch indices
venv/bin/python src/web_scraping/setup_elasticsearch.py

# 5. Run the scraper (small test — 2 products, with ES + MinIO)
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 2 --es --minio
```

After step 5, you should see images uploaded to MinIO and metadata indexed in Elasticsearch.

---

## Docker Services

All three services are defined in `docker-compose.yml` at the project root.

| Container | Service | URL | Purpose |
|---|---|---|---|
| `mi_elasticsearch` | Elasticsearch 8.17.0 | http://localhost:9200 | Metadata store (product + image indices) |
| `mi_kibana` | Kibana 8.17.0 | http://localhost:5601 | Web UI for browsing ES data |
| `mi_minio` | MinIO | http://localhost:9000 (API) / http://localhost:9001 (Console) | S3-compatible image storage |

### Common Commands

```bash
# Start all services
docker-compose up -d

# Check status
docker ps

# View logs
docker-compose logs -f elasticsearch
docker-compose logs -f minio

# Stop services (data is preserved in Docker volumes)
docker-compose down

# Stop AND delete all data (Elasticsearch indices + MinIO images)
docker-compose down -v
```

Data is persisted in Docker volumes (`esdata`, `miniodata`). Stopping containers does
**not** delete data. Only `docker-compose down -v` removes volumes.

### Default Credentials

| Service | Username | Password |
|---|---|---|
| MinIO Console | `minioadmin` | `minioadmin` |
| Elasticsearch | *(security disabled for local dev)* | |

---

## Python Environment

> **Important:** Virtual environments are **not portable** between machines or operating
> systems. Every team member must create their own `venv` locally after cloning.

**macOS / Linux / WSL:**

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Use `venv/bin/python` (or `venv\Scripts\python` on Windows) to run scripts. Shell
activation (`source venv/bin/activate`) works but is optional.

### Dependencies

| Package | Purpose |
|---|---|
| `requests` | HTTP client for API calls |
| `Pillow` | Image validation and dimension extraction |
| `beautifulsoup4` / `lxml` | HTML parsing (future scraping sources) |
| `elasticsearch` | Elasticsearch Python client |
| `boto3` | S3-compatible client for MinIO |
| `selenium` | Browser automation (future non-API sources) |

---

## Elasticsearch

Elasticsearch stores all product and image metadata in two indices.

### Setup

```bash
# Create indices (skips if they already exist)
venv/bin/python src/web_scraping/setup_elasticsearch.py

# Recreate indices (drops and rebuilds — destroys existing data)
venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate
```

You must `--recreate` if the field mappings in `setup_elasticsearch.py` have changed, since
Elasticsearch does not allow modifying field types on an existing index.

### Index Schemas

#### `mi_products`

One document per product from the CSV catalog.
- Document ID: `motion_product_id` (e.g., `s10807860`)
- Updated with `scrape_summary` after each scrape run (total images found, downloaded, avg score)

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

#### `mi_candidate_images`

One document per scraped image candidate (many-to-one with products).
- Document ID: `SHA1("{motion_product_id}:{image_url}")` — deterministic, so re-scraping upserts
- References the MinIO object key via `local_path` (when `storage_type` is `"minio"`)

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

Both indices use `dynamic: "strict"` — any field not in the mapping will be rejected.

### How the Scraper Writes to ES

The function `index_to_elasticsearch()` in `web_scraper.py` performs a three-step write
after each product is scraped:

1. **Index the product** into `mi_products` (upsert by `motion_product_id`)
2. **Bulk-index all candidate images** into `mi_candidate_images` (upsert by SHA1 doc ID)
3. **Update `scrape_summary`** on the product document with aggregate stats

#### Deterministic Document IDs

- Product docs use `motion_product_id` as their `_id`.
- Image docs use `SHA1("{motion_product_id}:{image_url}")` as their `_id`.

This means **re-scraping is safe** — running the scraper on the same product overwrites
existing documents rather than creating duplicates.

#### Denormalized Fields

`mfr_name` and `mfr_part_number` are copied onto each `mi_candidate_images` document.
This lets downstream consumers (React UI, query scripts) filter and display image results
without joining back to `mi_products`.

### Data Flow

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

### Querying with Kibana Dev Tools

Kibana Dev Tools provides an interactive console for running Elasticsearch queries
without `curl`. This is the easiest way to explore your data.

#### Getting Started

1. Open http://localhost:5601 in your browser
2. Click the **hamburger menu** (three lines icon) in the top-left corner
3. Scroll down to **Management** and click **Dev Tools**
4. You'll see a split-pane editor: **left** for requests, **right** for responses
5. Type a query in the left pane and click the **green play button** or press **Ctrl+Enter** to execute

#### Kibana Query Syntax

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

#### Common Queries (Kibana format)

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

### Querying with curl

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

### Index Management

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

---

## MinIO — Image Storage

MinIO is an S3-compatible object storage server. The scraper uploads downloaded images
to MinIO instead of the local filesystem (when the `--minio` flag is used).

### Accessing Images

**1. MinIO Web Console**

Open http://localhost:9001, log in with `minioadmin` / `minioadmin`, and browse the
`mi-images` bucket. You can preview, download, and manage images directly.

**2. CLI utility — `minio_es_match.py`**

```bash
# Show images for a product (with presigned download URLs)
venv/bin/python src/web_scraping/minio_es_match.py --product s10807860

# Download a product's images from MinIO to local disk
venv/bin/python src/web_scraping/minio_es_match.py --download s10807860

# List all objects in the bucket
venv/bin/python src/web_scraping/minio_es_match.py --list-bucket

# Verify MinIO and ES are in sync (detect orphaned/missing files)
venv/bin/python src/web_scraping/minio_es_match.py --verify

# Storage stats (combined MinIO + ES)
venv/bin/python src/web_scraping/minio_es_match.py --stats
```

**3. Programmatic access (presigned URLs)**

The ES `local_path` field stores the MinIO object key (e.g.,
`images/s10807860/s10807860_01_b2ed562c.jpg`). To generate a temporary download URL:

```python
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin')
url = s3.generate_presigned_url('get_object',
    Params={'Bucket': 'mi-images', 'Key': '<object_key_from_es>'},
    ExpiresIn=3600)
```

### Object Key Structure

```
mi-images/
  images/
    {product_id}/
      {product_id}_{index}_{url_hash}.{ext}
```

- `index`: zero-padded position (00, 01, ...)
- `url_hash`: first 8 chars of MD5 of the image URL
- `ext`: file extension from Pillow's detected format (not from the URL)

### Configuration

MinIO settings can be overridden via environment variables or CLI flags:

| Env Variable | Default | CLI Override |
|---|---|---|
| `MINIO_ENDPOINT` | `http://localhost:9000` | `--minio-endpoint` |
| `MINIO_ACCESS_KEY` | `minioadmin` | *(env only)* |
| `MINIO_SECRET_KEY` | `minioadmin` | *(env only)* |
| `MINIO_BUCKET` | `mi-images` | `--minio-bucket` |

---

## Running the Scraper

```bash
venv/bin/python src/web_scraping/web_scraper.py --csv <path_to_csv> [options]
```

### Required Arguments

| Flag | Description |
|---|---|
| `--csv` | Path to the product catalog CSV file |

### Optional Arguments

| Flag | Description | Default |
|---|---|---|
| `--product` | Filter by product ID or category keyword | *(all products)* |
| `--limit` | Max number of products to scrape | *(no limit)* |
| `--no-download` | Scrape metadata only, skip image downloads | *(downloads enabled)* |
| `--es` | Index results into Elasticsearch | *(disabled)* |
| `--es-host` | Elasticsearch host | `localhost` |
| `--es-port` | Elasticsearch port | `9200` |
| `--minio` | Upload images to MinIO instead of local disk | *(local storage)* |
| `--minio-endpoint` | MinIO endpoint URL | env or `http://localhost:9000` |
| `--minio-bucket` | MinIO bucket name | env or `mi-images` |

### Examples

```bash
# Scrape all 20 test products, save images locally, no ES
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv

# Scrape 3 products, upload to MinIO, index to ES
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --limit 3 --es --minio

# Scrape only SKF products, metadata only (no image download)
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --product SKF --no-download

# Full pipeline with all options
venv/bin/python src/web_scraping/web_scraper.py \
  --csv src/web_scraping/test_products_sample.csv \
  --es --minio --limit 5
```

### Output

The scraper always saves a JSON record per product to `output/json/`. Additionally:

- With `--minio`: images are uploaded to MinIO under `images/{product_id}/`
- Without `--minio`: images are saved to `output/images/`
- With `--es`: metadata is indexed into Elasticsearch

---

## File Descriptions

| File | Purpose |
|---|---|
| `docker-compose.yml` | Defines the Elasticsearch, Kibana, and MinIO Docker services with volumes, ports, and health checks |
| `requirements.txt` | Python dependencies (requests, Pillow, beautifulsoup4, elasticsearch, boto3, selenium, etc.) |
| `src/web_scraping/web_scraper.py` | Main scraper orchestration — loads CSV catalog, queries Wikimedia Commons and OpenVerse APIs, downloads images with Pillow validation, computes confidence scores, uploads to MinIO, and indexes metadata into Elasticsearch |
| `src/web_scraping/setup_elasticsearch.py` | Creates (or recreates with `--recreate`) the two Elasticsearch indices (`mi_products`, `mi_candidate_images`) with strict field mappings |
| `src/web_scraping/text_based_search.py` | Query builder using the `ProductVector` dataclass — converts product metadata into prioritized search query strings via `vector_to_query()` |
| `src/web_scraping/query_elasticsearch.py` | Terminal viewer for Elasticsearch data — displays product summaries, per-product detail views, and aggregate statistics |
| `src/web_scraping/minio_es_match.py` | MinIO/ES matching and verification utility — generates presigned URLs, downloads images from MinIO to local disk, detects orphaned or missing files, and reports storage stats |
| `src/web_scraping/manufacturer_scrapers.py` | Selenium-based framework for scraping manufacturer websites directly (not yet approved for use — requires site-specific authorization) |
| `src/web_scraping/test_manufacturer_scrapers.py` | Unit tests for the manufacturer scrapers module |
| `src/web_scraping/test_products_sample.csv` | 20-product test catalog from Motion Industries used for development and testing |
| `src/web_scraping/GCPAccess.md` | GCP service and IAM permission requirements for production deployment |

---

## Project Structure

```
MotionIndustries-ImageToProduct/
  docker-compose.yml              # Elasticsearch + Kibana + MinIO
  requirements.txt                # Python dependencies
  README.md                       # This file
  ELASTICSEARCH.md                # Detailed ES schema and query reference (legacy)
  INTEGRATION_PLAN.md             # Edge cases, pitfalls, and integration details (legacy)
  nsimon13-patch-2-merge.md       # Text-based search integration plan (legacy)
  src/
    web_scraping/
      web_scraper.py              # Main scraper (Wikimedia + OpenVerse APIs)
      setup_elasticsearch.py      # Creates ES indices (run once)
      text_based_search.py        # Query builder (ProductVector → search string)
      query_elasticsearch.py      # Terminal viewer for ES data
      minio_es_match.py           # MinIO <-> ES matching utility
      manufacturer_scrapers.py    # Selenium-based manufacturer site scraping
      test_manufacturer_scrapers.py  # Unit tests for manufacturer scrapers
      test_products_sample.csv    # 20-row test product catalog
      GCPAccess.md                # GCP deployment notes
  output/
    json/                         # Scrape records (one JSON per product per run)
    images/                       # Locally downloaded images (when not using MinIO)
```

---

## Integration Details

This section covers edge cases and pitfalls in the scraper/ES/MinIO integration.

### Strict Mapping Gotchas

Both indices use `dynamic: "strict"`. Elasticsearch will **reject any document that contains a field not defined in the mapping**.

- If you add a new field to the scraper's output (e.g., in `index_to_elasticsearch()`),
  you **must** also add it to `setup_elasticsearch.py` and recreate the indices.
- If you forget, bulk indexing will silently fail for those documents. The scraper logs
  the error count (e.g., `Indexed 0 candidate images for s10807860 (2 errors)`) but does
  not abort — other products continue processing.
- To fix: update the mapping in `setup_elasticsearch.py`, then run
  `venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate` and re-scrape.

Why strict mode? Permissive (`dynamic: true`) would silently accept typos or unexpected
fields and auto-create mappings with potentially wrong types. Strict mode catches these
mistakes early.

### Field Name Translation

The source APIs (Wikimedia, OpenVerse) return dimensions as `width` and `height`. The ES
mapping uses `api_width` and `api_height` to distinguish API-reported dimensions from
Pillow-verified `actual_width` and `actual_height`. The translation happens in
`index_to_elasticsearch()`:

```python
"api_width":  img.get("width"),
"api_height": img.get("height"),
```

If you index the raw enriched dict directly (e.g., using `**img`), the `width` and `height`
fields will be rejected by the strict mapping. Always use the explicit field mapping in
`index_to_elasticsearch()`.

### Document ID Determinism and Re-scraping

- **Products:** `_id` = `motion_product_id`. Re-scraping overwrites with fresh
  `catalog_loaded_at` and `scrape_summary`.
- **Images:** `_id` = `SHA1("{motion_product_id}:{image_url}")`. Same product + same URL
  always produces the same doc ID (upsert, no duplicates). If an image URL changes, a new
  document is created and the old one becomes orphaned. Use `minio_es_match.py --verify`
  to detect these.
- The SHA1 is computed over the **full image URL**, including query parameters. URLs that
  differ only in query string produce different doc IDs.
- **MinIO:** Re-uploading the same object key overwrites the file. No versioning by default.

To clean up stale data, use `setup_elasticsearch.py --recreate` to wipe indices and
re-scrape from scratch.

### Download Validation Pipeline

`download_image()` validates every downloaded file with Pillow before storing:

1. Checks that the HTTP response `Content-Type` starts with `image/`
2. Opens raw bytes with `Image.open()` and calls `img.verify()` to confirm validity
3. Re-opens after verify (Pillow requires this) to extract actual dimensions and format

If any step fails, `downloaded` is set to `False` and `download_error` captures the
exception. The image is still recorded in ES but no file is uploaded to MinIO.

| Error | Cause |
|---|---|
| `URL did not return an image` | Server returned HTML/JSON instead of an image |
| `HTTPError 403` | Source blocks scraper user-agent or hotlinking |
| `ConnectionTimeout` | Source server unresponsive |
| `UnidentifiedImageError` | Corrupt file or unsupported image format |

None of these abort the scraper — it continues to the next image.

### Bulk Indexing Partial Failures

`index_to_elasticsearch()` uses `bulk()` with `raise_on_error=False`:

- If some documents in the batch fail, the successful ones are still indexed.
- The function logs the error count but does **not** raise an exception.
- If 5 out of 7 images index successfully, you'll see `Indexed 5 candidate images for
  s10807860 (2 errors)`. The `scrape_summary` will still report all 7 as downloaded, but
  only 5 are queryable. Compare `scrape_summary.images_downloaded` against the actual
  document count in `mi_candidate_images` to detect mismatches.

### Rate Limiting and API Etiquette

The scraper includes a `REQUEST_DELAY` (1.5 seconds) between API calls.

- **Wikimedia Commons:** No strict rate limit, but requires `User-Agent` identification.
  The scraper includes a project-specific user agent.
- **OpenVerse:** Rate-limited; unauthenticated requests are throttled more aggressively.
  Increase `REQUEST_DELAY` if you see `429 Too Many Requests` errors.

---

## Text-Based Search Integration

The `text_based_search.py` module provides a `ProductVector` dataclass and `vector_to_query()`
function that converts product metadata into optimized search query strings.

### How It Works

A bridge function `product_to_vector()` converts the scraper's product dict into a
`ProductVector`, which `vector_to_query()` then uses to build a prioritized query string.

### Field Mapping (Scraper Dict to ProductVector)

| Scraper dict key | ProductVector field |
|---|---|
| `motion_product_id` | `id_number` |
| *(not stored)* | `image_name` |
| `item_number` | `item_number` |
| `enterprise_name` | `enterprise_number` |
| `mfr_name` | `manufacture_name` |
| `mfr_part_number` | `manufacture_part_number` |
| `web_desc` | `web_product_description` |
| `internal_description` | `motion_internal_desc` |
| `pgc` | `pgc` |
| `category` | `pgc_description` |

### Query Priority Order

`vector_to_query()` constructs the search string with fields ordered by specificity:

1. `manufacture_part_number` (most specific)
2. `manufacture_name`
3. `item_number`
4. `enterprise_number`
5. `pgc_description`
6. `web_product_description` (truncated to 8 words)
7. `motion_internal_desc` (truncated to 8 words)

**Example output** for product `s10807860`:
```
"21315 E/C3 SKF 02132770 ST.SM.SPHER.THRU SIZE 48 Spherical Roller Bearing 75 mm"
```

---

## Production Deployment (GCP)

For production deployment, the pipeline requires access to Google Cloud Platform services.

### Contact at Motion Industries

- anu.shrestha@motion.com
- george.baldwin@motion.com

### GCP Services Needed

| Service | Purpose |
|---|---|
| Cloud Storage | Image/object storage (replaces MinIO) |
| Secret Manager | API keys, scraper credentials, ES database credentials |
| Cloud Run | Containerized pipeline services and APIs |
| Artifact Registry | Docker image storage |
| Cloud Logging | Debugging and audit trails |
| Pub/Sub | Event-based triggers (e.g., requesting images for a product without manual scraper execution) |
| Vertex AI | Running image classifier and text models |
| Cloud Scheduler | Scheduled re-scraping for stale or low-quality images |

### IAM Permissions — Users

| Role | Purpose |
|---|---|
| Cloud Run Developer | Running cloud services |
| Artifact Registry Reader/Writer | Pushing Docker images |
| Storage Object Admin | Access to image storage bucket(s) |
| Secret Manager / Secret Accessor | Reading API keys and credentials |
| Logs Viewer | Debugging |
| Service Account User | Permission to use runtime service account |
| Vertex AI User | Invoking hosted model endpoints |

### IAM Roles — Pipeline Service Account

- Read/Write to image storage bucket
- Read access to secrets (API keys)
- Write logs (testing and debugging)
- Call Vertex AI endpoints for hosted models
- Publish to Pub/Sub (future use)

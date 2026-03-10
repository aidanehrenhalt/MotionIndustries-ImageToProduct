# ECE 4013 - Senior Design
## Image-to-Product Pipeline for Motion Industries

Automated pipeline that scrapes product images from open-source APIs, scores them with
confidence heuristics, stores images in MinIO (S3-compatible object storage), and indexes
metadata into Elasticsearch for search and review.

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
6. [MinIO — Image Storage](#minio--image-storage)
7. [Running the Scraper](#running-the-scraper)
8. [Utility Scripts](#utility-scripts)
9. [Project Structure](#project-structure)
10. [Further Documentation](#further-documentation)

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

### Indices

**`mi_products`** — One document per product from the CSV catalog.
- Document ID: `motion_product_id` (e.g., `s10807860`)
- Contains: product identifiers, manufacturer info, descriptions, search keywords
- Updated with `scrape_summary` after each scrape run (total images found, downloaded, avg score)

**`mi_candidate_images`** — One document per scraped image candidate (many-to-one with products).
- Document ID: `SHA1("{motion_product_id}:{image_url}")` — deterministic, so re-scraping upserts
- Contains: source metadata, license, dimensions, download results, confidence scores
- References the MinIO object key via `local_path` (when `storage_type` is `"minio"`)

Both indices use `dynamic: "strict"` — any field not in the mapping will be rejected.
See [ELASTICSEARCH.md](ELASTICSEARCH.md) for full field-by-field schema documentation.

### Quick Health Check

```bash
# Cluster health
curl 'http://localhost:9200/_cluster/health?pretty'

# Index document counts
curl 'http://localhost:9200/_cat/indices/mi_*?v'
```

### Kibana

Open http://localhost:5601 and navigate to **Management > Dev Tools** to run queries
interactively. See [ELASTICSEARCH.md](ELASTICSEARCH.md) for example queries.

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

## Utility Scripts

### `query_elasticsearch.py`

Terminal viewer for Elasticsearch results.

```bash
venv/bin/python src/web_scraping/query_elasticsearch.py
```

### `minio_es_match.py`

MinIO and Elasticsearch matching/verification utility. See [MinIO section](#minio--image-storage) for usage.

### `setup_elasticsearch.py`

Creates (or recreates) the two Elasticsearch indices. See [Elasticsearch section](#elasticsearch).

---

## Project Structure

```
MotionIndustries-ImageToProduct/
  docker-compose.yml              # Elasticsearch + Kibana + MinIO
  requirements.txt                # Python dependencies
  README.md                       # This file
  ELASTICSEARCH.md                # Detailed ES schema and query reference
  INTEGRATION_PLAN.md             # Edge cases, pitfalls, and integration details
  src/
    web_scraping/
      web_scraper.py              # Main scraper (Wikimedia + OpenVerse APIs)
      setup_elasticsearch.py      # Creates ES indices (run once)
      query_elasticsearch.py      # Terminal viewer for ES data
      minio_es_match.py           # MinIO <-> ES matching utility
      test_products_sample.csv    # 20-row test product catalog
      GCPAccess.md                # GCP service account setup notes
  output/
    json/                         # Scrape records (one JSON per product per run)
    images/                       # Locally downloaded images (when not using MinIO)
```

---

## Further Documentation

| Document | Contents |
|---|---|
| [ELASTICSEARCH.md](ELASTICSEARCH.md) | Full index schemas, query examples, data flow diagram |
| [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) | Edge cases, pitfalls, and integration details |
| [GCPAccess.md](src/web_scraping/GCPAccess.md) | GCP services and IAM setup for production deployment |

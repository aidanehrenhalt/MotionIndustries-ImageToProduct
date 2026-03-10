# Scraper → Elasticsearch & Image Storage Integration Plan

## Part 1: Elasticsearch Metadata Integration

### Problem 1: `download_meta` fields are silently dropped

In `scrape_product_images()` (`web_scraper.py` line ~421), `download_meta` is used to
compute `confidence_hints` but then only `downloaded` is kept in the `enriched` list.
The other five fields are thrown away before `enriched` is ever returned:

| Dropped field      | ES mapping field    |
|-------------------|---------------------|
| `local_path`      | `local_path`        |
| `file_size_bytes` | `file_size_bytes`   |
| `actual_width`    | `actual_width`      |
| `actual_height`   | `actual_height`     |
| `actual_format`   | `actual_format`     |
| `download_error`  | `download_error`    |

**Fix:** add these six fields to the `enriched.append()` call in `scrape_product_images()`.

---

### Problem 2: Field name mismatch

The `enriched` dict uses `width` and `height` for API-reported dimensions, but the
`mi_candidate_images` ES mapping uses `api_width` and `api_height`.

**Fix:** rename `width` → `api_width` and `height` → `api_height` in the
`enriched.append()` call.

---

### Problem 3: Product document is incomplete

`scrape_product_images()` returns a stripped `product` sub-dict with only 5 fields.
`mi_products` needs the full CSV row:

| Missing field          | Source                    |
|-----------------------|---------------------------|
| `item_number`         | `load_product_catalog()`  |
| `enterprise_name`     | `load_product_catalog()`  |
| `internal_description`| `load_product_catalog()`  |
| `pgc`                 | `load_product_catalog()`  |
| `search_keywords`     | `load_product_catalog()`  |

These are all available in the `product` dict built by `load_product_catalog()` — they
just never get passed through to `save_record()`.

**Fix:** pass the full `product` CSV row into the new ES write function (see Problem 4).

---

### Problem 4: No ES write step exists

`save_record()` only writes JSON. A new function is needed to handle the three-step ES
write sequence:

```
1. es.index()   →  index the product document into mi_products
2. bulk()       →  bulk-index all candidate images into mi_candidate_images
3. es.update()  →  update the product's scrape_summary sub-object
```

The JSON output from `save_record()` can remain alongside ES writes — useful for
debugging during the transition.

**Fix:** add `index_to_elasticsearch(es, product_row, record)` to `web_scraper.py`.

---

### Integration Steps (in order)

1. In `enriched.append()` — add the 6 `download_meta` fields; rename `width` → `api_width`
   and `height` → `api_height`
2. Pass the full product CSV row into the new ES write function alongside `record`
3. Add `index_to_elasticsearch(es, product_row, record)` function
4. Add `--es-url` CLI argument (default `http://localhost:9200`) and `--no-es` flag
5. In `__main__` — initialize the ES client if `--no-es` is not set; call
   `index_to_elasticsearch()` after each product scrape

---

### ES Write Pattern (reference)

```python
import hashlib
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch("http://localhost:9200")

# Step 1 — Index product document
es.index(index="mi_products", id=product_row["motion_product_id"], document={
    "motion_product_id":    product_row["motion_product_id"],
    "item_number":          product_row["item_number"],
    "enterprise_name":      product_row["enterprise_name"],
    "mfr_name":             product_row["mfr_name"],
    "mfr_name_text":        product_row["mfr_name"],
    "mfr_part_number":      product_row["mfr_part_number"],
    "mfr_part_number_text": product_row["mfr_part_number"],
    "description":          product_row["web_desc"],
    "internal_description": product_row.get("internal_description", ""),
    "pgc":                  product_row["pgc"],
    "category":             product_row["category"],
    "search_keywords":      product_row["search_keywords"],
    "catalog_loaded_at":    record["scraped_at"],
    "schema_version":       record["schema_version"],
})

# Step 2 — Bulk-index candidate images
actions = []
for img in record["candidate_images"]:
    doc_id = hashlib.sha1(
        f"{product_row['motion_product_id']}:{img['image_url']}".encode()
    ).hexdigest()
    actions.append({
        "_op_type": "index",
        "_index":   "mi_candidate_images",
        "_id":      doc_id,
        "_source":  {
            "motion_product_id": product_row["motion_product_id"],
            "mfr_name":          product_row["mfr_name"],
            "mfr_part_number":   product_row["mfr_part_number"],
            **img,   # all enriched image fields including the newly threaded download_meta fields
        },
    })
bulk(es, actions)

# Step 3 — Update scrape_summary on the product document
es.update(index="mi_products", id=product_row["motion_product_id"], doc={
    "scrape_summary": {**record["scrape_summary"], "last_scraped_at": record["scraped_at"]}
})
```

> **Re-scraping is safe.** Document IDs are deterministic — re-running the scraper on the
> same product upserts existing documents rather than creating duplicates.

---

---

## Part 2: Image File Storage

The scraper currently saves image files to the local `output/images/` folder. This works
on a single machine but breaks down for a shared team project — the review UI, ML pipeline,
and scraper would all need access to the same filesystem. The solution is **object storage**.

---

### Recommendation: MinIO for dev, S3-compatible API everywhere

**MinIO** is an open-source, S3-compatible object storage server that runs as a Docker
container alongside the existing Elasticsearch stack. The Python client (`boto3`) uses the
exact same API as AWS S3, Cloudflare R2, and Backblaze B2 — switching environments only
requires changing the endpoint URL, not any application code.

| Environment       | Service            | Cost                                  |
|------------------|--------------------|---------------------------------------|
| Local dev        | MinIO (Docker)     | Free                                  |
| Shared / prod    | Cloudflare R2      | Free up to 10 GB, no egress fees      |
| Shared / prod    | AWS S3             | Free tier 5 GB / 12 months; $0.023/GB |
| Shared / prod    | Backblaze B2       | $0.006/GB/month (cheapest paid)       |

**Cloudflare R2** is the most practical option for a senior design project — free up to
10 GB with no egress fees and S3-compatible, so no code changes when moving off local MinIO.

---

### How it fits into the scraper

Instead of writing raw bytes to a local file, `download_image()` uploads to the bucket
and returns the object key. The `local_path` field in `mi_candidate_images` stores the
object key (e.g., `images/s10807860_00_abc12345.jpg`). The review UI constructs
`{BASE_URL}/{object_key}` to display images — no shared filesystem needed.

---

### Adding MinIO to `docker-compose.yml`

```yaml
  minio:
    image: minio/minio
    container_name: mi_minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # MinIO web console (browse uploaded images in browser)
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data

volumes:
  esdata:
  minio_data:    # add alongside existing esdata volume
```

MinIO's web console at `http://localhost:9001` lets the team browse uploaded images
visually, similar to how Kibana works for ES documents.

---

### Required package

```
boto3>=1.34
```

---

### Upload pattern in `download_image()` (reference)

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",       # swap for real S3/R2 URL in production
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

object_key = f"images/{product_id}_{index:02d}_{url_hash}.{ext}"
s3.put_object(
    Bucket="mi-scraped-images",
    Key=object_key,
    Body=raw,
    ContentType=mime_type,
)
# Store object_key in the ES document instead of a local filepath
```

---

## Summary of Changes Required

| Task | File | Size |
|---|---|---|
| Thread `download_meta` fields into `enriched` | `web_scraper.py` | Small — 6 lines |
| Rename `width`/`height` → `api_width`/`api_height` | `web_scraper.py` | Trivial |
| Pass full product CSV row to ES write function | `web_scraper.py` | Small |
| Add `index_to_elasticsearch()` function | `web_scraper.py` | Medium |
| Add `--es-url` / `--no-es` CLI flags | `web_scraper.py` | Small |
| Add MinIO service to Docker Compose | `docker-compose.yml` | Small |
| Swap local file write → S3/MinIO upload in `download_image()` | `web_scraper.py` | Medium |
| Add `boto3` to requirements | `requirements.txt` | Trivial |

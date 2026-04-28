# MinIO

Reference for how the current pipeline uses MinIO for image storage.

Setup and run commands live in [README.md](../README.md).

## Role In The Pipeline

MinIO stores image binaries when the scraper runs with `--minio`.

- Elasticsearch keeps the metadata
- MinIO keeps the image object
- The Elasticsearch `local_path` field stores the MinIO object key

Default local endpoint:

- API: `http://localhost:9000`
- Console: `http://localhost:9001`

Default local credentials:

- Username: `minioadmin`
- Password: `minioadmin`

Default bucket:

- `mi-images`

## Object Key Layout

Object keys are written as:

```text
images/{product_id}/{product_id}_{index}_{url_hash}.{ext}
```

Example:

```text
images/s10807860/s10807860_00_c21b8dfa.jpg
```

Where:

- `product_id` is the Motion product ID
- `index` is the scraper candidate index
- `url_hash` is the first 8 chars of the image URL MD5
- `ext` comes from the detected image format

## How It Is Used

`src/web_scraping/web_scraper.py`:

- creates a boto3 S3 client pointed at MinIO
- auto-creates the bucket if it does not exist
- uploads image bytes with `put_object`
- stores the object key in the saved JSON record and Elasticsearch document

`src/Image_Classifier/classify_json_images.py`:

- integrated `classify_json_files()` can fetch MinIO-backed images in-memory for classification

`src/web_scraping/minio_es_match.py`:

- verifies MinIO and Elasticsearch consistency
- generates presigned URLs
- downloads MinIO images back to local disk

## Health Check

```bash
curl -sf http://localhost:9000/minio/health/live
```

## Common Local Checks

Verify MinIO and Elasticsearch match:

```bash
.venv/bin/python src/web_scraping/minio_es_match.py --verify
```

Show one product's MinIO-backed images:

```bash
.venv/bin/python src/web_scraping/minio_es_match.py --product s10807860
```

Show bucket and storage stats:

```bash
.venv/bin/python src/web_scraping/minio_es_match.py --list-bucket
.venv/bin/python src/web_scraping/minio_es_match.py --stats
```

Download a product's MinIO images locally:

```bash
.venv/bin/python src/web_scraping/minio_es_match.py --download s10807860
```

## Environment Variables

Defaults:

```bash
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=mi-images
```

## Notes That Matter

- Local Docker volumes preserve MinIO data across `docker-compose down`.
- `docker-compose down -v` deletes MinIO data.
- Re-uploading the same object key overwrites the existing object.
- The current local dev flow auto-creates the bucket; that is acceptable for local use, but production storage should be provisioned explicitly.

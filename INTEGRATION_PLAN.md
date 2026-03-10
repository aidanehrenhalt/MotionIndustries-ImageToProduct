# Integration Details — Edge Cases and Pitfalls

This document describes how the scraper, Elasticsearch, and MinIO integration works under
the hood, with emphasis on edge cases that can cause errors and patterns to avoid.

---

## Elasticsearch Strict Mapping

Both indices (`mi_products`, `mi_candidate_images`) use `dynamic: "strict"`. This means
Elasticsearch will **reject any document that contains a field not defined in the mapping**.

### What this means in practice

- If you add a new field to the scraper's output (e.g., in `index_to_elasticsearch()`),
  you **must** also add it to `setup_elasticsearch.py` and recreate the indices.
- If you forget, bulk indexing will silently fail for those documents. The scraper logs
  the error count (e.g., `Indexed 0 candidate images for s10807860 (2 errors)`) but does
  not abort — other products continue processing.
- To fix: update the mapping in `setup_elasticsearch.py`, then run
  `venv/bin/python src/web_scraping/setup_elasticsearch.py --recreate` and re-scrape.

### Why strict mode?

Permissive (`dynamic: true`) would silently accept typos or unexpected fields and auto-create
mappings with potentially wrong types (e.g., a number field might get mapped as `text`).
Strict mode catches these mistakes early.

---

## Field Name Mismatch: `width`/`height` vs. `api_width`/`api_height`

The source APIs (Wikimedia, OpenVerse) return dimensions as `width` and `height`. The
enriched image dict in `scrape_product_images()` keeps these names. However, the ES mapping
uses `api_width` and `api_height` to distinguish API-reported dimensions from
Pillow-verified `actual_width` and `actual_height`.

The translation happens in `index_to_elasticsearch()`:

```python
"api_width":  img.get("width"),
"api_height": img.get("height"),
```

If you index the raw enriched dict directly (e.g., using `**img`), the `width` and `height`
fields will be rejected by the strict mapping. Always use the explicit field mapping in
`index_to_elasticsearch()`.

---

## Document ID Determinism

### Products

Product document `_id` = `motion_product_id` (from the CSV). Re-scraping the same product
overwrites the existing document.

### Candidate Images

Image document `_id` = `SHA1("{motion_product_id}:{image_url}")`. This means:

- The same product + same image URL always produces the same doc ID (upsert, no duplicates).
- If an image URL changes (e.g., Wikimedia updates the file), it creates a new document
  and the old one becomes orphaned. Use `minio_es_match.py --verify` to detect these.
- The SHA1 is computed over the **full image URL**, including query parameters. Two URLs
  that differ only in query string (e.g., `?width=800` vs. `?width=1200`) produce
  different doc IDs.

---

## MinIO Object Key Structure

Images are stored in MinIO under: `images/{product_id}/{product_id}_{index}_{url_hash}.{ext}`

- `product_id`: the `motion_product_id` from the CSV
- `index`: zero-padded position in the scraper's result list (e.g., `00`, `01`)
- `url_hash`: first 8 characters of the MD5 hash of the image URL (for uniqueness)
- `ext`: file extension derived from Pillow's detected format (not from the URL)

The same object key is stored in the ES `local_path` field, so ES and MinIO stay in sync.

### Bucket auto-creation

`create_minio_client()` calls `head_bucket()` and creates the bucket if it doesn't exist.
This is safe for local dev but should be disabled in production — buckets should be
pre-provisioned with appropriate access policies.

---

## Download Validation

`download_image()` validates every downloaded file with Pillow before storing it:

1. Checks that the HTTP response `Content-Type` starts with `image/`
2. Opens the raw bytes with `Image.open()` and calls `img.verify()` to confirm it's a
   valid image (not a corrupt file or HTML error page)
3. Re-opens after verify (Pillow requires this) to extract actual dimensions and format

If any step fails, `downloaded` is set to `False` and `download_error` captures the
exception message. The image is still recorded in ES (with metadata from the API) but
no file is uploaded to MinIO.

### Common download failures

| Error | Cause | Effect |
|---|---|---|
| `URL did not return an image` | Server returned HTML/JSON instead of an image | Skipped, error logged |
| `HTTPError 403` | Source blocks scraper user-agent or hotlinking | Skipped, error logged |
| `ConnectionTimeout` | Source server unresponsive | Skipped, error logged |
| `UnidentifiedImageError` | File is corrupt or not a supported image format | Skipped, error logged |

None of these abort the scraper — it continues to the next image.

---

## Elasticsearch Bulk Indexing

`index_to_elasticsearch()` uses `bulk()` with `raise_on_error=False`. This means:

- If some documents in the batch fail (e.g., mapping violation), the successful ones are
  still indexed.
- The function logs the count of errors but does **not** raise an exception.
- If you need to debug bulk errors, temporarily set `raise_on_error=True` or inspect the
  `errors` list returned by `bulk()`.

### Partial failures are silent

If 5 out of 7 images index successfully and 2 fail, you'll see
`Indexed 5 candidate images for s10807860 (2 errors)`. The product's `scrape_summary`
will still report `total_images_found: 7` and `images_downloaded: 7` (since download
succeeded), but only 5 will be queryable in ES. This mismatch can be confusing.

To detect: compare `scrape_summary.images_downloaded` against the actual document count
for that product in `mi_candidate_images`.

---

## Global Variable Scope in `__main__`

The MinIO config variables (`MINIO_ENDPOINT`, `MINIO_BUCKET`) are defined at module level.
The `if __name__ == "__main__"` block is also at module level, so these variables can be
reassigned directly — no `global` declaration is needed.

Using `global` inside a conditional block at module scope causes a `SyntaxError` in
Python 3.12+. If you need to override these values from CLI args, assign them directly:

```python
if args.minio_endpoint:
    MINIO_ENDPOINT = args.minio_endpoint  # No "global" needed
```

The `global` keyword is only required inside **functions** that need to reassign a
module-level variable.

---

## Rate Limiting and API Etiquette

The scraper includes a `REQUEST_DELAY` (1.5 seconds) between API calls. Both Wikimedia
and OpenVerse have rate limits:

- **Wikimedia Commons:** No strict rate limit, but requests aggressive use of `User-Agent`
  identification. The scraper includes a project-specific user agent.
- **OpenVerse:** Rate-limited; unauthenticated requests are throttled more aggressively.
  If you see `429 Too Many Requests` errors, increase `REQUEST_DELAY`.

### Wikimedia title encoding

Wikimedia file pages use URL-encoded titles. The scraper uses `urllib.parse.quote()` to
construct `source_page` URLs. If a title contains special characters (e.g., parentheses,
Unicode), incorrect encoding will produce broken links in ES but won't cause scraper errors.

---

## Re-scraping Behavior

Because document IDs are deterministic:

- **Products:** Re-scraping overwrites the product document with fresh `catalog_loaded_at`
  and `scrape_summary`.
- **Images:** Re-scraping the same URL overwrites the existing image document. New images
  (new URLs) create new documents. Old images (URLs that no longer appear in API results)
  are **not deleted** — they remain in ES as stale records.
- **MinIO:** Re-uploading the same object key overwrites the file. There is no versioning
  enabled by default.

To clean up stale data, use `setup_elasticsearch.py --recreate` to wipe indices and
re-scrape from scratch.

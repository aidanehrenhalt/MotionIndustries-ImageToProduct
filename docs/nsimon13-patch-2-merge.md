# Integration Plan: `nsimon13-patch-2` into `WebScraper`

## Goal
Integrate the **Text-Based Search** from `nsimon13-patch-2` as the query builder that feeds into the Web Scraper, improving image search results before scraping. Defer `Model_Development/` and ML model integration for later.

## Pipeline Flow (Target)
```
CSV Catalog
  → load_product_catalog() (existing)
  → product_to_vector() (new bridge function)
  → vector_to_query() (from text_based_search.py)
  → optimized search query string
      ↓
Web Scraper (scrape_wikimedia, scrape_openverse, etc.)
  → candidate images
      ↓
Elasticsearch + MinIO (existing)
```

---

## Step 1 — Bring in the file (cherry-pick + rename)

Grab only `text_based_search (2).py` from the branch, renamed cleanly:

```bash
git show origin/nsimon13-patch-2:"text_based_search (2).py" > src/web_scraping/text_based_search.py
```

This avoids pulling in `Model_Development/`, the older `text_based_search.py`, `catalog_search (1).py`, and `image_search_ranker (3).py`.

---

## Step 2 — Bridge the data models

The scraper's product dict keys don't match `ProductVector` field names. Add a converter function in `web_scraper.py`:

```python
from text_based_search import ProductVector, vector_to_query

def product_to_vector(product: dict) -> ProductVector:
    """Convert scraper product dict to ProductVector for query building."""
    return ProductVector(
        id_number=product.get("motion_product_id", ""),
        image_name="",
        item_number=product.get("item_number", ""),
        enterprise_number=product.get("enterprise_name", ""),
        manufacture_name=product.get("mfr_name", ""),
        manufacture_part_number=product.get("mfr_part_number", ""),
        web_product_description=product.get("web_desc", ""),
        motion_internal_desc=product.get("internal_description", ""),
        pgc=product.get("pgc", ""),
        pgc_description=product.get("category", ""),
    )
```

### Field mapping reference

| Scraper dict key (`product[...]`) | ProductVector field |
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

---

## Step 3 — Replace keyword builder in `load_product_catalog()`

**Current behavior** (web_scraper.py lines 99-128): builds `search_keywords` with simple heuristics — first 5 words of description, manufacturer + description snippet.

**New behavior**: use `vector_to_query()` which produces a prioritized query string ordered by:
1. `manufacture_part_number` (most specific)
2. `manufacture_name`
3. `item_number`
4. `enterprise_number`
5. `pgc_description`
6. `web_product_description` (truncated to 8 words)
7. `motion_internal_desc` (truncated to 8 words)

Replace the keyword-building block in `load_product_catalog()` with:

```python
# Build the product dict first (same as current)
product_dict = {
    "motion_product_id": ...,
    "item_number": ...,
    # ... all existing fields ...
}

# Replace old keyword heuristics with vector_to_query
vector = product_to_vector(product_dict)
product_dict["search_keywords"] = [vector_to_query(vector)]

products.append(product_dict)
```

**Example output** for product `s10807860`:
```
"21315 E/C3 SKF 02132770 ST.SM.SPHER.THRU SIZE 48 Spherical Roller Bearing 75 mm"
```

---

## What stays the same (no changes needed)

- `scrape_wikimedia()`, `scrape_openverse()` — they already accept a keyword string
- `scrape_product_images()` — already iterates over `search_keywords`
- `download_image()`, `compute_confidence_hints()` — unchanged
- Elasticsearch indexing (`index_to_elasticsearch()`) — unchanged
- MinIO upload — unchanged
- CLI arguments — unchanged

---

## What is NOT included (deferred for later)

| File | Purpose | When to integrate |
|---|---|---|
| `catalog_search (1).py` | Excel/CSV loader with `CatalogSearch` class | When you need Excel support or interactive catalog search |
| `image_search_ranker (3).py` | Scores/ranks candidates against ProductVector | Replaces `compute_confidence_hints()` — integrate after text search is working |
| `text_based_search.py` (old, from main) | Simpler version, opens browser tabs | Superseded by `text_based_search (2).py` |
| `Model_Development/` | ML training notebook, checkpoint, datasets | Integrate from `main` branch when ready for ML pipeline |

---

## Requirements

`text_based_search.py` optionally depends on Playwright (for `fetch_first_image()`), but we are **not using that function** — we only use `vector_to_query()` which has zero additional dependencies.

No new pip installs required for this integration.

---

## Potential `requirements.txt` conflict

Both `WebScraper` and `nsimon13-patch-2` (which includes `main`) modified `requirements.txt` independently. When merging, combine both sets:
- **WebScraper additions**: elasticsearch, boto3, minio-related
- **main/nsimon additions**: torch, torchvision, ML-related

For now (text search only), no new requirements are needed.

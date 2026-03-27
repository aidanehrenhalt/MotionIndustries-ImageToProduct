# Project Backlog

Progress log covering `WebScraper`, `WebScraper_ClassifierIntegration`, and `Demo_MFR_Site` branches since January 2026.

---

## Week of Feb 16–22

**1 commit** — Initial scaffolding

- `6233dfc` — Started the web scraper; transferred existing code for scraping Wikimedia Commons and OpenVerse APIs between devices
- Set up project structure under `src/web_scraping/`

---

## Week of Feb 23–Mar 1

**2 commits** — Core scraper functional

- `8d0b4af` — Web scraper mostly complete; debugging API request issues with OpenVerse
- `735d06b` — Scraper successfully downloads images with JSON metadata
  - Identified need to fix image/JSON identifier pairing
  - Next steps noted: improve scraper, implement actual confidence scoring (with Rodrigo), start using ElasticSearch

---

## Week of Mar 2–8

**No commits** — Gap week: No major changes

---

## Week of Mar 9–15

**10 commits + uncommitted work** — Major sprint

### Mar 10 — ElasticSearch, MinIO, and housekeeping

- `acc65db` — Added README files for ElasticSearch setup
- `acd7ca7` — Rough MinIO implementation for image storage; created plan for GCP access (need to email Motion contacts)
- `9a56fe9` — Cleaned up `.md` docs; fixed MinIO implementation; corrected global variable usage (important for future GCP Cloud Storage migration)
- `124a99b` — Removed `venv/` from repo; updated `requirements.txt`; removed `test_scraper.py`
- `11c919c` — Pinned MinIO version in docker-compose (prevent breakage from auto-updates); set healthcheck to curl
- `b95a911` — Cleaned up repo; updated docs; tested full pipeline (web scraper → ElasticSearch → MinIO)

### Mar 11 — Non-API scraping, text search, documentation

- `fc2cb19` — Updated web scraper to scrape non-API sources; added `.md` outlining how to merge Nia × Ace branches
- `b7737a6` — Fixed `beautifulsoup4` import in `requirements.txt`; began integrating Nia's text-based search
- `b6f1a9f` — Integrated Nia's text-based search; testing showed queries were too narrow for current API sources; created a simpler search version that successfully returns output to ElasticSearch (viewable via curl or Kibana) and MinIO
- `db993fb` — Compiled all `.md` docs into a single `README.md` with installation, web scraping, and query instructions; added Motion Industries shared OneDrive folder link

### Uncommitted / In-Progress (as of Mar 15)

- **Manufacturer scrapers rewrite** (`manufacturer_scrapers.py`, +705 lines) — Two-tier architecture:
  - Tier 1: Documented manufacturer portals with known URL patterns and CSS selectors (requests + BeautifulSoup, no browser needed)
  - Tier 2: Generic DuckDuckGo fallback using Playwright for unknown manufacturers
  - Added thread-safe rate limiting, browser-like request headers, per-domain locks
- **Manufacturer scraper tests** (`test_manufacturer_scrapers.py`, +259 lines) — Test suite for the new two-tier scraper
- **Web scraper improvements** (`web_scraper.py`, +183 lines):
  - Improved keyword generation: uses part number + description, manufacturer + description
  - Better CSV parsing: handles both title-case and UPPER_SNAKE column names, `utf-8-sig` encoding, `[<ID>]` bracket format
  - Falls back to `description` field when `web_desc` is missing
- **Catalog ingestion script** (`ingest_catalog.py`, new) — Bulk-loads full product catalog CSV into ElasticSearch `mi_products` index without triggering image scraping; supports batch sizing and upsert semantics
- **ElasticSearch setup update** (`setup_elasticsearch.py`, +2 lines)
- **Requirements update** (`requirements.txt`, +3 lines)
- **94 JSON output files** in `src/web_scraping/output/json/` — Scrape results from test runs
- **CSV data files** — `ImageToProduct-Missing_Product_Images.csv`, `UniqueEnterpriseAndMFR.csv`
- **Source documentation** — `docs/SourceProductImages.md`

---

## Week of Mar 16–22 (`WebScraper_ClassifierIntegration`)

**1 commit** — CNN classifier integration

- `683280e` — Merged `scrapper+img_classifier` → `WebScraper_ClassifierIntegration`; integrated CNN image classifier into the WebScraper pipeline
  - **New `src/Image_Classifier/`**: `img_classifier.py` (8-class sequential CNN: 3 conv blocks, batch norm, dropout, 500×500 input, 1200→512→8 FC layers), `classify_json_images.py` (batch classifier, fully rewritten with dynamic paths and public API: `build_model()`, `load_model()`, `make_preprocess()`, `classify_image()`, `classify_json_dir()`), `trained_model.pth`, `cleaned_product_list.xlsx`
  - **New `Model_Development/`**: `training.py`, `training_notebook.ipynb`, `class_analysis.py`, `filtering_images.py`, checkpoint
  - `setup_elasticsearch.py`: added `predicted_class` field (int, -1 = unclassified) to `mi_candidate_images` index — **existing deployments must run `--recreate`**
  - `web_scraper.py`: added `--classify` flag; classifier loaded dynamically via `importlib.util` (torch optional); added `push_predictions_to_es()` to sync class predictions back to ES
  - Full one-shot pipeline: `ingest_catalog.py` → `web_scraper.py --classify --es --minio` → `classify_json_images.py`

---

## Week of Mar 23–27 (`WebScraper_ClassifierIntegration` + `Demo_MFR_Site`)

### Mar 24 — Merge from WebScraper + scraper–model pipeline scaffolding

- `21b44e6` — Updated `.gitignore` to exclude virtual environment files
- `a65e60c` — Standardized to `.venv/` for virtual environment directory name
- `3dce510`, `4065c31`, `65c95c7`, `3acb40b` — Merged most recent `WebScraper` changes into `WebScraper_ClassifierIntegration`; resolved conflicts in `README.md`, `.gitignore`, `requirements.txt`
- `fdb46ba` — Scraper → model pipeline scaffolding:
  - Added `src/Image_Classifier/assemble_dataset.py` (+321 lines) — builds training manifest CSV from scraped JSON + product CSV
  - Added `src/Image_Classifier/pgc_mapping.py` (+53 lines) — PGC class/index mapping
  - Added `src/Image_Classifier/train.py` (+252 lines) — new standalone training script
  - Updated `classify_json_images.py` (+66 lines)
  - Added `docs/PIPELINE_RUNBOOK.md` (+185 lines) — step-by-step runbook for full pipeline
  - Note: pipeline needs more extensive testing before relying on results

### Mar 26 — Pipeline review + test suite

- `a4368e3` — Full scraper/classifier pipeline review and tests:
  - Added `SCRAPER_ML_PIPELINE_REVIEW.md` (+782 lines) — comprehensive review of scraper ↔ classifier integration
  - Added `tests/test_scraper_classifier_pipeline.py` (+203 lines) — end-to-end test suite
  - Updated `classify_json_images.py` (+147 lines) and `web_scraper.py` (+55 lines)

### Mar 27 — Troubleshooting → pivot to demo site

- `92e0cf9` — Troubleshooting: real manufacturer portals block automated requests, making end-to-end pipeline testing against live sites infeasible
  - Added `PIPELINE_TEST_STEPS.md` (+186 lines) — documents troubleshooting steps and findings
  - Added `src/web_scraping/test_products_tier1_classifier.csv` — test product subset for Tier 1 scraper + classifier
  - Updated `manufacturer_scrapers.py` and `test_manufacturer_scrapers.py`
  - Cleaned up ~94 stale JSON output files and test images
  - **Decision**: pivot to building a demo manufacturer site to unblock local testing

**`Demo_MFR_Site` branch (branched from `WebScraper_ClassifierIntegration`, Mar 27)**

- `8184a8b` — Functioning demo site and demo pipeline:
  - `src/demo_mfr_site/scrape_ami_pages.py` — scraper targeting the local demo site
  - `src/demo_mfr_site/build_demo_site.py` (+476 lines) — generates a static HTML product catalog site
  - `src/demo_mfr_site/run_demo_pipeline.py` (+212 lines) — orchestrates demo pipeline (scrape → classify → output)
  - `src/demo_mfr_site/README.md` (+49 lines)
  - 4 sample AMI products: `mb2-10`, `mser204-12`, `ucst207-22`, `uct305` — with scraped HTML/Markdown/JSON, pipeline output JSON, product images, and a rendered static site

- `3710a4f` — Demo site finalization + documentation overhaul:
  - Removed `tempScraper+Model.md` (superseded by runbook) and `scrape_ami_pages.py` (logic folded into other scripts)
  - Reorganized docs: moved `ELASTICSEARCH.md`, `INTEGRATION_PLAN.md`, `GCPAccess.md`, `SourceProductImages.md`, `backlog.md` into `docs/`; rewrote `README.md` to project overview
  - Updated `docs/PIPELINE_RUNBOOK.md`; added `docs/MINIO.md` (+118 lines); added `docs/SCRAPER_ML_PIPELINE_REVIEW.md`

---

## Remaining Tasks

### Closed / resolved
- ~~Finalize and commit the manufacturer scrapers two-tier rewrite~~ — committed in `92e0cf9`
- ~~Finalize and commit the catalog ingestion pipeline (`ingest_catalog.py`)~~ — committed in `683280e`
- ~~Commit or clean up the 94 JSON test output files~~ — cleaned up in `92e0cf9`

### Active blockers
- **Real-site scraping blocked**: live manufacturer portals reject automated requests — `Demo_MFR_Site` unblocks local testing, but the gap against production sources remains open

### From code TODOs
- Switch to `vector_to_query(product_to_vector(product_dict))` for search queries (`web_scraper.py:185`)

### From commit messages and branch context
- Fix image/JSON identifier pairing so metadata and images are linked consistently
- Implement actual confidence scoring (collaboration with Rodrigo)
- Email Motion contacts about GCP access
- Migrate from local MinIO to GCP Cloud Storage (global variable refactor already done)
- Existing ElasticSearch deployments must run `python setup_elasticsearch.py --recreate` to apply the `predicted_class` field

### Open work
- Test the full pipeline end-to-end using the demo site; confirm `predicted_class` predictions flow to ES
- More extensive testing of scraper → model pipeline (noted in `fdb46ba`)
- Review and approve individual manufacturers in `MANUFACTURER_REGISTRY` (robots.txt + ToS check required per manufacturer)
- Install and configure Playwright for Tier 2 scraping (`pip install playwright && playwright install chromium`)
- Address narrow query issue from text-based search testing — queries need broadening for current API sources
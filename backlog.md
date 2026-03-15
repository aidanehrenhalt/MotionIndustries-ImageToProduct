# WebScraper Branch Backlog

Progress log for the `WebScraper` branch since January 2026.

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
- **Source documentation** — `SourceProductImages.md`

---

## Remaining Tasks

### From code TODOs
- Switch to `vector_to_query(product_to_vector(product_dict))` for search queries (`web_scraper.py:185`)

### From commit messages and branch context
- Fix image/JSON identifier pairing so metadata and images are linked consistently
- Implement actual confidence scoring (collaboration with Rodrigo)
- Email Motion contacts about GCP access
- Migrate from local MinIO to GCP Cloud Storage (global variable refactor already done in preparation)

### From uncommitted work
- Finalize and commit the manufacturer scrapers two-tier rewrite
- Finalize and commit the catalog ingestion pipeline (`ingest_catalog.py`)
- Review and approve individual manufacturers in `MANUFACTURER_REGISTRY` (robots.txt + ToS check required per manufacturer)
- Install and configure Playwright for Tier 2 scraping (`pip install playwright && playwright install chromium`)
- Address narrow query issue identified during text-based search testing — queries need broadening for current API sources
- Commit or clean up the 94 JSON test output files

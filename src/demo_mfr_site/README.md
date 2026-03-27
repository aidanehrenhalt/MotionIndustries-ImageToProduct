# AMI Demo Site

This folder contains a local AMI Bearings demo catalog built from 4 live product pages.

## Files

- `test_product_urls.md`: Source URL list for the AMI pages to scrape
- `scrape_ami_pages.py`: Uses `crawl4ai` to scrape the source pages into raw artifacts
- `scraped_data/`: Saved crawl output including raw HTML, markdown, and summary JSON
- `build_demo_site.py`: Normalizes the crawl output, downloads local product images, and builds the static demo site
- `site/`: Generated static website for local pipeline testing

## Rebuild Workflow

From the repo root:

```bash
CRAWL4_AI_BASE_DIRECTORY=/home/aceaid/MotionIndustries-ImageToProduct/.crawl4ai_runtime \
./.venv/bin/python src/demo_mfr_site/scrape_ami_pages.py

./.venv/bin/python src/demo_mfr_site/build_demo_site.py
```

## Serve Locally

```bash
cd src/demo_mfr_site/site
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/products/uct305.html`

## Data Contract

The normalized product dataset is written to:

- `site/assets/data/products.json`

Each product includes:

- `part_number`
- `item_name`
- `breadcrumbs`
- `specs`
- `images`
- `source_url`

# Web Scraper

Scrapes product images from open-licensed image sources (Wikimedia Commons, OpenVerse) and optionally manufacturer websites. Results are saved as JSON records and images stored locally or in MinIO. Supports indexing into Elasticsearch and post-scrape CNN classification.

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `requests`, `beautifulsoup4`, `Pillow`, `lxml`, `elasticsearch`, `boto3`

Optional: `selenium` + `chromedriver` (manufacturer site scraping), `torch` + `torchvision` (classification)

## Usage

```bash
python web_scraper.py --csv <file.csv> [OPTIONS]
python web_scraper.py --from-es [OPTIONS]
```

Either `--csv` or `--from-es` is required (mutually exclusive).

## Options

| Flag | Description |
|------|-------------|
| `--csv <path>` | Path to product catalog CSV file |
| `--from-es` | Pull product list from Elasticsearch (`mi_products` index) |
| `--mfr-filter <name>` | Filter by manufacturer name (used with `--from-es`) |
| `--enterprise-filter <name>` | Filter by enterprise name (used with `--from-es`) |
| `--product <keyword>` | Filter products by ID, manufacturer, or category keyword |
| `--limit <n>` | Limit number of products scraped (useful for testing) |
| `--no-download` | Skip image downloads; scrape metadata only |
| `--es` | Index results into Elasticsearch |
| `--es-host <host>` | Elasticsearch host (default: `localhost`) |
| `--es-port <port>` | Elasticsearch port (default: `9200`) |
| `--minio` | Upload images to MinIO instead of local filesystem |
| `--minio-endpoint <url>` | MinIO endpoint URL (default: env `MINIO_ENDPOINT`) |
| `--minio-bucket <name>` | MinIO bucket name (default: env `MINIO_BUCKET`) |
| `--mfr-scraping` | Enable Tier 1 manufacturer site scraping (requests + BS4) |
| `--mfr-tier2` | Enable Tier 2 `og:image` fallback for unknown manufacturers (requires Playwright) |
| `--mfr-only` | Skip Wikimedia/OpenVerse; use only manufacturer scrapers |
| `--manufacturer-sites` | Enable Selenium-based manufacturer site scraping |
| `--classify` | Run CNN image classifier on downloaded images after scraping |

## Image Sources

| Tier | Source | Method |
|------|--------|--------|
| Open APIs | Wikimedia Commons | REST API |
| Open APIs | OpenVerse | REST API (commercial + modification licenses) |
| Tier 1 | Manufacturer sites | requests + BeautifulSoup (`--mfr-scraping`) |
| Tier 2 | Unknown manufacturer sites | Playwright `og:image` fallback (`--mfr-tier2`) |
| Selenium | Manufacturer sites | Selenium WebDriver (`--manufacturer-sites`) |

## Output

```
output/
├── images/   # Downloaded image files (or MinIO: images/<product_id>/<filename>)
└── json/     # One JSON record per product (<product_id>_<timestamp>.json)
```

Each JSON record contains product metadata, a scrape summary, and a ranked list of candidate images with confidence hints (keyword match, license, resolution, source reliability).

## Elasticsearch Indices

| Index | Contents |
|-------|----------|
| `mi_products` | Product catalog with scrape summary |
| `mi_candidate_images` | Per-image records with confidence scores and (optionally) predicted class |

## MinIO Configuration

Set via environment variables or CLI flags:

| Variable | Default |
|----------|---------|
| `MINIO_ENDPOINT` | `http://localhost:9000` |
| `MINIO_ACCESS_KEY` | `minioadmin` |
| `MINIO_SECRET_KEY` | `minioadmin` |
| `MINIO_BUCKET` | `mi-images` |

Start MinIO and Elasticsearch locally with:
```bash
docker-compose up -d
```

## Examples

```bash
# Scrape all products from a CSV
python web_scraper.py --csv test_products_sample.csv

# Full pipeline: ES source, filter by manufacturer, upload to MinIO, classify
python web_scraper.py --from-es --mfr-filter SKF --es --minio --classify

# Test run — first 5 products, no downloads, print metadata only
python web_scraper.py --csv test_products_sample.csv --limit 5 --no-download

# Manufacturer scraping only, no open APIs
python web_scraper.py --csv test_products_sample.csv --mfr-scraping --mfr-only
```

## Notes

- Search queries are generated from manufacturer part number + first 5 words of product description, with a brand-context variant using manufacturer name. Falls back to PGC category if no description is available.
- A 1.5-second delay is applied between API requests to respect rate limits.
- Up to 3 images are fetched per source per product.
- The CNN classifier (`--classify`) requires `trained_model.pth` at `src/Image_Classifier/trained_model.pth` relative to the project root.

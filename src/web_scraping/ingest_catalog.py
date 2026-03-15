"""
Catalog Ingestion Script

Bulk-loads the full product catalog CSV into the Elasticsearch `mi_products`
index without triggering any image scraping. Run this once (or after catalog
updates) before using `web_scraper.py --from-es` to query products by
manufacturer or enterprise.

Usage:
    python ingest_catalog.py --csv /path/to/full_catalog.csv
    python ingest_catalog.py --csv /path/to/full_catalog.csv --es-host localhost --es-port 9200
    python ingest_catalog.py --csv /path/to/full_catalog.csv --batch-size 1000

Re-running is safe: uses index (upsert) semantics so no duplicates are created.
"""

import sys
import logging
import argparse

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Reuse the CSV parser already defined in web_scraper.py
from web_scraper import load_product_catalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


def _product_to_es_action(product: dict) -> dict:
    """Build an Elasticsearch bulk action for a single product dict."""
    pid = product.get("motion_product_id", "")
    return {
        "_op_type": "index",   # upsert: safe to re-run
        "_index": "mi_products",
        "_id": pid,
        "_source": {
            "motion_product_id": pid,
            "item_number": product.get("item_number", ""),
            "enterprise_name": product.get("enterprise_name", ""),
            "mfr_name": product.get("mfr_name", ""),
            "mfr_part_number": product.get("mfr_part_number", ""),
            "web_desc": product.get("web_desc", ""),
            "internal_description": product.get("internal_description", ""),
            "pgc": product.get("pgc", ""),
            "category": product.get("category", ""),
            "primary_image_filename": product.get("primary_image_filename", ""),
            # scrape_summary is seeded empty; web_scraper.py fills it later
            "scrape_summary": {},
        },
    }


def ingest(csv_path: str, es: Elasticsearch, batch_size: int = 500) -> None:
    """Load CSV and bulk-index all products into mi_products."""
    products = load_product_catalog(csv_path)
    if not products:
        log.error("No products loaded from CSV. Nothing to ingest.")
        sys.exit(1)

    actions = [_product_to_es_action(p) for p in products]
    log.info(f"Indexing {len(actions)} products into mi_products (batch_size={batch_size})...")

    success, errors = bulk(es, actions, chunk_size=batch_size, raise_on_error=False)
    log.info(f"Done — indexed {success} products, {len(errors)} error(s)")
    for err in errors[:10]:
        log.warning(f"  Bulk error: {err}")
    if len(errors) > 10:
        log.warning(f"  ... and {len(errors) - 10} more errors (check ES logs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk-load product catalog CSV into Elasticsearch mi_products index",
        epilog="Example: python ingest_catalog.py --csv /data/full_catalog.csv",
    )
    parser.add_argument("--csv", required=True, help="Path to product catalog CSV file")
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch host (default: localhost)")
    parser.add_argument("--es-port", default=9200, type=int, help="Elasticsearch port (default: 9200)")
    parser.add_argument("--batch-size", default=500, type=int, help="Bulk index batch size (default: 500)")
    args = parser.parse_args()

    es_url = f"http://{args.es_host}:{args.es_port}"
    es = Elasticsearch(es_url)
    try:
        info = es.info()
        log.info(f"Connected to Elasticsearch {info['version']['number']} at {es_url}")
    except Exception as e:
        log.error(f"Could not connect to Elasticsearch at {es_url}: {e}")
        log.error("Make sure it's running: docker-compose up -d")
        sys.exit(1)

    ingest(args.csv, es, batch_size=args.batch_size)

    count = es.count(index="mi_products")["count"]
    log.info(f"mi_products now contains {count} documents")

"""
Elasticsearch Index Setup Script
Motion Industries Image-to-Product Scraper — GT Senior Design

Creates two indices:
  - mi_products:          One document per product from the CSV catalog
  - mi_candidate_images:  One document per scraped candidate image (many-to-one with products)

Usage:
    python setup_elasticsearch.py                          # Create indices (skip if already exist)
    python setup_elasticsearch.py --recreate               # Drop and recreate both indices
    python setup_elasticsearch.py --host 10.0.0.5 --port 9200
"""

import argparse
import sys

from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError

# ── Index names ────────────────────────────────────────────────────────────────
PRODUCTS_INDEX = "mi_products"
IMAGES_INDEX   = "mi_candidate_images"

# ── Index definitions ──────────────────────────────────────────────────────────

PRODUCTS_SETTINGS = {
    "number_of_shards":   1,
    "number_of_replicas": 0,   # 0 replicas for single-node dev; raise to 1 for multi-node
    "analysis": {
        "analyzer": {
            "part_number_analyzer": {
                "type":      "custom",
                "tokenizer": "standard",
                "filter":    ["lowercase", "asciifolding"],
            }
        }
    },
}

PRODUCTS_MAPPINGS = {
    "dynamic": "strict",
    "properties": {
        # ── Product identifiers ────────────────────────────────────────────────
        "motion_product_id":    {"type": "keyword"},
        "item_number":          {"type": "keyword"},
        "enterprise_name":      {"type": "keyword"},

        # Manufacturer — keyword for exact filtering/aggregation, text for full-text search
        "mfr_name":             {"type": "keyword"},
        "mfr_name_text":        {"type": "text", "analyzer": "standard"},
        "mfr_part_number":      {"type": "keyword"},
        "mfr_part_number_text": {"type": "text", "analyzer": "part_number_analyzer"},

        # ── Descriptions ───────────────────────────────────────────────────────
        "description": {
            "type":     "text",
            "analyzer": "english",
            "fields":   {"keyword": {"type": "keyword", "ignore_above": 512}},
        },
        "web_desc":             {"type": "text", "analyzer": "english"},
        "internal_description": {"type": "text", "analyzer": "english"},
        "primary_image_filename": {"type": "keyword", "index": False},

        # ── Category / PGC ────────────────────────────────────────────────────
        "pgc":      {"type": "keyword"},
        "category": {
            "type":     "text",
            "analyzer": "english",
            "fields":   {"keyword": {"type": "keyword"}},
        },

        # ── Search context ────────────────────────────────────────────────────
        "search_keywords": {"type": "keyword"},   # Array — each keyword is an exact token

        # ── Timestamps ────────────────────────────────────────────────────────
        "catalog_loaded_at": {"type": "date", "format": "strict_date_time"},
        "schema_version":    {"type": "keyword"},

        # ── Scrape summary (updated after each scrape run) ────────────────────
        "scrape_summary": {
            "type": "object",
            "properties": {
                "total_images_found":    {"type": "integer"},
                "images_downloaded":     {"type": "integer"},
                "sources_queried":       {"type": "keyword"},
                "avg_preliminary_score": {"type": "float"},
                "last_scraped_at":       {"type": "date", "format": "strict_date_time"},
            },
        },
    },
}

# ──────────────────────────────────────────────────────────────────────────────

IMAGES_SETTINGS = {
    "number_of_shards":   1,
    "number_of_replicas": 0,
}

IMAGES_MAPPINGS = {
    "dynamic": "strict",
    "properties": {
        # ── Product back-reference ─────────────────────────────────────────────
        "motion_product_id": {"type": "keyword"},
        "mfr_name":          {"type": "keyword"},
        "mfr_part_number":   {"type": "keyword"},

        # ── Scrape metadata ────────────────────────────────────────────────────
        "candidate_index": {"type": "integer"},
        "scraped_at":      {"type": "date", "format": "strict_date_time"},
        "schema_version":  {"type": "keyword"},

        # ── Source info (URLs stored but not indexed — display only) ───────────
        "image_url":     {"type": "keyword", "index": False},
        "thumbnail_url": {"type": "keyword", "index": False},
        "source_page":   {"type": "keyword", "index": False},
        "source_name":   {"type": "keyword"},   # Indexed: "Wikimedia Commons", "OpenVerse / flickr"

        # ── Image metadata from the API ────────────────────────────────────────
        "title": {
            "type":     "text",
            "fields":   {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "license":     {"type": "keyword"},   # e.g. "CC BY-SA 4.0", "CC0"
        "attribution": {"type": "keyword"},
        "tags":        {"type": "keyword"},   # Array — exact-match aggregation
        "mime_type":   {"type": "keyword"},
        "api_width":   {"type": "integer"},   # Dimensions reported by the API at discovery time
        "api_height":  {"type": "integer"},

        # ── Download results ───────────────────────────────────────────────────
        "downloaded":      {"type": "boolean"},
        "storage_type":    {"type": "keyword"},                  # "local" or "minio"
        "local_path":      {"type": "keyword", "index": False},  # Filesystem path or MinIO object key
        "file_size_bytes": {"type": "integer"},
        "actual_width":    {"type": "integer"},   # Post-download Pillow-verified dimensions
        "actual_height":   {"type": "integer"},
        "actual_format":   {"type": "keyword"},   # e.g. "JPEG", "PNG"
        "download_error":  {"type": "text",    "index": False},  # Error string — display only

        # ── Confidence scoring ─────────────────────────────────────────────────
        "confidence_hints": {
            "type": "object",
            "properties": {
                "has_product_keywords_in_title": {"type": "boolean"},
                "is_permissively_licensed":      {"type": "boolean"},
                "meets_minimum_resolution":      {"type": "boolean"},
                "source_reliability":            {"type": "keyword"},
                "preliminary_score":             {"type": "float"},
            },
        },
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_client(host: str, port: int) -> Elasticsearch:
    es = Elasticsearch(f"http://{host}:{port}")
    try:
        info = es.info()
        print(f"Connected to Elasticsearch {info['version']['number']} at {host}:{port}")
    except ESConnectionError:
        print(f"ERROR: Could not connect to Elasticsearch at http://{host}:{port}")
        print("  Make sure the service is running:  docker-compose up -d")
        sys.exit(1)
    return es


def create_index(
    es: Elasticsearch,
    name: str,
    settings: dict,
    mappings: dict,
    recreate: bool = False,
) -> None:
    exists = es.indices.exists(index=name)

    if exists:
        if recreate:
            es.indices.delete(index=name)
            print(f"  Deleted existing index: {name}")
        else:
            print(f"  Index already exists (skipping): {name}  — use --recreate to drop and rebuild")
            return

    es.indices.create(index=name, settings=settings, mappings=mappings)
    print(f"  Created index: {name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Elasticsearch indices for the Motion Industries image scraper.",
        epilog="Example: python setup_elasticsearch.py --recreate",
    )
    parser.add_argument("--host",     default="localhost", help="Elasticsearch host (default: localhost)")
    parser.add_argument("--port",     default=9200, type=int, help="Elasticsearch port (default: 9200)")
    parser.add_argument("--recreate", action="store_true",   help="Drop and recreate indices if they already exist")
    args = parser.parse_args()

    es = get_client(args.host, args.port)

    print("\nSetting up indices...")
    create_index(es, PRODUCTS_INDEX, PRODUCTS_SETTINGS, PRODUCTS_MAPPINGS, recreate=args.recreate)
    create_index(es, IMAGES_INDEX,   IMAGES_SETTINGS,   IMAGES_MAPPINGS,   recreate=args.recreate)

    print("\nDone. Indices:")
    for idx in [PRODUCTS_INDEX, IMAGES_INDEX]:
        stats = es.indices.stats(index=idx)
        doc_count = stats["indices"][idx]["primaries"]["docs"]["count"]
        print(f"  {idx:<30} {doc_count} documents")


if __name__ == "__main__":
    main()

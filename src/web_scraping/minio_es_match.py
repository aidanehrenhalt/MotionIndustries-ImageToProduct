"""
MinIO ↔ ElasticSearch Image Matching Utility
Motion Industries Image-to-Product — GT Senior Design

Matches images stored in MinIO with their corresponding metadata in ElasticSearch.
Supports:
  - Listing all images for a product (metadata + MinIO presigned URLs)
  - Verifying MinIO ↔ ES consistency (orphaned images, missing files)
  - Downloading images from MinIO using ES metadata lookup
  - Generating presigned URLs for image access

Usage:
    python minio_es_match.py --product s10807860         # Show images for a product with MinIO URLs
    python minio_es_match.py --verify                     # Check MinIO ↔ ES consistency
    python minio_es_match.py --download s10807860         # Download a product's images from MinIO to local
    python minio_es_match.py --list-bucket                # List all objects in the MinIO bucket
    python minio_es_match.py --stats                      # Show storage stats (MinIO + ES)
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from elasticsearch import Elasticsearch

# ── Config ────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "mi-images")

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
IMAGES_INDEX = "mi_candidate_images"
PRODUCTS_INDEX = "mi_products"

DOWNLOAD_DIR = Path("output/minio_downloads")

# ── Clients ───────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )


def get_es_client():
    es = Elasticsearch(ES_URL)
    try:
        es.info()
    except Exception as e:
        print(f"ERROR: Could not connect to Elasticsearch at {ES_URL}: {e}")
        print("  Make sure it's running: docker-compose up -d")
        sys.exit(1)
    return es


# ── Core Functions ────────────────────────────────────────────────────────────

def get_product_images_from_es(es, product_id):
    """Query ES for all candidate images for a product, sorted by confidence score."""
    resp = es.search(index=IMAGES_INDEX, body={
        "query": {"term": {"motion_product_id": product_id}},
        "sort": [{"confidence_hints.preliminary_score": {"order": "desc"}}],
        "size": 100,
    })
    return resp["hits"]["hits"]


def get_all_minio_images_from_es(es):
    """Query ES for all candidate images stored in MinIO."""
    resp = es.search(index=IMAGES_INDEX, body={
        "query": {"term": {"storage_type": "minio"}},
        "sort": [{"motion_product_id": "asc"}],
        "size": 1000,
    })
    return resp["hits"]["hits"]


def list_minio_objects(s3, prefix="images/"):
    """List all objects in the MinIO bucket under a prefix."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(obj)
    return objects


def generate_presigned_url(s3, object_key, expiry=3600):
    """Generate a presigned URL for downloading an image from MinIO."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": MINIO_BUCKET, "Key": object_key},
        ExpiresIn=expiry,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_product(es, s3, product_id):
    """Show all images for a product, with MinIO presigned URLs for those stored in MinIO."""
    hits = get_product_images_from_es(es, product_id)
    if not hits:
        print(f"No images found for product '{product_id}' in ElasticSearch.")
        return

    print(f"\n{'='*90}")
    print(f" Images for {product_id} — {len(hits)} candidates")
    print(f"{'='*90}")

    for i, hit in enumerate(hits):
        s = hit["_source"]
        ch = s.get("confidence_hints", {})
        storage = s.get("storage_type", "local")
        obj_key = s.get("local_path", "")
        dl = "Yes" if s.get("downloaded") else "No"
        dims = (f"{s.get('actual_width','?')}x{s.get('actual_height','?')}"
                if s.get("actual_width")
                else f"{s.get('api_width','?')}x{s.get('api_height','?')}")
        size = s.get("file_size_bytes")
        size_str = f"{size/1024:.0f}KB" if size else "-"

        print(f"\n  [{i+1}] Score: {ch.get('preliminary_score', 0):.3f}  |  DL: {dl}  |  {dims}  |  {size_str}")
        print(f"      Source:   {s.get('source_name', '?')}")
        print(f"      Title:    {(s.get('title') or '(none)')[:70]}")
        print(f"      License:  {s.get('license', '?')}")
        print(f"      Storage:  {storage}")
        print(f"      Path/Key: {obj_key}")

        if storage == "minio" and obj_key and s.get("downloaded"):
            try:
                url = generate_presigned_url(s3, obj_key)
                print(f"      MinIO URL: {url}")
            except ClientError as e:
                print(f"      MinIO URL: ERROR - {e}")

        if s.get("download_error"):
            print(f"      Error:    {s['download_error'][:80]}")
    print()


def cmd_verify(es, s3):
    """Check consistency between MinIO objects and ES records."""
    print(f"\n{'='*70}")
    print(f" MinIO ↔ ElasticSearch Consistency Check")
    print(f"{'='*70}")

    # Get all MinIO-stored images from ES
    es_hits = get_all_minio_images_from_es(es)
    es_keys = {hit["_source"]["local_path"] for hit in es_hits if hit["_source"].get("local_path")}

    # Get all objects from MinIO
    minio_objects = list_minio_objects(s3)
    minio_keys = {obj["Key"] for obj in minio_objects}

    # Find mismatches
    in_es_not_minio = es_keys - minio_keys
    in_minio_not_es = minio_keys - es_keys
    matched = es_keys & minio_keys

    print(f"\n  Matched (in both ES and MinIO):     {len(matched)}")
    print(f"  In ES but missing from MinIO:       {len(in_es_not_minio)}")
    print(f"  In MinIO but missing from ES:       {len(in_minio_not_es)}")

    if in_es_not_minio:
        print(f"\n  Missing from MinIO:")
        for key in sorted(in_es_not_minio)[:20]:
            print(f"    - {key}")
        if len(in_es_not_minio) > 20:
            print(f"    ... and {len(in_es_not_minio) - 20} more")

    if in_minio_not_es:
        print(f"\n  Orphaned in MinIO (no ES record):")
        for key in sorted(in_minio_not_es)[:20]:
            print(f"    - {key}")
        if len(in_minio_not_es) > 20:
            print(f"    ... and {len(in_minio_not_es) - 20} more")

    print()


def cmd_download(es, s3, product_id):
    """Download all images for a product from MinIO to local disk."""
    hits = get_product_images_from_es(es, product_id)
    minio_hits = [h for h in hits
                  if h["_source"].get("storage_type") == "minio"
                  and h["_source"].get("downloaded")
                  and h["_source"].get("local_path")]

    if not minio_hits:
        print(f"No MinIO images found for product '{product_id}'.")
        return

    dest = DOWNLOAD_DIR / product_id
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(minio_hits)} images for {product_id} → {dest}/")
    for hit in minio_hits:
        obj_key = hit["_source"]["local_path"]
        filename = Path(obj_key).name
        local_path = dest / filename
        try:
            s3.download_file(MINIO_BUCKET, obj_key, str(local_path))
            size_kb = local_path.stat().st_size / 1024
            print(f"  Downloaded: {filename} ({size_kb:.0f}KB)")
        except ClientError as e:
            print(f"  FAILED: {filename} — {e}")

    print(f"\nDone. Files saved to {dest.resolve()}")


def cmd_list_bucket(s3):
    """List all objects in the MinIO bucket."""
    objects = list_minio_objects(s3, prefix="")
    if not objects:
        print(f"Bucket '{MINIO_BUCKET}' is empty.")
        return

    total_size = 0
    print(f"\n{'='*80}")
    print(f" MinIO Bucket: {MINIO_BUCKET} — {len(objects)} objects")
    print(f"{'='*80}")
    print(f" {'Key':<60} {'Size':>10}")
    print(f" {'-'*60} {'-'*10}")
    for obj in objects:
        size_str = f"{obj['Size']/1024:.0f}KB"
        total_size += obj["Size"]
        print(f" {obj['Key']:<60} {size_str:>10}")
    print(f"\n Total: {total_size/1024/1024:.2f} MB across {len(objects)} objects\n")


def cmd_stats(es, s3):
    """Show combined storage stats from MinIO and ES."""
    print(f"\n{'='*60}")
    print(f" Storage Stats")
    print(f"{'='*60}")

    # ES stats
    try:
        es_resp = es.search(index=IMAGES_INDEX, body={
            "size": 0,
            "aggs": {
                "by_storage": {"terms": {"field": "storage_type", "missing": "local"}},
                "total_size": {"sum": {"field": "file_size_bytes"}},
                "downloaded": {"terms": {"field": "downloaded"}},
                "by_product": {"cardinality": {"field": "motion_product_id"}},
            }
        })
        aggs = es_resp["aggregations"]
        total_images = es_resp["hits"]["total"]["value"]
        print(f"\n  ElasticSearch ({IMAGES_INDEX}):")
        print(f"    Total image records:   {total_images}")
        print(f"    Unique products:       {aggs['by_product']['value']}")
        total_bytes = aggs["total_size"]["value"] or 0
        print(f"    Total file size (ES):  {total_bytes/1024/1024:.2f} MB")
        print(f"\n    By Storage Type:")
        for b in aggs["by_storage"]["buckets"]:
            print(f"      {b['key']:<20} {b['doc_count']:>5} images")
    except Exception as e:
        print(f"\n  ElasticSearch: ERROR — {e}")

    # MinIO stats
    try:
        objects = list_minio_objects(s3, prefix="")
        total_minio_size = sum(o["Size"] for o in objects)
        print(f"\n  MinIO ({MINIO_BUCKET}):")
        print(f"    Total objects:         {len(objects)}")
        print(f"    Total size:            {total_minio_size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"\n  MinIO: ERROR — {e}")

    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Match and manage images between MinIO and ElasticSearch.",
        epilog="Example: python minio_es_match.py --product s10807860",
    )
    parser.add_argument("--product", type=str, help="Show images for a product (with MinIO URLs)")
    parser.add_argument("--verify", action="store_true", help="Check MinIO ↔ ES consistency")
    parser.add_argument("--download", type=str, metavar="PRODUCT_ID",
                        help="Download a product's images from MinIO to local")
    parser.add_argument("--list-bucket", action="store_true", help="List all objects in the MinIO bucket")
    parser.add_argument("--stats", action="store_true", help="Show storage stats")
    parser.add_argument("--es-url", default=None, help="Elasticsearch URL (default: env ES_URL or http://localhost:9200)")
    parser.add_argument("--minio-endpoint", default=None, help="MinIO endpoint (default: env MINIO_ENDPOINT or http://localhost:9000)")
    args = parser.parse_args()

    if args.es_url:
        global ES_URL
        ES_URL = args.es_url
    if args.minio_endpoint:
        global MINIO_ENDPOINT
        MINIO_ENDPOINT = args.minio_endpoint

    es = get_es_client()
    s3 = get_s3_client()

    if args.product:
        cmd_product(es, s3, args.product)
    elif args.verify:
        cmd_verify(es, s3)
    elif args.download:
        cmd_download(es, s3, args.download)
    elif args.list_bucket:
        cmd_list_bucket(s3)
    elif args.stats:
        cmd_stats(es, s3)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

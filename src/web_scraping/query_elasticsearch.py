"""
Quick terminal viewer for Elasticsearch scrape results.

Usage:
    python query_elasticsearch.py                        # Summary of all products + image counts
    python query_elasticsearch.py --product s10807860    # Images for a specific product
    python query_elasticsearch.py --images               # All images sorted by confidence score
    python query_elasticsearch.py --stats                # Aggregation stats (licenses, sources)
"""

import argparse
import json
import sys

from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
PRODUCTS_INDEX = "mi_products"
IMAGES_INDEX = "mi_candidate_images"


def connect():
    es = Elasticsearch(ES_URL)
    try:
        es.info()
    except Exception as e:
        print(f"ERROR: Could not connect to Elasticsearch at {ES_URL}: {e}")
        print("  Make sure it's running: docker-compose up -d")
        sys.exit(1)
    return es


def show_products(es):
    """List all products with scrape summary."""
    resp = es.search(index=PRODUCTS_INDEX, body={
        "query": {"match_all": {}},
        "sort": [{"motion_product_id": "asc"}],
        "size": 100,
    })
    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]
    print(f"\n{'='*90}")
    print(f" mi_products — {total} documents")
    print(f"{'='*90}")
    print(f" {'Product ID':<14} {'MFR':<16} {'Part #':<20} {'Images':>6} {'DL':>4} {'Avg Score':>9}")
    print(f" {'-'*14} {'-'*16} {'-'*20} {'-'*6} {'-'*4} {'-'*9}")
    for hit in hits:
        s = hit["_source"]
        ss = s.get("scrape_summary", {})
        print(f" {s['motion_product_id']:<14} "
              f"{s.get('mfr_name',''):<16} "
              f"{s.get('mfr_part_number',''):<20} "
              f"{ss.get('total_images_found', '-'):>6} "
              f"{ss.get('images_downloaded', '-'):>4} "
              f"{ss.get('avg_preliminary_score', '-'):>9}")
    print()


def show_product_images(es, product_id):
    """Show all candidate images for a specific product."""
    resp = es.search(index=IMAGES_INDEX, body={
        "query": {"term": {"motion_product_id": product_id}},
        "sort": [{"confidence_hints.preliminary_score": {"order": "desc"}}],
        "size": 50,
    })
    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]
    print(f"\n{'='*90}")
    print(f" Candidate images for {product_id} — {total} results")
    print(f"{'='*90}")
    for i, hit in enumerate(hits):
        s = hit["_source"]
        ch = s.get("confidence_hints", {})
        dl = "Yes" if s.get("downloaded") else "No"
        size = s.get("file_size_bytes")
        size_str = f"{size/1024:.0f}KB" if size else "-"
        dims = f"{s.get('actual_width','?')}x{s.get('actual_height','?')}" if s.get("actual_width") else f"{s.get('api_width','?')}x{s.get('api_height','?')}"
        print(f"\n  [{i+1}] Score: {ch.get('preliminary_score', 0):.3f}  |  DL: {dl}  |  {dims}  |  {size_str}")
        print(f"      Source:  {s.get('source_name', '?')}")
        print(f"      Title:   {(s.get('title') or '(none)')[:70]}")
        print(f"      License: {s.get('license', '?')}")
        print(f"      URL:     {(s.get('image_url') or '')[:80]}")
        if s.get("download_error"):
            print(f"      Error:   {s['download_error'][:80]}")
    print()


def show_all_images(es):
    """Show all images across products, sorted by score."""
    resp = es.search(index=IMAGES_INDEX, body={
        "query": {"match_all": {}},
        "sort": [{"confidence_hints.preliminary_score": {"order": "desc"}}],
        "size": 50,
    })
    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]
    print(f"\n{'='*100}")
    print(f" mi_candidate_images — {total} total (showing top {len(hits)} by score)")
    print(f"{'='*100}")
    print(f" {'#':>3} {'Score':>5} {'DL':>3} {'Product':<14} {'Source':<28} {'Title':<40}")
    print(f" {'-'*3} {'-'*5} {'-'*3} {'-'*14} {'-'*28} {'-'*40}")
    for i, hit in enumerate(hits):
        s = hit["_source"]
        ch = s.get("confidence_hints", {})
        dl = "Y" if s.get("downloaded") else "N"
        title = (s.get("title") or "")[:40]
        print(f" {i+1:>3} {ch.get('preliminary_score',0):>5.2f} {dl:>3} "
              f"{s.get('motion_product_id',''):<14} "
              f"{s.get('source_name',''):<28} "
              f"{title:<40}")
    print()


def show_stats(es):
    """Show aggregation stats."""
    resp = es.search(index=IMAGES_INDEX, body={
        "size": 0,
        "aggs": {
            "by_source": {"terms": {"field": "source_name"}},
            "by_license": {"terms": {"field": "license"}},
            "downloaded": {"terms": {"field": "downloaded"}},
            "avg_score": {"avg": {"field": "confidence_hints.preliminary_score"}},
        }
    })
    aggs = resp["aggregations"]
    total = resp["hits"]["total"]["value"]

    print(f"\n{'='*60}")
    print(f" Aggregation Stats — {total} total images")
    print(f"{'='*60}")

    print(f"\n  By Source:")
    for b in aggs["by_source"]["buckets"]:
        print(f"    {b['key']:<40} {b['doc_count']:>5}")

    print(f"\n  By License:")
    for b in aggs["by_license"]["buckets"]:
        print(f"    {b['key']:<40} {b['doc_count']:>5}")

    print(f"\n  Downloaded:")
    for b in aggs["downloaded"]["buckets"]:
        label = "Yes" if b["key_as_string"] == "true" else "No"
        print(f"    {label:<40} {b['doc_count']:>5}")

    avg = aggs["avg_score"]["value"]
    print(f"\n  Avg Preliminary Score: {avg:.3f}" if avg else "\n  Avg Preliminary Score: -")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query Elasticsearch scrape results from the terminal.")
    parser.add_argument("--product", type=str, help="Show images for a specific product ID")
    parser.add_argument("--images", action="store_true", help="Show all images sorted by score")
    parser.add_argument("--stats", action="store_true", help="Show aggregation stats")
    parser.add_argument("--raw", type=str, metavar="INDEX",
                        help="Dump raw JSON for an index (mi_products or mi_candidate_images)")
    args = parser.parse_args()

    es = connect()

    if args.product:
        show_product_images(es, args.product)
    elif args.images:
        show_all_images(es)
    elif args.stats:
        show_stats(es)
    elif args.raw:
        resp = es.search(index=args.raw, body={"query": {"match_all": {}}, "size": 10})
        print(json.dumps(resp.body, indent=2, default=str))
    else:
        show_products(es)


if __name__ == "__main__":
    main()

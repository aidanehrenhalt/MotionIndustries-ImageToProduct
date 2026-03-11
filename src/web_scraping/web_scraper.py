"""
Safe Sources (ToS Compliance)
1. Wikimedia Commons API
2. OpenVerse API

Setup (Run upon First Time Usage):
pip install requests beautifulsoup4 Pillow lxml

For now, no need for Selenium. Focusing on APIs.

Usage: 
python web_scraper.py --csv test-products_sample.csv        # Scrape images for all products listed and download images
"""

import os
import json
import time
import hashlib
import logging
import argparse
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from urllib.parse import quote, quote_plus, urlparse

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import boto3
from botocore.exceptions import ClientError

# Config
CHROMEDRIVER_PATH = None # Set path if manually installed, None for using webdriver-manager from pip install

OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
JSON_DIR = OUTPUT_DIR / "json"

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

# Request Headers
HEADERS = {
    "User-Agent": (
        "ImageToProduct-Scraper/1.0 "
        "(GT Senior Design x Motion Industries; "
        "Contact: aehrenhalt3@gatech.edu)"
    )
}

# MinIO / S3-compatible object storage config
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "mi-images")

REQUEST_DELAY = 1.5 # Seconds between Requests
# SELENIUM_PAGE_WAIT = 8 # Wait for Page to Load (Later for using Selenium)
MAX_IMAGES_PER_SOURCE = 3 # Max Images to Download per Source (Wikimedia, OpenVerse) per Product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")

# Load Product Catalog (from CSV test sample file)
def load_product_catalog(csv_path: str) -> list[dict]:
    """
    Load products from test sample (from Motion Industries .xslx file)

    Expected Columns:
    - <ID>: Unique Product Identifier (i.e. SKU)
    - PrimaryImageFilename (Reference image filename - not used for scraper)
    - Item Number (Motion Industry Internal Product Number)
    - Enterprise Name (Brand Parent Company)
    - Manufacturer Name (Brand/Manufacturer)
    - Manufacturer Part Number (MFR # - Key for Searching)
    - Web Product Description (Detailed Product Description)
    - Motion Internal Description (Internal Description)
    - PGC (Product Group Code)
    - PGC Description (Product Category)

    Returns a list of product dicts ready for scraping (loaded from csv file)
    """

    import csv
    products = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Generate search query from existing metadata
                mfr_name = row.get("Manufacturer Name", "").strip()
                mfr_part_number = row.get("Manufacturer Part Number", "").strip()
                category = row.get("PGC Description", "").strip()
                web_desc = row.get("Web Product Description", "").strip()

                # Search keywords: Priorities are MFR Name, MFR #, then Category/Description Context
                keywords = []
                if web_desc:
                    # Primary: First 5 words of web description, strip punctuation
                    desc_words = web_desc.replace(",", "").replace(".", "").split()
                    keywords.append(" ".join(desc_words[:5]))
                if mfr_name and web_desc:
                    desc_words = web_desc.replace(",", "").replace(".", "").split()
                    keywords.append(f"{mfr_name} {' '.join(desc_words[:4])}")
                # If no good keywords, use first 50 chars of web_desc
                if not keywords and category:
                    keywords.append(category)

                products.append({
                    "motion_product_id": (row.get("<ID>") or row.get("ID") or "").strip(),
                    "item_number": row.get("Item Number", "").strip(),
                    "enterprise_name": row.get("Enterprise Name", "").strip(),
                    "mfr_name": mfr_name,
                    "mfr_part_number": mfr_part_number,
                    "category": category,
                    "web_desc": web_desc,
                    "internal_description": row.get("Motion Internal Description", "").strip(),
                    "pgc": row.get("PGC", "").strip(),
                    "search_keywords": keywords
                })

            log.info(f"Loaded {len(products)} products from {csv_path}")

    except FileNotFoundError:
        log.error(f"CSV file not found: {csv_path}")
    except Exception as e:
        log.error(f"Error loading CSV file: {e}")
        exit(1)
        
    return products

# Selenium Driver (Later for non-API sources)
def build_driver(driver_path: str = None, headless: bool = True) -> any:
    """
    Create a Selenium WebDriver instance for manufacturer website scraping.

    Attempts to (in order):
    1. Explicit driver_path argument (manual install)
    2. Use chromedriver from CHROMEDRIVER_PATH
    3. Use webdriver-manager to auto-download chromedriver
    4. Selenium's built-in selenium-manager

    Chrome Options:
    --headless=new : No visible browser window (set headless=False for debugging)
    --no-sandbox : Disable sandboxing (required for some environments)
    --disable-dev-shm-usage : Avoid /dev/shm issues in Docker
    --disable-gpu : Disable GPU acceleration (required for some environments)
    --window-size : 1920,1080 : Set window size for consistent page rendering
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service

    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")

    resolved_path = driver_path or CHROMEDRIVER_PATH

    if resolved_path:
        log.info(f"Using ChromeDriver from path: {resolved_path}")
        service = Service(executable_path=resolved_path)
        return webdriver.Chrome(service=service, options=opts)

    # Try webdriver-manager
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        log.info("Using ChromeDriver from webdriver-manager")
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)
    except Exception as e:
        log.warning(f"webdriver-manager failed: {e}, fallback to selenium-manager")

    # Final fallback
    log.info("Using Selenium built-in selenium-manager for ChromeDriver")
    return webdriver.Chrome(options=opts)

##################################
### Wikimedia Commons REST API ###
##################################
def scrape_wikimedia(product: dict, keyword: str) -> list[dict]:
    """
    Search Wikimedia Commons API for product images.
    """
    results = []
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6, # File namespace
        "gsrsearch": keyword, 
        "gsrlimit": MAX_IMAGES_PER_SOURCE,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", # Image URL, Dimensions, MIME type, and File Extension Metadata
        "iiurlwidth": 800, # Get medium-sized thumbnail (800px width)
        "format": "json",
    }

    try:
        log.info(f"[Wikimedia] Searching for '{keyword}'")
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            for ii in page.get("imageinfo", []):
                if not ii.get("mime", "").startswith("image/"):
                    continue # Skip non-image results
                ext = ii.get("extmetadata", {})
                results.append({
                    "image_url": ii.get("url", ""),
                    "thumbnail_url": ii.get("thumbnailurl", ii.get("url", "")),
                    "source_page": f"https://commons.wikimedia.org/wiki/File:{quote(page.get('title', ''))}",
                    "source_name": "Wikimedia Commons",
                    "license": ext.get("License", {}).get("value", "Unknown"),
                    "attribution": ext.get("Artist", {}).get("value", "Unknown"),
                    "width": ii.get("width"), "height": ii.get("height"),
                    "mime_type": ii.get("mime"),
                    "title": page.get("title", ""), "tags": [],
                    "_source_fn": "wikimedia",
                })
            time.sleep(REQUEST_DELAY) # Delay between requests to respect API rate limits
    except Exception as e:
        log.warning(f"[Wikimedia] Error during API request: {e}")
    log.info(f"[Wikimedia] Found {len(results)} images for '{keyword}'")
    return results

################################
### OpenVerse API (requests) ###
################################
def scrape_openverse(product: dict, keyword: str) -> list[dict]:
    """
    Query OpenVerse API for product images.
    """
    results = []
    try:
        log.info(f"[OpenVerse] Searching for '{keyword}'")
        response = requests.get(
            "https://api.openverse.org/v1/images",
            params={
                "q": keyword, 
                "page_size": MAX_IMAGES_PER_SOURCE,
                "license_type": "commercial,modification"
                },
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        for item in response.json().get("results", []):
            results.append({
                "image_url": item.get("url", ""),
                "thumbnail_url": item.get("thumbnail", item.get("url", "")),
                "source_page": item.get("foreign_landing_url", ""),
                "source_name": f"OpenVerse / {item.get('source', 'Unknown')}",
                "license": item.get("license", "Unknown").upper(),
                "attribution": item.get("creator", "Unknown"),
                "width": item.get("width"), "height": item.get("height"),
                "mime_type": "image/jpeg", # OpenVerse doesn't provide MIME type, assume JPEG for now (could be improved by checking file extension)
                "title": item.get("title", ""),
                "tags": [t.get("name", "") for t in item.get("tags", [])[:10]],
                "_source_fn": "openverse",
            })
        time.sleep(REQUEST_DELAY) # Delay between requests to respect API rate limits
    except Exception as e:
        log.warning(f"[OpenVerse] Error during API request: {e}")
    log.info(f"[OpenVerse] Found {len(results)} images for '{keyword}'")
    return results

# MinIO Client Setup
def create_minio_client() -> boto3.client:
    """
    Create a boto3 S3 client configured for MinIO.
    Ensures the target bucket exists.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",  # Required by boto3, value doesn't matter for MinIO
    )
    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=MINIO_BUCKET)
        log.info(f"[MinIO] Bucket '{MINIO_BUCKET}' exists")
    except ClientError:
        s3.create_bucket(Bucket=MINIO_BUCKET)
        log.info(f"[MinIO] Created bucket '{MINIO_BUCKET}'")
    return s3

# Image Downloading / Processing
def download_image(image_url: str, product_id: str, index: int, s3_client=None) -> dict:
    """
    Download image from URL. If s3_client is provided, upload to MinIO/S3.
    Otherwise, save to local disk.
    Returns metadata dict with storage path and image info.
    """
    meta = {
        "downloaded": False,
        "local_path": None,
        "storage_type": "minio" if s3_client else "local",
        "file_size_bytes": None,
        "actual_width": None,
        "actual_height": None,
        "download_error": None
    }
    try:
        response = requests.get(
            image_url,
            headers=HEADERS,
            timeout=10,
            stream=True
        )
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").startswith("image/"):
            log.warning(f"URL did not return an image: {image_url}")
            meta["download_error"] = f"URL did not return an image. content-type: {response.headers.get('Content-Type', '')}"
            return meta
        raw = response.content
        img = Image.open(BytesIO(raw))
        img.verify() # Verify that it's a valid image
        img = Image.open(BytesIO(raw)) # Reopen after verify to get dimensions

        fmt = img.format or "UNKNOWN"
        ext = fmt.lower().replace("jpeg", "jpg") # Normalize jpeg to jpg
        content_type = response.headers.get("Content-Type", f"image/{ext}")
        url_hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()[:8] # Short hash of URL for uniqueness
        filename = f"{product_id}_{index:02d}_{url_hash}.{ext}"

        if s3_client:
            # Upload to MinIO/S3
            object_key = f"images/{product_id}/{filename}"
            s3_client.put_object(
                Bucket=MINIO_BUCKET,
                Key=object_key,
                Body=raw,
                ContentType=content_type,
            )
            log.info(f"[MinIO] Uploaded {object_key} ({len(raw)/1024:.0f}KB)")
            meta.update({
                "downloaded": True,
                "local_path": object_key,  # Store the object key, not a filesystem path
                "storage_type": "minio",
                "file_size_bytes": len(raw),
                "actual_width": img.width,
                "actual_height": img.height,
                "actual_format": fmt,
            })
        else:
            # Save to local filesystem (original behavior)
            filepath = IMAGES_DIR / filename
            with open(filepath, "wb") as f:
                f.write(raw)
            meta.update({
                "downloaded": True,
                "local_path": str(filepath),
                "storage_type": "local",
                "file_size_bytes": len(raw),
                "actual_width": img.width,
                "actual_height": img.height,
                "actual_format": fmt,
            })
    except Exception as e:
        meta["download_error"] = str(e)
        log.warning(f"Failed to download image from {image_url}: {e}")
    return meta

# Confidence Hints (Placeholder for Testing)
### NOTE: Lowkey AI slop because I'm focused on the scraper working not the actual model functionality (will work on with Rodrigo later)
def compute_confidence_hints(product: dict, image_meta: dict) -> dict:
    """
    AI slop for computing confidence hints based on product metadata and image metadata.
    For now, this is a temp solution (and I cannot speak to how well it works)
    Attempt at the idea is there though :)
    """
    title = (image_meta.get("title") or "").lower()
    keywords = [product["mfr_name"].lower(), product["category"].lower()]
    lic = (image_meta.get("license") or "").lower()
    w = image_meta.get("actual_width") or image_meta.get("width") or 0
    h = image_meta.get("actual_height") or image_meta.get("height") or 0
    source = image_meta.get("source_name", "").lower()

    kw_match = any(k in title for k in keywords)
    permissive = any(p in lic for p in ["cc", "public domain", "cc0", "cc by"]) # Add more permissive license indicators as needed
    min_res = (w >= 200 and h >= 200) # Arbitrary minimum resolution threshold for "decent" quality - 200x200 is probably dookie buns but it's temp
    reliability = ("high" if ("wikimedia" in source or "openverse" in source
                             or "manufacturer site" in source)
                    else "low")
    confidence_score = 0.0
    if kw_match:
        confidence_score += 0.35
    if permissive:
        confidence_score += 0.25
    if min_res:
        confidence_score += 0.25
    if reliability == "high":
        confidence_score += 0.15
    elif reliability == "low":
        confidence_score -= 0.15 # Lower confidence for low-reliability sources
    
    return {
        "has_product_keywords_in_title": kw_match,
        "is_permissively_licensed": permissive,
        "meets_minimum_resolution": min_res,
        "source_reliability": reliability,
        "preliminary_score": round(min(confidence_score, 1.0), 3) # Cap at 1.0 and round for readability
    }

# Duplication Helper
def _dedupe_by_url(images: list[dict]) -> list[dict]:
    """
    Helper function to remove duplicate image results by image_url
    """
    seen = set()
    unique = []
    for img in images:
        url = img.get("image_url")
        if url and url not in seen:
            seen.add(url)
            unique.append(img)
    return unique

# Main Scraping Function
def scrape_product_images(
    product: dict,
    driver=None,
    enable_mfr_sites: bool = False,
    download_images: bool = True,
    s3_client=None,
) -> dict:
    """
    Run all scraping steps for a single product and return compiled metadata and results.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Scraping images for Product ID: {product['motion_product_id']} | {product['mfr_name']} {product['mfr_part_number']} | {product['category']}")
    log.info(f"{'='*60}")

    scraped_at = datetime.now(timezone.utc).isoformat()

    keywords = product.get("search_keywords") or [f"{product['mfr_name']} {product['category']}"]
    log.info(f"Using {len(keywords)} search keywords: {keywords}")

    raw_images: list[dict] = []
    for keyword in keywords:
        raw_images += scrape_wikimedia(product, keyword)
        raw_images += scrape_openverse(product, keyword)

    if enable_mfr_sites and driver:
        from manufacturer_scrapers import scrape_manufacturer_site
        raw_images += scrape_manufacturer_site(product, driver=driver)

    raw_images = _dedupe_by_url(raw_images) # Remove duplicates by URL
    log.info(f"Total unique images found across sources: {len(raw_images)}")

    enriched: list[dict] = []
    for index, img in enumerate(raw_images):
        log.info(f"Processing image {index+1}/{len(raw_images)} from {img.get('source_name')}")
        download_meta = (
            download_image(
                img["image_url"],
                product["motion_product_id"],
                index,
                s3_client=s3_client,
            )
            if (download_images and img.get("image_url"))
            else {"downloaded": False, "storage_type": None}
        )
        confidence = compute_confidence_hints(product, {**img, **download_meta})
        enriched.append({
            "index": index,
            "image_url": img.get("image_url", ""),
            "thumbnail_url": img.get("thumbnail_url", ""),
            "source_page": img.get("source_page", ""),
            "source_name": img.get("source_name", ""),
            "title": img.get("title", ""),
            "license": img.get("license", ""),
            "attribution": img.get("attribution", "Unknown"),
            "tags": img.get("tags", []),
            "width": img.get("width"),
            "height": img.get("height"),
            "mime_type": img.get("mime_type", ""),
            "downloaded": download_meta.get("downloaded", False),
            "storage_type": download_meta.get("storage_type"),
            "local_path": download_meta.get("local_path"),
            "file_size_bytes": download_meta.get("file_size_bytes"),
            "actual_width": download_meta.get("actual_width"),
            "actual_height": download_meta.get("actual_height"),
            "actual_format": download_meta.get("actual_format"),
            "download_error": download_meta.get("download_error"),
            "confidence_hints": confidence,
            "scraped_at": scraped_at,
        })

    enriched.sort(key=lambda x: x["confidence_hints"]["preliminary_score"], reverse=True)

    return {
        "schema_version": "1.0", # Such New, Much Wow
        "scraped_at": scraped_at,
        "product": {
            "motion_product_id": product["motion_product_id"],
            "mfr_name": product["mfr_name"],
            "mfr_part_number": product["mfr_part_number"],
            "description": product["web_desc"],
            "category": product["category"],
        },
        "scrape_summary": {
            "total_images_found": len(raw_images),
            "images_downloaded": sum(1 for i in enriched if i["downloaded"]),
            "sources_queried": list(set(i["source_name"] for i in enriched)),
            "avg_preliminary_score": round(
                sum(i["confidence_hints"]["preliminary_score"] for i in enriched) / max(len(enriched), 1), 3
            ),
        },
        "candidate_images": enriched,
    }

def save_record(record: dict) -> Path:
    pid = record["product"]["motion_product_id"].replace("/", "_") # Sanitize for filesystem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = JSON_DIR / f"{pid}_{ts}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    log.info(f"Saved record to {filename}")
    return filename

def index_to_elasticsearch(record: dict, es: Elasticsearch, product_row: dict):
    """
    Index a scrape record into Elasticsearch (mi_products + mi_candidate_images).
    product_row is the original CSV-loaded dict with all catalog fields.
    """
    product = record["product"]
    pid = product["motion_product_id"]

    # Step 1 — Index/upsert the product document
    es.index(index="mi_products", id=pid, document={
        "motion_product_id":    pid,
        "item_number":          product_row.get("item_number", ""),
        "enterprise_name":      product_row.get("enterprise_name", ""),
        "mfr_name":             product["mfr_name"],
        "mfr_name_text":        product["mfr_name"],
        "mfr_part_number":      product["mfr_part_number"],
        "mfr_part_number_text": product["mfr_part_number"],
        "description":          product["description"],
        "internal_description": product_row.get("internal_description", ""),
        "pgc":                  product_row.get("pgc", ""),
        "category":             product["category"],
        "search_keywords":      product_row.get("search_keywords", []),
        "catalog_loaded_at":    record["scraped_at"],
        "schema_version":       record["schema_version"],
    })

    # Step 2 — Bulk-index candidate images
    actions = []
    for img in record["candidate_images"]:
        doc_id = hashlib.sha1(
            f"{pid}:{img['image_url']}".encode()
        ).hexdigest()
        actions.append({
            "_op_type": "index",
            "_index":   "mi_candidate_images",
            "_id":      doc_id,
            "_source": {
                "motion_product_id": pid,
                "mfr_name":          product["mfr_name"],
                "mfr_part_number":   product["mfr_part_number"],
                "candidate_index":   img["index"],
                "scraped_at":        img["scraped_at"],
                "schema_version":    record["schema_version"],
                "image_url":         img["image_url"],
                "thumbnail_url":     img["thumbnail_url"],
                "source_page":       img["source_page"],
                "source_name":       img["source_name"],
                "title":             img["title"],
                "license":           img["license"],
                "attribution":       img["attribution"],
                "tags":              img["tags"],
                "mime_type":         img["mime_type"],
                "api_width":         img.get("width"),
                "api_height":        img.get("height"),
                "downloaded":        img["downloaded"],
                "storage_type":      img.get("storage_type"),
                "local_path":        img.get("local_path"),
                "file_size_bytes":   img.get("file_size_bytes"),
                "actual_width":      img.get("actual_width"),
                "actual_height":     img.get("actual_height"),
                "actual_format":     img.get("actual_format"),
                "download_error":    img.get("download_error"),
                "confidence_hints":  img["confidence_hints"],
            },
        })
    if actions:
        success, errors = bulk(es, actions, raise_on_error=False)
        log.info(f"[ES] Indexed {success} candidate images for {pid}"
                 + (f" ({len(errors)} errors)" if errors else ""))

    # Step 3 — Update scrape_summary on the product
    es.update(index="mi_products", id=pid, doc={
        "scrape_summary": {
            "total_images_found":    record["scrape_summary"]["total_images_found"],
            "images_downloaded":     record["scrape_summary"]["images_downloaded"],
            "sources_queried":       record["scrape_summary"]["sources_queried"],
            "avg_preliminary_score": record["scrape_summary"]["avg_preliminary_score"],
            "last_scraped_at":       record["scraped_at"],
        }
    })
    log.info(f"[ES] Product {pid} indexed with scrape_summary")


def print_summary(record: dict):
    product = record["product"]
    summary = record["scrape_summary"]
    print(f"\n{'='*60}")
    print(f" [{product['motion_product_id']}] {product['mfr_name']} {product['mfr_part_number']} | {product['category']} ")
    print(f" {product['description'][:100]}... ")
    print(f" Images : {summary['total_images_found']} found, {summary['images_downloaded']} downloaded ")
    print(f" Avg Confidence Score: {summary['avg_preliminary_score']} ")
    print(f" Top Candidates:")
    for img in record["candidate_images"][:3]: # Show top 3 candidates
        downloaded = img["downloaded"]
        score = img["confidence_hints"]["preliminary_score"]
        status = "Downloaded" if downloaded else f"Download Failed: {img.get('download_error', '')[:45]}"
        print(f" [{score:.2f}] {img['source_name']:42s} {status}")
    print(f"{'='*60}")

# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image-to-Product Web Scraper for Motion Industries - GT Senior Design Project",
        epilog="Example Usage: python web_scraper.py --csv test-products_sample.csv --product SKF --no-download",
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to product catalog CSV file")
    parser.add_argument("--product", type=str, default=None, help="Specific product ID or category keyword to scrape")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products to scrape (for testing)")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading images, only scrape metadata")
    parser.add_argument("--es", action="store_true", help="Index results into Elasticsearch (must be running)")
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch host (default: localhost)")
    parser.add_argument("--es-port", default=9200, type=int, help="Elasticsearch port (default: 9200)")
    parser.add_argument("--minio", action="store_true", help="Upload images to MinIO instead of local filesystem")
    parser.add_argument("--minio-endpoint", default=None, help="MinIO endpoint URL (default: env MINIO_ENDPOINT or http://localhost:9000)")
    parser.add_argument("--minio-bucket", default=None, help="MinIO bucket name (default: env MINIO_BUCKET or mi-images)")
    parser.add_argument("--manufacturer-sites", action="store_true",
                        help="Enable scraping manufacturer websites via Selenium")
    args = parser.parse_args()

    # Load products from CSV
    products = load_product_catalog(args.csv)

    # Filtering
    if args.product:
        kw = args.product.lower()
        products = [
            p for p in products if 
            kw in p["mfr_name"].lower() or 
            kw in p["category"].lower()
        ]
        log.info(f"Filtered products with keyword '{args.product}': {len(products)} remaining")
    
    if args.limit:
        products = products[:args.limit]
        log.info(f"Limiting to first {args.limit} products for testing")
    
    if not products:
        log.error("No products to scrape after filtering. Exiting.")
        exit(1)

    # Connect to MinIO if requested
    s3 = None
    if args.minio:
        if args.minio_endpoint:
            MINIO_ENDPOINT = args.minio_endpoint
        if args.minio_bucket:
            MINIO_BUCKET = args.minio_bucket
        try:
            s3 = create_minio_client()
            log.info(f"Connected to MinIO at {MINIO_ENDPOINT}, bucket: {MINIO_BUCKET}")
        except Exception as e:
            log.error(f"Could not connect to MinIO at {MINIO_ENDPOINT}: {e}")
            log.error("Make sure it's running: docker-compose up -d")
            exit(1)

    # Connect to Elasticsearch if requested
    es = None
    if args.es:
        es_url = f"http://{args.es_host}:{args.es_port}"
        es = Elasticsearch(es_url)
        try:
            info = es.info()
            log.info(f"Connected to Elasticsearch {info['version']['number']} at {es_url}")
        except Exception as e:
            log.error(f"Could not connect to Elasticsearch at {es_url}: {e}")
            log.error("Make sure it's running: docker-compose up -d")
            exit(1)

    # Initialize Selenium driver if manufacturer scraping is enabled
    driver = None
    if args.manufacturer_sites:
        try:
            driver = build_driver()
            log.info("Selenium WebDriver initialized for manufacturer site scraping")
        except Exception as e:
            log.error(f"Could not init Selenium WebDriver: {e}")
            log.error("Manufacturer site scraping will be disabled. Install selenium and chromedriver.")

    saved_files = []

    try:
        for product in products:
            record = scrape_product_images(
                product,
                driver=driver,
                enable_mfr_sites=args.manufacturer_sites,
                download_images=not args.no_download,
                s3_client=s3,
            )
            saved_path = save_record(record)
            saved_files.append(saved_path)
            if es:
                index_to_elasticsearch(record, es, product)
            print_summary(record)
    finally:
        if driver:
            driver.quit()
            log.info("Selenium WebDriver closed")

    print(f"\nCompleted scraping {len(products)} products. Records saved to:")
    print(f"  JSON   {JSON_DIR.resolve()}")
    if s3:
        print(f"  Images MinIO {MINIO_ENDPOINT}/{MINIO_BUCKET}/images/")
    else:
        print(f"  Images {IMAGES_DIR.resolve()}")
    if es:
        print(f"  Elasticsearch  {es_url} (mi_products + mi_candidate_images)")
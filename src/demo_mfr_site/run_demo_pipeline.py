import json
import socket
import sys
import threading
import csv
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin
import shutil

import requests
from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "Image_Classifier"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "web_scraping"))

from classify_json_images import classify_json_files, rank_json_files, apply_final_ranking
from image_search_ranker import export_rankings_to_csv


SITE_DIR = ROOT / "site"
PRODUCT_DATA = SITE_DIR / "assets" / "data" / "products.json"
OUTPUT_DIR = ROOT / "pipeline_output"
IMAGES_DIR = OUTPUT_DIR / "images"
JSON_DIR = OUTPUT_DIR / "json"
RANKINGS_CSV = OUTPUT_DIR / "rankings.csv"
MODEL_PATH = Path("src/Image_Classifier/trained_model.pth")
SITE_REVIEW_IMAGES_DIR = SITE_DIR / "assets" / "review-images"
SITE_REVIEW_QUEUE_JSON = SITE_DIR / "assets" / "data" / "review_queue.json"
SITE_REVIEW_RANKINGS_CSV = SITE_DIR / "assets" / "data" / "review_rankings.csv"

HEADERS = {
    "User-Agent": "ImageToProduct-DemoPipeline/1.0",
}


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, IMAGES_DIR, JSON_DIR, SITE_REVIEW_IMAGES_DIR, SITE_REVIEW_QUEUE_JSON.parent):
        path.mkdir(parents=True, exist_ok=True)


def load_products() -> list[dict]:
    return json.loads(PRODUCT_DATA.read_text(encoding="utf-8"))


def choose_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    port = choose_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(SITE_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}"


def scrape_page(page_url: str) -> dict:
    response = requests.get(page_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.select_one(".product-title")
    breadcrumbs = [node.get_text(" ", strip=True) for node in soup.select(".breadcrumbs li")]
    specs = []
    for row in soup.select(".spec-table tr"):
        header = row.find("th")
        value = row.find("td")
        if header and value:
            specs.append(
                {
                    "label": header.get_text(" ", strip=True),
                    "value": value.get_text(" ", strip=True),
                }
            )

    image = soup.select_one(".gallery-grid img")
    image_src = urljoin(page_url, image["src"]) if image and image.get("src") else None

    return {
        "page_title": title.get_text(" ", strip=True) if title else "",
        "breadcrumbs": breadcrumbs,
        "specs": specs,
        "image_url": image_src,
        "html_length": len(response.text),
    }


def download_image(image_url: str, target: Path) -> dict:
    response = requests.get(image_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    target.write_bytes(response.content)
    with Image.open(target) as img:
        actual_width, actual_height = img.size
        actual_format = img.format
    return {
        "bytes": len(response.content),
        "content_type": response.headers.get("Content-Type", "image/jpeg"),
        "actual_width": actual_width,
        "actual_height": actual_height,
        "actual_format": actual_format,
    }


def build_record(
    product: dict,
    page_url: str,
    page_data: dict,
    image_rel_path: str,
    image_meta: dict,
    scraped_at: str,
) -> dict:
    return {
        "schema_version": "1.0",
        "scraped_at": scraped_at,
        "product": {
            "motion_product_id": product["part_number"],
            "mfr_name": "AMI Bearings Inc.",
            "mfr_part_number": product["part_number"],
            "description": product["item_name"],
            "category": product["breadcrumbs"][-1] if product.get("breadcrumbs") else "",
            "pgc": product.get("pgc", ""),
        },
        "scrape_summary": {
            "total_images_found": 1 if page_data["image_url"] else 0,
            "images_downloaded": 1 if page_data["image_url"] else 0,
            "sources_queried": ["Manufacturer Site / AMI BEARINGS INC"],
            "avg_preliminary_score": 1.0 if page_data["image_url"] else 0.0,
        },
        "candidate_images": [
            {
                "index": 0,
                "image_url": page_data["image_url"],
                "thumbnail_url": page_data["image_url"],
                "source_page": page_url,
                "source_name": "Manufacturer Site / AMI BEARINGS INC",
                "title": page_data["page_title"],
                "license": "Manufacturer catalog — proprietary, internal use only",
                "attribution": "AMI BEARINGS INC",
                "tags": page_data["breadcrumbs"],
                "width": None,
                "height": None,
                "mime_type": image_meta["content_type"],
                "downloaded": True,
                "storage_type": "local",
                "local_path": image_rel_path,
                "file_size_bytes": image_meta["bytes"],
                "actual_width": image_meta["actual_width"],
                "actual_height": image_meta["actual_height"],
                "actual_format": image_meta["actual_format"],
                "download_error": None,
                "confidence_hints": {
                    "has_product_keywords_in_title": True,
                    "is_permissively_licensed": False,
                    "meets_minimum_resolution": True,
                    "source_reliability": "high",
                    "preliminary_score": 1.0,
                },
                "scraped_at": scraped_at,
            }
        ],
    }


def normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def load_review_metadata_index() -> dict[str, dict]:
    if not PRODUCT_DATA.exists():
        return {}
    records = json.loads(PRODUCT_DATA.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for product in records:
        for key in {product.get("slug"), product.get("part_number")}:
            normalized = normalize_key(key)
            if normalized:
                index[normalized] = product
    return index


def export_review_dataset(json_dir: Path, rankings_csv: Path) -> dict:
    export_rankings_to_csv(json_dir, rankings_csv)
    shutil.copy2(rankings_csv, SITE_REVIEW_RANKINGS_CSV)

    metadata_index = load_review_metadata_index()
    products_by_key: dict[str, dict] = {}

    for json_path in sorted(json_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        product = data.get("product", {})
        product_id = product.get("motion_product_id") or json_path.stem
        key = normalize_key(product_id) or normalize_key(json_path.stem)
        metadata = metadata_index.get(key, {})
        candidates = data.get("candidate_images", [])
        candidate_by_name = {
            Path(candidate.get("local_path") or "").name.lower(): candidate
            for candidate in candidates
            if candidate.get("local_path")
        }
        products_by_key[key] = {
            "productId": product_id,
            "queueKey": product_id,
            "slug": metadata.get("slug") or json_path.stem,
            "productName": metadata.get("item_name") or product.get("description") or product_id,
            "manufacturer": product.get("mfr_name", ""),
            "manufacturerPartNumber": product.get("mfr_part_number", ""),
            "partNumber": metadata.get("part_number") or product.get("mfr_part_number") or product_id,
            "description": metadata.get("meta_description") or product.get("description") or "",
            "category": product.get("category", ""),
            "jsonFile": json_path.name,
            "jsonPath": str(json_path.relative_to(ROOT)),
            "sourceUrl": metadata.get("source_url", ""),
            "breadcrumbs": metadata.get("breadcrumbs", []),
            "highlights": metadata.get("highlights", []),
            "specs": metadata.get("specs", []),
            "candidateImages": [],
            "_candidateByName": candidate_by_name,
        }

    with rankings_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            product_key = normalize_key(row.get("motion_product_id"))
            if not product_key:
                continue
            product_entry = products_by_key.get(product_key)
            if not product_entry:
                continue

            image_filename = row.get("image_filename", "")
            candidate = product_entry["_candidateByName"].get(image_filename.lower(), {})
            local_source = Path(candidate.get("local_path") or "")
            review_image_name = image_filename or local_source.name
            review_image_target = SITE_REVIEW_IMAGES_DIR / review_image_name
            if local_source.exists():
                shutil.copy2(local_source, review_image_target)

            product_entry["candidateImages"].append(
                {
                    "id": f"{product_entry['queueKey']}-{review_image_name}",
                    "fileName": review_image_name,
                    "url": f"assets/review-images/{review_image_name}" if review_image_name else "",
                    "rank": int(row.get("image_rank") or 0),
                    "finalScore": float(row.get("final_score") or 0.0),
                    "finalScorePct": row.get("final_score_pct") or "",
                    "aiConfidence": float(row.get("ai_confidence") or 0.0),
                    "aiConfidencePct": row.get("ai_confidence_pct") or "",
                    "textScore": float(row.get("text_score") or 0.0),
                    "textScorePct": row.get("text_score_pct") or "",
                    "sourceName": row.get("source_name") or "",
                    "sourcePage": row.get("candidate_source_page") or candidate.get("source_page") or "",
                    "license": row.get("license") or candidate.get("license") or "",
                    "localPath": candidate.get("local_path") or "",
                    "predictedClass": row.get("ai_class_label") or "",
                }
            )

    queue = []
    for product_entry in products_by_key.values():
        product_entry["candidateImages"].sort(
            key=lambda image: (image.get("rank") or 9999, -image.get("finalScore") or 0.0)
        )
        product_entry["matchedImageCount"] = len(product_entry["candidateImages"])
        product_entry["bestConfidence"] = (
            product_entry["candidateImages"][0]["aiConfidence"]
            if product_entry["candidateImages"]
            else 0.0
        )
        product_entry["bestFinalScore"] = (
            product_entry["candidateImages"][0]["finalScore"]
            if product_entry["candidateImages"]
            else 0.0
        )
        product_entry.pop("_candidateByName", None)
        queue.append(product_entry)

    queue.sort(key=lambda item: item["productId"])
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "rankingsCsv": str(rankings_csv.relative_to(ROOT)),
            "jsonDir": str(json_dir.relative_to(ROOT)),
            "reviewRankingsCsv": str(SITE_REVIEW_RANKINGS_CSV.relative_to(ROOT)),
        },
        "products": queue,
    }
    SITE_REVIEW_QUEUE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "products": len(queue),
        "rankings_csv": str(rankings_csv),
        "review_queue_json": str(SITE_REVIEW_QUEUE_JSON),
    }


def main() -> None:
    ensure_dirs()
    products = load_products()
    server, thread, base_url = start_server()
    saved_json_files: list[Path] = []

    try:
        for product in products:
            page_url = f"{base_url}/products/{product['slug']}.html"
            page_data = scrape_page(page_url)
            if not page_data["image_url"]:
                raise RuntimeError(f"No gallery image found for {product['slug']}")

            image_name = f"{product['slug']}{Path(page_data['image_url']).suffix or '.jpg'}"
            image_path = IMAGES_DIR / image_name
            image_meta = download_image(page_data["image_url"], image_path)
            image_rel_path = str(image_path.relative_to(Path.cwd()))
            scraped_at = datetime.now(timezone.utc).isoformat()

            record = build_record(
                product,
                page_url,
                page_data,
                image_rel_path,
                image_meta,
                scraped_at,
            )
            json_path = JSON_DIR / f"{product['slug']}.json"
            json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            saved_json_files.append(json_path)

        # Pass 1 — CNN classifier (writes predicted_class + classifier_confidence)
        classify_json_files(saved_json_files, MODEL_PATH)
        # Pass 2 — text/metadata ranker (writes ranker_score + score_breakdown)
        rank_json_files(saved_json_files)
        # Pass 3 — combine both signals into final_score + final_rank
        apply_final_ranking(saved_json_files)
        review_summary = export_review_dataset(JSON_DIR, RANKINGS_CSV)

        summary = {
            "products_processed": len(saved_json_files),
            "images_dir": str(IMAGES_DIR),
            "json_dir": str(JSON_DIR),
            "json_files": [path.name for path in saved_json_files],
            "rankings_csv": str(RANKINGS_CSV),
            "review_queue_json": review_summary["review_queue_json"],
        }
        (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()

import json
import socket
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "Image_Classifier"))

from classify_json_images import classify_json_files, rank_json_files, apply_final_ranking


SITE_DIR = ROOT / "site"
PRODUCT_DATA = SITE_DIR / "assets" / "data" / "products.json"
OUTPUT_DIR = ROOT / "pipeline_output"
IMAGES_DIR = OUTPUT_DIR / "images"
JSON_DIR = OUTPUT_DIR / "json"
MODEL_PATH = Path("src/Image_Classifier/trained_model.pth")

HEADERS = {
    "User-Agent": "ImageToProduct-DemoPipeline/1.0",
}


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, IMAGES_DIR, JSON_DIR):
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

        summary = {
            "products_processed": len(saved_json_files),
            "images_dir": str(IMAGES_DIR),
            "json_dir": str(JSON_DIR),
            "json_files": [path.name for path in saved_json_files],
        }
        (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()

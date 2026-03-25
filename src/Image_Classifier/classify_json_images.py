"""
Image Classifier — Batch Classification of Scraped Images

Reads all JSON records produced by web_scraper.py, runs each downloaded image
through the pre-trained CNN, and writes the predicted class index back into
the JSON file under each candidate image's `predicted_class` field.

Can be run standalone (post-scrape pass) or called programmatically from
web_scraper.py via classify_json_dir().

Usage (standalone):
    python src/Image_Classifier/classify_json_images.py
    python src/Image_Classifier/classify_json_images.py --json-dir output/json --model src/Image_Classifier/trained_model.pth
    python src/Image_Classifier/classify_json_images.py --es  # also push predictions to Elasticsearch
"""

import hashlib
import json
import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms

log = logging.getLogger("classifier")

N_CLASSES = 8


# ── Model Architecture ────────────────────────────────────────────────────────

def build_model() -> nn.Sequential:
    """Return an uninitialised instance of the 8-class CNN."""
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(25),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(in_channels=50, out_channels=75, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(1200, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(model_path: Path) -> nn.Module:
    """Load and return the trained model from *model_path* in eval mode."""
    model = build_model()
    model.load_state_dict(
        torch.load(str(model_path), weights_only=True, map_location=torch.device("cpu"))
    )
    model.eval()
    log.info(f"[classifier] Loaded model from {model_path}")
    return model


def make_preprocess() -> transforms.Compose:
    """Return the standard preprocessing pipeline (resize + to-tensor)."""
    return transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])


def classify_image(model: nn.Module, image_path: Path, preprocess: transforms.Compose) -> int:
    """
    Classify a single image file.

    Returns the predicted class index (0-7), or -1 if the image cannot be read.
    """
    try:
        img = Image.open(str(image_path)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        return int(predicted.item())
    except Exception as e:
        log.warning(f"[classifier] Could not classify {image_path}: {e}")
        return -1


def classify_json_dir(json_dir: Path, model_path: Path) -> int:
    """
    Iterate every *.json file in *json_dir*, classify each candidate image
    whose `local_path` points to an existing file, and write the
    `predicted_class` field back into the JSON.

    Images stored in MinIO (storage_type == "minio") are skipped because
    their `local_path` is an object key, not a local filesystem path.

    Returns the total number of images classified.
    """
    model = load_model(model_path)
    preprocess = make_preprocess()

    # Resolve the project root so that relative local_paths from web_scraper
    # work regardless of where this script is invoked from.
    # File layout: <project_root>/src/Image_Classifier/classify_json_images.py
    project_root = Path(__file__).resolve().parent.parent.parent

    total_classified = 0

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[classifier] Skipping {json_file.name}: {e}")
            continue

        candidates = data.get("candidate_images", [])
        if not candidates:
            continue

        modified = False
        for img_info in candidates:
            local_path = img_info.get("local_path")
            storage_type = img_info.get("storage_type", "local")

            # Skip MinIO entries — we don't have them locally
            if storage_type == "minio" or not local_path:
                continue

            # Resolve against project root (web_scraper writes relative paths)
            full_path = project_root / local_path
            if not full_path.exists():
                log.warning(f"[classifier] Image not found: {full_path}")
                continue

            predicted = classify_image(model, full_path, preprocess)
            if predicted >= 0:
                img_info["predicted_class"] = predicted
                modified = True
                total_classified += 1

        if modified:
            json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            log.info(f"[classifier] Updated {json_file.name}")

    log.info(f"[classifier] Classified {total_classified} images across {json_dir}")
    return total_classified


def push_predictions_to_es(json_dir: Path, es_host: str = "localhost", es_port: int = 9200) -> int:
    """Push predicted_class values from JSON files into the mi_candidate_images
    Elasticsearch index.

    Uses the same SHA1(product_id:image_url) document-ID scheme as
    web_scraper.py so updates land on the correct documents.

    Returns the number of documents updated.
    """
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        log.error("[classifier] elasticsearch package not installed — cannot push to ES")
        return 0

    es = Elasticsearch(f"http://{es_host}:{es_port}")
    if not es.ping():
        log.error(f"[classifier] Cannot reach Elasticsearch at {es_host}:{es_port}")
        return 0

    updated = 0
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        pid = data.get("product", {}).get("motion_product_id", "")
        for img in data.get("candidate_images", []):
            if "predicted_class" not in img:
                continue
            doc_id = hashlib.sha1(
                f"{pid}:{img['image_url']}".encode()
            ).hexdigest()
            try:
                es.update(
                    index="mi_candidate_images",
                    id=doc_id,
                    doc={"predicted_class": img["predicted_class"]},
                )
                updated += 1
            except Exception:
                pass  # 404 or mapping error — skip silently

    log.info(f"[classifier] Pushed predicted_class for {updated} images into ES")
    return updated


# ── Standalone entry point ────────────────────────────────────────────────────

def _default_model_path() -> Path:
    """Return the model path relative to this file's location."""
    return Path(__file__).resolve().parent / "trained_model.pth"


def _default_json_dir() -> Path:
    """Return output/json relative to the project root."""
    return Path(__file__).resolve().parent.parent.parent / "output" / "json"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Batch-classify scraped images and write predicted_class to JSON records",
        epilog="Example: python src/Image_Classifier/classify_json_images.py --json-dir output/json",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=_default_json_dir(),
        help="Directory containing JSON records from web_scraper.py (default: output/json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_default_model_path(),
        help="Path to trained_model.pth (default: src/Image_Classifier/trained_model.pth)",
    )
    parser.add_argument(
        "--es", action="store_true",
        help="Push predicted_class values to Elasticsearch after classification",
    )
    parser.add_argument(
        "--es-host", default="localhost",
        help="Elasticsearch host (default: localhost)",
    )
    parser.add_argument(
        "--es-port", type=int, default=9200,
        help="Elasticsearch port (default: 9200)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")
    if not args.json_dir.is_dir():
        parser.error(f"JSON directory not found: {args.json_dir}")

    n = classify_json_dir(args.json_dir, args.model)
    print(f"Done — classified {n} images.")

    if args.es:
        updated = push_predictions_to_es(args.json_dir, args.es_host, args.es_port)
        print(f"Pushed {updated} predictions to Elasticsearch.")

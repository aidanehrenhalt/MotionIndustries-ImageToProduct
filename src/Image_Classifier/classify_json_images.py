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
"""

import io
import json
import os
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


def _get_s3_client():
    """
    Build a boto3 S3 client pointing at the MinIO instance.
    Reads the same environment variables used by web_scraper.py.
    Returns None if boto3 is not installed.
    """
    try:
        import boto3
    except ImportError:
        log.warning("[classifier] boto3 not installed; MinIO classification disabled")
        return None
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        region_name="us-east-1",
    )


def classify_image_from_bytes(
    model: nn.Module,
    img_bytes: bytes,
    preprocess: transforms.Compose,
) -> int:
    """
    Classify an image supplied as raw bytes (e.g. fetched from MinIO).

    Returns the predicted class index (0-7), or -1 on failure.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        return int(predicted.item())
    except Exception as e:
        log.warning(f"[classifier] Could not classify image from bytes: {e}")
        return -1


def classify_json_files(
    json_files: list,
    model_path: Path,
    s3_client=None,
) -> int:
    """
    Classify images referenced by a specific list of JSON files.

    Unlike classify_json_dir(), this function operates only on the
    files explicitly provided — it does not scan a directory.  Pass the
    list of paths returned by web_scraper.py's save_record() calls so
    that only the current run's artifacts are processed.

    MinIO-backed images (storage_type == "minio") are fetched in-memory
    via s3_client when provided.  If s3_client is None, a client is
    created lazily from the MINIO_* environment variables.  Images that
    cannot be fetched produce a warning and are skipped.

    Returns the total number of images classified.
    """
    model = load_model(model_path)
    preprocess = make_preprocess()

    bucket = os.environ.get("MINIO_BUCKET", "mi-images")
    project_root = Path(__file__).resolve().parent.parent.parent

    _s3 = s3_client
    total_classified = 0

    for json_file in json_files:
        json_file = Path(json_file)
        if not json_file.exists():
            log.warning(f"[classifier] JSON file not found, skipping: {json_file}")
            continue

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

            if not local_path:
                continue

            if storage_type == "minio":
                if _s3 is None:
                    _s3 = _get_s3_client()
                if _s3 is None:
                    log.warning(
                        f"[classifier] No S3 client available for MinIO image "
                        f"{local_path}; skipping"
                    )
                    continue
                try:
                    response = _s3.get_object(Bucket=bucket, Key=local_path)
                    img_bytes = response["Body"].read()
                except Exception as e:
                    log.warning(
                        f"[classifier] Could not fetch MinIO object "
                        f"{local_path}: {e}"
                    )
                    continue
                predicted = classify_image_from_bytes(model, img_bytes, preprocess)
            else:
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

    log.info(
        f"[classifier] Classified {total_classified} images "
        f"across {len(list(json_files))} file(s)"
    )
    return total_classified


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
    whose `local_path` points to an existing local file, and write the
    `predicted_class` field back into the JSON.

    MinIO-backed images (storage_type == "minio") are skipped — this
    function has no S3 client.  For MinIO support, or to restrict
    classification to only the files from the current scrape run, use
    classify_json_files() instead.

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
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")
    if not args.json_dir.is_dir():
        parser.error(f"JSON directory not found: {args.json_dir}")

    n = classify_json_dir(args.json_dir, args.model)
    print(f"Done — classified {n} images.")

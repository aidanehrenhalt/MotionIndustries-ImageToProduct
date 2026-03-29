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

import io
import hashlib
import json
import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image

# Allow importing sibling src packages without an editable install
_WEB_SCRAPING_DIR = Path(__file__).resolve().parent.parent / "web_scraping"
if str(_WEB_SCRAPING_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_SCRAPING_DIR))

from image_search_ranker import rank_candidates as _rank_candidates
from image_search_ranker import ProductVector as _RankerProductVector
from pgc_mapping import get_expected_classes

log = logging.getLogger("classifier")

N_CLASSES = 8


# ── Model Architecture ────────────────────────────────────────────────────────

def build_model():
    """Return an uninitialised instance of the 8-class CNN."""
    import torch.nn as nn
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

def load_model(model_path: Path):
    """Load and return the trained model from *model_path* in eval mode."""
    import torch
    model = build_model()
    model.load_state_dict(
        torch.load(str(model_path), weights_only=True, map_location=torch.device("cpu"))
    )
    model.eval()
    log.info(f"[classifier] Loaded model from {model_path}")
    return model


def make_preprocess():
    """Return the standard preprocessing pipeline (resize + to-tensor)."""
    from torchvision import transforms
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
    model,
    img_bytes: bytes,
    preprocess,
) -> dict:
    """
    Classify an image supplied as raw bytes (e.g. fetched from MinIO).

    Returns a dict with predicted_class (0-7) and classifier_confidence (0-1),
    or {"predicted_class": -1, "classifier_confidence": 0.0} on failure.
    """
    import torch
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return {
            "predicted_class": int(predicted.item()),
            "classifier_confidence": round(float(confidence.item()), 4),
        }
    except Exception as e:
        log.warning(f"[classifier] Could not classify image from bytes: {e}")
        return {"predicted_class": -1, "classifier_confidence": 0.0}


def compute_final_score(
    candidate: dict,
    expected_classes: set,
    w_confidence: float = 0.3,
    w_class_match: float = 0.2,
    w_text_similarity: float = 0.3,
    w_preliminary: float = 0.2,
) -> float:
    """Combine classifier and ranker signals into a single 0–1 final score.

    Formula (when PGC is known):
        final_score = 0.3 * classifier_confidence
                    + 0.2 * class_match         (1.0 if predicted_class in expected_classes)
                    + 0.3 * ranker_score         (from image_search_ranker)
                    + 0.2 * preliminary_score    (scrape-time heuristic)

    When expected_classes is empty (PGC unknown), the class_match weight is
    redistributed proportionally across the other three signals.
    """
    classifier_confidence = candidate.get("classifier_confidence", 0.0)
    predicted_class       = candidate.get("predicted_class", -1)
    ranker_score          = candidate.get("ranker_score", 0.0)
    preliminary_score     = candidate.get("confidence_hints", {}).get("preliminary_score", 0.0)

    if expected_classes:
        class_match = 1.0 if predicted_class in expected_classes else 0.0
        return round(
            w_confidence        * classifier_confidence
            + w_class_match     * class_match
            + w_text_similarity * ranker_score
            + w_preliminary     * preliminary_score,
            6,
        )
    else:
        remaining = w_confidence + w_text_similarity + w_preliminary
        if remaining == 0.0:
            return 0.0
        scale = 1.0 / remaining
        return round(
            (w_confidence        * classifier_confidence
             + w_text_similarity * ranker_score
             + w_preliminary     * preliminary_score) * scale,
            6,
        )


def _build_query_vector(data: dict) -> _RankerProductVector:
    """Build a ProductVector query from a JSON record's product metadata block."""
    prod = data.get("product", {})
    return _RankerProductVector(
        id_number               = prod.get("motion_product_id", ""),
        manufacture_name        = prod.get("mfr_name", ""),
        manufacture_part_number = prod.get("mfr_part_number", ""),
        web_product_description = prod.get("description", ""),
        pgc_description         = prod.get("category", ""),
    )


def rank_json_files(json_files: list) -> int:
    """Score each candidate image using image_search_ranker and write results to JSON.

    This is the parallel text/metadata ranker pass, independent from the CNN
    classifier.  Enriches each candidate with ranker_score, score_pct, and
    score_breakdown.  Call before apply_final_ranking().

    Returns the total number of candidates scored.
    """
    total_scored = 0
    for json_file in json_files:
        json_file = Path(json_file)
        if not json_file.exists():
            log.warning(f"[ranker] JSON file not found, skipping: {json_file}")
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[ranker] Skipping {json_file.name}: {e}")
            continue

        candidates = data.get("candidate_images", [])
        if not candidates:
            continue

        query  = _build_query_vector(data)

        # Backfill index on originals before ranking so the merge key is
        # always stable.  rank_candidates returns sorted copies — positional
        # fallback after sorting would map to the wrong candidate.
        for i, img in enumerate(candidates):
            if "index" not in img:
                log.warning(
                    f"[ranker] Candidate at position {i} in {json_file.name} "
                    "has no 'index' field; synthesizing index=%d", i
                )
                img["index"] = i

        ranked = _rank_candidates(query, candidates)

        # Merge ranker fields back into the original list using the stable index key.
        ranked_by_index = {r["index"]: r for r in ranked}
        for img_info in candidates:
            key = img_info["index"]
            if key in ranked_by_index:
                r = ranked_by_index[key]
                img_info["ranker_score"]    = r["ranker_score"]
                img_info["score_pct"]       = r["score_pct"]
                img_info["score_breakdown"] = r["score_breakdown"]
                total_scored += 1

        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info(f"[ranker] Scored {len(candidates)} candidate(s) in {json_file.name}")

    log.info(f"[ranker] Scored {total_scored} candidates across {len(list(json_files))} file(s)")
    return total_scored


def apply_final_ranking(json_files: list) -> int:
    """Compute final_score and final_rank for each candidate and re-save JSON.

    Must be called after both classify_json_files() and rank_json_files() have
    completed so that predicted_class, classifier_confidence, and ranker_score
    are all present.  Sorts candidates by final_score descending and assigns
    1-based final_rank (rank 1 = best match).

    Returns the total number of candidates ranked.
    """
    total_ranked = 0
    for json_file in json_files:
        json_file = Path(json_file)
        if not json_file.exists():
            log.warning(f"[ranker] JSON file not found, skipping: {json_file}")
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[ranker] Skipping {json_file.name}: {e}")
            continue

        candidates = data.get("candidate_images", [])
        if not candidates:
            continue

        expected_classes = get_expected_classes(data.get("product", {}))
        for img_info in candidates:
            img_info["final_score"] = compute_final_score(img_info, expected_classes)

        candidates.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
        for rank, img_info in enumerate(candidates, 1):
            img_info["final_rank"] = rank
            total_ranked += 1

        data["candidate_images"] = candidates
        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info(f"[ranker] Assigned final_rank to {len(candidates)} candidate(s) in {json_file.name}")

    log.info(f"[ranker] Final ranking complete for {total_ranked} candidates")
    return total_ranked


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

            if predicted["predicted_class"] >= 0:
                img_info["predicted_class"]       = predicted["predicted_class"]
                img_info["classifier_confidence"] = predicted["classifier_confidence"]
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


def classify_image(model, image_path: Path, preprocess) -> dict:
    """
    Classify a single image file.

    Returns a dict with predicted_class (0-7) and classifier_confidence (0-1),
    or {"predicted_class": -1, "classifier_confidence": 0.0} if the image cannot be read.
    """
    import torch
    try:
        img = Image.open(str(image_path)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return {
            "predicted_class": int(predicted.item()),
            "classifier_confidence": round(float(confidence.item()), 4),
        }
    except Exception as e:
        log.warning(f"[classifier] Could not classify {image_path}: {e}")
        return {"predicted_class": -1, "classifier_confidence": 0.0}


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
            if predicted["predicted_class"] >= 0:
                img_info["predicted_class"]       = predicted["predicted_class"]
                img_info["classifier_confidence"] = predicted["classifier_confidence"]
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

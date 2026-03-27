"""
Integration tests for the scraper → classifier pipeline boundary.

Covers:
1. Local image classification writes predicted_class
2. MinIO-backed classification writes predicted_class (mocked boto3)
3. ES sync updates mi_candidate_images (mocked ES client)
4. Classification scoped to saved_files only (not the full JSON directory)
5. Missing MinIO object produces a warning but does not crash the run

Run with:
    pytest -q tests/test_scraper_classifier_pipeline.py
"""

import hashlib
import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make both src packages importable without an editable install
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "Image_Classifier"))
sys.path.insert(0, str(_ROOT / "src" / "web_scraping"))

import classify_json_images as cji  # noqa: E402 (import after sys.path mutation)
import web_scraper as ws            # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(color=(128, 64, 32)) -> bytes:
    """Return a minimal valid JPEG as raw bytes."""
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (500, 500), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _write_local_record(tmp_path: Path, pid: str, img_name: str) -> tuple:
    """
    Create a local image file and a JSON record pointing at it.
    Returns (json_file_path, local_img_path).
    """
    img_path = tmp_path / "images" / img_name
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(_make_jpeg_bytes())

    record = {
        "product": {"motion_product_id": pid},
        "candidate_images": [{
            "image_url": f"http://example.com/{img_name}",
            "local_path": str(img_path),
            "storage_type": "local",
        }],
    }
    json_file = tmp_path / f"{pid}.json"
    json_file.write_text(json.dumps(record), encoding="utf-8")
    return json_file, img_path


def _mock_model(winning_class: int = 3):
    """Return a MagicMock that behaves like a trained torch model."""
    import torch
    mock = MagicMock()
    out = torch.zeros(1, 8)
    out[0][winning_class] = 10.0
    mock.return_value = out
    return mock


# ── Test 1: local classification writes predicted_class ───────────────────────

def test_local_classification_writes_predicted_class(tmp_path):
    json_file, _ = _write_local_record(tmp_path, "LOCAL001", "local.jpg")

    with patch.object(cji, "load_model", return_value=_mock_model(3)), \
         patch.object(cji, "make_preprocess", return_value=cji.make_preprocess()):
        n = cji.classify_json_files([json_file], Path("dummy_model.pth"))

    assert n == 1
    data = json.loads(json_file.read_text())
    assert data["candidate_images"][0]["predicted_class"] == 3


# ── Test 2: MinIO classification writes predicted_class ───────────────────────

def test_minio_classification_writes_predicted_class(tmp_path):
    record = {
        "product": {"motion_product_id": "MINIO001"},
        "candidate_images": [{
            "image_url": "http://minio.local/img.jpg",
            "local_path": "images/MINIO001/img.jpg",
            "storage_type": "minio",
        }],
    }
    json_file = tmp_path / "MINIO001.json"
    json_file.write_text(json.dumps(record), encoding="utf-8")

    fake_bytes = _make_jpeg_bytes(color=(10, 20, 30))
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(fake_bytes)}

    with patch.object(cji, "load_model", return_value=_mock_model(5)), \
         patch.dict("os.environ", {"MINIO_BUCKET": "mi-images"}):
        n = cji.classify_json_files(
            [json_file], Path("dummy_model.pth"), s3_client=mock_s3
        )

    assert n == 1
    data = json.loads(json_file.read_text())
    assert data["candidate_images"][0]["predicted_class"] == 5
    mock_s3.get_object.assert_called_once_with(
        Bucket="mi-images", Key="images/MINIO001/img.jpg"
    )


# ── Test 3: ES sync updates mi_candidate_images ───────────────────────────────

def test_es_sync_updates_candidate_images(tmp_path):
    pid = "P999"
    url = "http://example.com/img.jpg"
    expected_doc_id = hashlib.sha1(f"{pid}:{url}".encode()).hexdigest()

    record = {
        "product": {"motion_product_id": pid},
        "candidate_images": [{
            "image_url": url,
            "predicted_class": 2,
            "storage_type": "local",
            "local_path": "images/p999.jpg",
        }],
    }
    json_file = tmp_path / f"{pid}.json"
    json_file.write_text(json.dumps(record), encoding="utf-8")

    mock_es = MagicMock()
    ws.push_predictions_to_es([json_file], mock_es)

    mock_es.update.assert_called_once_with(
        index="mi_candidate_images",
        id=expected_doc_id,
        doc={"predicted_class": 2},
    )


# ── Test 4: classification scoped to saved_files only ─────────────────────────

def test_classification_scoped_to_saved_files_only(tmp_path):
    """
    Two JSON files exist on disk; only one is passed to classify_json_files.
    The file NOT passed must not gain a predicted_class field.
    """
    current_file, _ = _write_local_record(tmp_path, "CURRENT", "current.jpg")
    other_file, _   = _write_local_record(tmp_path, "OTHER",   "other.jpg")

    with patch.object(cji, "load_model", return_value=_mock_model(1)), \
         patch.object(cji, "make_preprocess", return_value=cji.make_preprocess()):
        n = cji.classify_json_files([current_file], Path("dummy_model.pth"))

    assert n == 1

    other_data = json.loads(other_file.read_text())
    assert "predicted_class" not in other_data["candidate_images"][0]


# ── Test 5: missing MinIO object warns and does not crash ─────────────────────

def test_missing_minio_object_warns_and_continues(tmp_path, caplog):
    from botocore.exceptions import ClientError

    record = {
        "product": {"motion_product_id": "MISSING001"},
        "candidate_images": [{
            "image_url": "http://minio.local/missing.jpg",
            "local_path": "images/MISSING001/missing.jpg",
            "storage_type": "minio",
        }],
    }
    json_file = tmp_path / "MISSING001.json"
    json_file.write_text(json.dumps(record), encoding="utf-8")

    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        "GetObject",
    )

    with patch.object(cji, "load_model", return_value=_mock_model()), \
         caplog.at_level(logging.WARNING, logger="classifier"):
        n = cji.classify_json_files(
            [json_file], Path("dummy_model.pth"), s3_client=mock_s3
        )

    assert n == 0
    assert "Could not fetch MinIO object" in caplog.text

    data = json.loads(json_file.read_text())
    assert "predicted_class" not in data["candidate_images"][0]


# ── Test 6: classifier_confidence is written alongside predicted_class ─────────

def test_classifier_confidence_is_written(tmp_path):
    """Phase 1: classify_json_files must write classifier_confidence alongside predicted_class."""
    json_file, _ = _write_local_record(tmp_path, "CONF001", "conf.jpg")

    with patch.object(cji, "load_model", return_value=_mock_model(2)), \
         patch.object(cji, "make_preprocess", return_value=cji.make_preprocess()):
        cji.classify_json_files([json_file], Path("dummy_model.pth"))

    data = json.loads(json_file.read_text())
    candidate = data["candidate_images"][0]
    assert candidate["predicted_class"] == 2
    assert "classifier_confidence" in candidate
    assert 0.0 <= candidate["classifier_confidence"] <= 1.0

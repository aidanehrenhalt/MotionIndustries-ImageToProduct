"""
Pipeline API Server
-------------------
Accepts file uploads from the React frontend, triggers the scrape → classify → rank
pipeline, and serves the resulting review_queue.json.

Usage:
    .venv/bin/python src/api/server.py

Endpoints:
    POST /api/upload                        Upload and parse a product catalog file
    POST /api/pipeline/run                  Start the pipeline for an uploaded file
    GET  /api/pipeline/status/<job_id>      Poll pipeline progress
    GET  /api/review-queue                  Return the latest review_queue.json
    GET  /api/review-queue/<job_id>         Return review_queue.json for a specific job
    GET  /api/jobs                          List all jobs (debug)
    GET  /health                            Health check

Notes:
- web_scraper.py writes to output/ relative to its cwd (project root).
- All stages run with cwd=ROOT so output paths are consistent.
- PYTHONPATH is set to include src/web_scraping so the scraper can import
  text_based_search without an editable install.
- Concurrent jobs are not isolated (they share ROOT/output/). This is
  intentional for a dev/local environment. For production, each job should
  run with its own output directory.
"""

import csv as csv_mod
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = ROOT / "uploads"
JOBS_DIR = ROOT / "uploads" / "jobs"
DEBUG_UPLOAD_DIR = UPLOAD_DIR / "debug"

MODEL_PATH = ROOT / "src" / "Image_Classifier" / "trained_model.pth"
SCRAPER_SCRIPT = ROOT / "src" / "web_scraping" / "web_scraper.py"
CLASSIFIER_SCRIPT = ROOT / "src" / "Image_Classifier" / "classify_json_images.py"
RANKER_SCRIPT = ROOT / "src" / "web_scraping" / "image_search_ranker.py"

# Standard scraper output directories (relative to ROOT, as web_scraper.py writes them)
SCRAPER_JSON_DIR = ROOT / "output" / "json"
SCRAPER_IMAGES_DIR = ROOT / "output" / "images"
SCRAPER_RANKINGS_CSV = ROOT / "output" / "rankings.csv"

VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── Column normalisation map ───────────────────────────────────────────────────
# Maps common upload column header variations → canonical pipeline field names.
COLUMN_MAP = {
    # motion_product_id
    "motion product id": "motion_product_id",
    "motion_product_id": "motion_product_id",
    "product id": "motion_product_id",
    "product_id": "motion_product_id",
    "id": "motion_product_id",

    # item_number
    "item number": "item_number",
    "item_number": "item_number",
    "item no": "item_number",
    "item#": "item_number",

    # enterprise_name
    "enterprise name": "enterprise_name",
    "enterprise_name": "enterprise_name",
    "enterprise": "enterprise_name",
    "enterprise number": "enterprise_name",
    "enterprise_number": "enterprise_name",

    # mfr_name
    "manufacturer name": "mfr_name",
    "manufacturer_name": "mfr_name",
    "mfr name": "mfr_name",
    "mfr_name": "mfr_name",
    "manufacturer": "mfr_name",
    "brand": "mfr_name",

    # mfr_part_number
    "manufacturer part number": "mfr_part_number",
    "manufacturer_part_number": "mfr_part_number",
    "mfr part number": "mfr_part_number",
    "mfr_part_number": "mfr_part_number",
    "part number": "mfr_part_number",
    "part_number": "mfr_part_number",
    "mpn": "mfr_part_number",

    # web_desc
    "web product description": "web_desc",
    "web_product_description": "web_desc",
    "web desc": "web_desc",
    "web_desc": "web_desc",
    "description": "web_desc",
    "product description": "web_desc",
    "product_description": "web_desc",

    # internal_description
    "motion internal desc": "internal_description",
    "motion_internal_desc": "internal_description",
    "internal description": "internal_description",
    "internal_description": "internal_description",
    "internal desc": "internal_description",

    # pgc
    "pgc": "pgc",
    "pgc code": "pgc",
    "pgc_code": "pgc",
    "product group code": "pgc",

    # category / pgc_description
    "pgc description": "category",
    "pgc_description": "category",
    "category": "category",
    "product category": "category",

    # image filename (optional)
    "primaryimagefilename": "primary_image_filename",
    "primary image filename": "primary_image_filename",
    "image filename": "primary_image_filename",
    "image_filename": "primary_image_filename",
}

REQUIRED_OUTPUT_COLUMNS = [
    "motion_product_id",
    "mfr_name",
    "mfr_part_number",
    "web_desc",
]

# ── Logging & app ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# In-memory job registry: { job_id: { status, log, csv_path, ... } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _preload_persisted_jobs() -> None:
    """
    Scan uploads/jobs/ for any directories that contain both job_meta.json and
    review_queue.json. These are pre-built jobs (e.g. from run_demo_pipeline.py)
    that should be immediately available without a pipeline run.
    """
    for meta_path in sorted(JOBS_DIR.glob("*/job_meta.json")):
        job_id = meta_path.parent.name
        review_queue_path = meta_path.parent / "review_queue.json"
        if not review_queue_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        with _jobs_lock:
            if job_id not in _jobs:
                _jobs[job_id] = {
                    "status": "done",
                    "csv_path": str(meta_path.parent / "input.csv"),
                    "review_queue_path": str(review_queue_path),
                    "row_count": meta.get("row_count", 0),
                    "headers": meta.get("headers", []),
                    "log": [meta.get("description", f"Pre-loaded job: {job_id}")],
                    "created_at": meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "finished_at": meta.get("finished_at", datetime.now(timezone.utc).isoformat()),
                }
                log.info("Pre-loaded persisted job: %s", job_id)


_preload_persisted_jobs()


# ── File parsing helpers ───────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        mapped = COLUMN_MAP.get(key)
        if mapped:
            rename[col] = mapped
    return df.rename(columns=rename)


def _validate_df(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_OUTPUT_COLUMNS if c not in df.columns]

TEXT_UPLOAD_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def _upload_trace_enabled() -> bool:
    return os.environ.get("UPLOAD_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def _trace_upload(message: str, *args) -> None:
    if _upload_trace_enabled():
        log.info("[upload-trace] " + message, *args)


def _write_failed_upload_sample(raw: bytes, sha256: str, filename: str) -> Path:
    suffix = Path(filename).suffix or ".bin"
    DEBUG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEBUG_UPLOAD_DIR / f"{sha256}{suffix}"
    output_path.write_bytes(raw)
    return output_path


def _reject_binary_text_upload(raw: bytes, label: str) -> None:
    """Reject obviously binary uploads before permissive decoders like latin-1 accept them."""
    if b"\x00" in raw:
        raise ValueError(f"{label} upload appears to be binary data, not a text file")


def _decode_text_with_fallbacks(raw: bytes, label: str) -> str:
    """Decode uploaded text bytes using common spreadsheet/export encodings."""
    _reject_binary_text_upload(raw, label)
    decode_errors = []

    for encoding in TEXT_UPLOAD_ENCODINGS:
        try:
            text = raw.decode(encoding)
            _trace_upload("%s decode succeeded with %s", label, encoding)
            return text
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")
            _trace_upload("%s decode failed with %s: %s", label, encoding, exc)

    raise ValueError(
        f"Unable to decode {label} upload with common encodings "
        f"({', '.join(TEXT_UPLOAD_ENCODINGS)}). Details: {' | '.join(decode_errors)}"
    )


def _read_csv_with_fallbacks(raw: bytes) -> pd.DataFrame:
    """Parse CSV bytes using common encodings used by spreadsheet exports."""
    csv_kwargs = {"dtype": str, "keep_default_na": False}
    parse_errors = []

    # UTF-16 files (Excel "Unicode Text" export) have null bytes between every
    # ASCII character.  Try UTF-16 variants before the binary-data guard so we
    # don't reject them as binary.
    utf16_encodings = ["utf-16", "utf-16-le", "utf-16-be"]
    if b"\x00" in raw:
        for enc in utf16_encodings:
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError as exc:
                # Some exports produce isolated invalid surrogates — replace them
                # rather than failing on the whole file.
                try:
                    text = raw.decode(enc, errors="replace")
                except Exception as exc2:
                    parse_errors.append(f"{enc}: {exc} | replace-fallback: {exc2}")
                    _trace_upload("CSV UTF-16 parse failed with %s: %s", enc, exc2)
                    continue
                parse_errors.append(f"{enc} (strict): {exc}")
                _trace_upload("CSV UTF-16 strict decode failed with %s, retrying with replace: %s", enc, exc)
            try:
                df = pd.read_csv(io.StringIO(text), **csv_kwargs)
                _trace_upload("CSV parse succeeded with %s (UTF-16 path)", enc)
                return df
            except Exception as exc:
                parse_errors.append(f"{enc} csv: {exc}")
                _trace_upload("CSV UTF-16 parse failed with %s: %s", enc, exc)
        # Null bytes remain and no UTF-16 variant worked — truly binary.
        raise ValueError(
            "CSV upload contains null bytes and could not be decoded as UTF-16. "
            f"Details: {' | '.join(parse_errors)}"
        )

    _reject_binary_text_upload(raw, "CSV")

    for encoding in TEXT_UPLOAD_ENCODINGS:
        try:
            # Decode to str with Python's codec first, then give pandas a StringIO.
            # This sidesteps pandas' internal C-level encoder which can raise
            # UnicodeDecodeError even when encoding='latin-1' is passed.
            text = raw.decode(encoding)
            df = pd.read_csv(io.StringIO(text), **csv_kwargs)
            _trace_upload("CSV parse succeeded with %s", encoding)
            return df
        except UnicodeDecodeError as exc:
            parse_errors.append(f"{encoding}: {exc}")
            _trace_upload("CSV parse unicode decode failed with %s: %s", encoding, exc)
        except Exception as exc:
            parse_errors.append(f"{encoding}: {exc}")
            _trace_upload("CSV parse failed with %s: %s: %s", encoding, type(exc).__name__, exc)

    raise ValueError(
        "Unable to decode CSV upload with common encodings "
        f"({', '.join(TEXT_UPLOAD_ENCODINGS)}). Details: {' | '.join(parse_errors)}"
    )


def _read_json_with_fallbacks(raw: bytes):
    """Parse JSON bytes using the same text-encoding fallbacks as CSV uploads."""
    text = _decode_text_with_fallbacks(raw, "JSON")
    try:
        data = json.loads(text)
        _trace_upload("JSON parse succeeded after text decode")
        return data
    except json.JSONDecodeError as exc:
        _trace_upload("JSON parse failed after text decode: %s", exc)
        raise ValueError(f"Invalid JSON upload: {exc}") from exc


def _parse_upload_bytes(raw: bytes, filename: str) -> tuple[pd.DataFrame, list[str]]:
    filename = filename.lower()
    buf = io.BytesIO(raw)

    if filename.endswith(".csv"):
        _trace_upload("Parser branch selected: csv")
        df = _read_csv_with_fallbacks(raw)
    elif filename.endswith((".xlsx", ".xls")):
        _trace_upload("Parser branch selected: excel")
        df = pd.read_excel(buf, dtype=str, keep_default_na=False)
    elif filename.endswith(".json"):
        _trace_upload("Parser branch selected: json")
        data = _read_json_with_fallbacks(raw)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "products" in data:
            df = pd.DataFrame(data["products"])
        else:
            raise ValueError("JSON must be an array of objects or {products: [...]}")
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    df = df.fillna("").astype(str)
    df = _normalise_columns(df)
    return df, df.columns.tolist()


def _parse_upload(file_storage) -> tuple[pd.DataFrame, list[str]]:
    return _parse_upload_bytes(file_storage.read(), file_storage.filename)


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _subprocess_env() -> dict:
    """Build environment with PYTHONPATH including web_scraping directory."""
    env = os.environ.copy()
    extra_path = str(ROOT / "src" / "web_scraping")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extra_path}:{existing}" if existing else extra_path
    return env


def _job_log(job_id: str, line: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["log"].append(line)
    log.info("[job:%s] %s", job_id, line)


def _local_path_to_http_url(local_path: str) -> str:
    """
    Convert a local filesystem image path to an HTTP URL served by this API.

    web_scraper.py writes local_path as a relative path like:
        output/images/<product_id>/<filename>.jpg

    We serve these via /images/<relative-path-after-output/images/>.
    """
    if not local_path:
        return ""
    p = Path(local_path)
    try:
        # Try to get the path relative to output/images/
        rel = p.relative_to(ROOT / "output" / "images")
        return f"http://localhost:5050/images/{rel.as_posix()}"
    except ValueError:
        pass
    # Fallback: check if it contains the output/images segment
    parts = p.parts
    try:
        idx = next(i for i in range(len(parts) - 1) if parts[i] == "output" and parts[i + 1] == "images")
        rel = "/".join(parts[idx + 2:])
        return f"http://localhost:5050/images/{rel}"
    except StopIteration:
        return ""


def _build_review_queue(json_dir: Path, rankings_csv: Path, output_path: Path) -> None:
    """
    Convert per-product JSON files + rankings CSV into review_queue.json
    in the format expected by pipelineFolderParser.js.
    """
    products_by_id: dict[str, dict] = {}

    for json_path in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        product = data.get("product", {})
        product_id = product.get("motion_product_id") or json_path.stem

        raw_candidates = {
            Path(c.get("local_path") or "").name.lower(): c
            for c in data.get("candidate_images", [])
            if c.get("local_path")
        }
        products_by_id[product_id] = {
            "productId": product_id,
            "queueKey": product_id,
            "slug": product_id.lower(),
            "productName": product.get("description") or product.get("web_desc") or product_id,
            "manufacturer": product.get("mfr_name", ""),
            "manufacturerPartNumber": product.get("mfr_part_number", ""),
            "partNumber": product.get("mfr_part_number") or product_id,
            "description": product.get("web_desc") or product.get("description") or "",
            "category": product.get("category", ""),
            "jsonFile": json_path.name,
            "candidateImages": [],
            "_rawCandidates": raw_candidates,
        }

    if rankings_csv.exists():
        with rankings_csv.open(encoding="utf-8", newline="") as fh:
            for row in csv_mod.DictReader(fh):
                pid = row.get("motion_product_id", "").strip()
                entry = products_by_id.get(pid)
                if not entry:
                    continue
                fname = row.get("image_filename", "")
                raw = entry["_rawCandidates"].get(fname.lower(), {})
                local_path = raw.get("local_path", "")
                image_http_url = _local_path_to_http_url(local_path)
                entry["candidateImages"].append({
                    "id": f"{pid}-{fname}",
                    "fileName": fname,
                    "url": image_http_url,
                    "rank": int(row.get("image_rank") or 0),
                    "finalScore": float(row.get("final_score") or 0.0),
                    "finalScorePct": row.get("final_score_pct") or "",
                    "aiConfidence": float(row.get("ai_confidence") or 0.0),
                    "aiConfidencePct": row.get("ai_confidence_pct") or "",
                    "textScore": float(row.get("text_score") or 0.0),
                    "textScorePct": row.get("text_score_pct") or "",
                    "sourceName": row.get("source_name") or "",
                    "license": row.get("license") or "",
                    "predictedClass": row.get("ai_class_label") or "",
                    "localPath": local_path,
                })
    else:
        # No rankings CSV — fall back to preliminary scores from raw JSON
        for entry in products_by_id.values():
            for raw in entry["_rawCandidates"].values():
                hints = raw.get("confidence_hints", {})
                local_path = raw.get("local_path", "")
                fname = Path(local_path).name if local_path else ""
                prelim = float(hints.get("preliminary_score") or 0.0)
                ai_conf = float(raw.get("classifier_confidence") or 0.0)
                image_http_url = _local_path_to_http_url(local_path)
                entry["candidateImages"].append({
                    "id": f"{entry['productId']}-{fname}",
                    "fileName": fname,
                    "url": image_http_url,
                    "rank": 1,
                    "finalScore": prelim,
                    "finalScorePct": f"{prelim * 100:.1f}%",
                    "aiConfidence": ai_conf,
                    "aiConfidencePct": f"{ai_conf * 100:.1f}%" if ai_conf else "",
                    "textScore": float(raw.get("ranker_score") or 0.0),
                    "textScorePct": "",
                    "sourceName": raw.get("source_name", ""),
                    "license": raw.get("license", ""),
                    "predictedClass": str(raw.get("predicted_class", "")),
                    "localPath": local_path,
                })

    queue = []
    for entry in products_by_id.values():
        entry["candidateImages"].sort(
            key=lambda img: (img.get("rank") or 9999, -(img.get("finalScore") or 0.0))
        )
        entry["matchedImageCount"] = len(entry["candidateImages"])
        entry["bestConfidence"] = (
            entry["candidateImages"][0]["aiConfidence"] if entry["candidateImages"] else 0.0
        )
        entry.pop("_rawCandidates", None)
        queue.append(entry)

    queue.sort(key=lambda item: item["productId"])
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "rankingsCsv": str(rankings_csv),
            "jsonDir": str(json_dir),
        },
        "products": queue,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_pipeline(job_id: str, csv_path: Path, flags: dict) -> None:
    """
    Background thread: scrape → classify → rank → export review_queue.json.

    All stages run from ROOT so web_scraper.py writes to ROOT/output/ (its
    hardcoded relative path). Results are copied to the job directory after
    completion so each job has its own review_queue.json snapshot.
    """
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    env = _subprocess_env()
    use_es = bool(flags.get("es", False))
    use_minio = bool(flags.get("minio", False))
    limit = flags.get("limit")

    job_dir = JOBS_DIR / job_id
    review_queue_json = job_dir / "review_queue.json"

    try:
        # ── Stage 1: Scrape ────────────────────────────────────────────────────
        _job_log(job_id, "Stage 1/3: Scraping images (Wikimedia + OpenVerse)...")
        scraper_cmd = [
            str(VENV_PYTHON), str(SCRAPER_SCRIPT),
            "--csv", str(csv_path),
            "--classify",
        ]
        if limit:
            scraper_cmd += ["--limit", str(limit)]
        if use_es:
            scraper_cmd.append("--es")
        if use_minio:
            scraper_cmd.append("--minio")

        result = subprocess.run(
            scraper_cmd,
            capture_output=True, text=True,
            cwd=str(ROOT),
            env=env,
            timeout=600,
        )
        stdout_tail = result.stdout[-800:] if result.stdout else ""
        stderr_tail = result.stderr[-800:] if result.stderr else ""
        _job_log(job_id, f"Scraper exit={result.returncode}")
        if stdout_tail:
            _job_log(job_id, f"stdout: {stdout_tail}")
        if result.returncode != 0:
            raise RuntimeError(f"Scraper failed (exit {result.returncode}):\n{stderr_tail}")

        json_files = list(SCRAPER_JSON_DIR.glob("*.json"))
        if not json_files:
            raise RuntimeError(
                f"Scraper produced no JSON output in {SCRAPER_JSON_DIR}. "
                "Check that the products have descriptions and the APIs are reachable."
            )
        _job_log(job_id, f"Scraper wrote {len(json_files)} JSON file(s).")

        # ── Stage 2: Rank ──────────────────────────────────────────────────────
        # Note: --classify above already ran the CNN classifier inside web_scraper.py.
        # This stage runs the text-similarity ranker and computes final_score.
        _job_log(job_id, "Stage 2/3: Ranking candidates (text similarity + AI fusion)...")
        rank_cmd = [
            str(VENV_PYTHON), str(RANKER_SCRIPT),
            "--json-dir", str(SCRAPER_JSON_DIR),
            "--output", str(SCRAPER_RANKINGS_CSV),
        ]
        result = subprocess.run(
            rank_cmd, capture_output=True, text=True,
            cwd=str(ROOT), env=env, timeout=300,
        )
        if result.returncode != 0:
            _job_log(job_id, f"Ranker exited {result.returncode} — continuing (non-fatal)")
            _job_log(job_id, f"Ranker stderr: {result.stderr[-500:]}")
        else:
            _job_log(job_id, "Ranker done.")

        # ── Stage 3: Build review_queue.json ──────────────────────────────────
        _job_log(job_id, "Stage 3/3: Building review_queue.json...")
        _build_review_queue(SCRAPER_JSON_DIR, SCRAPER_RANKINGS_CSV, review_queue_json)
        _job_log(job_id, f"review_queue.json → {review_queue_json}")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["review_queue_path"] = str(review_queue_json)
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            _jobs[job_id]["json_count"] = len(json_files)

    except Exception as exc:
        _job_log(job_id, f"PIPELINE ERROR: {exc}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accept a multipart file upload. Validate, parse, normalise columns, and
    persist as CSV in uploads/jobs/<job_id>/input.csv.

    Returns:
        { jobId, fileName, rowCount, headers, missingRequired }
    """
    _trace_upload("Incoming request content_type=%s, has_file=%s", request.content_type, "file" in request.files)

    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    raw = file.read()
    sha256 = hashlib.sha256(raw).hexdigest()
    hex_preview = raw[:32].hex()
    _trace_upload(
        "Received filename=%s content_type=%s ext=%s bytes=%d sha256=%s head32=%s",
        file.filename,
        file.content_type,
        ext,
        len(raw),
        sha256,
        hex_preview,
    )

    if ext not in ("xlsx", "xls", "csv", "json"):
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    try:
        df, headers = _parse_upload_bytes(raw, file.filename)
    except Exception as exc:
        _trace_upload("Upload parse failed: %s: %s", type(exc).__name__, exc)
        if _upload_trace_enabled():
            sample_path = _write_failed_upload_sample(raw, sha256, file.filename)
            _trace_upload("Saved failing upload sample to %s", sample_path)
        return jsonify({"error": f"Parse error: {exc}"}), 422

    if len(df) == 0:
        return jsonify({"error": "File has headers but no data rows"}), 422

    missing = _validate_df(df)

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    csv_path = job_dir / "input.csv"
    df.to_csv(csv_path, index=False)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "uploaded",
            "csv_path": str(csv_path),
            "row_count": len(df),
            "headers": headers,
            "log": [f"Uploaded {file.filename} ({len(df)} rows)"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    log.info("Uploaded job %s: %d rows, headers=%s", job_id, len(df), headers)
    return jsonify({
        "jobId": job_id,
        "fileName": file.filename,
        "rowCount": len(df),
        "headers": headers,
        "missingRequired": missing,
    })


@app.route("/api/pipeline/run", methods=["POST"])
def pipeline_run():
    """
    Start the pipeline for an uploaded job.

    Body (JSON):
        { jobId: string, limit?: number, es?: bool, minio?: bool }
    """
    body = request.get_json(force=True) or {}
    job_id = body.get("jobId")
    if not job_id:
        return jsonify({"error": "jobId required"}), 400

    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Unknown jobId"}), 404
    if job["status"] == "running":
        return jsonify({"jobId": job_id, "status": "already_running"}), 200

    csv_path = Path(job["csv_path"])
    flags = {
        "es": bool(body.get("es", False)),
        "minio": bool(body.get("minio", False)),
        "limit": body.get("limit"),
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, csv_path, flags),
        daemon=True,
    )
    thread.start()

    return jsonify({"jobId": job_id, "status": "started"})


@app.route("/api/pipeline/status/<job_id>", methods=["GET"])
def pipeline_status(job_id: str):
    """Poll pipeline progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Unknown jobId"}), 404

    return jsonify({
        "jobId": job_id,
        "status": job.get("status"),
        "log": job.get("log", []),
        "rowCount": job.get("row_count"),
        "jsonCount": job.get("json_count"),
        "createdAt": job.get("created_at"),
        "startedAt": job.get("started_at"),
        "finishedAt": job.get("finished_at"),
        "error": job.get("error"),
    })


@app.route("/api/review-queue", methods=["GET"])
def review_queue_latest():
    """Return the most recently completed job's review_queue.json."""
    with _jobs_lock:
        done_jobs = [
            (jid, j) for jid, j in _jobs.items()
            if j.get("status") == "done" and j.get("review_queue_path")
        ]

    if not done_jobs:
        return jsonify({"error": "No completed pipeline runs yet"}), 404

    done_jobs.sort(key=lambda pair: pair[1].get("finished_at", ""), reverse=True)
    _, latest = done_jobs[0]
    path = Path(latest["review_queue_path"])
    if not path.exists():
        return jsonify({"error": "review_queue.json not found on disk"}), 404

    return send_file(str(path), mimetype="application/json")


@app.route("/api/review-queue/<job_id>", methods=["GET"])
def review_queue_by_job(job_id: str):
    """Return the review_queue.json for a specific completed job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Unknown jobId"}), 404
    if job.get("status") != "done":
        return jsonify({
            "error": f"Job status is '{job.get('status')}', not done",
            "status": job.get("status"),
        }), 409

    path = Path(job.get("review_queue_path", ""))
    if not path.exists():
        return jsonify({"error": "review_queue.json not found"}), 404

    return send_file(str(path), mimetype="application/json")


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all jobs (debug)."""
    with _jobs_lock:
        summary = {
            jid: {
                "status": j.get("status"),
                "rowCount": j.get("row_count"),
                "createdAt": j.get("created_at"),
                "finishedAt": j.get("finished_at"),
                "error": j.get("error"),
            }
            for jid, j in _jobs.items()
        }
    return jsonify(summary)


@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename: str):
    """Serve downloaded images over HTTP so the browser can load them."""
    return send_file(str(ROOT / "output" / "images" / filename))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_exists": MODEL_PATH.exists(),
        "scraper_exists": SCRAPER_SCRIPT.exists(),
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 5050))
    log.info("Pipeline API server starting on http://localhost:%d", port)
    log.info("ROOT:    %s", ROOT)
    log.info("Scraper: %s (exists=%s)", SCRAPER_SCRIPT, SCRAPER_SCRIPT.exists())
    log.info("Model:   %s (exists=%s)", MODEL_PATH, MODEL_PATH.exists())
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

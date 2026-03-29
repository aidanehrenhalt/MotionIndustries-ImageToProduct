#!/usr/bin/env python3
"""
Image Search & Ranking System  (v2 — Unified Ranker)
=====================================================
Scores and ranks candidate images by fusing two independent signals:

  1. **AI Classifier signal** — CNN-predicted class + confidence written
     into each candidate's JSON record by
     ``src/Image_Classifier/classify_json_images.py``.

  2. **Text-based search signal** — token-overlap similarity between the
     product's metadata and each candidate's scraped title / tags / URL,
     computed directly from the JSON record or from results returned by
     ``text_based_search.py``.

Both signals are combined into a single ``final_score`` per candidate.
Results are written back to each JSON record **and** exported to a flat
CSV file that the website interface can read directly.

──────────────────────────────────────────────────
CSV output columns
──────────────────────────────────────────────────
  motion_product_id     – ID that links to the database record / folder
  image_rank            – 1-based rank within this product (1 = best)
  image_filename        – basename of the local image file
  image_url             – original source URL
  final_score_pct       – fused score as "74.3%"
  final_score           – fused score as a 0-1 float
  ai_confidence_pct     – CNN confidence as "82.1%"
  ai_confidence         – CNN confidence raw float
  ai_predicted_class    – 0-indexed CNN class index (-1 = not run)
  ai_class_label        – human-readable PGC1 label ("BEARINGS", …)
  text_score_pct        – text-similarity score as "61.0%"
  text_score            – text-similarity score raw float
  preliminary_score     – scrape-time heuristic (0-1)
  source_name           – scrape source ("Wikimedia Commons", …)
  license               – image license string
  image_width           – pixel width
  image_height          – pixel height
  scraped_at            – ISO timestamp

──────────────────────────────────────────────────
Scoring formula
──────────────────────────────────────────────────
When the product PGC is known and the expected CNN class is available:

  final_score = W_AI_CONF    * ai_confidence        (default 0.30)
              + W_CLASS_MATCH * class_match_flag     (default 0.20)
              + W_TEXT        * text_similarity      (default 0.30)
              + W_PRELIM      * preliminary_score    (default 0.20)

When PGC is unknown (no expected class), ``W_CLASS_MATCH`` is
redistributed proportionally across the remaining three weights so the
formula still sums to 1.

──────────────────────────────────────────────────
Standalone usage
──────────────────────────────────────────────────
  # Rank everything in output/json, emit CSV:
  python image_search_ranker.py

  # Custom paths:
  python image_search_ranker.py \\
      --json-dir output/json \\
      --output   rankings.csv \\
      --min-score 0.05

  # Preview table without writing CSV:
  python image_search_ranker.py --dry-run

──────────────────────────────────────────────────
Programmatic usage
──────────────────────────────────────────────────
  from image_search_ranker import (
      rank_candidates,          # enrich + sort a flat list of candidate dicts
      rank_json_files,          # text-ranker pass over a list of JSON Paths
      apply_final_ranking,      # fuse AI + text → final_score, write JSON
      export_rankings_to_csv,   # flatten all JSON records → CSV
      score_against_vector,     # low-level: score one vector vs another
      ImageSearcher,            # catalog-based standalone search
      ProductVector,
  )
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("image_search_ranker")


# ─────────────────────────────────────────────────────────────────────────────
# ProductVector  (shared canonical 10-field struct)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from motion_image_pipeline import ProductVector  # type: ignore
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

    @dataclass
    class ProductVector:
        """
        Canonical 10-field product vector.

        Field index / name reference
        ─────────────────────────────
        [0]  id_number               – Primary product key  (motion_product_id)
        [1]  image_name              – Expected / known image filename
        [2]  item_number             – Motion internal item number
        [3]  enterprise_number       – Enterprise / brand parent ID
        [4]  manufacture_name        – Manufacturer / brand name
        [5]  manufacture_part_number – Manufacturer's own part number
        [6]  web_product_description – Full web-facing description
        [7]  motion_internal_desc    – Motion's internal description
        [8]  pgc                     – Product Group Code  (e.g. "1110")
        [9]  pgc_description         – Human-readable product category
        """
        id_number:               str = ""
        image_name:              str = ""
        item_number:             str = ""
        enterprise_number:       str = ""
        manufacture_name:        str = ""
        manufacture_part_number: str = ""
        web_product_description: str = ""
        motion_internal_desc:    str = ""
        pgc:                     str = ""
        pgc_description:         str = ""

        @classmethod
        def from_list(cls, vec: list) -> "ProductVector":
            padded = (list(vec) + [""] * 10)[:10]
            return cls(*[str(v).strip() for v in padded])

        def to_dict(self) -> dict:
            return {
                "id_number":               self.id_number,
                "image_name":              self.image_name,
                "item_number":             self.item_number,
                "enterprise_number":       self.enterprise_number,
                "manufacture_name":        self.manufacture_name,
                "manufacture_part_number": self.manufacture_part_number,
                "web_product_description": self.web_product_description,
                "motion_internal_desc":    self.motion_internal_desc,
                "pgc":                     self.pgc,
                "pgc_description":         self.pgc_description,
            }


# ─────────────────────────────────────────────────────────────────────────────
# PGC label map  (mirrors pgc_mapping.py — no import dependency needed here)
# ─────────────────────────────────────────────────────────────────────────────

PGC1_LABELS: dict[int, str] = {
    1: "BEARINGS",
    2: "SEALS AND ACCESSORIES",
    3: "POWER TRANSMISSION",
    4: "ELECTRICAL & MAT'L HAND'G",
    5: "HOSE AND FITTINGS",
    6: "FLUID POWER",
    7: "PROCESS PUMPS AND EQUIPMENT",
    8: "INDUSTRIAL SUPPLIES",
}


def _pgc_to_expected_classes(pgc_raw: str) -> set:
    """
    Return the set of 0-indexed CNN class indices that match this PGC code.
    Returns an empty set when the PGC is absent, invalid, or PGC1 == 9.
    """
    s = str(pgc_raw).strip() if pgc_raw else ""
    if not s:
        return set()
    try:
        first_digit = int(s[0])
    except (ValueError, IndexError):
        return set()
    if first_digit not in PGC1_LABELS:
        return set()
    return {first_digit - 1}   # 0-indexed


def _class_label(class_index: int) -> str:
    """Convert a 0-indexed CNN class (0-7) to its human-readable PGC1 label."""
    if class_index < 0:
        return "NOT CLASSIFIED"
    return PGC1_LABELS.get(class_index + 1, "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights  (tune here to adjust signal importance)
# ─────────────────────────────────────────────────────────────────────────────

W_AI_CONF:    float = 0.30   # CNN classifier confidence
W_CLASS_MATCH: float = 0.20  # PGC class correctness flag (1.0 / 0.0)
W_TEXT:        float = 0.30  # text-similarity / ranker score
W_PRELIM:      float = 0.20  # scrape-time preliminary heuristic score

# Field weights for the text-similarity sub-scorer (ProductVector comparison)
FIELD_WEIGHTS: dict = {
    "id_number":               10.0,
    "item_number":              8.0,
    "manufacture_part_number":  8.0,
    "enterprise_number":        6.0,
    "image_name":               5.0,
    "manufacture_name":         4.0,
    "pgc":                      3.0,
    "pgc_description":          3.0,
    "web_product_description":  2.0,
    "motion_internal_desc":     2.0,
}
TOTAL_WEIGHT: float = sum(FIELD_WEIGHTS.values())

# Tier thresholds for text scoring
EXACT_BONUS:    float = 1.00
CONTAINS_BONUS: float = 0.60
TOKEN_BONUS:    float = 0.30

# Maximum words used from free-text description fields in text queries
_DESCRIPTION_WORD_LIMIT = 8


# ─────────────────────────────────────────────────────────────────────────────
# Low-level text-similarity primitives
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lower-case and split on non-alphanumeric chars; drop empty tokens."""
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def _field_score(query_val: str, catalog_val: str) -> float:
    """
    Compare two field strings and return a 0-1 relevance score.

    Tiers (best wins):
      1.00 — exact case-insensitive match
      0.60 — one string wholly contains the other
      0-0.30 — token overlap (geometric-mean normalised Jaccard)
    """
    if not query_val or not catalog_val:
        return 0.0
    q, c = query_val.strip().lower(), catalog_val.strip().lower()
    if q == c:
        return EXACT_BONUS
    score = 0.0
    if q in c or c in q:
        score = max(score, CONTAINS_BONUS)
    qt, ct = _tokenize(query_val), _tokenize(catalog_val)
    if qt and ct:
        overlap = len(qt & ct) / math.sqrt(len(qt) * len(ct))
        score = max(score, TOKEN_BONUS * overlap)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# RankedResult  (used by the standalone ImageSearcher)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    """A catalog ProductVector paired with its weighted relevance score."""
    record:    ProductVector
    score:     float
    breakdown: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RankedResult(image='{self.record.image_name}', "
            f"score={self.score:.4f}, {self.score * 100:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core vector–vs–vector scorer  (shared by pipeline and standalone modes)
# ─────────────────────────────────────────────────────────────────────────────

def score_against_vector(
    query: ProductVector,
    catalog: ProductVector,
) -> RankedResult:
    """
    Score one catalog ProductVector against a query ProductVector.

    Returns a RankedResult with:
      .score      — weighted composite, normalised to [0, 1]
      .breakdown  — per-field dict for explainability
    """
    breakdown: dict = {}
    weighted_sum = 0.0
    query_dict   = query.to_dict()
    catalog_dict = catalog.to_dict()

    for fname, weight in FIELD_WEIGHTS.items():
        fs = _field_score(query_dict[fname], catalog_dict[fname])
        contrib = fs * weight
        breakdown[fname] = {
            "raw_score":    round(fs, 4),
            "weight":       weight,
            "contribution": round(contrib, 4),
        }
        weighted_sum += contrib

    return RankedResult(
        record    = catalog,
        score     = weighted_sum / TOTAL_WEIGHT,
        breakdown = breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text-similarity scorer for a raw candidate dict  (pipeline integration)
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_dict_to_vector(candidate: dict) -> ProductVector:
    """
    Build a pseudo-ProductVector from a scraped image candidate dict.

    Field mapping from candidate JSON:
      image_name              ← title
      manufacture_name        ← first token of keyword_used (if present)
      manufacture_part_number ← keyword_used
      web_product_description ← title + space-joined tags
      pgc_description         ← tags joined
    """
    title = candidate.get("title", "")
    tags  = " ".join(candidate.get("tags", []))
    kw    = candidate.get("keyword_used", "")

    return ProductVector(
        image_name              = title,
        manufacture_name        = kw.split()[0] if kw else "",
        manufacture_part_number = kw,
        web_product_description = f"{title} {tags}".strip(),
        pgc_description         = tags,
    )


def _text_score_from_candidate(query: ProductVector, candidate: dict) -> float:
    """
    Return a 0-1 text-similarity score for one candidate dict.

    Prefers the already-stored ``ranker_score`` field (written by a
    previous ranking pass) so we never re-compute unnecessarily.
    Falls back to computing from the candidate's title / tags / keyword fields.
    """
    if "ranker_score" in candidate:
        return float(candidate["ranker_score"])

    cat_vec = _candidate_dict_to_vector(candidate)
    result  = score_against_vector(query, cat_vec)
    return result.score


# ─────────────────────────────────────────────────────────────────────────────
# Final fused score  (AI + text combined)
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(
    candidate: dict,
    expected_classes: set,
    *,
    w_ai_conf:    float = W_AI_CONF,
    w_class_match: float = W_CLASS_MATCH,
    w_text:        float = W_TEXT,
    w_prelim:      float = W_PRELIM,
) -> float:
    """
    Fuse AI classifier + text-similarity signals into a single 0-1 score.

    Scoring formula (PGC known):
        final = w_ai_conf    * classifier_confidence
              + w_class_match * class_match           (1.0 if correct PGC)
              + w_text        * text_similarity_score
              + w_prelim      * preliminary_score

    When PGC is unknown (expected_classes is empty), ``w_class_match`` is
    redistributed proportionally across the other three weights so the
    formula still sums to 1.

    Args:
        candidate:        Candidate image dict (enriched by scraper/classifier).
        expected_classes: Set of 0-indexed CNN classes valid for this product.
        w_*:              Weight overrides (use defaults in most cases).

    Returns:
        Fused score as a float in [0, 1].
    """
    ai_confidence  = float(candidate.get("classifier_confidence", 0.0))
    predicted_cls  = int(candidate.get("predicted_class", -1))
    text_score     = float(
        candidate.get("ranker_score") or
        candidate.get("text_similarity_score", 0.0)
    )
    prelim         = float(
        candidate.get("confidence_hints", {}).get("preliminary_score", 0.0)
    )

    if expected_classes:
        class_match = 1.0 if predicted_cls in expected_classes else 0.0
        return round(
            w_ai_conf     * ai_confidence
            + w_class_match * class_match
            + w_text        * text_score
            + w_prelim      * prelim,
            6,
        )
    else:
        # No PGC — redistribute class_match weight
        remaining = w_ai_conf + w_text + w_prelim
        if remaining == 0.0:
            return 0.0
        scale = (w_ai_conf + w_class_match + w_text + w_prelim) / remaining
        return round(
            (w_ai_conf * ai_confidence + w_text * text_score + w_prelim * prelim)
            * scale,
            6,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration — rank a flat list of candidate dicts
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    query: ProductVector,
    candidates: list,
    min_score: float = 0.0,
) -> list:
    """
    Score and rank a list of candidate image dicts against a query vector.

    This is the primary pipeline integration point.  Each candidate dict is
    enriched **in-place** with:
      ``ranker_score``    – text-similarity composite score (0-1 float)
      ``score_pct``       – e.g. "42.7%"
      ``score_breakdown`` – per-field detail for explainability

    A copy of each enriched dict is returned in descending score order.
    The original list is not reordered.

    Args:
        query:      ProductVector for the product we want an image for.
        candidates: List of dicts from the scraper / text_based_search.
        min_score:  Discard candidates below this threshold (default 0).

    Returns:
        New list of enriched dicts sorted by ranker_score descending.
    """
    scored: list = []

    for candidate in candidates:
        cat_vec = _candidate_dict_to_vector(candidate)
        result  = score_against_vector(query, cat_vec)

        if result.score < min_score:
            continue

        enriched = dict(candidate)
        enriched["ranker_score"]    = round(result.score, 6)
        enriched["score_pct"]       = f"{result.score * 100:.1f}%"
        enriched["score_breakdown"] = result.breakdown
        scored.append(enriched)

    scored.sort(key=lambda x: x["ranker_score"], reverse=True)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# JSON-file level passes  (called by the pipeline runner)
# ─────────────────────────────────────────────────────────────────────────────

def _build_query_vector_from_json(data: dict) -> ProductVector:
    """Build a ProductVector from a JSON record's ``product`` block."""
    prod = data.get("product", {})
    return ProductVector(
        id_number               = prod.get("motion_product_id", ""),
        manufacture_name        = prod.get("mfr_name", ""),
        manufacture_part_number = prod.get("mfr_part_number", ""),
        web_product_description = prod.get("description", ""),
        pgc_description         = prod.get("category", ""),
        pgc                     = prod.get("pgc", ""),
    )


def rank_json_files(json_files) -> int:
    """
    Text-ranker pass: compute ``ranker_score`` / ``score_pct`` /
    ``score_breakdown`` for every candidate in each JSON file and write
    results back.

    This pass is **independent of the CNN classifier** — it only uses
    metadata text similarity.  Run it before ``apply_final_ranking()``.

    Args:
        json_files: Iterable of Path objects pointing at scraper JSON records.

    Returns:
        Total number of candidates scored.
    """
    total_scored = 0

    for json_file in json_files:
        json_file = Path(json_file)
        if not json_file.exists():
            log.warning("[ranker] JSON not found, skipping: %s", json_file)
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("[ranker] Skipping %s: %s", json_file.name, exc)
            continue

        candidates = data.get("candidate_images", [])
        if not candidates:
            continue

        query = _build_query_vector_from_json(data)

        # Ensure every candidate has a stable index key before we sort copies.
        for i, img in enumerate(candidates):
            if "index" not in img:
                log.warning(
                    "[ranker] Candidate at position %d in %s has no 'index' field; "
                    "synthesizing index=%d", i, json_file.name, i
                )
                img["index"] = i

        ranked = rank_candidates(query, candidates)

        # Merge ranker fields back by the stable index key (not position).
        ranked_by_index = {r["index"]: r for r in ranked}
        for img_info in candidates:
            key = img_info["index"]
            if key in ranked_by_index:
                r = ranked_by_index[key]
                img_info["ranker_score"]          = r["ranker_score"]
                img_info["score_pct"]             = r["score_pct"]
                img_info["score_breakdown"]       = r["score_breakdown"]
                img_info["text_similarity_score"] = r["ranker_score"]
                total_scored += 1

        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("[ranker] Text-scored %d candidate(s) in %s",
                 len(candidates), json_file.name)

    log.info("[ranker] Text-scored %d candidates total.", total_scored)
    return total_scored


def apply_final_ranking(json_files) -> int:
    """
    Fuse AI classifier + text-similarity signals into ``final_score`` for
    every candidate in each JSON file, sort candidates by ``final_score``
    descending, assign 1-based ``final_rank``, and write results back.

    Must be called **after** both the classifier pass
    (``classify_json_images.classify_json_files()``) and ``rank_json_files()``
    have completed so that ``predicted_class``, ``classifier_confidence``,
    and ``ranker_score`` are all present.

    Args:
        json_files: Iterable of Path objects pointing at scraper JSON records.

    Returns:
        Total number of candidates ranked.
    """
    total_ranked = 0

    for json_file in json_files:
        json_file = Path(json_file)
        if not json_file.exists():
            log.warning("[ranker] JSON not found, skipping: %s", json_file)
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("[ranker] Skipping %s: %s", json_file.name, exc)
            continue

        candidates = data.get("candidate_images", [])
        if not candidates:
            continue

        pgc_raw          = data.get("product", {}).get("pgc", "")
        expected_classes = _pgc_to_expected_classes(pgc_raw)

        for img_info in candidates:
            img_info["final_score"] = compute_final_score(img_info, expected_classes)
            img_info["final_score_pct"] = f"{img_info['final_score'] * 100:.1f}%"

        candidates.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)

        for rank, img_info in enumerate(candidates, 1):
            img_info["final_rank"] = rank
            total_ranked += 1

        data["candidate_images"] = candidates
        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("[ranker] Final-ranked %d candidate(s) in %s",
                 len(candidates), json_file.name)

    log.info("[ranker] Final-ranked %d candidates total.", total_ranked)
    return total_ranked


# ─────────────────────────────────────────────────────────────────────────────
# CSV export  (flat file for the website interface)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical column order for the output CSV
CSV_COLUMNS: list = [
    "motion_product_id",
    "image_rank",
    "image_filename",
    "image_url",
    "final_score_pct",
    "final_score",
    "ai_confidence_pct",
    "ai_confidence",
    "ai_predicted_class",
    "ai_class_label",
    "text_score_pct",
    "text_score",
    "preliminary_score",
    "source_name",
    "license",
    "image_width",
    "image_height",
    "scraped_at",
]


def _extract_csv_row(product_id: str, rank: int, img: dict) -> dict:
    """
    Flatten one candidate image dict into a CSV-ready row dict.

    All fields are safe strings / numbers suitable for csv.DictWriter.
    """
    ai_conf     = float(img.get("classifier_confidence", 0.0))
    ai_cls      = int(img.get("predicted_class", -1))
    text_score  = float(
        img.get("text_similarity_score") or img.get("ranker_score", 0.0)
    )
    prelim      = float(
        img.get("confidence_hints", {}).get("preliminary_score", 0.0)
    )
    final_score = float(img.get("final_score", 0.0))

    # Derive image filename from local_path or image_url
    local_path = img.get("local_path", "") or ""
    filename   = Path(local_path).name if local_path else ""
    if not filename:
        url_path = img.get("image_url", "") or ""
        filename = Path(url_path.split("?")[0]).name

    return {
        "motion_product_id":  product_id,
        "image_rank":         rank,
        "image_filename":     filename,
        "image_url":          img.get("image_url", ""),
        "final_score_pct":    f"{final_score * 100:.1f}%",
        "final_score":        round(final_score, 6),
        "ai_confidence_pct":  f"{ai_conf * 100:.1f}%",
        "ai_confidence":      round(ai_conf, 4),
        "ai_predicted_class": ai_cls,
        "ai_class_label":     _class_label(ai_cls),
        "text_score_pct":     f"{text_score * 100:.1f}%",
        "text_score":         round(text_score, 6),
        "preliminary_score":  round(prelim, 4),
        "source_name":        img.get("source_name", ""),
        "license":            img.get("license", ""),
        "image_width":        img.get("actual_width") or img.get("width", ""),
        "image_height":       img.get("actual_height") or img.get("height", ""),
        "scraped_at":         img.get("scraped_at", ""),
    }


def export_rankings_to_csv(
    json_dir: Path,
    output_csv: Path,
    min_score: float = 0.0,
) -> int:
    """
    Read all JSON records in *json_dir*, collect every ranked candidate image,
    and write a flat CSV to *output_csv*.

    Candidates are ordered by product ID (alphabetically) then by
    ``final_rank`` ascending (rank 1 = best match first).

    Args:
        json_dir:   Directory containing the scraper/classifier JSON records.
        output_csv: Destination CSV file path.
        min_score:  Skip rows whose ``final_score`` is below this threshold.

    Returns:
        Total number of rows written to the CSV.
    """
    rows: list = []

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("[csv] Skipping %s: %s", json_file.name, exc)
            continue

        product_id = data.get("product", {}).get("motion_product_id", json_file.stem)
        candidates = data.get("candidate_images", [])

        # Sort by final_rank if present, else by final_score descending
        candidates = sorted(
            candidates,
            key=lambda c: (c.get("final_rank", 9999), -c.get("final_score", 0.0)),
        )

        for img in candidates:
            final_score = float(img.get("final_score", 0.0))
            if final_score < min_score:
                continue
            rank = int(img.get("final_rank", 0))
            rows.append(_extract_csv_row(product_id, rank, img))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("[csv] Wrote %d rows → %s", len(rows), output_csv)
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# One-shot convenience wrapper  (text-rank → final-rank → CSV)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_ranking_pipeline(
    json_dir: Path,
    output_csv: Path,
    min_score: float = 0.0,
    dry_run: bool = False,
) -> int:
    """
    Complete ranking pipeline in three steps:

    1. ``rank_json_files()``   — compute text-similarity scores
    2. ``apply_final_ranking()`` — fuse AI + text into final_score
    3. ``export_rankings_to_csv()`` — write flat CSV

    Args:
        json_dir:   Directory with scraper JSON records.
        output_csv: Destination CSV path.
        min_score:  Filter threshold for CSV rows.
        dry_run:    If True, skip writing JSON/CSV and only print a summary.

    Returns:
        Number of CSV rows written (0 in dry-run mode).
    """
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        log.warning("[pipeline] No JSON files found in %s", json_dir)
        return 0

    log.info("[pipeline] Found %d JSON file(s) in %s", len(json_files), json_dir)

    if not dry_run:
        n_text   = rank_json_files(json_files)
        n_ranked = apply_final_ranking(json_files)
        log.info("[pipeline] Text-scored: %d  |  Final-ranked: %d", n_text, n_ranked)
        n_rows = export_rankings_to_csv(json_dir, output_csv, min_score=min_score)
        print(f"\n✓ Ranked {n_ranked} candidate image(s) across {len(json_files)} product(s).")
        print(f"✓ CSV written → {output_csv}  ({n_rows} row(s))")
        return n_rows
    else:
        # Dry run: just print a human-readable summary from existing JSON state
        _dry_run_summary(json_files)
        return 0


def _dry_run_summary(json_files: list) -> None:
    """Print a ranked summary table for every product without modifying files."""
    for json_file in json_files:
        try:
            data = json.loads(Path(json_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        prod       = data.get("product", {})
        product_id = prod.get("motion_product_id", json_file.stem)
        candidates = data.get("candidate_images", [])

        if not candidates:
            print(f"\n  {product_id}  — no candidates")
            continue

        print(f"\n  {'═'*70}")
        print(f"  Product : {product_id}  |  {prod.get('mfr_name','')} {prod.get('mfr_part_number','')}")
        print(f"  Desc    : {prod.get('description','')[:80]}")
        print(f"  {'─'*70}")
        print(f"  {'Rank':<5} {'Final':>7} {'AI Conf':>8} {'AI Class':<28} {'Text':>7}  Image")
        print(f"  {'─'*70}")

        candidates_sorted = sorted(
            candidates,
            key=lambda c: (c.get("final_rank", 9999), -c.get("final_score", 0.0)),
        )
        for img in candidates_sorted[:10]:   # cap preview at 10
            rank       = img.get("final_rank", "—")
            final      = f"{img.get('final_score', 0)*100:.1f}%"
            ai_conf    = f"{img.get('classifier_confidence', 0)*100:.1f}%"
            ai_cls     = _class_label(img.get("predicted_class", -1))[:27]
            text       = f"{img.get('ranker_score', img.get('text_similarity_score', 0))*100:.1f}%"
            fname      = Path(img.get("local_path", "") or "").name or "(no local file)"
            print(f"  {str(rank):<5} {final:>7} {ai_conf:>8} {ai_cls:<28} {text:>7}  {fname}")

    print(f"\n  {'═'*70}")
    print("  (dry-run: no files written)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone ImageSearcher  (catalog-based workflow outside the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class ImageSearcher:
    """
    Maintains an in-memory catalog of ProductVectors and ranks them against
    a query.

    Use this when working outside the scraper pipeline — e.g. scoring an
    internal pre-built image catalog instead of live scrape output.
    """

    def __init__(self) -> None:
        self.catalog: List[ProductVector] = []

    # ── Catalog management ────────────────────────────────────────────────

    def add_record(self, vec: ProductVector) -> None:
        self.catalog.append(vec)

    def add_from_list(self, vec: list) -> None:
        fields = [
            "id_number", "image_name", "item_number", "enterprise_number",
            "manufacture_name", "manufacture_part_number",
            "web_product_description", "motion_internal_desc",
            "pgc", "pgc_description",
        ]
        padded = (list(vec) + [""] * 10)[:10]
        kwargs = {f: str(v).strip() for f, v in zip(fields, padded)}
        self.catalog.append(ProductVector(**kwargs))

    def load_catalog(self, records: List[list]) -> None:
        for rec in records:
            self.add_from_list(rec)

    def load_from_vectors(self, vectors: List[ProductVector]) -> None:
        self.catalog.extend(vectors)

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: ProductVector,
        top_n: int = 5,
        min_score: float = 0.0,
    ) -> List[RankedResult]:
        """
        Rank the in-memory catalog against the query vector.

        Returns a list of RankedResult objects sorted by score descending.
        """
        results = [score_against_vector(query, rec) for rec in self.catalog]
        results = [r for r in results if r.score >= min_score]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def search_from_list(
        self,
        query_vec: list,
        top_n: int = 5,
        min_score: float = 0.0,
    ) -> List[RankedResult]:
        """Convenience wrapper — accepts a raw 10-element list as the query."""
        fields = [
            "id_number", "image_name", "item_number", "enterprise_number",
            "manufacture_name", "manufacture_part_number",
            "web_product_description", "motion_internal_desc",
            "pgc", "pgc_description",
        ]
        padded = (list(query_vec) + [""] * 10)[:10]
        kwargs = {f: str(v).strip() for f, v in zip(fields, padded)}
        return self.search(ProductVector(**kwargs), top_n, min_score)

    # ── Reporting ─────────────────────────────────────────────────────────

    @staticmethod
    def print_results(
        results: List[RankedResult],
        show_breakdown: bool = False,
    ) -> None:
        if not results:
            print("No matching images found.")
            return

        print(f"\n{'='*60}")
        print(f"  TOP {len(results)} IMAGE MATCH(ES)")
        print(f"{'='*60}")

        for rank, r in enumerate(results, 1):
            rec     = r.record
            bar_len = int(r.score * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            print(f"\n  Rank #{rank}  [{bar}]  {r.score * 100:.1f}%")
            print(f"  Image Name   : {rec.image_name or '(none)'}")
            print(f"  ID           : {rec.id_number}")
            print(f"  Item #       : {rec.item_number}")
            print(f"  Mfr          : {rec.manufacture_name}")
            print(f"  Mfr Part #   : {rec.manufacture_part_number}")
            print(f"  PGC          : {rec.pgc} — {rec.pgc_description}")
            print(f"  Web Desc     : {rec.web_product_description[:80]}")

            if show_breakdown:
                print("  --- Score Breakdown ---")
                for fname, detail in r.breakdown.items():
                    if detail["raw_score"] > 0:
                        print(
                            f"    {fname:<28} "
                            f"raw={detail['raw_score']:.3f}  "
                            f"w={detail['weight']:.1f}  "
                            f"contrib={detail['contribution']:.4f}"
                        )

        print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _default_json_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "output" / "json"


def _default_output_csv() -> Path:
    return Path(__file__).resolve().parent.parent / "output" / "rankings.csv"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Unified image ranker — fuses AI classifier + text-similarity signals, "
            "writes results back to JSON, and exports a ranked CSV for the web interface."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json-dir", "-j", type=Path, default=_default_json_dir(),
        help="Directory containing scraper JSON records (default: output/json)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=_default_output_csv(),
        help="Destination CSV file (default: output/rankings.csv)",
    )
    parser.add_argument(
        "--min-score", "-m", type=float, default=0.0,
        help="Exclude candidates below this final_score threshold (0-1, default: 0)",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print a ranked summary table without writing any files",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Run only the text-similarity pass (skip final fused ranking and CSV export)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run the built-in standalone demo with synthetic catalog data",
    )

    args = parser.parse_args()

    # ── Built-in demo ─────────────────────────────────────────────────────
    if args.demo:
        _run_demo()
        return

    # ── Validate json-dir ─────────────────────────────────────────────────
    if not args.json_dir.is_dir():
        parser.error(f"JSON directory not found: {args.json_dir}")

    json_files = sorted(args.json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.json_dir}")
        return

    print(f"\nFound {len(json_files)} JSON record(s) in {args.json_dir}")

    if args.text_only:
        n = rank_json_files(json_files)
        print(f"✓ Text-scored {n} candidate image(s).")
        return

    run_full_ranking_pipeline(
        json_dir   = args.json_dir,
        output_csv = args.output,
        min_score  = args.min_score,
        dry_run    = args.dry_run,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demo / self-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_demo() -> None:
    """Demonstrate both the catalog-searcher and pipeline rank_candidates paths."""

    catalog_data = [
        ["001", "bearing_6205_SKF.jpg",  "ITM-6205", "ENT-100", "SKF",      "6205-2RS1",
         "Deep groove ball bearing 25mm bore",   "SKF 6205 sealed bearing",        "1110", "Ball Bearings"],
        ["002", "bearing_6205_FAG.jpg",  "ITM-6206", "ENT-100", "FAG",      "6205-2RSR",
         "Deep groove ball bearing 25mm sealed", "FAG 6205 rubber sealed",         "1110", "Ball Bearings"],
        ["003", "seal_CR_12345.jpg",     "ITM-9901", "ENT-200", "CR Seals", "12345",
         "Oil seal shaft 40mm",                  "Rotary shaft seal 40mm",          "2200", "Shaft Seals"],
        ["004", "coupling_jaw_8mm.jpg",  "ITM-4401", "ENT-300", "Lovejoy",  "L-050",
         "Jaw coupling 8mm bore aluminum",       "Lovejoy jaw flex coupling 8mm",  "3300", "Couplings"],
        ["005", "motor_1HP_WEG.jpg",     "ITM-7701", "ENT-400", "WEG",      "00118ET3E143T",
         "1HP TEFC motor 1800RPM 143T frame",    "WEG 1hp 3ph 1800rpm motor",      "4400", "Electric Motors"],
        ["006", "bearing_6205_NTN.jpg",  "ITM-6207", "ENT-100", "NTN",      "6205LLU",
         "Ball bearing 6205 double sealed NTN",  "NTN 6205 bearing double lip seal","1110", "Ball Bearings"],
    ]

    searcher = ImageSearcher()
    searcher.load_catalog(catalog_data)

    query_vec = [
        "",            # id_number
        "",            # image_name
        "ITM-6205",    # item_number
        "ENT-100",     # enterprise_number
        "SKF",         # manufacture_name
        "6205-2RS1",   # manufacture_part_number
        "deep groove ball bearing 25mm",
        "",            # motion_internal_desc
        "1110",        # pgc
        "Ball Bearings",
    ]

    LABELS = [
        "ID Number", "Image Name", "Item Number", "Enterprise Number",
        "Manufacture Name", "Mfr Part Number", "Web Description",
        "Motion Internal Desc", "PGC", "PGC Description",
    ]
    print("\n  QUERY VECTOR:")
    print(f"  {'Field':<28} Value")
    print(f"  {'-'*28} {'-'*30}")
    for label, val in zip(LABELS, query_vec):
        if val:
            print(f"  {label:<28} {val}")

    results = searcher.search_from_list(query_vec, top_n=5, min_score=0.05)
    searcher.print_results(results, show_breakdown=True)

    # ── Pipeline-style rank_candidates demo ───────────────────────────────
    print("  --- rank_candidates() pipeline demo (text-similarity only) ---")
    pipeline_candidates = [
        {
            "index":        0,
            "image_url":    "https://example.com/skf_6205.jpg",
            "title":        "SKF 6205-2RS1 Deep Groove Ball Bearing",
            "source_name":  "Wikimedia Commons",
            "license":      "CC BY-SA",
            "tags":         ["bearing", "ball bearing", "SKF"],
            "keyword_used": "SKF 6205-2RS1",
            "width": 800, "height": 600,
            "downloaded": False,
            # Simulate AI classifier output
            "predicted_class":       0,
            "classifier_confidence": 0.82,
            "confidence_hints":      {"preliminary_score": 0.65},
        },
        {
            "index":        1,
            "image_url":    "https://example.com/generic_bearing.jpg",
            "title":        "Generic industrial bearing photo",
            "source_name":  "OpenVerse / Flickr",
            "license":      "CC0",
            "tags":         ["bearing", "industrial"],
            "keyword_used": "ball bearing industrial",
            "width": 400, "height": 300,
            "downloaded": False,
            "predicted_class":       0,
            "classifier_confidence": 0.45,
            "confidence_hints":      {"preliminary_score": 0.15},
        },
    ]

    fields = [
        "id_number", "image_name", "item_number", "enterprise_number",
        "manufacture_name", "manufacture_part_number",
        "web_product_description", "motion_internal_desc",
        "pgc", "pgc_description",
    ]
    query = ProductVector(**{f: str(v).strip() for f, v in zip(fields, (list(query_vec) + [""] * 10)[:10])})
    ranked = rank_candidates(query, pipeline_candidates)

    print(f"\n  {'Score':>8}  {'AI Conf':>8}  {'Final*':>8}  Title")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*40}")
    expected_classes = _pgc_to_expected_classes("1110")
    for r in ranked:
        final = compute_final_score(r, expected_classes)
        print(f"  {r['score_pct']:>8}  {r.get('classifier_confidence',0)*100:>7.1f}%  {final*100:>7.1f}%  {r['title']}")

    print("\n  * Final score = 30% AI confidence + 20% class match + 30% text + 20% preliminary")
    print(f"\n  Run  python image_search_ranker.py --json-dir output/json  to rank real data.\n")


if __name__ == "__main__":
    main()

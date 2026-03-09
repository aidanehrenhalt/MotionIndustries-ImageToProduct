"""
Image Search & Ranking System
==============================
Scores and ranks candidate images against a product's word vector.

Word Vector Fields (canonical 10-field spec):
  [0]  id_number               – Primary product key
  [1]  image_name              – Expected / known image filename
  [2]  item_number             – Motion internal item number
  [3]  enterprise_number       – Enterprise / brand parent ID
  [4]  manufacture_name        – Manufacturer / brand name
  [5]  manufacture_part_number – Manufacturer's own part number
  [6]  web_product_description – Full web-facing description
  [7]  motion_internal_desc    – Motion's internal description
  [8]  pgc                     – Product Group Code
  [9]  pgc_description         – Human-readable product category

Compatibility
-------------
This module works in two modes:

1. **Integrated** (default when used with the pipeline):
   - Imports `ProductVector` from `motion_image_pipeline`.
   - `rank_candidates(vector, candidates)` accepts the plain-dict
     candidate list produced by the pipeline's `fetch_candidates()` and
     returns the same list enriched with ranking keys:
       "ranker_score", "score_pct", "score_breakdown"

2. **Standalone** (no pipeline present):
   - Defines its own `ProductVector` dataclass (identical spec).
   - `ImageSearcher` class provides a catalog-based search workflow
     for scripts that maintain their own in-memory record list.

Standalone usage
----------------
  python image_search_ranker.py          # runs built-in demo

Programmatic usage (pipeline integration)
-----------------------------------------
  from image_search_ranker import rank_candidates, ImageSearcher
  ranked = rank_candidates(product_vector, candidate_dicts)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# ProductVector — import from pipeline if available, else define locally
# ---------------------------------------------------------------------------
try:
    from motion_image_pipeline import ProductVector
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

    @dataclass
    class ProductVector:
        """
        10-field canonical product vector.
        Identical to the definition in motion_image_pipeline.py.
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
            """Build a ProductVector from an ordered list of up to 10 values."""
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


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

# Field weights — tune to reflect business priority.
# Higher weight → stronger pull on the final composite score.
FIELD_WEIGHTS: dict = {
    "id_number":               10.0,   # exact primary key — decisive
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

# Per-field score tiers
EXACT_BONUS:    float = 1.00   # strings match exactly (case-insensitive)
CONTAINS_BONUS: float = 0.60   # one string wholly contains the other
TOKEN_BONUS:    float = 0.30   # scaled by token-overlap ratio (0–1)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Lower-case, split on non-alphanumeric chars, drop empty tokens."""
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def _field_score(query_val: str, catalog_val: str) -> float:
    """
    Compare two field strings and return a 0–1 relevance score.

    Scoring tiers (best-wins):
      1.0  exact match (case-insensitive)
      0.6  one value contains the other
      0–0.3  token overlap (Jaccard-style, normalised by geometric mean)
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


# ---------------------------------------------------------------------------
# RankedResult — used by the standalone ImageSearcher
# ---------------------------------------------------------------------------

@dataclass
class RankedResult:
    """A catalog ProductVector paired with its weighted relevance score."""
    record:    ProductVector
    score:     float
    breakdown: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RankedResult(image='{self.record.image_name}', "
            f"score={self.score:.4f}, {self.score*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Core scoring function — shared by both modes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline integration — rank a list of candidate dicts
# ---------------------------------------------------------------------------

def _candidate_dict_to_vector(candidate: dict) -> ProductVector:
    """
    Build a pseudo-ProductVector from a scraped image candidate dict
    (as returned by motion_image_pipeline.fetch_candidates).

    Populated fields:
      image_name              ← title (Wikimedia page name / OpenVerse title)
      manufacture_name        ← first token of keyword_used
      manufacture_part_number ← full keyword_used string
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


def rank_candidates(
    query: ProductVector,
    candidates: list,
    min_score: float = 0.0,
) -> list:
    """
    Score and rank a list of candidate image dicts against a query ProductVector.

    This is the primary integration point for motion_image_pipeline.py.
    Each candidate dict is enriched in-place with:
      "ranker_score"    : float 0–1
      "score_pct"       : str  e.g. "42.7%"
      "score_breakdown" : dict  per-field detail for explainability

    Args:
        query:      ProductVector representing the product we want an image for.
        candidates: list of dicts from fetch_candidates() (pipeline) or any
                    dict with at least "title", "keyword_used", "tags" keys.
        min_score:  Discard candidates below this threshold (default 0 = keep all).

    Returns:
        New list of candidate dicts sorted by ranker_score descending.
    """
    scored: list = []

    for candidate in candidates:
        cat_vec = _candidate_dict_to_vector(candidate)
        result  = score_against_vector(query, cat_vec)

        if result.score < min_score:
            continue

        enriched = dict(candidate)   # copy so we don't mutate the original
        enriched["ranker_score"]    = round(result.score, 6)
        enriched["score_pct"]       = f"{result.score * 100:.1f}%"
        enriched["score_breakdown"] = result.breakdown
        scored.append(enriched)

    scored.sort(key=lambda x: x["ranker_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# ImageSearcher — standalone catalog-based workflow
# ---------------------------------------------------------------------------

class ImageSearcher:
    """
    Maintains an in-memory catalog of ProductVectors and ranks them
    against a query.  Use this when working outside the pipeline
    (e.g. a pre-built internal image catalog).
    """

    def __init__(self) -> None:
        self.catalog: List[ProductVector] = []

    # --- Catalog management -----------------------------------------------

    def add_record(self, vec: ProductVector) -> None:
        """Add a single ProductVector to the catalog."""
        self.catalog.append(vec)

    def add_from_list(self, vec: list) -> None:
        """Add a ProductVector built from a raw 10-element list."""
        padded = (list(vec) + [""] * 10)[:10]
        fields = [
            "id_number", "image_name", "item_number", "enterprise_number",
            "manufacture_name", "manufacture_part_number",
            "web_product_description", "motion_internal_desc",
            "pgc", "pgc_description",
        ]
        kwargs = {f: str(v).strip() for f, v in zip(fields, padded)}
        self.catalog.append(ProductVector(**kwargs))

    def load_catalog(self, records: List[list]) -> None:
        """Bulk-load a list of raw 10-element lists."""
        for rec in records:
            self.add_from_list(rec)

    def load_from_vectors(self, vectors: List[ProductVector]) -> None:
        """Bulk-load from an existing list of ProductVectors (e.g. from CatalogSearch)."""
        self.catalog.extend(vectors)

    # --- Search -----------------------------------------------------------

    def search(
        self,
        query: ProductVector,
        top_n: int = 5,
        min_score: float = 0.0,
    ) -> List[RankedResult]:
        """
        Rank the in-memory catalog against the query vector.

        Args:
            query:     ProductVector to match against.
            top_n:     Maximum results to return.
            min_score: Minimum score threshold (0–1).

        Returns:
            list[RankedResult] sorted descending by score.
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
        padded = (list(query_vec) + [""] * 10)[:10]
        fields = [
            "id_number", "image_name", "item_number", "enterprise_number",
            "manufacture_name", "manufacture_part_number",
            "web_product_description", "motion_internal_desc",
            "pgc", "pgc_description",
        ]
        kwargs = {f: str(v).strip() for f, v in zip(fields, padded)}
        return self.search(ProductVector(**kwargs), top_n, min_score)

    # --- Reporting --------------------------------------------------------

    @staticmethod
    def print_results(
        results: List[RankedResult],
        show_breakdown: bool = False,
    ) -> None:
        """Pretty-print ranked results to stdout."""
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
            print(f"\n  Rank #{rank}  [{bar}]  {r.score*100:.1f}%")
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


# ---------------------------------------------------------------------------
# Built-in demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Build a small sample catalog
    catalog_data = [
        ["001", "bearing_6205_SKF.jpg",  "ITM-6205", "ENT-100", "SKF",      "6205-2RS1",    "Deep groove ball bearing 25mm bore",         "SKF 6205 sealed bearing",          "B1100", "Ball Bearings"],
        ["002", "bearing_6205_FAG.jpg",  "ITM-6206", "ENT-100", "FAG",      "6205-2RSR",    "Deep groove ball bearing 25mm sealed",        "FAG 6205 rubber sealed",           "B1100", "Ball Bearings"],
        ["003", "seal_CR_12345.jpg",     "ITM-9901", "ENT-200", "CR Seals", "12345",        "Oil seal shaft 40mm",                        "Rotary shaft seal 40mm",           "S2200", "Shaft Seals"],
        ["004", "coupling_jaw_8mm.jpg",  "ITM-4401", "ENT-300", "Lovejoy",  "L-050",        "Jaw coupling 8mm bore aluminum",             "Lovejoy jaw flex coupling 8mm",    "C3300", "Couplings"],
        ["005", "motor_1HP_WEG.jpg",     "ITM-7701", "ENT-400", "WEG",      "00118ET3E143T","1HP TEFC motor 1800RPM 143T frame",          "WEG 1hp 3ph 1800rpm motor",        "M4400", "Electric Motors"],
        ["006", "bearing_6205_NTN.jpg",  "ITM-6207", "ENT-100", "NTN",      "6205LLU",      "Ball bearing 6205 double sealed NTN",        "NTN 6205 bearing double lip seal", "B1100", "Ball Bearings"],
        ["007", "gear_reducer_5_1.jpg",  "ITM-5501", "ENT-500", "Dodge",    "175873",       "Shaft mount gear reducer 5:1 ratio",         "Dodge Quantis 5:1 shaft mount",    "G5500", "Gear Reducers"],
        ["008", "vbelt_AX38.jpg",        "ITM-3301", "ENT-600", "Gates",    "AX38",         "V-Belt cogged A-section 38 inch",            "Gates AX38 Hi-Power II belt",      "V6600", "V-Belts"],
    ]

    searcher = ImageSearcher()
    searcher.load_catalog(catalog_data)

    # 2. Query vector (partial — blanks are fine)
    query_vec = [
        "",            # id_number
        "",            # image_name  (this is what we're looking for)
        "ITM-6205",    # item_number
        "ENT-100",     # enterprise_number
        "SKF",         # manufacture_name
        "6205-2RS1",   # manufacture_part_number
        "deep groove ball bearing 25mm",
        "",            # motion_internal_desc
        "B1100",       # pgc
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

    # 3. Search + display
    results = searcher.search_from_list(query_vec, top_n=5, min_score=0.05)
    searcher.print_results(results, show_breakdown=True)

    # 4. Pipeline-style rank_candidates demo
    print("\n  --- rank_candidates() pipeline demo ---")
    pipeline_candidates = [
        {
            "image_url":    "https://example.com/skf_6205.jpg",
            "title":        "SKF 6205-2RS1 Deep Groove Ball Bearing",
            "source_name":  "Wikimedia Commons",
            "license":      "CC BY-SA",
            "tags":         ["bearing", "ball bearing", "SKF"],
            "keyword_used": "SKF 6205-2RS1",
            "width": 800, "height": 600,
            "downloaded": False, "candidate_index": 0,
        },
        {
            "image_url":    "https://example.com/generic_bearing.jpg",
            "title":        "Generic industrial bearing",
            "source_name":  "OpenVerse / Flickr",
            "license":      "CC0",
            "tags":         ["bearing", "industrial"],
            "keyword_used": "ball bearing industrial",
            "width": 400, "height": 300,
            "downloaded": False, "candidate_index": 1,
        },
    ]

    _fields = [
        "id_number", "image_name", "item_number", "enterprise_number",
        "manufacture_name", "manufacture_part_number",
        "web_product_description", "motion_internal_desc",
        "pgc", "pgc_description",
    ]
    query = ProductVector(**{f: str(v).strip() for f, v in zip(_fields, (list(query_vec) + [""]*10)[:10])})
    ranked = rank_candidates(query, pipeline_candidates)
    for r in ranked:
        print(f"  {r['score_pct']:>7}  {r['title']}")
        for fname, detail in r["score_breakdown"].items():
            if detail["raw_score"] > 0:
                print(
                    f"           {fname:<28} "
                    f"raw={detail['raw_score']:.3f}  "
                    f"contrib={detail['contribution']:.4f}"
                )
    print()

    # 5. Fuzzy / partial query demo
    print("  --- Fuzzy partial query ---")
    fuzzy = ProductVector(
        manufacture_name        = "WEG",
        web_product_description = "electric motor 1HP TEFC",
        pgc_description         = "motors",
    )
    searcher.print_results(searcher.search(fuzzy, top_n=3))

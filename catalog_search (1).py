#!/usr/bin/env python3
"""
Catalog Database Search Tool
=============================
Loads an Excel or CSV product catalog and exposes search utilities that
return rows as ProductVector objects — the same structure used by
motion_image_pipeline.py and image_search_ranker.py.

Standalone usage
----------------
  python catalog_search.py                        # interactive mode
  python catalog_search.py --catalog products.xlsx --query "SKF bearing"
  python catalog_search.py --catalog products.csv  --missing-only

Programmatic usage (from the pipeline or other scripts)
-------------------------------------------------------
  from catalog_search import CatalogSearch
  cs = CatalogSearch("catalog.xlsx")
  results = cs.search("SKF")                       # returns list[ProductVector]
  missing = cs.missing_image_products()            # only rows with no image set
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared ProductVector import — use the pipeline's definition when available,
# otherwise define a local compatible copy so this file works standalone.
# ---------------------------------------------------------------------------
try:
    from motion_image_pipeline import ProductVector, COLUMN_ALIASES
    _IMPORTED_FROM_PIPELINE = True
except ImportError:
    _IMPORTED_FROM_PIPELINE = False
    from dataclasses import dataclass

    @dataclass
    class ProductVector:
        """
        10-field canonical product vector.
        Field order matches the word-vector spec used by the ranker:
          [0] id_number  [1] image_name       [2] item_number
          [3] enterprise_number               [4] manufacture_name
          [5] manufacture_part_number         [6] web_product_description
          [7] motion_internal_desc            [8] pgc
          [9] pgc_description
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

        def search_keywords(self) -> list:
            """Generate prioritised keyword strings for web queries."""
            kws = []
            if self.manufacture_name and self.manufacture_part_number:
                kws.append(f"{self.manufacture_name} {self.manufacture_part_number}")
            if self.manufacture_part_number:
                kws.append(self.manufacture_part_number)
            if self.manufacture_name and self.web_product_description:
                words = self.web_product_description.split()[:5]
                kws.append(f"{self.manufacture_name} {' '.join(words)}")
            if self.pgc_description and self.manufacture_name:
                kws.append(f"{self.manufacture_name} {self.pgc_description}")
            if self.pgc_description and not kws:
                kws.append(self.pgc_description)
            seen, unique = set(), []
            for k in kws:
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    unique.append(k)
            return unique

    # Canonical column-name aliases (mirrors pipeline's COLUMN_ALIASES exactly)
    COLUMN_ALIASES: dict = {
        "id":                      ["id", "<id>", "motion_product_id", "product_id", "sku"],
        "image_name":              ["primaryimagefilename", "image_name", "image", "photo"],
        "item_number":             ["item number", "item_number", "itemnumber"],
        "enterprise_number":       ["enterprise name", "enterprise_name", "enterprise_number"],
        "manufacture_name":        ["manufacturer name", "manufacturer_name", "mfr_name", "manufacture_name"],
        "manufacture_part_number": ["manufacturer part number", "manufacturer_part_number",
                                    "mfr_part_number", "manufacture_part_number", "mfr part number"],
        "web_product_description": ["web product description", "web_product_description", "web_desc"],
        "motion_internal_desc":    ["motion internal description", "motion_internal_desc",
                                    "motion internal desc", "internal description"],
        "pgc":                     ["pgc"],
        "pgc_description":         ["pgc description", "pgc_description"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Normalise a column name for alias matching."""
    return name.strip().lower().replace("_", " ")


def _build_col_map(raw_columns: list) -> dict:
    """
    Map canonical field names → actual column names found in the file.
    Returns {canonical: actual_col_name} for every field that matched.
    """
    norm_to_raw = {_norm(c): c for c in raw_columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in norm_to_raw:
                mapping[canonical] = norm_to_raw[alias]
                break
    return mapping


def _row_to_vector(row: dict, col_map: dict) -> ProductVector:
    """Convert one raw row dict into a ProductVector."""
    def get(field: str) -> str:
        col = col_map.get(field, "")
        return str(row.get(col, "") or "").strip()

    return ProductVector(
        id_number               = get("id"),
        image_name              = get("image_name"),
        item_number             = get("item_number"),
        enterprise_number       = get("enterprise_number"),
        manufacture_name        = get("manufacture_name"),
        manufacture_part_number = get("manufacture_part_number"),
        web_product_description = get("web_product_description"),
        motion_internal_desc    = get("motion_internal_desc"),
        pgc                     = get("pgc"),
        pgc_description         = get("pgc_description"),
    )


# ---------------------------------------------------------------------------
# CatalogSearch — main public class
# ---------------------------------------------------------------------------

class CatalogSearch:
    """
    Load a product catalog (Excel or CSV) and search it, returning
    ProductVector objects compatible with the motion_image_pipeline.
    """

    def __init__(self, catalog_path: str):
        """
        Args:
            catalog_path: Path to an .xlsx, .xls, or .csv file.
        """
        self.catalog_path = catalog_path
        self._vectors: list[ProductVector] = []
        self._col_map: dict = {}
        self._raw_columns: list = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Parse the catalog file and populate internal vector list."""
        path = Path(self.catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")

        suffix = path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("Install pandas + openpyxl: pip install pandas openpyxl")
            df = pd.read_excel(self.catalog_path, dtype=str).fillna("")
            self._raw_columns = list(df.columns)
            rows = df.to_dict(orient="records")

        elif suffix == ".csv":
            with open(self.catalog_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self._raw_columns = list(reader.fieldnames or [])
                rows = [dict(r) for r in reader]

        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Use .xlsx or .csv")

        self._col_map = _build_col_map(self._raw_columns)
        self._vectors = [_row_to_vector(r, self._col_map) for r in rows]

        print(f"✓ Loaded {len(self._vectors)} products from '{path.name}'")
        mapped = list(self._col_map.keys())
        missing = [k for k in COLUMN_ALIASES if k not in self._col_map]
        print(f"✓ Mapped fields  : {', '.join(mapped)}")
        if missing:
            print(f"⚠ Unmapped fields: {', '.join(missing)}")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def all_products(self) -> list:
        """All ProductVectors loaded from the catalog."""
        return list(self._vectors)

    @property
    def columns(self) -> list:
        """Raw column names from the source file."""
        return list(self._raw_columns)

    def missing_image_products(self) -> list:
        """
        Return only rows where image_name is blank —
        i.e., products that still need an image sourced.
        """
        return [v for v in self._vectors if not v.image_name]

    # ------------------------------------------------------------------
    # Search methods — all return list[ProductVector]
    # ------------------------------------------------------------------

    def search(self, query: str, case_sensitive: bool = False) -> list:
        """
        Full-text search across ALL 10 vector fields.

        Args:
            query:          Term to search for.
            case_sensitive: Default False (case-insensitive).

        Returns:
            list[ProductVector] — matching products.
        """
        q = query if case_sensitive else query.lower()
        results = []
        for vec in self._vectors:
            haystack = " ".join(vec.to_dict().values())
            if not case_sensitive:
                haystack = haystack.lower()
            if q in haystack:
                results.append(vec)
        return results

    def search_field(
        self,
        field: str,
        query: str,
        case_sensitive: bool = False,
        exact_match: bool = False,
    ) -> list:
        """
        Search a specific canonical field.

        Args:
            field:          One of the 10 canonical field names
                            (e.g. 'manufacture_name', 'pgc_description').
            query:          Search term.
            case_sensitive: Default False.
            exact_match:    If True, field value must equal query exactly.

        Returns:
            list[ProductVector]
        """
        if field not in COLUMN_ALIASES:
            raise ValueError(
                f"Unknown field '{field}'. "
                f"Valid fields: {', '.join(COLUMN_ALIASES.keys())}"
            )
        q = query if case_sensitive else query.lower()
        results = []
        for vec in self._vectors:
            val = vec.to_dict().get(field, "")
            v = val if case_sensitive else val.lower()
            if exact_match:
                if v == q:
                    results.append(vec)
            else:
                if q in v:
                    results.append(vec)
        return results

    def search_multi(self, criteria: dict, case_sensitive: bool = False) -> list:
        """
        AND-logic multi-field search.

        Args:
            criteria:  {canonical_field: query_value, ...}
                       e.g. {"manufacture_name": "SKF", "pgc_description": "bearing"}
            case_sensitive: Default False.

        Returns:
            list[ProductVector] — rows matching ALL criteria.
        """
        results = self._vectors
        for field, query in criteria.items():
            results = self.search_field(
                field, query, case_sensitive=case_sensitive
            )
            # narrow down iteratively — rebuild from current result set
            if not results:
                break
        return results

    def find_by_id(self, product_id: str) -> ProductVector | None:
        """Return the ProductVector with matching id_number, or None."""
        for vec in self._vectors:
            if vec.id_number == product_id:
                return vec
        return None

    # ------------------------------------------------------------------
    # Image-directory matching (local filesystem)
    # ------------------------------------------------------------------

    def match_local_images(
        self,
        results: list,
        image_directory: str,
        extensions: list = None,
    ) -> dict:
        """
        Find local image files whose filenames contain the product's id_number.

        Args:
            results:         list[ProductVector] to match.
            image_directory: Directory to scan for images.
            extensions:      e.g. ['.jpg', '.png']. Defaults to common formats.

        Returns:
            {id_number: [filepath, ...], ...}  — only products with matches.
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]

        img_dir = Path(image_directory)
        if not img_dir.exists():
            print(f"⚠ Image directory not found: {image_directory}")
            return {}

        # Collect all image files once
        all_images = []
        for ext in extensions:
            all_images.extend(img_dir.glob(f"**/*{ext}"))
            all_images.extend(img_dir.glob(f"**/*{ext.upper()}"))

        image_map: dict = {}
        for vec in results:
            identifier = vec.id_number or vec.image_name
            if not identifier:
                continue
            matched = [
                str(p) for p in all_images
                if identifier.lower() in p.stem.lower()
            ]
            if matched:
                image_map[vec.id_number] = matched

        return image_map

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display(self, results: list, max_results: int = 10) -> None:
        """Pretty-print a list of ProductVectors."""
        if not results:
            print("\n❌ No matches found.")
            return

        print(f"\n✓ {len(results)} result(s) — showing up to {max_results}")
        show = results[:max_results]
        for i, vec in enumerate(show, 1):
            d = vec.to_dict()
            print(f"\n  [{i}] {vec.id_number or '(no id)'}")
            print(f"  {'─'*60}")
            for k, v in d.items():
                if v:
                    print(f"    {k:<28}: {v}")
        if len(results) > max_results:
            print(f"\n  … and {len(results) - max_results} more.")


# ---------------------------------------------------------------------------
# Interactive / CLI entry point
# ---------------------------------------------------------------------------

def _run_interactive(cs: CatalogSearch) -> None:
    """Simple REPL for manual exploration."""
    while True:
        print(f"\n{'='*60}")
        print("  CATALOG SEARCH — OPTIONS")
        print("  1. Full-text search (all fields)")
        print("  2. Field search")
        print("  3. Multi-field search (AND)")
        print("  4. Show missing-image products")
        print("  5. Show ALL products")
        print("  6. Exit")
        choice = input("\n  Select (1-6): ").strip()

        if choice == "1":
            q = input("  Search term: ").strip()
            if q:
                cs.display(cs.search(q))

        elif choice == "2":
            print(f"\n  Available fields: {', '.join(COLUMN_ALIASES.keys())}")
            fld = input("  Field name: ").strip()
            q   = input("  Value: ").strip()
            try:
                cs.display(cs.search_field(fld, q))
            except ValueError as e:
                print(f"  ⚠ {e}")

        elif choice == "3":
            print(f"\n  Enter field=value pairs one per line. Blank line to finish.")
            print(f"  Available fields: {', '.join(COLUMN_ALIASES.keys())}")
            criteria: dict = {}
            while True:
                entry = input("  criterion: ").strip()
                if not entry:
                    break
                if "=" in entry:
                    f, v = entry.split("=", 1)
                    criteria[f.strip()] = v.strip()
            if criteria:
                try:
                    cs.display(cs.search_multi(criteria))
                except ValueError as e:
                    print(f"  ⚠ {e}")

        elif choice == "4":
            missing = cs.missing_image_products()
            print(f"\n  Products missing images: {len(missing)}")
            cs.display(missing)

        elif choice == "5":
            cs.display(cs.all_products, max_results=20)

        elif choice == "6":
            print("  Goodbye!")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Motion Industries Catalog Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--catalog", "-c", help="Path to .xlsx or .csv catalog file")
    parser.add_argument("--query",   "-q", help="Search term (searches all fields)")
    parser.add_argument(
        "--field", "-f",
        help="Restrict search to a specific canonical field "
             f"({', '.join(COLUMN_ALIASES.keys())})",
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Only show products with no image filename set",
    )
    parser.add_argument(
        "--max", type=int, default=10,
        help="Max results to display (default: 10)",
    )
    args = parser.parse_args()

    # Prompt for catalog path if not supplied
    catalog = args.catalog
    if not catalog:
        catalog = input("Enter path to catalog file (.xlsx / .csv): ").strip()
    if not catalog or not os.path.exists(catalog):
        print(f"Error: file not found: {catalog}")
        sys.exit(1)

    cs = CatalogSearch(catalog)

    if args.query:
        # Non-interactive: run the requested query and exit
        if args.field:
            results = cs.search_field(args.field, args.query)
        elif args.missing_only:
            results = [
                v for v in cs.search(args.query)
                if not v.image_name
            ]
        else:
            results = cs.search(args.query)
        cs.display(results, max_results=args.max)
    elif args.missing_only:
        cs.display(cs.missing_image_products(), max_results=args.max)
    else:
        # Drop into interactive mode
        _run_interactive(cs)


if __name__ == "__main__":
    main()

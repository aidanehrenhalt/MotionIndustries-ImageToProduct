#!/usr/bin/env python3
"""
Text-Based Product Image Search
=================================
Builds an optimised web image-search URL from product attributes and,
optionally, uses Playwright to fetch and download the first result image.

The query is built from a ProductVector (the canonical 10-field struct
shared across the Motion Industries pipeline) so that search terms
match exactly what the ranker uses to score results.

Standalone usage
----------------
  python text_based_search.py --item-name "SKF bearing" --print-only
  python text_based_search.py -mn "6205-2RS1" -n "Ball Bearing" --engine bing
  python text_based_search.py --vector-json '{"manufacture_name":"SKF","manufacture_part_number":"6205-2RS1"}'

Programmatic usage (pipeline / other scripts)
---------------------------------------------
  from text_based_search import build_search_query, build_bing_image_url, vector_to_query
  from text_based_search import fetch_first_image

  query = vector_to_query(product_vector)
  url   = build_bing_image_url(query)
  image_url, local_path = fetch_first_image(query, output_dir="./images")

Requirements (for fetch_first_image only)
-----------------------------------------
  pip install playwright
  playwright install chromium
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# sync_playwright imported at module level so tests can patch 'text_based_search.sync_playwright'
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None  # type: ignore  — guarded inside fetch_first_image

# ---------------------------------------------------------------------------
# ProductVector — import from pipeline if available, else define locally
# ---------------------------------------------------------------------------
try:
    from motion_image_pipeline import ProductVector
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    from dataclasses import dataclass

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
# Query builders
# ---------------------------------------------------------------------------

# Priority order used when building search terms from a ProductVector.
# Earlier fields are more specific and appear first in the query.
_VECTOR_FIELD_PRIORITY = [
    "manufacture_part_number",   # most specific — part number
    "manufacture_name",          # brand name
    "item_number",               # internal item number
    "enterprise_number",         # enterprise ID
    "pgc_description",           # product category
    "web_product_description",   # broad description (truncated)
    "motion_internal_desc",      # internal description (truncated)
]
_DESCRIPTION_WORD_LIMIT = 8   # cap on description tokens to keep queries tight


def vector_to_query(vector: ProductVector) -> str:
    """
    Build an optimised search query string from a ProductVector.

    Fields are added in priority order (most specific first).
    Description fields are truncated to the first 8 words.

    Args:
        vector: ProductVector with any combination of fields populated.

    Returns:
        A clean query string (no double spaces).
    """
    parts: list = []
    d = vector.to_dict()

    for field in _VECTOR_FIELD_PRIORITY:
        val = d.get(field, "").strip()
        if not val:
            continue
        if field in ("web_product_description", "motion_internal_desc"):
            words = val.split()[:_DESCRIPTION_WORD_LIMIT]
            val   = " ".join(words)
        if val and val not in parts:   # deduplicate identical tokens
            parts.append(val)

    return " ".join(parts).strip()


def build_search_query(args) -> str:
    """
    Build a prioritised search query from CLI Namespace args.

    Priority order (highest → lowest):
      manufacture_number → manufacture_vin → enterprise_product_number
      → enterprise_vin → item_name → item_size → product_description

    Args:
        args: argparse.Namespace with optional fields:
              item_name, item_size, manufacture_number, manufacture_vin,
              enterprise_product_number, enterprise_vin, product_description

    Returns:
        Query string (may be empty if all args are None/blank).
    """
    parts: list = []

    for attr in (
        "manufacture_number",
        "manufacture_vin",
        "enterprise_product_number",
        "enterprise_vin",
        "item_name",
        "item_size",
    ):
        val = getattr(args, attr, None)
        if val:
            parts.append(str(val).strip())

    desc = getattr(args, "product_description", None)
    if desc:
        words = str(desc).split()[:_DESCRIPTION_WORD_LIMIT]
        parts.append(" ".join(words))

    return " ".join(parts)


def build_bing_image_url(query: str) -> str:
    """Return a Bing Images search URL for the given query string."""
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.bing.com/images/search?q={encoded}"


def build_search_urls(query: str) -> dict:
    """
    Return image-search URLs for multiple engines.

    Returns:
        {"Google Images": url, "Bing Images": url, "DuckDuckGo Images": url}
    """
    encoded = urllib.parse.quote_plus(query)
    return {
        "Google Images":      f"https://www.google.com/search?tbm=isch&q={encoded}",
        "Bing Images":        f"https://www.bing.com/images/search?q={encoded}",
        "DuckDuckGo Images":  f"https://duckduckgo.com/?iax=images&ia=images&q={encoded}",
    }


# ---------------------------------------------------------------------------
# Image fetcher (Playwright — optional dependency)
# ---------------------------------------------------------------------------

def fetch_first_image(
    query: str,
    output_dir: str = ".",
    headless: bool = True,
) -> tuple:
    """
    Use Playwright (headless Chromium) to open the Bing Images search
    results page and download the first image found.

    Fallback chain for the image URL:
      1. `src` of the first `.mimg` thumbnail element (full-URL only)
      2. `m` attribute of the nearest anchor tag → parsed for `murl`

    Args:
        query:      Search string.
        output_dir: Directory to save the downloaded image.
        headless:   Run browser without a visible window (default True).

    Returns:
        (image_url: str, local_path: str)

    Raises:
        SystemExit if no image URL can be found.

    Note:
        Playwright must be installed:
            pip install playwright && playwright install chromium
    """
    if sync_playwright is None:
        print(
            "Playwright is required for fetch_first_image.\n"
            "Install with: pip install playwright && playwright install chromium"
        )
        sys.exit(1)

    search_url = build_bing_image_url(query)
    image_url  = None
    local_path = None

    # Uses module-level sync_playwright so tests can patch 'text_based_search.sync_playwright'
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.goto(search_url, wait_until="networkidle", timeout=20_000)

        # --- Try 1: first thumbnail with an absolute src ---
        img_el = page.query_selector("img.mimg")
        if img_el:
            src = img_el.get_attribute("src") or ""
            if src.startswith("http"):
                image_url = src

        # --- Try 2: anchor's `m` JSON attribute → murl ---
        if not image_url:
            anchor = page.query_selector("a.iusc")
            if anchor:
                m_raw = anchor.get_attribute("m")
                if m_raw:
                    try:
                        m_data = json.loads(m_raw)
                        image_url = m_data.get("murl", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

        browser.close()

    if not image_url:
        print(f"No image found for query: '{query}'")
        sys.exit(1)

    # --- Download the image ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive a safe filename from the query + URL extension
    safe_query = re.sub(r"[^\w\-]", "_", query)[:50]
    ext_match  = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", image_url)
    ext        = ext_match.group(1).lower() if ext_match else ""
    if ext not in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
        ext = "jpg"

    filename   = f"{safe_query}.{ext}"
    local_path = str(out_dir / filename)

    with urllib.request.urlopen(image_url) as response:
        image_data = response.read()
    with open(local_path, "wb") as f:
        f.write(image_data)

    return image_url, local_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a product image search query and optionally fetch the first result.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__,
    )

    # ---- Individual product-attribute flags (mirrors original CLI) ----
    parser.add_argument("--item-name",                "-n",  metavar="NAME",
                        help="Product / item name")
    parser.add_argument("--item-size",                "-s",  metavar="SIZE",
                        help="Item size (e.g. '10mm', 'Large')")
    parser.add_argument("--manufacture-number",       "-mn", metavar="MFR_NUM",
                        help="Manufacturer part / model number  [→ manufacture_part_number]")
    parser.add_argument("--manufacture-vin",          "-mv", metavar="MFR_VIN",
                        help="Manufacturer VIN / serial")
    parser.add_argument("--enterprise-product-number","-ep", metavar="EPN",
                        help="Enterprise product number  [→ enterprise_number]")
    parser.add_argument("--enterprise-vin",           "-ev", metavar="EVIN",
                        help="Enterprise VIN")
    parser.add_argument("--product-description",      "-d",  metavar="DESC",
                        help="Product description (first 8 words used)  [→ web_product_description]")

    # ---- ProductVector JSON input (pipeline integration shortcut) ----
    parser.add_argument(
        "--vector-json", "-vj", metavar="JSON",
        help=(
            "Pass a ProductVector as a JSON object instead of individual flags.\n"
            "Example: '{\"manufacture_name\":\"SKF\",\"manufacture_part_number\":\"6205-2RS1\"}'"
        ),
    )

    # ---- Behaviour flags ----
    parser.add_argument(
        "--engine", "-e",
        choices=["google", "bing", "duckduckgo", "all"],
        default="bing",
        help="Image search engine to use (default: bing)",
    )
    parser.add_argument(
        "--print-only", "-p", action="store_true",
        help="Print the Bing URL only; do not fetch or open browser",
    )
    parser.add_argument(
        "--output-dir", "-o", metavar="DIR", default=".",
        help="Directory to save downloaded images (default: current directory)",
    )
    parser.add_argument(
        "--no-headless", action="store_true",
        help="Show the browser window when fetching images",
    )

    args = parser.parse_args()

    # --- Build the query ---
    if args.vector_json:
        try:
            fields = json.loads(args.vector_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing --vector-json: {e}")
            sys.exit(1)
        vec   = ProductVector(**{k: str(v) for k, v in fields.items()})
        query = vector_to_query(vec)
    else:
        query = build_search_query(args)

    if not query:
        parser.print_help()
        sys.exit(1)

    # Always print the query and Bing URL
    print(f"\nSearch Query: {query}\n")
    print(f"  Bing Images:\n  {build_bing_image_url(query)}\n")

    if args.print_only:
        print("(--print-only mode: URLs not opened in browser)")
        return

    # Normal run: fetch and download the first image result
    image_url, local_path = fetch_first_image(
        query,
        output_dir=args.output_dir,
        headless=not args.no_headless,
    )
    print(f"  Image URL  : {image_url}")
    print(f"  Saved to   : {local_path}")


if __name__ == "__main__":
    main()

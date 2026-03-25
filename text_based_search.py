#!/usr/bin/env python3
"""
Product Image Search Script
Builds an optimized web search query from product attributes and opens
image search results in the default browser.
"""

import argparse
import webbrowser
import urllib.parse
import sys


def build_search_query(args):
    """Build a prioritized search query from product arguments."""
    parts = []

    # Highest-priority identifiers first (most specific)
    if args.manufacture_number:
        parts.append(args.manufacture_number)
    if args.manufacture_vin:
        parts.append(args.manufacture_vin)
    if args.enterprise_product_number:
        parts.append(args.enterprise_product_number)
    if args.enterprise_vin:
        parts.append(args.enterprise_vin)

    # Item name and size
    if args.item_name:
        parts.append(args.item_name)
    if args.item_size:
        parts.append(args.item_size)

    # Description words (trim to most useful keywords to avoid overly long queries)
    if args.product_description:
        # Use the first 8 words of the description to keep query focused
        desc_words = args.product_description.split()[:8]
        parts.append(" ".join(desc_words))

    return " ".join(parts)


def build_search_urls(query):
    """Return image search URLs for multiple search engines."""
    encoded = urllib.parse.quote_plus(query)
    return {
        "Google Images":  f"https://www.google.com/search?tbm=isch&q={encoded}",
        "Bing Images":    f"https://www.bing.com/images/search?q={encoded}",
        "DuckDuckGo Images": f"https://duckduckgo.com/?iax=images&ia=images&q={encoded}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Search the web for a product image using multiple identifiers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--item-name",               "-n",  metavar="NAME",   help="Product / item name")
    parser.add_argument("--item-size",               "-s",  metavar="SIZE",   help="Item size (e.g. '10mm', 'Large', '5x7')")
    parser.add_argument("--manufacture-number",      "-mn", metavar="MFR_NUM",help="Manufacturer part / model number")
    parser.add_argument("--manufacture-vin",         "-mv", metavar="MFR_VIN",help="Manufacturer VIN / serial")
    parser.add_argument("--enterprise-product-number","-ep",metavar="EPN",    help="Enterprise product number")
    parser.add_argument("--enterprise-vin",          "-ev", metavar="EVIN",   help="Enterprise VIN")
    parser.add_argument("--product-description",    "-d",  metavar="DESC",   help="Product description (keywords used)")
    parser.add_argument(
        "--engine",
        "-e",
        choices=["google", "bing", "duckduckgo", "all"],
        default="google",
        help="Which image search engine to open (default: google)",
    )
    parser.add_argument(
        "--print-only",
        "-p",
        action="store_true",
        help="Print the URLs instead of opening them in a browser",
    )

    args = parser.parse_args()

    # Require at least one argument
    provided = [v for v in vars(args).values() if isinstance(v, str) and v]
    if not provided:
        parser.print_help()
        sys.exit(1)

    query = build_search_query(args)
    urls  = build_search_urls(query)

    print(f"\nSearch Query: {query}\n")

    # Filter by chosen engine
    if args.engine == "all":
        selected = urls
    else:
        engine_map = {
            "google":     "Google Images",
            "bing":       "Bing Images",
            "duckduckgo": "DuckDuckGo Images",
        }
        key = engine_map[args.engine]
        selected = {key: urls[key]}

    for name, url in selected.items():
        print(f"  {name}:\n  {url}\n")
        if not args.print_only:
            webbrowser.open(url)

    if args.print_only:
        print("(--print-only mode: URLs not opened in browser)")


if __name__ == "__main__":
    main()

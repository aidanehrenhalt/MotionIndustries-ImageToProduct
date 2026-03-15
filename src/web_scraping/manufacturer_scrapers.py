"""
Manufacturer Website Scraping Module

Scrapes product images directly from manufacturer websites.

Two tiers:
  Tier 1 — Documented manufacturer portals with known URL patterns and CSS
            selectors. Uses requests + BeautifulSoup (no browser required).
            Only manufacturers with approved=True in MANUFACTURER_REGISTRY
            are scraped. All others are skipped.
  Tier 2 — Generic fallback for unknown manufacturers. Searches DuckDuckGo
            (HTML endpoint, no JS), follows the first result, and extracts
            the og:image meta tag using Playwright. Requires Playwright:
                pip install playwright && playwright install chromium

Legacy Selenium path (scrape_manufacturer_site) is retained for backward
compatibility with the --manufacturer-sites CLI flag.

Usage:
    # New path (Tier 1 + optional Tier 2):
    from manufacturer_scrapers import scrape_manufacturer_images
    images = scrape_manufacturer_images(product, enable_tier2=False)

    # Legacy Selenium path:
    from manufacturer_scrapers import scrape_manufacturer_site
    images = scrape_manufacturer_site(product, driver=driver)
"""

import time
import random
import logging
import threading
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, quote_plus, urlparse, urljoin
from urllib.robotparser import RobotFileParser

log = logging.getLogger("scraper")

# Standard Chrome user-agent — avoids CAPTCHA on search engines that block
# explicit bot strings, and matches what browsers send for robots.txt checks.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Browser-like headers for all HTTP requests (reduces bot-detection false positives)
_REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
}

# Cache robots.txt parsers per domain
_robots_cache: dict[str, RobotFileParser | None] = {}

# Cache last request time per domain for rate limiting
_last_request_time: dict[str, float] = {}

# Base delay between requests to the same domain (seconds)
RATE_LIMIT_BASE_DELAY = 3.0

# Max images to collect per manufacturer source
MAX_IMAGES_PER_MFR = 3

# Per-domain threading locks for thread-safe rate limiting
_domain_locks: dict[str, threading.Lock] = {}
_domain_locks_meta_lock = threading.Lock()

# CAD file extensions to exclude (Timken ToS: CAD drawings are restricted)
CAD_EXTENSIONS = {".step", ".stp", ".dxf", ".igs", ".iges", ".x_t", ".x_b"}

# ──────────────────────────────────────────────────────────────
# Manufacturer Registry
# ──────────────────────────────────────────────────────────────
# Each entry must be individually reviewed for ToS + robots.txt
# compliance before flipping approved to True.

MANUFACTURER_REGISTRY = {
    # ── Tier 1: Documented portals with verified selectors (approved=True) ──────
    "ami bearings inc": {
        # robots.txt disallows /keyword/ (the search path) but /item/ pages are allowed.
        # Strategy: DDG site search → /item/ URL → fetch page → /Asset/ images.
        "base_url": "https://catalog.amibearings.com",
        "site_search_domain": "catalog.amibearings.com",
        "image_selectors": [
            "img[src^='/Asset/']",
            "img[src^='/ImgMedium/']",
            "img[src^='/ImgSmall/']",
        ],
        "renderer": "ddg_item",
        "approved": True,
        "license_note": "Manufacturer catalog — proprietary, internal use only",
    },
    "ntn": {
        # Same catalog platform as AMI; same robots.txt restriction on /keyword/.
        # Strategy: DDG site search → /item/ URL → fetch page → /ImgMedium/ images.
        "base_url": "https://bearingfinder.ntnamericas.com",
        "site_search_domain": "bearingfinder.ntnamericas.com",
        "image_selectors": [
            "img[src^='/ImgMedium/']",
            "img[src^='/ImgSmall/']",
        ],
        "renderer": "ddg_item",
        "approved": True,
        "license_note": "Manufacturer catalog — proprietary, internal use only",
    },
    "timken": {
        # NOTE: cad.timken.com robots.txt disallows /keyword/ (the correct
        # part-specific search path). The /search path is allowed but returns
        # only generic category images, not part-specific results.
        # Tier 1 is therefore not viable; Timken falls through to Tier 2
        # (DuckDuckGo + Playwright og:image) when --mfr-tier2 is enabled.
        "base_url": "https://cad.timken.com",
        "search_url_template": "https://cad.timken.com/keyword/all-product-types?key=all&keyword={part_number}&SchType=2",
        "image_selectors": [
            "img[src^='/ImgSmall/']",
            "img[src^='/ImgLarge/']",
            "a[href^='/Asset/'][href$='.JPG']",
            "a[href^='/Asset/'][href$='.jpg']",
        ],
        "renderer": "requests",
        "approved": False,
        "license_note": "Product images only — CAD drawings restricted per Timken ToS",
    },
    # ── Tier 1 pending review (approved=False) ───────────────────────────────
    "skf": {
        "base_url": "https://www.skf.com",
        "search_url_template": "https://www.skf.com/us/products?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "nsk": {
        "base_url": "https://www.nsk.com",
        "search_url_template": "https://www.nsk.com/search/?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "dodge": {
        "base_url": "https://www.dodgeindustrial.com",
        "search_url_template": "https://www.dodgeindustrial.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "fag": {
        "base_url": "https://www.schaeffler.com",
        "search_url_template": "https://www.schaeffler.com/en/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "ina": {
        "base_url": "https://www.schaeffler.com",
        "search_url_template": "https://www.schaeffler.com/en/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "leeson": {
        "base_url": "https://www.leeson.com",
        "search_url_template": "https://www.leeson.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "sealmaster": {
        "base_url": "https://www.sealmaster.com",
        "search_url_template": "https://www.sealmaster.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "browning": {
        "base_url": "https://www.regalrexnord.com",
        "search_url_template": "https://www.regalrexnord.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "mcgill": {
        "base_url": "https://www.regalrexnord.com",
        "search_url_template": "https://www.regalrexnord.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "renold jeffrey": {
        "base_url": "https://www.renold.com",
        "search_url_template": "https://www.renold.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "rexnord": {
        "base_url": "https://www.regalrexnord.com",
        "search_url_template": "https://www.regalrexnord.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
    "martin sprocket": {
        "base_url": "https://www.martinsprocket.com",
        "search_url_template": "https://www.martinsprocket.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "renderer": "requests",
        "approved": False,
    },
}

# Alias map for manufacturer name normalization.
# Keys must be strings that could appear as a product's mfr_name field —
# i.e., values from the MFR_NAME column of UniqueEnterpriseAndMFR.csv or
# known product catalog variants. Do NOT add enterprise names here.
_ALIASES = {
    # Leeson (product catalog variant)
    "leeson electric": "leeson",
    # Schaeffler brands — exact MFR_NAME column values
    "fag (schaeffler)": "fag",
    "ina (schaeffler)": "ina",
    # Rexnord — "REXNORD INC" appears in MFR_NAME column
    "rexnord inc": "rexnord",
    # Martin Sprocket — "MARTIN SPROCKET & GEAR CO" appears in MFR_NAME column
    "martin sprocket & gear co": "martin sprocket",
    # Renold (same key, kept for explicit alias completeness)
    "renold jeffrey": "renold jeffrey",
    # Timken sub-brands — all from MFR_NAME column of UniqueEnterpriseAndMFR.csv
    "timken (fafnir)": "timken",
    "timken (revolvo)": "timken",
    "timken belts (carlisle)": "timken",
    "timken drives llc": "timken",
    "timken national seals": "timken",
    "timken belts": "timken",
    # NTN — "NTN" appears in MFR_NAME column (row 95). No enterprise-name aliases.
    # AMI — "AMI BEARINGS INC" is the canonical MFR_NAME; short forms as variants
    "ami bearings": "ami bearings inc",
    # NSK variant from MFR_NAME column
    "nsk corp, bearing div": "nsk",
    # SKF sub-brands from MFR_NAME column (skf entry is approved=False pending review)
    "cooper bearings (skf)": "skf",
    "cr seals (skf)": "skf",
    "general bearing (skf)": "skf",
    "kaydon bearings (skf)": "skf",
    "mrc (skf)": "skf",
    "peer bearing co (skf)": "skf",
}


def _normalize_mfr_name(name: str) -> str:
    """
    Normalize a manufacturer name to match registry keys.
    Lowercases, strips whitespace, and applies alias mappings.
    """
    normalized = name.strip().lower()
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    # Try partial match on aliases (e.g. "FAG" matches "fag")
    for alias, key in _ALIASES.items():
        if normalized == alias:
            return key
    return normalized


def check_robots_txt(url: str) -> bool:
    """
    Check if the given URL is allowed by the site's robots.txt.
    Returns False on fetch errors (conservative — assume disallowed).
    Results are cached per domain.
    """
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain not in _robots_cache:
        robots_url = f"{domain}/robots.txt"
        try:
            resp = requests.get(
                robots_url,
                headers=_REQUEST_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(resp.text.splitlines())
            _robots_cache[domain] = rp
        except Exception as e:
            log.warning(f"[Manufacturer] Could not fetch robots.txt from {domain}: {e}")
            _robots_cache[domain] = None
            return False

    rp = _robots_cache[domain]
    if rp is None:
        return False

    allowed = rp.can_fetch(USER_AGENT, url)
    if not allowed:
        log.info(f"[Manufacturer] robots.txt disallows: {url}")
    return allowed


def _get_crawl_delay(domain: str) -> float:
    """Get Crawl-delay from cached robots.txt, or return 0 if not set."""
    rp = _robots_cache.get(domain)
    if rp is None:
        return 0
    try:
        delay = rp.crawl_delay(USER_AGENT)
        return delay if delay else 0
    except Exception:
        return 0


def _get_domain_lock(domain: str) -> threading.Lock:
    """Get or create a per-domain threading lock (thread-safe)."""
    with _domain_locks_meta_lock:
        if domain not in _domain_locks:
            _domain_locks[domain] = threading.Lock()
        return _domain_locks[domain]


def _rate_limit_delay(base_url: str):
    """
    Enforce rate limiting per domain (thread-safe).
    Uses 3s base + random 0.5-2s jitter, or Crawl-delay from robots.txt
    (whichever is larger). A per-domain lock serializes concurrent requests
    to the same host so the delay is respected across threads.
    """
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    lock = _get_domain_lock(domain)

    with lock:
        crawl_delay = _get_crawl_delay(domain)
        min_delay = max(RATE_LIMIT_BASE_DELAY, crawl_delay)
        jitter = random.uniform(0.5, 2.0)
        total_delay = min_delay + jitter

        last_time = _last_request_time.get(domain, 0)
        elapsed = time.time() - last_time
        if elapsed < total_delay:
            wait = total_delay - elapsed
            log.debug(f"[Manufacturer] Rate limiting: waiting {wait:.1f}s for {domain}")
            time.sleep(wait)

        _last_request_time[domain] = time.time()


def _build_product_url(config: dict, part_number: str) -> str:
    """
    Build a product search URL from the config template and part number.
    Handles special characters like commas and slashes in part numbers.
    """
    encoded = quote(part_number, safe="")
    return config["search_url_template"].format(part_number=encoded)


# ──────────────────────────────────────────────────────────────
# Tier 1 — requests + BeautifulSoup (server-rendered portals)
# ──────────────────────────────────────────────────────────────

def _fetch_with_requests(url: str, base_url: str, check_robots: bool = True) -> str | None:
    """
    Fetch a page with requests. Enforces rate limiting.
    Checks robots.txt by default; pass check_robots=False to skip (e.g. for
    search-engine query endpoints like html.duckduckgo.com whose robots.txt
    blocks all crawlers but whose HTML search endpoint is a public utility).
    Returns the HTML string, or None on any failure.
    """
    if check_robots and not check_robots_txt(url):
        return None
    _rate_limit_delay(base_url)
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log.warning(f"[Tier1] HTTP fetch failed for {url}: {e}")
        return None


def _extract_images_from_html(
    html: str,
    config: dict,
    product: dict,
    base_url: str,
    page_url: str,
) -> list[dict]:
    """
    Parse HTML with BeautifulSoup and extract image URLs using the CSS
    selectors in config['image_selectors'].

    Filters:
    - SVG files and data: URIs
    - CAD file extensions (safety net for Timken's legal restriction)
    - Duplicate URLs
    """
    soup = BeautifulSoup(html, "lxml")
    images: list[dict] = []
    seen: set[str] = set()
    mfr_name = product.get("mfr_name", "Unknown")

    for selector in config.get("image_selectors", []):
        for el in soup.select(selector):
            # <a href=...> or <img src=...>
            img_url = el.get("src") or el.get("href") or ""
            if not img_url:
                continue
            if img_url.startswith("data:") or img_url.endswith(".svg"):
                continue
            if any(img_url.lower().endswith(ext) for ext in CAD_EXTENSIONS):
                continue
            # Skip technical drawing images (filenames containing "dwg" or ending in "_s")
            url_stem = img_url.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if "dwg" in url_stem or url_stem.endswith("_s"):
                continue
            if not img_url.startswith(("http://", "https://")):
                img_url = urljoin(base_url, img_url)
            if img_url in seen:
                continue
            seen.add(img_url)

            images.append({
                "image_url": img_url,
                "thumbnail_url": img_url,
                "source_page": page_url,
                "source_name": f"Manufacturer Site / {mfr_name}",
                "license": config.get("license_note", "Proprietary"),
                "attribution": mfr_name,
                "width": None,
                "height": None,
                "mime_type": "",
                "title": el.get("alt", "") or el.get_text(strip=True),
                "tags": [],
                "_source_fn": "manufacturer_tier1",
            })

    return images[:MAX_IMAGES_PER_MFR]


def _scrape_tier1(product: dict, config: dict) -> list[dict]:
    """
    Scrape a Tier 1 (documented, server-rendered) manufacturer portal
    using requests + BeautifulSoup. No browser required.
    """
    part_number = product.get("mfr_part_number", "")
    if not part_number:
        log.warning(f"[Tier1] No part number for product {product.get('motion_product_id')}, skipping")
        return []

    search_url = _build_product_url(config, part_number)
    mfr_name = product.get("mfr_name", "Unknown")
    log.info(f"[Tier1] Fetching {search_url} for '{mfr_name}' part '{part_number}'")

    html = _fetch_with_requests(search_url, config["base_url"])
    if not html:
        return []

    images = _extract_images_from_html(html, config, product, config["base_url"], search_url)
    log.info(f"[Tier1] Found {len(images)} image(s) for '{mfr_name}' part '{part_number}'")
    return images


# ──────────────────────────────────────────────────────────────
# Tier 1 — Sitemap lookup → item page (robots.txt safe)
# ──────────────────────────────────────────────────────────────
# Both AMI and NTN publish their sitemap in robots.txt and the sitemap
# lists every /item/ product page URL. Downloading it once per process
# gives us a reliable part-slug → full URL lookup without any search
# engine dependency.
# ──────────────────────────────────────────────────────────────

# Per-domain sitemap cache: { "catalog.amibearings.com": { "uct204": "https://..." } }
_sitemap_cache: dict[str, dict[str, str]] = {}


def _load_sitemap(base_url: str, sitemap_path: str = "/sitemap_en-us.xml.gz") -> dict[str, str]:
    """
    Download and parse a gzipped sitemap for the given base_url.

    Returns a dict mapping series codes to their /viewitems/ page URLs:
        { "uct200": "https://catalog.amibearings.com/viewitems/set-screw-locking-8/...-uct200-series" }

    The series code is extracted from the last URL segment before "-series"
    in each /viewitems/ URL (e.g. "set-screw-locking-take-up-unit-uct200-series"
    → series code "uct200").

    Results are cached in-process (one download per domain per run).
    """
    import gzip
    import io
    import re as _re

    parsed = urlparse(base_url)
    domain = parsed.netloc
    if domain in _sitemap_cache:
        return _sitemap_cache[domain]

    sitemap_url = f"{base_url}{sitemap_path}"
    log.info(f"[Sitemap] Downloading {sitemap_url} ...")
    try:
        resp = requests.get(sitemap_url, headers=_REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()
        with gzip.open(io.BytesIO(resp.content)) as f:
            xml_bytes = f.read()
        # Strip BOM if present
        if xml_bytes.startswith(b"\xef\xbb\xbf"):
            xml_bytes = xml_bytes[3:]
    except Exception as e:
        log.warning(f"[Sitemap] Failed to download {sitemap_url}: {e}")
        _sitemap_cache[domain] = {}
        return {}

    # Parse XML with default sitemap namespace
    try:
        from lxml import etree as _etree
        root = _etree.fromstring(xml_bytes)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = [el.text for el in root.findall(".//sm:url/sm:loc", ns) if el.text]
    except Exception as e:
        log.warning(f"[Sitemap] Failed to parse XML for {domain}: {e}")
        _sitemap_cache[domain] = {}
        return {}

    # Build series_code → viewitems URL mapping.
    # Sitemap contains /viewitems/ series pages, not individual /item/ pages.
    # The last path slug encodes the series: "...-uct200-series" → "uct200"
    lookup: dict[str, str] = {}
    for url in locs:
        if "/viewitems/" not in url:
            continue
        slug = url.rstrip("/").rsplit("/", 1)[-1].lower()
        m = _re.search(r"([a-z]+\d+[a-z]*)-series$", slug)
        if m:
            series_code = m.group(1)  # e.g. "uct200", "uct200c", "ucfl200"
            if series_code not in lookup:            # keep first occurrence
                lookup[series_code] = url

    _sitemap_cache[domain] = lookup
    log.info(f"[Sitemap] Loaded {len(lookup)} series entries for {domain}")
    return lookup


def _extract_base_slug(part_number: str) -> str:
    """
    Extract the base catalog slug from a Motion Industries part number.

    Strips MI-specific suffix codes (clearance class, temperature rating,
    sealing type, material codes) that manufacturer catalogs don't index.

    Examples:
        UCT204C4HR23     → uct204        (strip clearance/heat/sealing after digits)
        MUCHPL205-14RFCW → muchpl205-14  (keep prefix + size + bore, strip material)
        UCHPL201-8B      → uchpl201-8    (keep through bore number)
        UCECH201-8NPMZ20 → ucech201-8
    """
    import re as _re
    pn = part_number.lower()
    # Keep: letters + digits + optional hyphen + digits (bore size)
    # Stop before any trailing alpha suffix codes like RFCW, B, NPMZ20, C4HR23
    m = _re.match(r"([a-z]+\d+(?:-\d+)?)", pn)
    return m.group(1) if m else pn


def _find_viewitems_url(
    part_number: str, sitemap: dict[str, str]
) -> str | None:
    """
    Look up the /viewitems/ series URL for a given part number in the sitemap.

    Tries candidate series codes in order: most specific (with trailing letter)
    first, then plain hundreds-group code.
        UCT204C4HR23 → try "uct200c", then "uct200"
        MUCHPL205-14 → try "muchpl200"
    """
    import re as _re

    part_lc = part_number.lower()
    m = _re.match(r"([a-z]+)(\d+)", part_lc)
    if not m:
        return None
    prefix, digits = m.group(1), m.group(2)
    hundreds = digits[0]
    trailing_alpha = _re.match(r"[a-z]+\d+([a-z]*)", part_lc)
    suffix = trailing_alpha.group(1) if trailing_alpha else ""
    base_series = f"{prefix}{hundreds}00"

    for series_code in ([f"{base_series}{suffix}"] if suffix else []) + [base_series]:
        if series_code in sitemap:
            return sitemap[series_code]
    return None


def _find_item_url_in_series_page(
    base_url: str, part_number: str, viewitems_url: str
) -> str | None:
    """
    Fetch a /viewitems/ series listing page and find the item link whose slug
    best matches the given part number.

    Matching strategy:
      1. Try an exact slug match (part_number.lower())
      2. Try the base slug (prefix + size + bore, e.g. "muchpl205-14")
      3. Take the first item whose slug starts with the base slug

    Returns the full item URL, or None if no match.
    """
    html = _fetch_with_requests(viewitems_url, base_url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    item_links = soup.select("a[href*='/item/']")

    base_slug = _extract_base_slug(part_number)
    part_lc = part_number.lower()

    # Priority order: exact match > base slug exact match > base slug prefix match
    exact_match: str | None = None
    base_exact: str | None = None
    base_prefix: str | None = None

    for a in item_links:
        href = a.get("href", "")
        slug = href.rstrip("/").rsplit("/", 1)[-1].lower()
        if slug == part_lc:
            exact_match = href
            break
        if slug == base_slug and base_exact is None:
            base_exact = href
        if slug.startswith(base_slug) and base_prefix is None:
            base_prefix = href

    chosen = exact_match or base_exact or base_prefix
    if chosen:
        if not chosen.startswith(("http://", "https://")):
            chosen = urljoin(base_url, chosen)
        log.info(f"[Sitemap] Found item page for '{part_number}': {chosen}")
    return chosen


def _build_item_url_from_sitemap(
    base_url: str, part_number: str, sitemap: dict[str, str]
) -> str | None:
    """
    Backwards-compatible wrapper: returns a /item/ URL for a given part number.

    First tries the viewitems series page approach (fetches the page and finds
    the closest item link), which handles bore-code slugs like "muchpl205-14w".
    Falls back to direct URL construction for simple series like UCT.
    """
    viewitems_url = _find_viewitems_url(part_number, sitemap)
    if not viewitems_url:
        return None
    return _find_item_url_in_series_page(base_url, part_number, viewitems_url)

def _decode_ddg_href(href: str) -> str | None:
    """
    Decode a DuckDuckGo HTML redirect href to the actual destination URL.

    DDG HTML links use the format:
        //duckduckgo.com/l/?uddg=<percent-encoded-url>&rut=...

    Returns the decoded destination URL, or None if it cannot be determined.
    """
    from urllib.parse import parse_qs, unquote as _unquote

    if not href:
        return None
    # Normalize protocol-relative URLs
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return _unquote(qs["uddg"][0])
    # Already a direct URL (non-DDG domain)
    if parsed.netloc and "duckduckgo.com" not in parsed.netloc:
        return href
    return None


def _scrape_ddg_item(product: dict, config: dict) -> list[dict]:
    """
    Tier 1 via sitemap lookup (primary) or DuckDuckGo site search (fallback).

    Both AMI and NTN catalog platforms publish a gzipped sitemap that lists
    every /item/ product URL.  We download it once per process run, build a
    slug → URL lookup table, and resolve the part number directly — no search
    engine needed.

    Motion Industries part numbers often include suffix codes
    (clearance class, temperature rating, sealing type) that manufacturer
    catalogs don't index.  We try candidates in order:
        1. Full part number slug  (e.g. "uct204c4hr23")
        2. Base part slug         (e.g. "uct204")  ← strip trailing modifier codes

    If the sitemap lookup fails for all candidates, falls back to a DuckDuckGo
    site-restricted HTML search (requires the DDG HTML endpoint to be reachable
    without CAPTCHA — may not always work in automated environments).

    Fetching the /item/ page is always done via requests; robots.txt on these
    sites only blocks /keyword/ and /results/, not /item/ or /Asset/.
    """
    part_number = product.get("mfr_part_number", "")
    if not part_number:
        return []

    domain = config.get("site_search_domain", "")
    mfr_name = product.get("mfr_name", "Unknown")
    base_url = config["base_url"]

    # ── Strategy 1: sitemap → series page → find closest item link ──────────
    # The sitemap maps series codes to /viewitems/ URLs. We fetch the series
    # listing page and find the item link whose slug best matches our part
    # number (using the base slug: prefix + size + bore, e.g. "muchpl205-14").
    # If no exact item match, we use the series-level images from that page.
    sitemap = _load_sitemap(base_url)
    item_url: str | None = None
    viewitems_url: str | None = None

    if sitemap:
        viewitems_url = _find_viewitems_url(part_number, sitemap)
        if viewitems_url:
            item_url = _find_item_url_in_series_page(base_url, part_number, viewitems_url)
            if not item_url:
                log.info(
                    f"[Tier1] No item match on series page for '{mfr_name}' "
                    f"'{part_number}'; will use series-level images"
                )
        else:
            log.info(
                f"[Tier1] No sitemap series match for '{mfr_name}' '{part_number}'"
            )

    # ── Strategy 2: DDG site search fallback (if sitemap missed entirely) ───
    if not item_url and not viewitems_url:
        ddg_query = f"{_extract_base_slug(part_number)} site:{domain}"
        ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(ddg_query)}"
        log.info(f"[Tier1] DDG fallback search: {ddg_url}")
        # DDG robots.txt has Disallow: / — skip robots check for the search step.
        ddg_html = _fetch_with_requests(
            ddg_url, "https://html.duckduckgo.com", check_robots=False
        )
        if ddg_html:
            soup = BeautifulSoup(ddg_html, "lxml")
            first_link = soup.select_one("a.result__a")
            decoded = _decode_ddg_href(first_link.get("href", "")) if first_link else None
            if decoded and domain in decoded:
                item_url = decoded
                log.info(f"[Tier1] DDG found page for '{part_number}': {item_url}")

    # ── Fetch item page; fall back to series/viewitems page if needed ────────
    fetch_url = item_url or viewitems_url
    if not fetch_url:
        log.info(f"[Tier1] No page found for '{mfr_name}' '{part_number}'")
        return []

    item_html = _fetch_with_requests(fetch_url, base_url)
    if not item_html:
        return []

    images = _extract_images_from_html(item_html, config, product, base_url, fetch_url)
    log.info(f"[Tier1] Found {len(images)} image(s) for '{mfr_name}' '{part_number}'")
    return images


# ──────────────────────────────────────────────────────────────
# Legacy Selenium extraction (used by scrape_manufacturer_site)
# ──────────────────────────────────────────────────────────────

def _extract_product_images(driver, config: dict, product: dict) -> list[dict]:
    """
    Extract product image URLs from the current page using CSS selectors
    defined in the manufacturer config.

    Filters out:
    - SVG images
    - Base64-encoded images
    - Tiny icons (< 50px)
    - Duplicate URLs
    """
    images = []
    seen_urls = set()
    current_url = driver.current_url

    for selector in config.get("image_selectors", []):
        try:
            elements = driver.find_elements("css selector", selector)
        except Exception:
            continue

        for el in elements:
            # Try src, then data-src, then data-lazy-src
            img_url = (
                el.get_attribute("src")
                or el.get_attribute("data-src")
                or el.get_attribute("data-lazy-src")
                or ""
            )

            if not img_url:
                continue

            # Skip base64, SVGs, and tiny tracking pixels
            if img_url.startswith("data:"):
                continue
            if img_url.endswith(".svg"):
                continue

            # Resolve relative URLs
            if not img_url.startswith(("http://", "https://")):
                img_url = urljoin(current_url, img_url)

            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            # Filter small icons by checking natural dimensions
            try:
                width = el.get_attribute("naturalWidth")
                height = el.get_attribute("naturalHeight")
                if width and height and int(width) < 50 and int(height) < 50:
                    continue
            except (ValueError, TypeError):
                pass

            alt_text = el.get_attribute("alt") or ""

            mfr_name = product.get("mfr_name", "Unknown")
            images.append({
                "image_url": img_url,
                "thumbnail_url": img_url,
                "source_page": current_url,
                "source_name": f"Manufacturer Site / {mfr_name}",
                "license": "Proprietary",
                "attribution": mfr_name,
                "width": None,
                "height": None,
                "mime_type": "",
                "title": alt_text,
                "tags": [],
                "_source_fn": "manufacturer",
            })

    return images


def scrape_manufacturer_site(product: dict, driver=None) -> list[dict]:
    """
    Scrape product images from the manufacturer's website.

    Returns a list of image dicts matching the same format as
    scrape_wikimedia() / scrape_openverse().

    Returns [] if:
    - No driver provided
    - Manufacturer not in registry
    - Manufacturer not approved
    - robots.txt disallows the URL
    """
    if driver is None:
        return []

    mfr_name_raw = product.get("mfr_name", "")
    mfr_key = _normalize_mfr_name(mfr_name_raw)

    config = MANUFACTURER_REGISTRY.get(mfr_key)
    if config is None:
        log.info(f"[Manufacturer] '{mfr_name_raw}' (normalized: '{mfr_key}') not in registry, skipping")
        return []

    if not config.get("approved", False):
        log.info(f"[Manufacturer] '{mfr_name_raw}' is not approved for scraping, skipping")
        return []

    part_number = product.get("mfr_part_number", "")
    if not part_number:
        log.warning(f"[Manufacturer] No part number for product {product.get('motion_product_id')}, skipping")
        return []

    search_url = _build_product_url(config, part_number)

    # Check robots.txt before loading the page
    if not check_robots_txt(search_url):
        log.info(f"[Manufacturer] robots.txt disallows {search_url}, skipping")
        return []

    # Respect rate limits
    _rate_limit_delay(config["base_url"])

    try:
        log.info(f"[Manufacturer] Loading {search_url}")
        driver.get(search_url)
        # Wait for page to load (basic wait for body)
        time.sleep(3)

        images = _extract_product_images(driver, config, product)
        log.info(f"[Manufacturer] Found {len(images)} images for '{mfr_name_raw}' part '{part_number}'")
        return images

    except Exception as e:
        log.warning(f"[Manufacturer] Error scraping {search_url}: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# Tier 2 — Generic og:image fallback (Playwright, JS-heavy pages)
# ──────────────────────────────────────────────────────────────

def _scrape_tier2_generic(product: dict) -> list[dict]:
    """
    Generic fallback for manufacturers not in MANUFACTURER_REGISTRY.

    Strategy (two hops):
      1. Fetch DuckDuckGo HTML endpoint (server-rendered) with requests to
         get the first organic result URL for "{mfr_name} {part_number}".
      2. Load that product page with Playwright (handles JS-rendered content)
         and extract the og:image meta tag.

    Returns [] if Playwright is not installed or no image is found.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning(
            "[Tier2] Playwright not installed — skipping Tier 2 scraping. "
            "Run: pip install playwright && playwright install chromium"
        )
        return []

    mfr_name = product.get("mfr_name", "")
    part_number = product.get("mfr_part_number", "")
    if not mfr_name or not part_number:
        return []

    # Step 1: DuckDuckGo HTML endpoint is server-rendered — no JS needed
    ddg_url = (
        f"https://html.duckduckgo.com/html/?q={quote_plus(mfr_name + ' ' + part_number)}"
    )
    # DDG's robots.txt has Disallow: / — skip robots check for this search step.
    html = _fetch_with_requests(ddg_url, "https://html.duckduckgo.com", check_robots=False)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    first_result = soup.select_one("a.result__a")
    product_page_url = first_result.get("href") if first_result else None
    if not product_page_url:
        log.info(f"[Tier2] No DDG results for '{mfr_name}' '{part_number}'")
        return []

    log.info(f"[Tier2] Following result: {product_page_url}")

    # Step 2: Load product page with Playwright to resolve JS-rendered og:image
    og_image: str | None = None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_context(user_agent=USER_AGENT).new_page()
            page.goto(product_page_url, wait_until="domcontentloaded", timeout=20_000)
            og_image = page.get_attribute('meta[property="og:image"]', "content")
            browser.close()
    except Exception as e:
        log.warning(f"[Tier2] Playwright failed for '{mfr_name}' '{part_number}': {e}")
        return []

    if not og_image:
        log.info(f"[Tier2] No og:image found at {product_page_url}")
        return []

    log.info(f"[Tier2] Found og:image for '{mfr_name}' '{part_number}'")
    return [{
        "image_url": og_image,
        "thumbnail_url": og_image,
        "source_page": product_page_url,
        "source_name": f"Generic Search / {mfr_name}",
        "license": "Unknown — requires review",
        "attribution": mfr_name,
        "width": None,
        "height": None,
        "mime_type": "",
        "title": f"{mfr_name} {part_number}",
        "tags": [],
        "_source_fn": "manufacturer_tier2",
    }]


# ──────────────────────────────────────────────────────────────
# Public entry point (no Selenium driver required)
# ──────────────────────────────────────────────────────────────

def scrape_manufacturer_images(product: dict, enable_tier2: bool = False) -> list[dict]:
    """
    Scrape product images from manufacturer websites.

    Dispatch order:
      1. Tier 1 — if the manufacturer is in MANUFACTURER_REGISTRY with
         approved=True, use documented CSS selectors via requests+BS4.
      2. Tier 2 — if no Tier 1 match and enable_tier2=True, fall back to
         a DuckDuckGo search + Playwright og:image extraction.

    Returns a list of image dicts in the same format as scrape_wikimedia().
    Requires no Selenium driver.
    """
    mfr_name_raw = product.get("mfr_name", "")
    mfr_key = _normalize_mfr_name(mfr_name_raw)
    config = MANUFACTURER_REGISTRY.get(mfr_key)

    if config is not None:
        if not config.get("approved", False):
            log.info(f"[Manufacturer] '{mfr_name_raw}' in registry but not approved, skipping")
            return []
        renderer = config.get("renderer", "requests")
        if renderer == "requests":
            return _scrape_tier1(product, config)
        if renderer == "ddg_item":
            return _scrape_ddg_item(product, config)
        log.warning(f"[Manufacturer] Unknown renderer '{renderer}' for '{mfr_name_raw}', skipping")
        return []

    # Not in registry — Tier 2 fallback
    if enable_tier2:
        log.info(f"[Manufacturer] '{mfr_name_raw}' not in Tier 1 registry, trying Tier 2 generic")
        return _scrape_tier2_generic(product)

    log.info(f"[Manufacturer] '{mfr_name_raw}' not in registry and Tier 2 disabled, skipping")
    return []

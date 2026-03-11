"""
Manufacturer Website Scraping Module

Scrapes product images directly from manufacturer websites using Selenium.
All manufacturers start with approved=False and must be individually reviewed
(robots.txt + ToS) before enabling.

Usage:
    from manufacturer_scrapers import scrape_manufacturer_site
    images = scrape_manufacturer_site(product, driver=driver)
"""

import time
import random
import logging
from urllib.parse import quote, urlparse, urljoin
from urllib.robotparser import RobotFileParser

log = logging.getLogger("scraper")

# User-Agent string for robots.txt checking (must match what Selenium sends)
USER_AGENT = "Mozilla/5.0 (compatible; ImageToProduct-Scraper/1.0)"

# Cache robots.txt parsers per domain
_robots_cache: dict[str, RobotFileParser | None] = {}

# Cache last request time per domain for rate limiting
_last_request_time: dict[str, float] = {}

# Base delay between requests to the same domain (seconds)
RATE_LIMIT_BASE_DELAY = 3.0

# ──────────────────────────────────────────────────────────────
# Manufacturer Registry
# ──────────────────────────────────────────────────────────────
# Each entry must be individually reviewed for ToS + robots.txt
# compliance before flipping approved to True.

MANUFACTURER_REGISTRY = {
    "skf": {
        "base_url": "https://www.skf.com",
        "search_url_template": "https://www.skf.com/us/products?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "timken": {
        "base_url": "https://www.timken.com",
        "search_url_template": "https://www.timken.com/search/?searchTerms={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "nsk": {
        "base_url": "https://www.nsk.com",
        "search_url_template": "https://www.nsk.com/search/?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "dodge": {
        "base_url": "https://www.dodgeindustrial.com",
        "search_url_template": "https://www.dodgeindustrial.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "fag": {
        "base_url": "https://www.schaeffler.com",
        "search_url_template": "https://www.schaeffler.com/en/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "ina": {
        "base_url": "https://www.schaeffler.com",
        "search_url_template": "https://www.schaeffler.com/en/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "leeson": {
        "base_url": "https://www.leeson.com",
        "search_url_template": "https://www.leeson.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "sealmaster": {
        "base_url": "https://www.sealmaster.com",
        "search_url_template": "https://www.sealmaster.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "browning": {
        "base_url": "https://www.regalrexnord.com",
        "search_url_template": "https://www.regalrexnord.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "mcgill": {
        "base_url": "https://www.regalrexnord.com",
        "search_url_template": "https://www.regalrexnord.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "renold jeffrey": {
        "base_url": "https://www.renold.com",
        "search_url_template": "https://www.renold.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "rexnord": {
        "base_url": "https://www.regalrexnord.com",
        "search_url_template": "https://www.regalrexnord.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
    "martin sprocket": {
        "base_url": "https://www.martinsprocket.com",
        "search_url_template": "https://www.martinsprocket.com/search?q={part_number}",
        "image_selectors": ["img.product-image", "img[data-src*='product']"],
        "approved": False,
    },
}

# Alias map for manufacturer name normalization
_ALIASES = {
    "leeson electric": "leeson",
    "fag (schaeffler)": "fag",
    "ina (schaeffler)": "ina",
    "rexnord inc": "rexnord",
    "martin sprocket & gear co": "martin sprocket",
    "renold jeffrey": "renold jeffrey",
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
        rp = RobotFileParser()
        robots_url = f"{domain}/robots.txt"
        rp.set_url(robots_url)
        try:
            rp.read()
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


def _rate_limit_delay(base_url: str):
    """
    Enforce rate limiting per domain.
    Uses 3s base + random 0.5-2s jitter, or Crawl-delay from robots.txt
    (whichever is larger).
    """
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

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

"""
Safe Sources (ToS Compliance)
1. Wikimedia Commons API
2. OpenVerse API

Setup (Run upon First Time Usage):
pip install requests beautifulsoup4 Pillow lxml

For now, no need for Selenium. Focusing on APIs.

Usage: 
python web_scraper.py --csv test-products_sample.csv        # Scrape images for all products listed and download images
"""

import os
import json
import time
import hashlib
import logging
import argparse
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse

# Config
CHROMEDRIVER_PATH = None # Set path if manually installed, None for using webdriver-manager from pip install

OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
JSON_DIR = OUTPUT_DIR / "json"

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

# Request Headers
HEADERS = {
    "User-Agent": (
        "ImageToProduct-Scraper/1.0 "
        "(GT Senior Design x Motion Industries; "
        "Contact: aehrenhalt3@gatech.edu)"
    )
}

REQUEST_DELAY = 1.5 # Seconds between Requests
# SELENIUM_PAGE_WAIT = 8 # Wait for Page to Load (Later for using Selenium)
MAX_IMAGES_PER_SOURCE = 3 # Max Images to Download per Source (Wikimedia, OpenVerse) per Product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    dateFormat="%H:%M:%S",
)
log = logging.getLogger("scraper")

# Load Product Catalog (from CSV test sample file)
def load_product_catalog(csv_path: str) -> list[dict]:
    """
    Load products from test sample (from Motion Industries .xslx file)

    Expected Columns:
    - <ID>: Unique Product Identifier (i.e. SKU)
    - PrimaryImageFilename (Reference image filename - not used for scraper)
    - Item Number (Motion Industry Internal Product Number)
    - Enterprise Name (Brand Parent Company)
    - Manufacturer Name (Brand/Manufacturer)
    - Manufacturer Part Number (MFR # - Key for Searching)
    - Web Product Description (Detailed Product Description)
    - Motion Internal Description (Internal Description)
    - PGC (Product Group Code)
    - PGC Description (Product Category)

    Returns a list of product dicts ready for scraping (loaded from csv file)
    """

    import csv
    products = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Generate search query from existing metadata
                mfr_name = row.get("Manufacturer Name", "").strip()
                mfr_part_number = row.get("Manufacturer Part Number", "").strip()
                category = row.get("PGC Description", "").strip()
                web_desc = row.get("Web Product Description", "").strip()

                # Search keywords: Priorities are MFR Name, MFR #, then Category/Description Context
                keywords = []
                if mfr_name and mfr_part:
                    keywords.append(f"{mfr_name} {mfr_part_number}")
                if mfr_part and category:
                    # Extra first few woords from category to focus search
                    category_shortened = " ".join(category.split()[:3]) # Take first 3 words of category
                    keywords.append(f"{mfr_part_number} {category_shortened}")

                # If no good keywords, use first 50 chars of web_desc
                if not keywords and web_desc:
                    keywords.append(web_desc[:50])

                products.append({
                    "motion_product_id": row.get("ID", "").strip(),
                    "item_number": row.get("Item Number", "").strip(),
                    "mfr_name": mfr_name
                    "mfr_part_number": mfr_part_number,
                    "category": category,
                    "web_desc": web_desc,
                    "pgc": row.get("PGC", "").strip(),
                    "search_keywords": keywords
                })

            log.info(f"Loaded {len(products)} products from {csv_path}")

        except FileNotFoundError:
            log.error(f"CSV file not found: {csv_path}")
        except Exception as e:
            log.error(f"Error loading CSV file: {e}")
            exit(1)
        
        return products

# Selenium Driver (Later for non-API sources)
def build_driver(driver_path: str = None) -> webdriver.Chrome:
    """
    Create a Selenium WebDriver instance (for later use with non-API sources)

    Attempts to (in order):
    1. Explicit driver_path argument (manual install)
    2. Use chromedriver from CHROMEDRIVER_PATH
    3. Use webdriver-manager to auto-download chromedriver
    4. Selenium's built-in selenium-manager)

    Chrome Options:
    --headless=new : No visible browser window
    --no-sandbox : Disable sandboxing (required for some environments)
    --disable-dev-shm-usage : Avoid /dev/shm issues in Docker
    --disable-gpu : Disable GPU acceleration (required for some environments)
    --window-size : 1920,1080 : Set window size for consistent page rendering
    images prefs disabled : Skip loading image files - Only need DOM metadata for scraping, saves bandwidth and speeds up loading   
    """
    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled") # Avoid detection as bot (probably should remove later)

    # Disable image loading to speed up page loads (only need DOM metadata for scraping)
    opts.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"]) # Avoid detection as bot (probably should remove later)
    opts.add_experimental_option("useAutomationExtension", False) # Avoid detection as bot (probably should remove later)

    resolved_path = driver_path or CHROMEDRIVER_PATH

    if resolved_path:
        log.info(f"Using ChromeDriver from path: {resolved_path}")
        service = Service(executable_path=resolved_path)
        driver = webdriver.Chrome(service=service, options=opts)
        return webdriver.Chrome(service=service, options=opts)

    # Try webdriver-manager
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        log.info("Using ChromeDriver from webdriver-manager")
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)
    except Exception as e:
        log.warning(f"webdriver-manager failed: {e}, fallback to selenium-manager")

    # Final fallback
    log.info(f"Using Selenium built-in selenium-manager for ChromeDriver")
    return webdriver.Chrome(options=opts)

# Wikimedia Commons REST API
def search_wikimedia(product: dict) -> list[dict]:
    """
    Search Wikimedia Commons API for product images.
    """
    results = []
    keyword = f"{product["mfr_name"]} {product["category"]}"

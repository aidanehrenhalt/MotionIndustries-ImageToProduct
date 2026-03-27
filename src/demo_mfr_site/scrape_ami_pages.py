import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


ROOT = Path(__file__).resolve().parent
URLS_FILE = ROOT / "test_product_urls.md"
OUTPUT_DIR = ROOT / "scraped_data"
JSON_DIR = OUTPUT_DIR / "json"
HTML_DIR = OUTPUT_DIR / "html"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"


def load_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"https?://\S+", line)
        if match:
            urls.append(match.group(0).rstrip(").,"))
    return urls


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    tail = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
    tail = re.sub(r"[^a-zA-Z0-9_-]+", "-", tail).strip("-").lower()
    return tail or "page"


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, JSON_DIR, HTML_DIR, MARKDOWN_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_summary(result) -> dict:
    metadata = result.metadata or {}
    media = result.media or {}
    links = result.links or {}
    markdown = result.markdown

    return {
        "url": result.url,
        "success": result.success,
        "status_code": result.status_code,
        "redirected_url": result.redirected_url,
        "redirected_status_code": result.redirected_status_code,
        "page_title": metadata.get("title"),
        "description": metadata.get("description"),
        "keywords": metadata.get("keywords"),
        "og_image": metadata.get("og:image") or metadata.get("og_image"),
        "image_count": len(media.get("images", [])),
        "internal_link_count": len(links.get("internal", [])),
        "external_link_count": len(links.get("external", [])),
        "table_count": len(result.tables or []),
        "markdown_length": len(markdown.raw_markdown) if markdown else 0,
        "html_length": len(result.html or ""),
        "cleaned_html_length": len(result.cleaned_html or ""),
        "error_message": result.error_message,
    }


async def scrape_url(crawler: AsyncWebCrawler, url: str) -> dict:
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=90000,
        delay_before_return_html=2,
        scan_full_page=True,
        verbose=True,
    )
    result = await crawler.arun(url=url, config=run_config)

    slug = slugify_url(url)
    markdown = result.markdown.raw_markdown if result.markdown else ""
    html = result.html or ""
    cleaned_html = result.cleaned_html or ""

    payload = {
        "summary": build_summary(result),
        "metadata": result.metadata or {},
        "media": result.media or {},
        "links": result.links or {},
        "tables": result.tables or [],
        "response_headers": result.response_headers or {},
        "extracted_content": result.extracted_content,
        "console_messages": result.console_messages or [],
        "network_requests": result.network_requests or [],
        "artifacts": {
            "html": str((HTML_DIR / f"{slug}.html").relative_to(ROOT)),
            "cleaned_html": str((HTML_DIR / f"{slug}.cleaned.html").relative_to(ROOT)),
            "markdown": str((MARKDOWN_DIR / f"{slug}.md").relative_to(ROOT)),
        },
    }

    (HTML_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    (HTML_DIR / f"{slug}.cleaned.html").write_text(cleaned_html, encoding="utf-8")
    (MARKDOWN_DIR / f"{slug}.md").write_text(markdown, encoding="utf-8")
    (JSON_DIR / f"{slug}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload["summary"]


async def main() -> None:
    ensure_dirs()
    urls = load_urls(URLS_FILE)
    if not urls:
        raise SystemExit(f"No URLs found in {URLS_FILE}")

    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
    )

    summaries: list[dict] = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            print(f"Scraping: {url}")
            summary = await scrape_url(crawler, url)
            summaries.append(summary)
            print(json.dumps(summary, indent=2))

    (OUTPUT_DIR / "index.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(summaries)} crawl summaries to {OUTPUT_DIR / 'index.json'}")


if __name__ == "__main__":
    asyncio.run(main())

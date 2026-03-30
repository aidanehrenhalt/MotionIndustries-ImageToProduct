import json
import re
from html import escape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


ROOT = Path(__file__).resolve().parent
SCRAPED_JSON_DIR = ROOT / "scraped_data" / "json"
SITE_DIR = ROOT / "site"
PRODUCTS_DIR = SITE_DIR / "products"
ASSETS_DIR = SITE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
DATA_DIR = ASSETS_DIR / "data"

SOURCE_BASE = "https://catalog.amibearings.com"
HEADERS = {
    "User-Agent": (
        "ImageToProduct-DemoSiteBuilder/1.0 "
        "(GT Senior Design x Motion Industries)"
    )
}


def ensure_dirs() -> None:
    for path in (SITE_DIR, PRODUCTS_DIR, ASSETS_DIR, IMAGES_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_scraped_records() -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for path in sorted(SCRAPED_JSON_DIR.glob("*.json")):
        records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not records:
        raise SystemExit(f"No scraped JSON files found in {SCRAPED_JSON_DIR}")
    return records


def clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("N/A", "")).strip()


def parse_title(title: str) -> tuple[str, str]:
    match = re.match(r"Item #\s*([^,]+),\s*(.+?)\s+On\s+AMI Bearings Inc\.", title)
    if not match:
        return title, title
    return match.group(1).strip(), match.group(2).strip()


def parse_breadcrumbs(url: str) -> list[str]:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if "item" not in parts:
        return ["Catalog"]
    item_index = parts.index("item")
    relevant = parts[item_index + 1 : -1]
    labels = ["Catalog"]
    for part in relevant:
        labels.append(part.replace("-", " ").title())
    return labels


def extract_specs(tables: list[dict]) -> list[dict]:
    specs: list[dict] = []
    seen: set[str] = set()
    for table in tables or []:
        for row in table.get("rows", []):
            if len(row) < 2:
                continue
            key = clean_value(str(row[0]))
            value = clean_value(str(row[1]))
            if not key or not value or key in seen:
                continue
            seen.add(key)
            specs.append({"label": key, "value": value})
    return specs


def choose_images(record: dict) -> list[dict]:
    media_images = record.get("media", {}).get("images", [])
    selected: list[dict] = []
    seen: set[str] = set()

    candidates = []
    og_image = record.get("metadata", {}).get("og:image")
    if og_image:
        candidates.append({"src": og_image, "alt": "Primary product image"})

    for image in media_images:
        src = image.get("src") or ""
        if not src:
            continue
        if src.startswith("/ImgMedium/") and src.lower().endswith(".jpg"):
            candidates.append(image)
        elif src.startswith("https://fci.navigator.traceparts.com/") and image.get("alt") == "3D":
            candidates.append(image)

    for image in candidates:
        raw_src = image.get("src") or ""
        abs_src = urljoin(SOURCE_BASE, raw_src.replace("\\", "/"))
        if abs_src in seen:
            continue
        seen.add(abs_src)
        selected.append(
            {
                "source_url": abs_src,
                "alt": image.get("alt") or "Product image",
            }
        )
        if len(selected) == 3:
            break
    return selected


def download_image(url: str, target: Path) -> bool:
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        target.write_bytes(response.content)
        return True
    except Exception:
        return False


def local_image_name(slug: str, url: str, index: int) -> str:
    suffix = Path(urlparse(url).path).suffix or ".jpg"
    return f"{slug}-{index}{suffix.lower()}"


def normalize_product(path: Path, record: dict) -> dict:
    slug = path.stem
    title = record.get("metadata", {}).get("title") or record.get("summary", {}).get("page_title") or slug
    part_number, item_name = parse_title(title)
    breadcrumbs = parse_breadcrumbs(record["summary"]["url"])
    specs = extract_specs(record.get("tables", []))
    images = choose_images(record)

    highlights = specs[:4]
    if not highlights:
        highlights = [{"label": "Product", "value": item_name}]

    return {
        "slug": slug,
        "part_number": part_number,
        "item_name": item_name,
        "page_title": title,
        "source_url": record["summary"]["url"],
        "source_domain": "catalog.amibearings.com",
        "breadcrumbs": breadcrumbs,
        "meta_description": record.get("metadata", {}).get("description", ""),
        "specs": specs,
        "highlights": highlights,
        "images": images,
        "documents": [
            {"label": "Download PDF", "url": record["summary"]["url"]},
            {"label": "Printable Page", "url": record["summary"]["url"]},
        ],
    }


def render_specs(specs: list[dict]) -> str:
    return "\n".join(
        f"<tr><th>{escape(spec['label'])}</th><td>{escape(spec['value'])}</td></tr>" for spec in specs
    )


def render_breadcrumbs(breadcrumbs: list[str], part_number: str) -> str:
    crumbs = breadcrumbs + [part_number]
    return "".join(f"<li>{escape(crumb)}</li>" for crumb in crumbs)


def render_gallery(images: list[dict]) -> str:
    if not images:
        return "<div class='gallery-empty'>No images available.</div>"
    return "\n".join(
        (
            "<figure class='gallery-card'>"
            f"<img src='../assets/images/{escape(image['local_name'])}' alt='{escape(image['alt'])}'>"
            f"<figcaption>{escape(image['alt'])}</figcaption>"
            "</figure>"
        )
        for image in images
    )


def render_highlights(highlights: list[dict]) -> str:
    return "\n".join(
        (
            "<div class='highlight'>"
            f"<span class='label'>{escape(item['label'])}</span>"
            f"<strong>{escape(item['value'])}</strong>"
            "</div>"
        )
        for item in highlights
    )


def render_index_cards(products: list[dict]) -> str:
    cards = []
    for product in products:
        image_src = (
            f"assets/images/{product['images'][0]['local_name']}"
            if product["images"]
            else ""
        )
        image_tag = (
            f"<img src='{escape(image_src)}' alt='{escape(product['item_name'])}'>"
            if image_src
            else ""
        )
        highlights = ", ".join(f"{item['label']}: {item['value']}" for item in product["highlights"][:2])
        cards.append(
            "<article class='product-card'>"
            f"<a class='product-card-link' href='products/{escape(product['slug'])}.html'>"
            f"<div class='thumb'>{image_tag}</div>"
            f"<p class='eyebrow'>AMI Bearings Demo</p>"
            f"<h2>{escape(product['part_number'])}</h2>"
            f"<p class='product-name'>{escape(product['item_name'])}</p>"
            f"<p class='meta'>{escape(highlights)}</p>"
            "</a>"
            "</article>"
        )
    return "\n".join(cards)


def render_topbar(home_prefix: str = "") -> str:
    return (
        '<header class="topbar">'
        '<div class="site-shell topbar-inner">'
        f'<a class="brand" href="{home_prefix}index.html">Motion Industries Demo<strong>AMI Bearings</strong></a>'
        '<nav class="topnav">'
        f'<a href="{home_prefix}index.html">Catalog</a>'
        f'<a href="{home_prefix}review.html">Review Queue</a>'
        "</nav>"
        "</div>"
        "</header>"
    )


def write_stylesheet() -> None:
    styles = """\
:root {
  --bg: #f3efe6;
  --panel: #fffaf1;
  --ink: #13242f;
  --muted: #5d6b73;
  --line: #d8d2c4;
  --accent: #0b6973;
  --accent-2: #d8822f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(216,130,47,0.16), transparent 28%),
    linear-gradient(180deg, #efe7d6 0%, var(--bg) 100%);
}
a { color: inherit; text-decoration: none; }
img { display: block; max-width: 100%; }
.site-shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; }
.topbar {
  border-bottom: 1px solid rgba(19,36,47,0.12);
  background: rgba(255,250,241,0.84);
  backdrop-filter: blur(10px);
}
.topbar-inner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 18px 0;
}
.brand { font-size: 0.95rem; letter-spacing: 0.16em; text-transform: uppercase; }
.brand strong { display: block; font-size: 1.2rem; letter-spacing: 0.04em; }
.topnav {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.topnav a {
  padding: 8px 12px;
  border: 1px solid rgba(19,36,47,0.12);
  border-radius: 999px;
  background: rgba(255,250,241,0.7);
  font-size: 0.9rem;
}
.masthead { padding: 56px 0 28px; }
.masthead-grid {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 24px;
  align-items: end;
}
.kicker, .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.78rem; }
.hero-title, .product-title {
  margin: 10px 0 12px;
  font-size: clamp(2.4rem, 5vw, 4.6rem);
  line-height: 0.95;
}
.hero-copy, .product-copy { color: var(--muted); font-size: 1.02rem; line-height: 1.65; max-width: 62ch; }
.summary-panel, .detail-panel, .gallery-card, .product-card {
  background: var(--panel);
  border: 1px solid rgba(19,36,47,0.08);
  box-shadow: 0 14px 36px rgba(19,36,47,0.08);
}
.summary-panel { padding: 22px; border-radius: 22px; }
.product-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
  padding: 12px 0 56px;
}
.product-card { border-radius: 22px; overflow: hidden; }
.product-card-link { display: block; }
.thumb {
  aspect-ratio: 4 / 3;
  background: linear-gradient(135deg, #dfe9ea, #f6efe2);
  padding: 18px;
}
.thumb img { width: 100%; height: 100%; object-fit: contain; }
.product-card h2 { margin: 0 0 8px; font-size: 2rem; }
.product-card .product-name, .product-card .meta { margin: 0; color: var(--muted); }
.product-card-link > *:not(.thumb) { padding-left: 24px; padding-right: 24px; }
.product-card-link .eyebrow { padding-top: 22px; }
.product-card-link .meta { padding-bottom: 24px; padding-top: 14px; }
.breadcrumbs { list-style: none; padding: 22px 0 0; margin: 0; display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 0.92rem; }
.breadcrumbs li::after { content: "/"; margin-left: 10px; color: var(--line); }
.breadcrumbs li:last-child::after { content: ""; margin: 0; }
.detail-layout {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 28px;
  padding: 26px 0 56px;
}
.detail-panel { border-radius: 26px; padding: 28px; }
.highlights { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 28px 0; }
.highlight {
  padding: 16px;
  border-radius: 18px;
  background: rgba(11,105,115,0.06);
  border: 1px solid rgba(11,105,115,0.12);
}
.highlight .label { display: block; color: var(--muted); font-size: 0.84rem; margin-bottom: 6px; }
.gallery-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.gallery-card { padding: 16px; border-radius: 18px; }
.gallery-card img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; }
.gallery-card figcaption { margin-top: 10px; color: var(--muted); font-size: 0.9rem; }
.spec-table { width: 100%; border-collapse: collapse; }
.spec-table th, .spec-table td { padding: 12px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
.spec-table th { width: 42%; text-align: left; font-weight: 600; }
.doc-list { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 22px; }
.doc-link {
  display: inline-flex;
  align-items: center;
  padding: 10px 14px;
  border-radius: 999px;
  background: var(--accent);
  color: white;
}
.review-layout {
  display: grid;
  grid-template-columns: 0.9fr 1.1fr;
  gap: 24px;
  padding: 18px 0 48px;
}
.review-panel {
  background: var(--panel);
  border: 1px solid rgba(19,36,47,0.08);
  box-shadow: 0 14px 36px rgba(19,36,47,0.08);
  border-radius: 24px;
  padding: 24px;
}
.review-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.review-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin: 20px 0 24px;
}
.stat-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(11,105,115,0.06);
  border: 1px solid rgba(11,105,115,0.12);
}
.stat-card strong {
  display: block;
  font-size: 1.4rem;
  margin-top: 6px;
}
.review-product-list {
  display: grid;
  gap: 12px;
}
.review-product-button {
  width: 100%;
  text-align: left;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid rgba(19,36,47,0.12);
  background: white;
  cursor: pointer;
}
.review-product-button.active {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(11,105,115,0.14);
}
.review-product-button small,
.muted {
  color: var(--muted);
}
.review-grid {
  display: grid;
  gap: 18px;
}
.review-image-card {
  border-radius: 20px;
  border: 1px solid rgba(19,36,47,0.12);
  background: white;
  overflow: hidden;
}
.review-image-frame {
  aspect-ratio: 4 / 3;
  background: linear-gradient(135deg, #dfe9ea, #f6efe2);
  padding: 18px;
}
.review-image-frame img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.review-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 18px;
}
.metric {
  padding: 12px;
  border-radius: 16px;
  background: rgba(216,130,47,0.08);
}
.metric-label {
  display: block;
  font-size: 0.78rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.candidate-table {
  width: 100%;
  border-collapse: collapse;
}
.candidate-table th,
.candidate-table td {
  padding: 12px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
}
.candidate-row {
  cursor: pointer;
}
.candidate-row.active {
  background: rgba(11,105,115,0.08);
}
.review-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 18px;
}
.action-button {
  border: none;
  border-radius: 999px;
  padding: 12px 18px;
  font: inherit;
  cursor: pointer;
}
.action-approve { background: var(--accent); color: white; }
.action-reject { background: #8c3b2a; color: white; }
.action-skip { background: #d8d2c4; color: var(--ink); }
.action-export { background: var(--accent-2); color: white; }
.review-feedback {
  width: 100%;
  min-height: 110px;
  border-radius: 18px;
  border: 1px solid rgba(19,36,47,0.16);
  padding: 14px;
  font: inherit;
  background: #fffdf9;
}
.review-status {
  margin-top: 14px;
  color: var(--muted);
}
.empty-state {
  padding: 26px;
  border-radius: 20px;
  background: rgba(216,130,47,0.1);
  border: 1px solid rgba(216,130,47,0.2);
}
.footer { padding: 24px 0 48px; color: var(--muted); font-size: 0.94rem; }
@media (max-width: 860px) {
  .masthead-grid, .detail-layout, .product-grid, .highlights, .gallery-grid, .review-layout, .review-summary, .review-metrics {
    grid-template-columns: 1fr;
  }
  .topbar-inner {
    align-items: flex-start;
    flex-direction: column;
  }
}
"""
    (ASSETS_DIR / "styles.css").write_text(styles, encoding="utf-8")


def write_index(products: list[dict]) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AMI Bearings Demo Catalog</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
  {render_topbar("")}
  <main class="site-shell">
    <section class="masthead">
      <div class="masthead-grid">
        <div>
          <p class="kicker">Manufacturer Demo Site</p>
          <h1 class="hero-title">Catalog pages rebuilt from collected AMI source artifacts.</h1>
          <p class="hero-copy">This local site preserves the product-page patterns your scraper needs: breadcrumbs, part number, product title, image gallery, document actions, and specification tables.</p>
        </div>
        <aside class="summary-panel">
          <p class="eyebrow">Included Pages</p>
          <h2>4 AMI products</h2>
          <p>Each page was scraped from the AMI catalog, normalized into JSON, then rendered locally with downloaded image assets.</p>
        </aside>
      </div>
    </section>
    <section class="product-grid">
      {render_index_cards(products)}
    </section>
  </main>
  <footer class="site-shell footer">
    <p>Source artifacts live in <code>src/demo_mfr_site/scraped_data</code>. Normalized dataset is available at <code>assets/data/products.json</code>. Ranked review queue data is written after the demo pipeline runs.</p>
  </footer>
</body>
</html>
"""
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")


def write_product_page(product: dict) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(product['page_title'])}</title>
  <meta name="description" content="{escape(product['meta_description'])}">
  <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
  {render_topbar("../")}
  <main class="site-shell">
    <ol class="breadcrumbs">{render_breadcrumbs(product['breadcrumbs'], product['part_number'])}</ol>
    <section class="detail-layout">
      <article class="detail-panel">
        <p class="eyebrow">Item # {escape(product['part_number'])}</p>
        <h1 class="product-title">{escape(product['item_name'])}</h1>
        <p class="product-copy">{escape(product['meta_description'])}</p>
        <section class="highlights">{render_highlights(product['highlights'])}</section>
        <div class="doc-list">
          {''.join(f"<a class='doc-link' href='{escape(doc['url'])}'>{escape(doc['label'])}</a>" for doc in product['documents'])}
        </div>
      </article>
      <aside class="detail-panel">
        <p class="eyebrow">Image Gallery</p>
        <div class="gallery-grid">{render_gallery(product['images'])}</div>
      </aside>
    </section>
    <section class="detail-layout">
      <article class="detail-panel">
        <p class="eyebrow">Specifications</p>
        <table class="spec-table">
          <tbody>
            {render_specs(product['specs'])}
          </tbody>
        </table>
      </article>
      <aside class="detail-panel">
        <p class="eyebrow">Source</p>
        <p class="product-copy">Original page: <a href="{escape(product['source_url'])}">{escape(product['source_domain'])}</a></p>
        <p class="product-copy">Local data endpoint: <code>../assets/data/products.json</code></p>
      </aside>
    </section>
  </main>
  <footer class="site-shell footer">
    <p><a href="../index.html">Back to catalog index</a></p>
  </footer>
</body>
</html>
"""
    (PRODUCTS_DIR / f"{product['slug']}.html").write_text(html, encoding="utf-8")


def write_review_page() -> None:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AMI Bearings Review Queue</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
  """ + render_topbar("") + """
  <main class="site-shell">
    <section class="masthead">
      <div class="masthead-grid">
        <div>
          <p class="kicker">Review Frontend</p>
          <h1 class="hero-title">Manual review queue built from ranked demo pipeline output.</h1>
          <p class="hero-copy">This page reads the normalized review dataset generated from pipeline rankings, JSON records, and output images. Review decisions stay local and can be exported as JSON.</p>
        </div>
        <aside class="summary-panel">
          <p class="eyebrow">Review Source</p>
          <h2>Pipeline-backed queue</h2>
          <p>Run <code>./run_pipeline_b.sh</code> first so the demo pipeline can regenerate the ranked artifacts and review dataset.</p>
        </aside>
      </div>
    </section>
    <section id="review-app" class="review-layout">
      <article class="review-panel">
        <p class="eyebrow">Queue</p>
        <div class="empty-state">Loading review queue…</div>
      </article>
      <aside class="review-panel">
        <p class="eyebrow">Selected Product</p>
        <div class="empty-state">Waiting for review data…</div>
      </aside>
    </section>
  </main>
  <footer class="site-shell footer">
    <p>The review dataset is expected at <code>assets/data/review_queue.json</code>. Exported decisions remain local unless they are written back by a later backend integration.</p>
  </footer>
  <script src="assets/review.js"></script>
</body>
</html>
"""
    (SITE_DIR / "review.html").write_text(html, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    products = []
    for path, record in load_scraped_records():
        product = normalize_product(path, record)
        localized_images = []
        for index, image in enumerate(product["images"], start=1):
            local_name = local_image_name(product["slug"], image["source_url"], index)
            target = IMAGES_DIR / local_name
            if not target.exists():
                download_image(image["source_url"], target)
            if target.exists():
                image = dict(image)
                image["local_name"] = local_name
                localized_images.append(image)
        product["images"] = localized_images
        products.append(product)

    write_stylesheet()
    write_index(products)
    for product in products:
        write_product_page(product)
    write_review_page()
    (DATA_DIR / "products.json").write_text(json.dumps(products, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

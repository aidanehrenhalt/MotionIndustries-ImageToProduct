# Sourcing Product Images for Motion Industries SKUs from Web-Accessible Sources

## Executive summary

You can usually fill “missing image” gaps for industrial/MRO SKUs by harvesting images from three escalating tiers of sources: (a) the distributor’s own product pages and image CDN, (b) official manufacturer catalogs / CAD portals / PIM libraries, and (c) other large distributors and marketplaces that may have broader long-tail coverage. The main operational constraint is that many high-value sites explicitly restrict automated access in Terms of Use or robots rules (or enforce bot protection), so “technically scrapable” does not reliably mean “contractually permitted.” For example, Motion’s published Terms of Use prohibits automated access and screen/database scraping without prior written consent. citeturn32view0 eBay’s robots policy also explicitly prohibits automated access without express permission. citeturn41view0 Grainger’s robots rules broadly disallow bots from key paths and explicitly disallow multiple AI crawlers. citeturn39view0 MSC’s robots rules exist, but the site may still block automated requests with bot protection (observed as an Incapsula/Imperva incident on a product page). citeturn40view0turn55view0

Given those realities, the most robust and lowest-risk approach for a spreadsheet-enrichment workflow is:

1) **Prefer official/primary sources first** (manufacturer catalogs and Motion’s own image CDN) because they’re the most likely to be correct for the exact MPN/SKU, and they tend to offer consistent “asset URL” patterns (often `/Asset/...` and/or stable CDNs). citeturn22view0turn65view0  
2) **Use sanctioned APIs when available** instead of scraping high-friction sites: Amazon Product Advertising API explicitly exposes image URLs in multiple sizes, and eBay’s Browse API returns primary and additional image URLs. citeturn68search0turn68search2  
3) **Treat cross-site distributor scraping as opportunistic**: it can help coverage, but you need strict throttling, caching, legal review, and strong quality controls (deduplication + placeholder detection + “close enough” matching safeguards). citeturn39view0turn55view0turn41view0

## Priority source list with extraction recipes

The entries below focus on **repeatable URL patterns**, **practical selectors**, and **API endpoints** that can be operationalized for bulk enrichment. “Priority” here means: expected accuracy for the exact SKU/MPN + consistency of extraction + practical likelihood of obtaining an image at scale.

### Source priority overview

| Priority intent | Source (examples) | Why it’s high value | Typical access strategy |
|---|---|---|---|
| Primary distributor | entity["company","Motion Industries","industrial distributor us"] | Direct mapping to Motion item numbers; images often served from a dedicated CDN with resize parameters. citeturn26search0turn2view0 | Crawl product pages by Motion item; extract CDN image URLs. (Compliance note: Terms of Use restrict automation.) citeturn32view0 |
| Primary manufacturer catalog / CAD portal | entity["company","The Timken Company","bearing manufacturer us"], entity["company","AMI Bearings, Inc.","mounted bearings maker us"], entity["company","NTN Corporation","bearing maker japan"] | Official part-number pages commonly expose stable `/Asset/...` or catalog image URLs and sometimes dimensions/variants; best accuracy when you have MPN. citeturn22view0turn38view0turn65view0turn54search8 | Query by MPN → extract “asset” image links and/or CAD drawing images. |
| Major distributor | entity["company","W.W. Grainger","mro distributor us"] | Often have high-resolution images hosted on a stable image CDN (`static.grainger.com/rp/s/is/image/...`). citeturn62search8turn57view0 | Map SKU/MPN → Grainger item page → extract `og:image` / primary image URL (or derive CDN URL if exposed). |
| Major distributor | entity["company","Fastenal Company","industrial distributor us"] | Broad SKU coverage for industrial supplies; images often hosted on a separate domain (`img2.fastenal.com/productimages/...`). citeturn43view0turn50view0turn48search0 | Use headless browser for JS pages; or harvest known image CDN URLs when discoverable. |
| Major distributor (high friction) | entity["company","MSC Industrial Supply Co.","mro distributor us"] | Strong assortment, but automation may be blocked (Incapsula/Imperva observed on product page). citeturn55view0turn40view0 | Prefer partner feeds/APIs if available; otherwise expect frequent blocks. |
| Marketplace / API | entity["company","Amazon.com, Inc.","ecommerce company us"] | Very broad catalog; “Images” resource provides URLs and dimensions in Small/Medium/Large via PA-API. citeturn68search0 | Use Product Advertising API (requires license/affiliate compliance). citeturn68search1 |
| Marketplace / API | entity["company","eBay Inc.","ecommerce marketplace us"] | Very broad, but direct scraping is discouraged by robots policy; Browse API returns `image.imageUrl` and additional image URLs. citeturn41view0turn68search2 | Use eBay Buy Browse API; comply with eBay API License Agreement. citeturn68search3 |

### Motion product pages and Motion image CDN

**Observed page pattern (product detail)**  
Motion product pages commonly follow the SKU format and show the Motion item number (“MI ITEM #…”) alongside manufacturer part number (“MFR #…”). citeturn26search0turn29view0

```text
Pattern:
  https://www.motion.com/products/sku/<MI_ITEM>

Example:
  https://www.motion.com/products/sku/03770969   (UCT308 example page)
```

**Observed “search results” page pattern (useful when you have MPN but not MI item)**  
Motion search results URLs are structured with semicolon-delimited parameters (not standard `?q=`), and the results list includes MI item numbers and MFR numbers inline. citeturn28view0

```text
Pattern:
  https://www.motion.com/products/search;q=<QUERY>;origin=search

Example:
  https://www.motion.com/products/search;q=industrial%20hose;origin=search
```

**Image URL behavior (key operational detail)**  
On at least some Motion product pages, image links resolve to `content.motion.com` and include Cloudflare-style resizing parameters (`/cdn-cgi/image/width=...`). citeturn2view0turn29view0

```text
Observed image URL shape:
  https://content.motion.com/cdn-cgi/image/width=256,fit=scale-down,format=auto,metadata=none/.../motion3/fsdb/images/item/<ASSET>.jpg

Example (from a Motion product image link):
  https://content.motion.com/cdn-cgi/image/width=256,fit=scale-down,format=auto,metadata=none/https://content.motion.com/motion3/fsdb/images/item/AMI_UCT300.jpg

Practical implication:
  If permitted, you can often request higher widths (e.g., 800–1600) by changing the width parameter.
```

**Extraction selectors (practical, robust-first)**  
Use layered extraction so a small markup shift doesn’t break you:

```text
1) Primary (when present):
   CSS:  img[src*="content.motion.com"][src*="/fsdb/images/item/"]
   XPath: //img[contains(@src,"content.motion.com") and contains(@src,"/fsdb/images/item/")]

2) Fallback (if images are exposed as links rather than <img>):
   CSS:  a[href*="content.motion.com"][href*="/fsdb/images/item/"]
   XPath: //a[contains(@href,"content.motion.com") and contains(@href,"/fsdb/images/item/")]

3) Last-resort (common ecommerce pattern):
   CSS: meta[property="og:image"], meta[name="og:image"]
   XPath: //meta[@property="og:image"]/@content
```

### Manufacturer catalogs and OEM portals with stable “Asset” image patterns

A highly repeatable pattern across multiple manufacturer catalog portals is the presence of direct `/Asset/...` image links on the part-number page (often alongside “Download PDF” or “View CAD drawing”). citeturn22view0turn65view0turn54search8

#### Timken CAD portal (`cad.timken.com`)

The Timken CAD part page is explicitly keyed by the part number in the URL, and the page exposes image-related UI and downloadable resources. citeturn22view0turn38view0

```text
Part page pattern:
  https://cad.timken.com/item/<category-path>/<family>/<part-number>

Example:
  https://cad.timken.com/item/u-series---take-up-mounted-bearings--uct-200--uct/uct-take-up-units-2/uct308
```

**What to scrape for images**
- Product/related-product images referenced as `/Asset/<filename>.JPG` and/or small images such as `/ImgSmall/<filename>.JPG` have been observed. citeturn22view0turn24view2turn25search9

**Selectors**
```text
CSS:   a[href^="/Asset/"][href$=".JPG"], a[href^="/Asset/"][href$=".jpg"]
XPath: //a[starts-with(@href,"/Asset/") and (contains(@href,".jpg") or contains(@href,".JPG"))]

CSS:   img[src^="/ImgSmall/"], img[src^="/ImgLarge/"]
XPath: //img[starts-with(@src,"/ImgSmall/") or starts-with(@src,"/ImgLarge/")]
```

#### AMI Bearings catalog portal (`catalog.amibearings.com`)

AMI’s catalog page for a part number shows a direct ` /Asset/...` image reference (example: `/Asset/UCT300.jpg`). citeturn65view0

```text
Part page pattern:
  https://catalog.amibearings.com/item/<category-path>/<family>/<part-number>

Example:
  https://catalog.amibearings.com/item/set-screw-locking-8/set-screw-locking-take-up-unit-uct300-series/uct308

Observed asset link shape on page:
  /Asset/UCT300.jpg
```

**Selectors**
```text
CSS:   a[href^="/Asset/"], img[src^="/Asset/"]
XPath: //a[starts-with(@href,"/Asset/")] | //img[starts-with(@src,"/Asset/")]
```

#### NTN “Bearing Finder” portal (`bearingfinder.ntnamericas.com`)

NTN’s product pages similarly expose asset images directly using `/Asset/...png` on the page. citeturn54search8

```text
Part page pattern:
  https://bearingfinder.ntnamericas.com/item/<category-path>/<family>/<item-number>

Observed asset link shape on page:
  /Asset/<...>.png
```

**Selectors**
```text
CSS:   a[href^="/Asset/"][href$=".png"], img[src^="/Asset/"][src$=".png"]
XPath: //a[starts-with(@href,"/Asset/") and contains(@href,".png")] | //img[starts-with(@src,"/Asset/") and contains(@src,".png")]
```

### Grainger product pages and image CDN

A Grainger product page can be fully server-rendered and includes a “Main product photo” field in the rendered content. citeturn57view0 Separately, Grainger product imagery is commonly hosted on a stable “Scene7-style” CDN structure: `static.grainger.com/rp/s/is/image/Grainger/<asset>?$adapimg$&hei=...&wid=...`. citeturn62search8turn62search16

```text
Product page example:
  https://www.grainger.com/product/IPTCI-BEARINGS-Take-Up-Ball-Bearing-Mounted-825CG3

Observed image CDN pattern:
  https://static.grainger.com/rp/s/is/image/Grainger/<ASSET_ID>?$adapimg$&hei=1072&wid=1072
```

**Extraction approach**
1) On the product page HTML, look for OpenGraph / schema metadata first (most stable).  
2) If Grainger exposes a direct CDN URL in markup (or via a JSON blob), capture it and store the **normalized high-res variant**.

**Selectors**
```text
Primary:
  CSS:   meta[property="og:image"]
  XPath: //meta[@property="og:image"]/@content

Fallback:
  CSS:   img[alt*="Main product photo" i]
  XPath: //img[contains(translate(@alt,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"main product photo")]/@src
```

### Fastenal pages and image hosting

Some Fastenal pages are JS-heavy (“You need to enable JavaScript to run this app”), which implies that **static HTML scraping may fail** without a rendering step. citeturn43view0turn46view0 However, product imagery can appear on separate image hosts such as `img2.fastenal.com/productimages/<id>.jpg`. citeturn50view0turn48search0

```text
Observed image host form:
  https://img2.fastenal.com/productimages/<IMAGE_ID>.jpg
Example (verified as an accessible image URL):
  https://img2.fastenal.com/productimages/0762603.jpg
```

**Practical selectors (when you render with a browser)**
```text
1) OpenGraph:
   meta[property="og:image"]

2) Primary image element patterns:
   img[src*="fastenal.com"][src*="productimages"]
   img[src*="img2.fastenal.com/productimages/"]
```

### MSC Direct: robots rules exist, but bot protection may still block

MSC publishes a robots.txt with many disallowed paths and a sitemap reference. citeturn40view0 In practice, even a simple product-page fetch may return an “Incapsula incident” response, consistent with bot mitigation. citeturn55view0

```text
Example blocked response signature:
  "Request unsuccessful. Incapsula incident ID: ..."
```

**Operational guidance**
- Treat MSC as **“API/partner-feed preferred.”**  
- If you need MSC coverage, plan for: allowlisting, contractual permission, or a commercially supported data access method.

### Marketplaces: prefer APIs over scraping

#### Amazon Product Advertising API (PA-API 5.0)

Amazon’s PA-API “Images” resource explicitly returns **image URLs in Small/Medium/Large** and includes height/width metadata. citeturn68search0 The licensing documentation points you to the locale-specific Associates Program IP License governing use of PA-API content. citeturn68search1

```text
Docs:
  https://webservices.amazon.com/paapi5/documentation/images.html

Key response concept:
  Images.*.Large.URL (and associated width/height)
```

#### eBay Browse API

eBay’s Buy Browse API `getItem` response includes the primary image URL in `image.imageUrl` and returns `additionalImages[].imageUrl` for extra images. citeturn68search2 eBay also provides an API License Agreement governing use of eBay Content. citeturn68search3 Additionally, eBay’s robots policy explicitly prohibits robots/automated means without express permission (outside limited search-engine use). citeturn41view0

```text
Docs:
  https://developer.ebay.com/api-docs/buy/browse/resources/item/methods/getItem

Key response fields:
  image.imageUrl
  additionalImages[].imageUrl
```

## Legal, licensing, and respectful scraping guardrails

Because licensing/jurisdiction is unspecified, treat nearly all product photography and many drawings as **copyrighted content** where your right to copy/store/redistribute depends on the site’s license terms or explicit permission.

### Contractual and access constraints that materially affect feasibility

- **Motion Terms of Use**: explicitly prohibits using automated means to access the services or collect information (including robots/spiders/scripts), and prohibits screen scraping / database scraping without prior written consent. citeturn32view0  
- **eBay robots policy**: explicitly states automated access without express permission is strictly prohibited (beyond limited indexing for public search engines). citeturn41view0  
- **Grainger robots**: disallows many paths (including `/api/`) and explicitly disallows multiple AI crawler user-agents. citeturn39view0  
- **MSC**: robots rules are published, but bot protection can still block requests (Incapsula incident observed). citeturn40view0turn55view0  
- **Timken CAD portal**: includes an on-page statement restricting the CAD drawings to personal one-time use and prohibiting redistribution/display on other websites or media. citeturn22view0  

### Respectful scraping practices when access is permitted

Even when robots/terms allow your use case (or you’ve obtained permission), implement the following as defaults:

- **Robots and policy checks**: retrieve and cache robots rules and key terms pages; fail closed when uncertain. (Grainger and MSC both publish robots files; eBay uses robots to declare automation restrictions.) citeturn39view0turn40view0turn41view0  
- **Rate limits**: set conservative per-host concurrency (e.g., 1–2 concurrent requests) and bounded QPS; honor any `Retry-After` on 429/503 responses.  
- **Identification and headers**: send a descriptive `User-Agent` (company/project contact), include `From` or contact header where appropriate, prefer `Accept: text/html,image/*`.  
- **Caching and conditional requests**: persist fetched HTML and images with `ETag` / `Last-Modified` and revalidate; avoid re-downloading unchanged assets.  
- **Backoff and circuit breakers**: exponential backoff with jitter; if bot-block signatures appear (e.g., Incapsula incident pages), stop and route to an alternate source rather than escalating. citeturn55view0  

## SKU-to-image matching and disambiguation methods

Industrial SKU matching fails most often because different sites index the same product under different identifiers (internal distributor SKUs vs. manufacturer part numbers vs. GTIN), and because many part numbers contain punctuation, spaces, or pack-size suffixes.

### Normalization and canonical identifiers

A practical canonicalization stack is:

- **MPN-first canonical key**: manufacturer name + normalized manufacturer part number (uppercased; remove repeated whitespace; normalize hyphen/slash variants; preserve meaningful suffixes like “-24” where they indicate bore size).  
- **Distributor item key**: distributor item number (e.g., Motion “MI ITEM #…”) when available on the source page. Motion product pages expose both MI item and MFR numbers. citeturn26search0turn29view0  
- **GTIN/EAN/UPC**: when present, treat as a strong cross-site join (but beware packaging-level GTINs and distributor repacks).

### Search strategies when SKUs differ across sites

Using the Motion website as an example, if you start from an MPN without a Motion item number:

1) Query Motion’s search results endpoint with the MPN and/or manufacturer name. The rendered results list MI item numbers and MFR numbers in-line, enabling a deterministic jump to `.../products/sku/<MI_ITEM>`. citeturn28view0  
2) If Motion search yields multiple hits, rank candidates by:
   - exact MFR # match (post-normalization),
   - brand/manufacturer match,
   - key attribute overlaps (bore, dimensions, ratio, voltage) if available in listing text.

For manufacturer catalogs like Timken’s CAD portal, the quick start guide explicitly notes “Search catalog by keyword or part number,” which implies straightforward “MPN as query” entry points are first-class. citeturn38view0

### Fuzzy matching and verification (to avoid wrong images)

Use fuzzy matching only to propose candidates; require a verification step before committing:

- **Text similarity**: token-set ratio (handles reordering) on normalized descriptions + MPN.  
- **Attribute agreement**: extract 2–5 high-signal attributes (e.g., bore diameter, ratio, frame size) when present and require agreement to break ties.  
- **Variant handling**: many portals enumerate multiple images/variants; prefer “primary image” plus one secondary if you need multi-angle coverage (eBay APIs distinguish primary vs. additional images). citeturn68search2  

### Image deduplication and “placeholder” detection

At scale, you will see: repeated catalog drawings, repeated product-family photos, and placeholder images.

- **Perceptual hashes**: compute pHash/dHash and group by small Hamming distance to deduplicate near-identical angles/crops.  
- **Exact checksums**: store SHA-256 to avoid duplicate downloads and to detect asset changes over time.  
- **Placeholder heuristics**:
  - repeated hash across thousands of SKUs,
  - very small pixel dimensions,
  - filenames or alt-text indicating “filler,” “no image,” or generic category art.

Also note that some distributors explicitly warn that photos may not represent actual items—Motion includes this disclaimer on product pages—so you may want to flag those as “illustrative” in metadata. citeturn29view0turn26search0

## Automation architecture and operational controls

### Recommended tooling

A pragmatic stack for this job usually needs **both** a fast HTTP crawler and a browser renderer:

- **HTTP-first extraction**: Scrapy or a `httpx`/requests + lxml pipeline for sites that are server-rendered (many Motion pages and at least some Grainger pages are readable as HTML). citeturn29view0turn57view0  
- **Headless rendering fallback**: Playwright for JS-heavy sites (Fastenal often requires JS to render). citeturn43view0turn46view0  
- **API connectors**: direct REST calls for eBay Browse API and Amazon PA-API where licensed. citeturn68search0turn68search2  
- **Image processing**: Pillow + imagehash; store to object storage (S3/GCS/Azure Blob) with content-addressed paths.

### Workflow pseudocode

```pseudo
for each SKU_row in input_spreadsheet:
  candidates = []

  # 1) Manufacturer-first (official portals)
  candidates += lookup_manufacturer_portals(MPN, manufacturer_name)

  # 2) Motion (if permitted)
  mi_item = find_motion_item(MPN)  # via Motion search results endpoint
  if mi_item:
     candidates += extract_motion_images(mi_item)

  # 3) Other distributors (Grainger/Fastenal) if allowed
  candidates += lookup_grainger(MPN or keyword bundle)
  candidates += lookup_fastenal(MPN or keyword bundle)

  # 4) Marketplaces via APIs (licensed)
  if have_gtin_or_asin:
     candidates += amazon_paapi_images(asin)
  if have_ebay_item_id:
     candidates += ebay_browse_images(item_id)

  # Rank + verify
  best = select_best_candidate(candidates, ruleset)
  if best passes validation:
     download_image(best.url) with caching + checksum
     write_back_to_spreadsheet(best + metadata)
  else:
     flag_for_manual_review(SKU_row)
```

### Minimal `curl` examples for API-based image retrieval

**eBay Browse API (`getItem`)**: returns `image.imageUrl` and `additionalImages[].imageUrl`. citeturn68search2

```bash
curl -sS \
  -H "Authorization: Bearer $EBAY_OAUTH_TOKEN" \
  "https://api.ebay.com/buy/browse/v1/item/{item_id}"
```

**Amazon PA-API (Images resource concept)**: the Images resource returns URLs for Small/Medium/Large and includes dimensions. citeturn68search0  
(Implementation requires signed requests and compliance with the governing license terms. citeturn68search1)

```text
Docs reference:
  https://webservices.amazon.com/paapi5/documentation/images.html
```

### Error handling patterns that matter for this problem

- **429 / throttling**: exponential backoff + respect `Retry-After`.  
- **Bot blocks**: detect “incident ID”/challenge pages and stop (MSC example). citeturn55view0  
- **Timeouts on large assets**: use streaming downloads, max size caps, and resumable storage writes (AMI asset retrieval may require longer timeouts). citeturn65view0  
- **Asset drift**: periodically revalidate images (ETag/Last-Modified) rather than assuming permanence.

### Mermaid flowchart of the scraping workflow

```mermaid
flowchart TD
  A[Input SKU rows] --> B[Normalize identifiers<br/>MPN, brand, GTIN]
  B --> C{Official manufacturer portal<br/>has image?}
  C -- Yes --> D[Extract asset URL(s)<br/>/Asset, ImgSmall, etc.]
  C -- No --> E{Motion item mapping available?}
  E -- Yes --> F[Fetch Motion product page<br/>extract content.motion.com URLs]
  E -- No --> G{Other distributors allowed?}
  G -- Yes --> H[Grainger/Fastenal crawl<br/>or rendered fetch]
  G -- No --> I{Marketplace API allowed?}
  I -- Yes --> J[Amazon/eBay API image URLs]
  I -- No --> K[Manual review queue]
  D --> L[Validate + dedupe images]
  F --> L
  H --> L
  J --> L
  L --> M[Download + checksum + metadata]
  M --> N[Write back to spreadsheet<br/>image URL + provenance]
```

## Pitfalls, quality controls, and metadata schema

### Pitfalls and mitigations

**Duplicate or generic “family” images**  
Mitigation: perceptual hashing clusters; prefer manufacturer portal images over distributor family images when available.

**Placeholder / missing-image artifacts (common at scale)**  
Mitigation: maintain a rolling “known placeholder hash set” and automatically reject repeated placeholders; enforce minimum resolution thresholds.

**“Photo may not represent actual item” disclaimers**  
Motion product pages explicitly include this warning. citeturn29view0turn26search0  
Mitigation: carry an “illustrative_only” boolean in metadata and consider routing these to a secondary-source attempt (manufacturer portal) before accepting.

**Bot protection / WAF blocks**  
MSC example shows an Incapsula incident response on a product URL. citeturn55view0  
Mitigation: do not escalate with evasive tactics; switch sources or pursue authorized access.

**Robots/ToS conflicts**  
Grainger robots disallows wide crawling areas and blocks multiple bot user-agents; eBay robots prohibits automated access without permission; Motion Terms of Use prohibits automation/screen scraping without consent. citeturn39view0turn41view0turn32view0  
Mitigation: prioritize permitted sources; use official APIs (eBay Browse API; Amazon PA-API) where licensing allows. citeturn68search2turn68search0

### Reliability, resolution, and coverage expectations by source

| Source class | Reliability (correct product match) | Typical resolution expectation | Coverage likelihood |
|---|---|---|---|
| Official manufacturer catalog/CAD portals (`/Asset/...`, part-number URLs) | High when MPN matches exactly citeturn22view0turn65view0turn54search8 | Medium–high; often drawings + some photos; depends on manufacturer | High within that manufacturer; zero outside it |
| Motion product pages + Motion image CDN | High for Motion MI item matches; CDN resizing suggests flexible output widths citeturn2view0turn26search0 | Medium–high (resizable variants may exist) citeturn2view0 | High for Motion-listed SKUs |
| Grainger product pages + `static.grainger.com` image CDN | High for Grainger item matches; CDN suggests high-res options citeturn62search8turn57view0 | High (examples show ~1000px renders) citeturn62search8 | High for common MRO items |
| Fastenal (JS site + `img2.fastenal.com/productimages/...`) | Medium–high; discovery is harder if JS blocks automation citeturn43view0turn50view0 | Medium–high (varies by SKU) | Medium–high in fastener/safety/MRO categories |
| MSC Direct | Potentially high if accessible; may be blocked by bot protection citeturn55view0turn40view0 | Medium–high | Medium–high, but practically constrained |
| Amazon via PA-API | Medium (depends on ASIN/GTIN mapping quality); API returns sized images citeturn68search0 | Medium/Large variants provided citeturn68search0 | Very high catalog breadth; match quality varies |
| eBay via Browse API | Medium (listing-level variability); API returns primary + additional image URLs citeturn68search2 | Medium–high | Very high, but noisy; best as last-mile fallback |

### Recommended metadata fields to capture per image

Store enough provenance to (a) support auditability and (b) avoid rework.

| Field | Why it matters |
|---|---|
| `internal_sku` / `row_id` | Join back to the spreadsheet row deterministically |
| `normalized_mpn`, `manufacturer_name` | Drives matching + dedupe across sources |
| `source_name` and `source_type` (manufacturer / distributor / marketplace / api) | Supports trust ranking and policy decisions |
| `source_page_url` | Human-auditable provenance |
| `image_url_original` | Re-fetch reference |
| `image_url_canonical` | Normalized “best” variant (e.g., chosen size/resolution) |
| `retrieved_at_utc` | Change tracking and cache invalidation |
| `http_status`, `etag`, `last_modified` | Efficient revalidation and debugging |
| `width_px`, `height_px`, `bytes` | Quality gates and UI sizing |
| `sha256` (or similar checksum) | Exact dedupe and integrity checks |
| `phash` (perceptual hash) | Near-duplicate detection and placeholder filtering |
| `license_terms_ref` (URL or doc reference) | Connects the asset to governing terms (important when licensing is unspecified) |
| `usage_flag` (e.g., internal-only / public-ok / unknown) | Prevents accidental redistribution beyond rights |
| `notes` / `match_confidence` | Supports manual QA and continuous improvement |


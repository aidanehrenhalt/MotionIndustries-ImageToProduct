const DEFAULT_REVIEW_QUEUE_URL = "assets/data/review_queue.json";
const AUTO_REFRESH_MS = 15_000;

function normalizeConfidence(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  if (num >= 0 && num <= 1) return Math.round(num * 1000) / 10;
  return Math.round(num * 10) / 10;
}

function mapCandidateImage(img, baseUrl) {
  const imageUrl = img.url
    ? new URL(img.url, baseUrl).href
    : "";

  return {
    id: img.id || `${img.fileName}-${img.rank}`,
    fileName: img.fileName || "",
    url: imageUrl,
    thumbnailUrl: imageUrl,
    confidence: normalizeConfidence(img.aiConfidence),
    textSimilarity: normalizeConfidence(img.textScore),
    finalScore: normalizeConfidence(img.finalScore),
    finalScorePct: img.finalScorePct || "",
    aiConfidencePct: img.aiConfidencePct || "",
    textScorePct: img.textScorePct || "",
    source: img.sourceName || "",
    sourceReliability: img.aiConfidence >= 0.9 ? "High" : img.aiConfidence >= 0.7 ? "Medium" : "Low",
    title: img.fileName ? img.fileName.replace(/\.[^.]+$/, "") : "",
    rank: img.rank || 0,
  };
}

function mapProduct(product, baseUrl) {
  const candidates = (product.candidateImages || []).map((img) =>
    mapCandidateImage(img, baseUrl)
  );

  return {
    queueKey: product.queueKey || product.productId,
    itemNumber: product.productId || product.partNumber || "",
    productName: product.productName || product.productId || "",
    description: product.description || "",
    manufacturer: product.manufacturer || "",
    manufacturerPartNumber: product.manufacturerPartNumber || "",
    partNumber: product.partNumber || product.productId || "",
    category: product.category || "",
    candidateImages: candidates,
    bestConfidence: candidates[0]?.confidence || 0,
    matchedImageCount: product.matchedImageCount || candidates.length,
    jsonFileNames: product.jsonFile ? [product.jsonFile] : [],
    sourceLayout: "review-queue-json",
    attributes: {
      ...(product.slug ? { slug: product.slug } : {}),
      ...(product.sourceUrl ? { "Source URL": product.sourceUrl } : {}),
      ...(product.specs?.length ? { Specs: product.specs.map((s) => `${s.label}: ${s.value}`).join(", ") } : {}),
      ...(product.breadcrumbs?.length ? { Breadcrumbs: product.breadcrumbs.join(" > ") } : {}),
      ...(product.highlights?.length ? { Highlights: product.highlights.join(", ") } : {}),
    },
  };
}

export async function fetchReviewQueue(url) {
  const fetchUrl = url || DEFAULT_REVIEW_QUEUE_URL;
  const response = await fetch(fetchUrl);
  if (!response.ok) {
    throw new Error(`Failed to load review queue: ${response.status} ${response.statusText}`);
  }
  const payload = await response.json();
  const baseUrl = fetchUrl.replace(/[^/]*$/, "");
  const products = payload.products || [];
  return products.map((p) => mapProduct(p, baseUrl));
}

export { DEFAULT_REVIEW_QUEUE_URL, AUTO_REFRESH_MS };

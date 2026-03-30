function normalizeKey(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function findValueByAliases(record, aliases) {
  if (!record || typeof record !== "object" || Array.isArray(record)) return undefined;
  const entries = Object.entries(record);
  for (const [key, value] of entries) {
    const normalized = normalizeKey(key);
    if (aliases.some((alias) => normalized === alias || normalized.includes(alias))) return value;
  }
  return undefined;
}

function isPlainObject(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function tryParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function getFileExtension(name) {
  const parts = String(name || "").split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

function stripExtension(name) {
  return String(name || "").replace(/\.[^.]+$/, "");
}

function looksLikeImageFile(name) {
  return ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"].includes(getFileExtension(name));
}

function looksLikeJsonFile(name) {
  return getFileExtension(name) === "json";
}

function pickRelativePath(fileLike) {
  return fileLike.relativePath || fileLike.webkitRelativePath || fileLike.name || "";
}

function splitRelativePath(path) {
  return String(path || "").replace(/^\/+|\/+$/g, "").split("/").filter(Boolean);
}

function classifyRootFile(fileLike) {
  const relativePath = pickRelativePath(fileLike);
  const segments = splitRelativePath(relativePath);
  const lowered = segments.map((segment) => String(segment).toLowerCase());
  const jsonIndex = lowered.lastIndexOf("json");
  const imagesIndex = lowered.lastIndexOf("images");

  if (jsonIndex !== -1 && jsonIndex < segments.length - 1 && looksLikeJsonFile(fileLike.name)) {
    return {
      kind: "json",
      relativePath,
      rootFolder: segments.slice(0, jsonIndex).join("/"),
      subPath: segments.slice(jsonIndex + 1).join("/"),
    };
  }

  if (imagesIndex !== -1 && imagesIndex < segments.length - 1 && looksLikeImageFile(fileLike.name)) {
    return {
      kind: "image",
      relativePath,
      rootFolder: segments.slice(0, imagesIndex).join("/"),
      subPath: segments.slice(imagesIndex + 1).join("/"),
    };
  }

  return {
    kind: "other",
    relativePath,
    rootFolder: segments.slice(0, -1).join("/"),
    subPath: segments.slice(-1).join("/"),
  };
}

function flattenForDisplay(value, prefix = "", output = {}) {
  if (value == null) return output;
  if (Array.isArray(value)) {
    if (value.length > 0 && value.every((item) => typeof item !== "object" || item == null)) {
      output[prefix || "value"] = value.join(", ");
    }
    return output;
  }
  if (!isPlainObject(value)) {
    output[prefix || "value"] = String(value);
    return output;
  }

  Object.entries(value).forEach(([key, child]) => {
    const nextPrefix = prefix ? `${prefix} · ${key}` : key;
    if (child == null || child === "") return;
    if (isPlainObject(child)) {
      flattenForDisplay(child, nextPrefix, output);
      return;
    }
    if (Array.isArray(child)) {
      if (child.length === 0) return;
      if (child.every((item) => typeof item !== "object" || item == null)) {
        output[nextPrefix] = child.join(", ");
      }
      return;
    }
    output[nextPrefix] = String(child);
  });
  return output;
}

function collectImageHints(value, results = []) {
  if (Array.isArray(value)) {
    value.forEach((item) => collectImageHints(item, results));
    return results;
  }
  if (!isPlainObject(value)) return results;

  const maybeImageRef =
    findValueByAliases(value, ["filename", "filepath", "imagefile", "imagefilename", "name"]) ||
    findValueByAliases(value, ["imageurl", "url", "imagelink", "thumbnailurl", "src"]);

  if (maybeImageRef) {
    results.push({
      ref: String(maybeImageRef),
      confidence: findValueByAliases(value, ["confidence", "score", "rankscore", "matchscore", "aiscore"]),
      textSimilarity: findValueByAliases(value, ["textsimilarity", "textsim", "textscore", "similarity"]),
      imageQuality: findValueByAliases(value, ["imagequality", "quality", "qualityscore"]),
      source: findValueByAliases(value, ["source", "domain", "website", "origin"]),
      sourceReliability: findValueByAliases(value, ["reliability", "trust", "sourcereliability"]),
      title: findValueByAliases(value, ["title", "label", "caption"]),
    });
  }

  Object.values(value).forEach((child) => collectImageHints(child, results));
  return results;
}

function normalizeScore(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  if (num >= 0 && num <= 1) return Math.round(num * 1000) / 10;
  return Math.round(num * 10) / 10;
}

function normalizeReliability(value) {
  const text = String(value || "").trim();
  if (!text) return "Medium";
  const lower = text.toLowerCase();
  if (lower.includes("high")) return "High";
  if (lower.includes("low")) return "Low";
  return "Medium";
}

function addKeyVariants(target, rawValue) {
  const input = String(rawValue || "").trim();
  if (!input) return;

  const values = new Set();
  values.add(input);
  values.add(stripExtension(input));

  const lastSegment = stripExtension(input.split(/[\\/]/).pop() || "");
  if (lastSegment) values.add(lastSegment);

  const normalized = normalizeKey(lastSegment || input);
  if (normalized) values.add(normalized);

  const stripped = String(lastSegment || input)
    .toLowerCase()
    .replace(/(?:[_\-\s]?)(image|img|candidate|rank|result|photo|pic|scrape|scraped|output|main|primary|front|hero|thumb|thumbnail|match)+[_\-\s]*\d*$/g, "")
    .replace(/[_\-\s]*\d+$/g, "")
    .trim();
  if (stripped) {
    values.add(stripped);
    const strippedNormalized = normalizeKey(stripped);
    if (strippedNormalized) values.add(strippedNormalized);
  }

  const tokens = String(lastSegment || input).split(/[^a-zA-Z0-9]+/).filter(Boolean);
  tokens.forEach((token) => values.add(token));

  values.forEach((value) => {
    const text = String(value || "").trim().toLowerCase();
    if (!text) return;
    const compact = normalizeKey(text);
    if ((compact.length >= 4) || /^\d{4,}$/.test(compact)) target.add(compact);
  });
}

function createMatchKeySet(values) {
  const keys = new Set();
  values.forEach((value) => addKeyVariants(keys, value));
  return keys;
}

function mergeMetadata(jsonObjects) {
  const merged = {};
  jsonObjects.forEach((obj) => {
    if (!obj) return;
    if (Array.isArray(obj)) {
      const firstObject = obj.find((item) => isPlainObject(item));
      if (firstObject) Object.assign(merged, firstObject);
      return;
    }
    if (isPlainObject(obj)) Object.assign(merged, obj);
  });
  return merged;
}

function extractProductFields(metadata, fallbackItemNumber) {
  const itemNumber = String(
    findValueByAliases(metadata, ["itemnumber", "itemno", "motionnumber", "motionindustriesnumber", "productid", "minumber"]) ||
    fallbackItemNumber ||
    ""
  ).trim();

  const productName = String(
    findValueByAliases(metadata, ["productname", "name", "title", "itemname", "partname"]) ||
    itemNumber ||
    ""
  ).trim();

  const description = String(
    findValueByAliases(metadata, ["description", "desc", "details", "productdescription", "longdescription"]) ||
    ""
  ).trim();

  const manufacturer = String(
    findValueByAliases(metadata, ["manufacturer", "mfr", "brand", "vendor"]) ||
    ""
  ).trim();

  const manufacturerPartNumber = String(
    findValueByAliases(metadata, ["manufacturerpartnumber", "mfrpartnumber", "mfrpartno", "mfgpartnumber"]) ||
    ""
  ).trim();

  const partNumber = String(
    findValueByAliases(metadata, ["partnumber", "partno", "part", "sku", "pn"]) ||
    manufacturerPartNumber ||
    itemNumber ||
    ""
  ).trim();

  const category = String(
    findValueByAliases(metadata, ["category", "class", "type", "group", "series", "productgroup"]) ||
    ""
  ).trim();

  return { itemNumber, productName, description, manufacturer, manufacturerPartNumber, partNumber, category };
}

function buildImageEntry(fileLike, classification) {
  const relativePath = classification.relativePath;
  const subPath = classification.subPath || fileLike.name;
  const basename = stripExtension(fileLike.name);
  const pathSegments = splitRelativePath(subPath);

  const matchKeys = createMatchKeySet([
    fileLike.name,
    basename,
    relativePath,
    subPath,
    ...pathSegments,
  ]);

  return {
    fileLike,
    relativePath,
    subPath,
    basename,
    compactBase: normalizeKey(basename),
    compactPath: normalizeKey(relativePath),
    matchKeys,
  };
}

function buildProductSeed(parsed, fileLike, classification) {
  const metadata = mergeMetadata([parsed]);
  const fields = extractProductFields(metadata, stripExtension(fileLike.name));
  const imageHints = collectImageHints(metadata);

  const explicitImageRefs = imageHints.map((hint) => hint.ref).filter(Boolean);
  const matchKeys = createMatchKeySet([
    fields.itemNumber,
    fields.partNumber,
    fields.manufacturerPartNumber,
    fields.productName,
    stripExtension(fileLike.name),
    classification.subPath,
    ...explicitImageRefs,
  ]);

  const primaryKey =
    normalizeKey(fields.itemNumber) ||
    normalizeKey(fields.partNumber) ||
    normalizeKey(fields.manufacturerPartNumber) ||
    normalizeKey(stripExtension(fileLike.name));

  const queueKey = normalizeKey(classification.subPath || pickRelativePath(fileLike) || fileLike.name) || `${primaryKey}-${Date.now()}`;

  return {
    queueKey,
    primaryKey,
    metadataList: [parsed],
    jsonFiles: [fileLike],
    imageHints,
    matchKeys,
    explicitImageRefs,
  };
}

function calculateMatchScore(productSeed, imageEntry) {
  let score = 0;

  if (productSeed.primaryKey && imageEntry.compactBase === productSeed.primaryKey) score += 120;

  productSeed.explicitImageRefs.forEach((ref) => {
    const refKey = normalizeKey(stripExtension(String(ref).split(/[\\/]/).pop() || ref));
    if (!refKey) return;
    if (refKey === imageEntry.compactBase) score += 200;
    else if (imageEntry.matchKeys.has(refKey)) score += 140;
  });

  productSeed.matchKeys.forEach((key) => {
    if (imageEntry.matchKeys.has(key)) score += 40;
  });

  if (productSeed.primaryKey && imageEntry.compactPath.includes(productSeed.primaryKey)) score += 30;

  return score;
}

function findHintForImage(imageHints, imageEntry) {
  let bestHint = null;
  let bestScore = 0;

  imageHints.forEach((hint) => {
    const refName = String(hint.ref || "").split(/[\/]/).pop() || "";
    const exactRef = normalizeKey(stripExtension(refName));
    const genericRef = normalizeKey(stripExtension(refName).replace(/(?:[_\-\s]?)(image|img|candidate|rank|result|photo|pic|scrape|scraped|output|main|primary|front|hero|thumb|thumbnail)+[_\-\s]*\d*$/g, "").replace(/[_\-\s]*\d+$/g, ""));
    const hasSpecificFileReference = !!(exactRef && genericRef && exactRef !== genericRef);

    let score = 0;
    if (exactRef && exactRef === imageEntry.compactBase) score += 100;
    if (!hasSpecificFileReference && genericRef && genericRef !== exactRef && imageEntry.matchKeys.has(genericRef)) score += 10;

    if (score > bestScore) {
      bestScore = score;
      bestHint = hint;
    }
  });

  return bestScore > 0 ? bestHint : null;
}

function createProductImage(fileLike, hint, urlFactory, rank, productKey) {
  const objectUrl = urlFactory(fileLike);
  const confidence = normalizeScore(hint?.confidence);
  const textSimilarity = Number(hint?.textSimilarity);
  const imageQuality = Number(hint?.imageQuality);
  return {
    id: `${productKey || "product"}-${fileLike.name}-${rank}`,
    fileName: fileLike.name,
    url: objectUrl,
    thumbnailUrl: objectUrl,
    confidence,
    textSimilarity: Number.isFinite(textSimilarity) ? Math.round(textSimilarity * 100) / 100 : 0,
    imageQuality: Number.isFinite(imageQuality) ? Math.round(imageQuality * 10) / 10 : 0,
    source: hint?.source ? String(hint.source) : "Images folder",
    sourceReliability: normalizeReliability(hint?.sourceReliability),
    title: hint?.title ? String(hint.title) : stripExtension(fileLike.name),
    rank,
  };
}

function buildProduct(seed, assignedImages, urlFactory) {
  const metadata = mergeMetadata(seed.metadataList);
  const fields = extractProductFields(metadata, stripExtension(seed.jsonFiles[0]?.name || seed.primaryKey || "PART"));
  const flattened = flattenForDisplay(metadata);

  const images = assignedImages
    .map((imageEntry) => createProductImage(imageEntry.fileLike, findHintForImage(seed.imageHints, imageEntry), urlFactory, 0, seed.queueKey || seed.primaryKey))
    .sort((a, b) => {
      if (b.confidence !== a.confidence) return b.confidence - a.confidence;
      return a.fileName.localeCompare(b.fileName);
    })
    .map((image, index) => ({ ...image, rank: index + 1 }));

  return {
    queueKey: seed.queueKey,
    ...fields,
    candidateImages: images,
    bestConfidence: images[0]?.confidence || 0,
    attributes: flattened,
    jsonFileNames: seed.jsonFiles.map((file) => file.name),
    jsonRelativePaths: seed.jsonFiles.map((file) => pickRelativePath(file)),
    matchedImageCount: images.length,
    sourceLayout: "json-images-root",
    updatedAt: new Date().toISOString(),
  };
}

export async function parsePipelineFolderFiles(files, urlFactory) {
  const productSeeds = [];
  const images = [];
  let foundJsonFolderFile = false;
  let foundImagesFolderFile = false;

  for (const fileLike of files) {
    const classification = classifyRootFile(fileLike);
    fileLike.relativePath = classification.relativePath;

    if (classification.kind === "json") {
      foundJsonFolderFile = true;
      const parsed = tryParseJson(await fileLike.text());
      if (!parsed) continue;

      const seed = buildProductSeed(parsed, fileLike, classification);
      if (!seed.primaryKey && !seed.queueKey) continue;
      productSeeds.push(seed);
      continue;
    }

    if (classification.kind === "image") {
      foundImagesFolderFile = true;
      images.push(buildImageEntry(fileLike, classification));
    }
  }

  if (!foundJsonFolderFile) {
    throw new Error("No JSON files were found inside a /json folder. Select the root output folder that contains both /json and /images.");
  }

  if (!foundImagesFolderFile) {
    throw new Error("No image files were found inside an /images folder. Select the root output folder that contains both /json and /images.");
  }

  const seeds = Array.from(productSeeds);
  const assignedImages = new Map(seeds.map((seed) => [seed.queueKey || seed.primaryKey, []]));

  images.forEach((imageEntry) => {
    let bestSeed = null;
    let bestScore = 0;

    seeds.forEach((seed) => {
      const score = calculateMatchScore(seed, imageEntry);
      if (score > bestScore) {
        bestScore = score;
        bestSeed = seed;
      }
    });

    if (bestSeed && bestScore > 0) assignedImages.get(bestSeed.queueKey || bestSeed.primaryKey).push(imageEntry);
  });

  return seeds
    .map((seed) => buildProduct(seed, assignedImages.get(seed.queueKey || seed.primaryKey) || [], urlFactory))
    .filter((product) => product.itemNumber || product.productName || product.partNumber)
    .sort((a, b) => String(a.itemNumber || a.partNumber || a.productName).localeCompare(String(b.itemNumber || b.partNumber || b.productName), undefined, { numeric: true, sensitivity: "base" }));
}

async function readDirectoryEntries(handle, basePath = "") {
  const collected = [];
  for await (const [name, entry] of handle.entries()) {
    const nextPath = basePath ? `${basePath}/${name}` : name;
    if (entry.kind === "directory") {
      const childFiles = await readDirectoryEntries(entry, nextPath);
      collected.push(...childFiles);
      continue;
    }
    const file = await entry.getFile();
    file.relativePath = nextPath;
    collected.push(file);
  }
  return collected;
}

export async function readFilesFromDirectoryHandle(handle) {
  return readDirectoryEntries(handle, "");
}

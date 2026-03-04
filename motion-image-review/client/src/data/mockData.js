const SOURCES = [
  { domain: "skf.com", reliability: "High" },
  { domain: "mcmaster.com", reliability: "High" },
  { domain: "amazon.com", reliability: "Medium" },
  { domain: "ebay.com", reliability: "Low" },
  { domain: "grainger.com", reliability: "High" },
];

export const REJECT_REASONS = [
  "Wrong product",
  "Low quality",
  "Watermarked",
  "Competitor logo",
  "Irrelevant",
  "Duplicate",
];

export const ACCEPT_REASONS = [
  "Exact match",
  "Close enough",
  "Best available",
];

function generateCandidateImages(productName, count) {
  const images = [];
  const shortName = productName.split(" ").slice(0, 3).join("+");

  for (let i = 0; i < count; i++) {
    const conf = Math.max(40, Math.min(99, 95 - i * (8 + Math.random() * 7) + (Math.random() * 5 - 2.5)));
    const textSim = Math.max(0.25, Math.min(0.99, (conf / 100) * 0.95 + (Math.random() * 0.15 - 0.075)));
    const imgQuality = Math.max(3, Math.min(9.8, 8.5 - i * 0.6 + (Math.random() * 1.5 - 0.75)));
    const source = SOURCES[Math.floor(Math.random() * SOURCES.length)];
    const dims = [400, 500, 600][Math.floor(Math.random() * 3)];

    images.push({
      id: `img-${Date.now()}-${i}-${Math.random().toString(36).substr(2, 5)}`,
      url: `https://placehold.co/${dims}x${dims}/e8eef2/1a1f36?text=${shortName}+${i + 1}`,
      thumbnailUrl: `https://placehold.co/120x120/e8eef2/1a1f36?text=${shortName}+${i + 1}`,
      confidence: Math.round(conf * 10) / 10,
      textSimilarity: Math.round(textSim * 100) / 100,
      imageQuality: Math.round(imgQuality * 10) / 10,
      source: source.domain,
      sourceReliability: source.reliability,
      rank: i + 1,
    });
  }
  return images;
}

const PRODUCT_TEMPLATES = [
  { name: "SKF 6205-2RSH Deep Groove Ball Bearing", mfr: "SKF", partNum: "6205-2RSH", cat: "Bearings", desc: "Single row deep groove ball bearing with contact seals on both sides. 25mm bore, 52mm OD, 15mm width. Suitable for high-speed applications with moderate radial loads." },
  { name: "Gates 8PK1700 Serpentine Belt", mfr: "Gates", partNum: "8PK1700", cat: "Belts & Pulleys", desc: "Micro-V serpentine belt, 8 ribs, 1700mm effective length. EPDM rubber compound for extended service life. Fits various automotive and industrial applications." },
  { name: "Parker 10143-8-8 Hydraulic Fitting", mfr: "Parker", partNum: "10143-8-8", cat: "Hydraulic Fittings", desc: 'Male pipe swivel 90° elbow fitting. 1/2" tube OD x 1/2" male NPT. Steel with zinc-nickel plating for corrosion resistance.' },
  { name: "Timken 32210 Tapered Roller Bearing", mfr: "Timken", partNum: "32210", cat: "Bearings", desc: "Single row tapered roller bearing. 50mm bore, 90mm OD, 24.75mm width. Designed for combined radial and axial loads in heavy-duty industrial equipment." },
  { name: "Rexnord BS226839 Roller Chain", mfr: "Rexnord", partNum: "BS226839", cat: "Gear Drives", desc: "ANSI #60 single strand roller chain, 10ft box. 3/4\" pitch, 0.469\" roller diameter. Riveted with solid rollers for smooth operation." },
  { name: "Dodge P2B-IP-207RE Pillow Block", mfr: "Dodge", partNum: "P2B-IP-207RE", cat: "Bearings", desc: "Type E pillow block bearing unit. 1-7/16\" bore, cast iron housing with set screw locking collar. Relubricatable with standard grease fitting." },
  { name: "Baldor EM3611T Electric Motor", mfr: "Baldor", partNum: "EM3611T", cat: "Electrical Motors", desc: "3HP, 1760 RPM, 3-phase, 182T frame TEFC electric motor. 208-230/460V, 60Hz. Premium efficient design meets NEMA Premium specification." },
  { name: "Festo DSBC-50-200-PPVA-N3 Cylinder", mfr: "Festo", partNum: "DSBC-50-200", cat: "Pneumatic Cylinders", desc: "ISO 15552 pneumatic cylinder. 50mm bore, 200mm stroke. Double-acting with adjustable cushioning at both ends." },
  { name: "Bando 5VX1000 V-Belt", mfr: "Bando", partNum: "5VX1000", cat: "Belts & Pulleys", desc: "Power King Cog belt, 5VX cross section. 100\" outside length. Cogged design for improved flexibility and heat dissipation." },
  { name: "Martin 50BS18HT Sprocket", mfr: "Martin Sprocket", partNum: "50BS18HT", cat: "Gear Drives", desc: "Type B sprocket for #50 chain. 18 teeth, 1\" bore with keyway. Hardened teeth for extended wear life." },
];

export function generateQueueData() {
  return PRODUCT_TEMPLATES.map((p, i) => ({
    itemNumber: `MI-${(100000 + i * 1337 + 42).toString().slice(0, 6)}`,
    productName: p.name,
    partNumber: `PN-${(20000 + i * 731).toString()}`,
    manufacturer: p.mfr,
    manufacturerPartNumber: p.partNum,
    description: p.desc,
    category: p.cat,
    candidateImages: generateCandidateImages(p.name, 4 + Math.floor(Math.random() * 4)),
  }));
}

export function generateHistoryData() {
  const decisions = ["accepted", "rejected", "skipped"];
  const items = [];

  for (let i = 0; i < 20; i++) {
    const tmpl = PRODUCT_TEMPLATES[i % PRODUCT_TEMPLATES.length];
    const decision = decisions[Math.floor(Math.random() * 3)];
    const conf = Math.round((65 + Math.random() * 33) * 10) / 10;
    const daysAgo = Math.floor(Math.random() * 30);
    let feedback = "";
    let feedbackTags = [];

    if (decision === "accepted") {
      feedbackTags = [ACCEPT_REASONS[Math.floor(Math.random() * ACCEPT_REASONS.length)]];
      if (Math.random() > 0.5) feedback = "Image matches catalog spec. Good resolution.";
    } else if (decision === "rejected") {
      feedbackTags = [REJECT_REASONS[Math.floor(Math.random() * REJECT_REASONS.length)]];
      if (Math.random() > 0.4) feedback = "Image shows different variant of the product.";
    }

    const shortName = tmpl.name.split(" ").slice(0, 3).join("+");
    items.push({
      itemNumber: `MI-${(200000 + i * 997).toString().slice(0, 6)}`,
      productName: tmpl.name,
      partNumber: `PN-${(40000 + i * 503).toString()}`,
      manufacturer: tmpl.mfr,
      manufacturerPartNumber: tmpl.partNum,
      description: tmpl.desc,
      category: tmpl.cat,
      decision,
      confidence: conf,
      feedback,
      feedbackTags,
      reviewedAt: new Date(Date.now() - daysAgo * 86400000).toISOString(),
      imageUrl: `https://placehold.co/120x120/e8eef2/1a1f36?text=${shortName}`,
      candidateImages: generateCandidateImages(tmpl.name, 4 + Math.floor(Math.random() * 3)),
    });
  }

  return items.sort((a, b) => new Date(b.reviewedAt) - new Date(a.reviewedAt));
}

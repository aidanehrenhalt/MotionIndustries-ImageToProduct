let XLSX = null;

export function loadXLSX() {
  return new Promise((resolve, reject) => {
    if (XLSX) { resolve(XLSX); return; }
    const script = document.createElement("script");
    script.src = "https://cdn.sheetjs.com/xlsx-0.20.1/package/dist/xlsx.full.min.js";
    script.onload = () => { XLSX = window.XLSX; resolve(XLSX); };
    script.onerror = () => reject(new Error("Failed to load SheetJS library"));
    document.head.appendChild(script);
  });
}

// Parse any supported file → { headers: string[], rows: any[][] }
export async function parseFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();

  if (ext === "json") {
    const text = await file.text();
    const data = JSON.parse(text);
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && !Array.isArray(data[0])) {
      const headers = Object.keys(data[0]);
      const rows = data.map((item) => headers.map((h) => item[h] ?? ""));
      return { headers, rows };
    }
    if (Array.isArray(data) && Array.isArray(data[0])) {
      return { headers: data[0].map(String), rows: data.slice(1) };
    }
    throw new Error("JSON must be an array of objects or array of arrays");
  }

  const xlsx = await loadXLSX();
  let workbook;
  if (ext === "csv") {
    const text = await file.text();
    workbook = xlsx.read(text, { type: "string" });
  } else {
    const buffer = await file.arrayBuffer();
    workbook = xlsx.read(buffer, { type: "array" });
  }
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  const raw = xlsx.utils.sheet_to_json(sheet, { header: 1, defval: "" });
  if (raw.length === 0) throw new Error("File is empty");
  return { headers: raw[0].map(String), rows: raw.slice(1) };
}

export function validateFileType(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["xlsx", "xls", "csv", "json"].includes(ext)) {
    return "Unsupported file type. Upload .xlsx, .xls, .csv, or .json";
  }
  return null;
}

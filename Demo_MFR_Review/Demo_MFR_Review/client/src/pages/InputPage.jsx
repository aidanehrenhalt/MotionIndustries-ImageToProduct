import React, { useState, useCallback, useMemo } from "react";
import { COLORS, fontStack, monoStack, Icons } from "../components/Layout";
import { parseFile, validateFileType, loadXLSX } from "../utils/fileParser";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:5050";

export default function InputPage({ rawHeaders, rawRows, onLoadExcel }) {
  const [step, setStep] = useState(rawRows.length > 0 ? "view" : "upload");
  const [fileHeaders, setFileHeaders] = useState(rawHeaders || []);
  const [fileRows, setFileRows] = useState(rawRows || []);
  const [fileName, setFileName] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [search, setSearch] = useState("");

  // Pipeline submission state
  const [jobId, setJobId] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null); // null | "uploading" | "running" | "done" | "error"
  const [pipelineLog, setPipelineLog] = useState([]);
  const [pipelineError, setPipelineError] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);
  const [limitValue, setLimitValue] = useState("");

  const processFile = useCallback(async (file) => {
    if (!file) return;
    const typeError = validateFileType(file);
    if (typeError) { setError(typeError); return; }
    setLoading(true); setError(null);
    try {
      const { headers, rows } = await parseFile(file);
      if (rows.length === 0) { setError("File has headers but no data rows."); setLoading(false); return; }
      setFileHeaders(headers);
      setFileRows(rows);
      setFileName(file.name);
      setCurrentFile(file);
      onLoadExcel(headers, rows);
      setStep("view");
      // Reset any previous pipeline run state when a new file is loaded
      setJobId(null);
      setPipelineStatus(null);
      setPipelineLog([]);
      setPipelineError(null);
    } catch (err) {
      setError("Failed to read file."); console.error(err);
    } finally { setLoading(false); }
  }, [onLoadExcel]);

  const handleDragOver = useCallback((e) => { e.preventDefault(); setDragging(true); }, []);
  const handleDragLeave = useCallback((e) => { e.preventDefault(); setDragging(false); }, []);
  const handleDrop = useCallback((e) => { e.preventDefault(); setDragging(false); processFile(e.dataTransfer.files[0]); }, [processFile]);

  const displayHeaders = step === "view" ? (rawHeaders.length > 0 ? rawHeaders : fileHeaders) : fileHeaders;
  const displayRows = step === "view" ? (rawRows.length > 0 ? rawRows : fileRows) : fileRows;

  const filteredRows = useMemo(() => {
    if (!search) return displayRows;
    const q = search.toLowerCase();
    return displayRows.filter((row) => row.some((cell) => String(cell).toLowerCase().includes(q)));
  }, [displayRows, search]);

  const handleDownload = useCallback(async () => {
    try {
      const xlsx = await loadXLSX();
      const ws = xlsx.utils.aoa_to_sheet([displayHeaders, ...displayRows]);
      const wb = xlsx.utils.book_new();
      xlsx.utils.book_append_sheet(wb, ws, "Products");
      xlsx.writeFile(wb, "product_database.xlsx");
    } catch (err) { console.error(err); }
  }, [displayHeaders, displayRows]);

  // ── Pipeline submission ───────────────────────────────────────────────────

  const pollStatus = useCallback(async (id) => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/status/${id}`);
        const data = await res.json();
        setPipelineLog(data.log || []);
        if (data.status === "done") {
          setPipelineStatus("done");
          return;
        }
        if (data.status === "error") {
          setPipelineStatus("error");
          setPipelineError(data.error || "Unknown pipeline error.");
          return;
        }
        setTimeout(poll, 2500);
      } catch {
        setPipelineStatus("error");
        setPipelineError("Lost connection to API server. Is it running?");
      }
    };
    poll();
  }, []);

  const handleRunPipeline = useCallback(async () => {
    if (!currentFile) return;
    setPipelineStatus("uploading");
    setPipelineError(null);
    setPipelineLog([]);
    setJobId(null);

    try {
      // Step 1: upload file to backend
      const formData = new FormData();
      formData.append("file", currentFile);
      console.info("[upload-trace] file metadata", {
        name: currentFile.name,
        type: currentFile.type,
        size: currentFile.size,
        lastModified: currentFile.lastModified,
      });
      
      const uploadRes = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const uploadText = await uploadRes.text();
      let uploadData = null;
      try {
        uploadData = uploadText ? JSON.parse(uploadText) : {};
      } catch (parseErr) {
        console.error("[upload-trace] non-JSON upload response", {
          status: uploadRes.status,
          statusText: uploadRes.statusText,
          contentType: uploadRes.headers.get("content-type"),
          body: uploadText,
          error: String(parseErr),
        });
        throw new Error(`Upload returned a non-JSON response (${uploadRes.status})`);
      }
      if (!uploadRes.ok) {
        throw new Error(uploadData.error || "Upload failed");
      }

      const id = uploadData.jobId;
      setJobId(id);

      if (uploadData.missingRequired?.length > 0) {
        console.warn("Missing recommended columns:", uploadData.missingRequired);
      }

      // Step 2: trigger pipeline
      setPipelineStatus("running");
      const runBody = { jobId: id };
      if (limitValue && !isNaN(Number(limitValue))) {
        runBody.limit = Number(limitValue);
      }
      const runRes = await fetch(`${API_BASE}/api/pipeline/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runBody),
      });
      const runData = await runRes.json();
      if (!runRes.ok) {
        throw new Error(runData.error || "Failed to start pipeline");
      }

      // Step 3: poll for completion
      pollStatus(id);
    } catch (err) {
      setPipelineStatus("error");
      setPipelineError(err.message || "Unexpected error");
    }
  }, [currentFile, limitValue, pollStatus]);

  // ── Upload screen ─────────────────────────────────────────────────────────

  if (step === "upload") {
    return (
      <div style={{ fontFamily: fontStack }}>
        <div
          onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
          onClick={() => document.getElementById("product-excel-upload").click()}
          style={{
            background: dragging ? COLORS.bluePale : COLORS.white,
            border: `2px dashed ${dragging ? COLORS.blue : COLORS.g300}`,
            borderRadius: 16, padding: "80px 40px", textAlign: "center",
            cursor: "pointer", transition: "all 0.2s ease",
          }}
        >
          <input id="product-excel-upload" type="file" accept=".xlsx,.xls,.csv,.json" onChange={(e) => { processFile(e.target.files[0]); e.target.value = ""; }} style={{ display: "none" }} />
          <div style={{ color: dragging ? COLORS.blue : COLORS.g400, marginBottom: 16 }}><Icons.Upload /></div>
          <div style={{ fontSize: 16, fontWeight: 600, color: COLORS.g800, marginBottom: 8 }}>
            {dragging ? "Drop your file here" : "Upload Product Excel"}
          </div>
          <div style={{ fontSize: 13, color: COLORS.g500, marginBottom: 6 }}>
            This file is the product database — it feeds into the web scraper pipeline.
          </div>
          <div style={{ fontSize: 12, color: COLORS.g400, marginBottom: 20 }}>
            Supports .xlsx, .xls, .csv, and .json
          </div>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: COLORS.blue, color: "#fff", padding: "10px 24px", borderRadius: 10, fontSize: 14, fontWeight: 600 }}>
            <Icons.Upload /> Choose File
          </div>
          {loading && <div style={{ marginTop: 20, color: COLORS.blue, fontSize: 14 }}>Reading file...</div>}
          {error && <div style={{ marginTop: 20, color: COLORS.red, fontSize: 13 }}>{error}</div>}
        </div>
      </div>
    );
  }

  // ── Database + pipeline view ───────────────────────────────────────────────

  const isPipelineRunning = pipelineStatus === "uploading" || pipelineStatus === "running";
  const reviewQueueUrl = jobId ? `${API_BASE}/api/review-queue/${jobId}` : null;

  return (
    <div style={{ fontFamily: fontStack }}>
      {/* Toolbar */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
        <div style={{ position: "relative", flex: "1 1 200px", maxWidth: 360 }}>
          <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: COLORS.g400 }}><Icons.Search /></span>
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search products..." style={{ width: "100%", padding: "9px 12px 9px 36px", borderRadius: 8, border: `1px solid ${COLORS.g200}`, fontSize: 13, fontFamily: fontStack, outline: "none", boxSizing: "border-box" }} />
        </div>
        <span style={{ fontSize: 12, color: COLORS.g400 }}>{displayRows.length} row{displayRows.length !== 1 ? "s" : ""} · {displayHeaders.length} columns</span>
        <div style={{ flex: 1 }} />
        <button onClick={handleDownload} style={{ background: COLORS.blue, color: "#fff", border: "none", borderRadius: 8, padding: "9px 18px", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: fontStack }}>Export Excel</button>
        <button onClick={() => setStep("upload")} style={{ background: COLORS.g100, color: COLORS.g600, border: `1px solid ${COLORS.g300}`, borderRadius: 8, padding: "9px 18px", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: fontStack }}>Upload New File</button>
      </div>

      {/* Info banner */}
      <div style={{ background: COLORS.bluePale, border: `1px solid ${COLORS.blue}30`, borderRadius: 10, padding: "12px 20px", marginBottom: 16, fontSize: 13, color: COLORS.blue }}>
        Product database loaded: <strong>{fileName}</strong> — {displayRows.length} row{displayRows.length !== 1 ? "s" : ""}.
        Use <strong>Run Pipeline</strong> below to scrape, classify, and rank images for these products.
      </div>

      {/* Pipeline trigger panel */}
      <div style={{ background: COLORS.white, border: `1px solid ${COLORS.g200}`, borderRadius: 12, padding: "18px 20px", marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.g900, marginBottom: 10 }}>
          Run Pipeline
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label style={{ fontSize: 12, color: COLORS.g600 }}>Limit rows (optional):</label>
            <input
              type="number"
              min="1"
              value={limitValue}
              onChange={(e) => setLimitValue(e.target.value)}
              placeholder="all"
              style={{ width: 80, padding: "6px 10px", borderRadius: 6, border: `1px solid ${COLORS.g300}`, fontSize: 13, fontFamily: fontStack }}
              disabled={isPipelineRunning}
            />
          </div>
          <button
            onClick={handleRunPipeline}
            disabled={isPipelineRunning || !currentFile}
            style={{
              background: isPipelineRunning ? COLORS.g300 : COLORS.blue,
              color: "#fff", border: "none", borderRadius: 8,
              padding: "10px 22px", fontSize: 14, fontWeight: 700,
              cursor: isPipelineRunning ? "default" : "pointer",
              fontFamily: fontStack,
            }}
          >
            {pipelineStatus === "uploading" ? "Uploading..." : pipelineStatus === "running" ? "Pipeline running..." : "Run Pipeline"}
          </button>

          {pipelineStatus === "done" && reviewQueueUrl && (
            <div style={{ fontSize: 13, color: COLORS.g600 }}>
              Done — copy this URL into the <strong>Output UI</strong> tab:
              <code style={{ marginLeft: 8, padding: "2px 8px", background: COLORS.g100, borderRadius: 4, fontSize: 12 }}>
                {reviewQueueUrl}
              </code>
            </div>
          )}
        </div>

        {/* Pipeline log */}
        {pipelineLog.length > 0 && (
          <div style={{ marginTop: 12, background: COLORS.g50, borderRadius: 8, padding: "10px 14px", fontSize: 12, fontFamily: monoStack, color: COLORS.g700, maxHeight: 180, overflowY: "auto" }}>
            {pipelineLog.map((line, i) => (
              <div key={i} style={{ marginBottom: 2 }}>{line}</div>
            ))}
          </div>
        )}

        {pipelineError && (
          <div style={{ marginTop: 10, color: COLORS.red, fontSize: 13 }}>
            Error: {pipelineError}
          </div>
        )}

        {pipelineStatus === "done" && (
          <div style={{ marginTop: 10, color: "#15803d", fontSize: 13, fontWeight: 600 }}>
            Pipeline complete. Go to the <strong>Output UI</strong> tab and paste the review queue URL above.
          </div>
        )}
      </div>

      {/* Data table */}
      <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, overflow: "hidden" }}>
        <div style={{ overflowX: "auto", maxHeight: "55vh", overflowY: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, fontFamily: fontStack }}>
            <thead>
              <tr style={{ background: COLORS.g50, position: "sticky", top: 0, zIndex: 2 }}>
                <th style={{ padding: "10px 12px", textAlign: "center", fontSize: 11, color: COLORS.g400, fontWeight: 600, borderBottom: `1px solid ${COLORS.g200}`, borderRight: `1px solid ${COLORS.g200}`, background: COLORS.g50, width: 50 }}>#</th>
                {displayHeaders.map((h, i) => (
                  <th key={i} style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, color: COLORS.g700, fontWeight: 600, borderBottom: `1px solid ${COLORS.g200}`, background: COLORS.g50, whiteSpace: "nowrap", minWidth: 100 }}>
                    {h || `Column ${i + 1}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row, ri) => (
                <tr key={ri} style={{ borderBottom: `1px solid ${COLORS.g100}` }}>
                  <td style={{ padding: "8px 12px", textAlign: "center", fontSize: 11, color: COLORS.g400, fontFamily: monoStack, borderRight: `1px solid ${COLORS.g200}`, background: COLORS.g50 }}>{ri + 1}</td>
                  {displayHeaders.map((_, ci) => (
                    <td key={ci} style={{ padding: "8px 14px", color: COLORS.g800, maxWidth: 250, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={String(row[ci] ?? "")}>
                      {String(row[ci] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
              {filteredRows.length === 0 && (
                <tr><td colSpan={displayHeaders.length + 1} style={{ padding: 40, textAlign: "center", color: COLORS.g400 }}>No rows match</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useMemo } from "react";
import { COLORS, fontStack, monoStack, Icons, confidenceColor, decisionBadge, formatDate } from "./Layout";

export default function HistoryTable({ history, onSelectItem }) {
  const [search, setSearch] = useState("");
  const [filterDecision, setFilterDecision] = useState("all");
  const [sortField, setSortField] = useState("reviewedAt");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);
  const perPage = 12;

  const filtered = useMemo(() => {
    let items = [...history];
    if (search) {
      const q = search.toLowerCase();
      items = items.filter((h) =>
        h.productName.toLowerCase().includes(q) ||
        h.itemNumber.toLowerCase().includes(q) ||
        h.partNumber.toLowerCase().includes(q)
      );
    }
    if (filterDecision !== "all") {
      items = items.filter((h) => h.decision === filterDecision);
    }
    items.sort((a, b) => {
      let av = a[sortField], bv = b[sortField];
      if (sortField === "reviewedAt") { av = new Date(av); bv = new Date(bv); }
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return items;
  }, [history, search, filterDecision, sortField, sortDir]);

  const pageCount = Math.ceil(filtered.length / perPage);
  const pageItems = filtered.slice(page * perPage, (page + 1) * perPage);

  const handleSort = (field) => {
    if (sortField === field) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortField(field); setSortDir("desc"); }
  };

  const columns = [
    { key: "itemNumber", label: "Item #" },
    { key: null, label: "Image" },
    { key: "productName", label: "Product Name" },
    { key: "decision", label: "Decision" },
    { key: "confidence", label: "Confidence" },
    { key: null, label: "Feedback" },
    { key: "reviewedAt", label: "Reviewed" },
  ];

  return (
    <div>
      {/* Search & Filter */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
        <div style={{ position: "relative", flex: "1 1 280px", maxWidth: 360 }}>
          <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: COLORS.g400 }}><Icons.Search /></span>
          <input value={search} onChange={(e) => { setSearch(e.target.value); setPage(0); }} placeholder="Search by product name, item #, or part #..." style={{ width: "100%", padding: "10px 12px 10px 36px", borderRadius: 8, border: `1px solid ${COLORS.g200}`, fontSize: 13, fontFamily: fontStack, outline: "none", boxSizing: "border-box" }} />
        </div>
        <select value={filterDecision} onChange={(e) => { setFilterDecision(e.target.value); setPage(0); }} style={{ padding: "10px 14px", borderRadius: 8, border: `1px solid ${COLORS.g200}`, fontSize: 13, fontFamily: fontStack, background: COLORS.white, cursor: "pointer", color: COLORS.g700 }}>
          <option value="all">All Decisions</option>
          <option value="accepted">Accepted</option>
          <option value="rejected">Rejected</option>
          <option value="skipped">Skipped</option>
        </select>
        <span style={{ fontSize: 12, color: COLORS.g400 }}>{filtered.length} result{filtered.length !== 1 ? "s" : ""}</span>
      </div>

      {/* Table */}
      <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, overflow: "hidden", marginBottom: 16 }}>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, fontFamily: fontStack }}>
            <thead>
              <tr style={{ background: COLORS.g50 }}>
                {columns.map((col, i) => (
                  <th key={i} onClick={() => col.key && handleSort(col.key)} style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, color: COLORS.g400, fontWeight: 600, cursor: col.key ? "pointer" : "default", whiteSpace: "nowrap", userSelect: "none" }}>
                    {col.label}
                    {sortField === col.key && (
                      <span style={{ marginLeft: 4, display: "inline-flex", transform: sortDir === "asc" ? "rotate(0)" : "rotate(180deg)", transition: "transform 0.15s" }}><Icons.ArrowUp /></span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageItems.map((item) => {
                const b = decisionBadge(item.decision);
                return (
                  <tr key={item.itemNumber + item.reviewedAt} onClick={() => onSelectItem(item)} style={{ cursor: "pointer", borderBottom: `1px solid ${COLORS.g100}` }}>
                    <td style={{ padding: "10px 14px", fontFamily: monoStack, fontWeight: 600, color: COLORS.blue, fontSize: 12 }}>{item.itemNumber}</td>
                    <td style={{ padding: "10px 14px" }}><img src={item.imageUrl} alt="" style={{ width: 40, height: 40, borderRadius: 6, objectFit: "cover", background: COLORS.g100 }} /></td>
                    <td style={{ padding: "10px 14px", fontWeight: 500, color: COLORS.g800, maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.productName}</td>
                    <td style={{ padding: "10px 14px" }}><span style={{ padding: "3px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: b.bg, color: b.color, border: `1px solid ${b.border}`, whiteSpace: "nowrap" }}>{b.label}</span></td>
                    <td style={{ padding: "10px 14px" }}><span style={{ fontFamily: monoStack, fontWeight: 700, color: confidenceColor(item.confidence), fontSize: 12 }}>{item.confidence}%</span></td>
                    <td style={{ padding: "10px 14px", fontSize: 12, color: COLORS.g500, maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.feedbackTags?.join(", ") || item.feedback || "—"}</td>
                    <td style={{ padding: "10px 14px", fontSize: 12, color: COLORS.g500, whiteSpace: "nowrap" }}>{formatDate(item.reviewedAt)}</td>
                  </tr>
                );
              })}
              {pageItems.length === 0 && (
                <tr><td colSpan={7} style={{ padding: 40, textAlign: "center", color: COLORS.g400 }}>No results found</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div style={{ display: "flex", justifyContent: "center", gap: 6 }}>
          {Array.from({ length: pageCount }, (_, i) => (
            <button key={i} onClick={() => setPage(i)} style={{ width: 34, height: 34, borderRadius: 8, border: page === i ? `1.5px solid ${COLORS.blue}` : `1px solid ${COLORS.g200}`, background: page === i ? COLORS.bluePale : COLORS.white, color: page === i ? COLORS.blue : COLORS.g500, fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: monoStack }}>{i + 1}</button>
          ))}
        </div>
      )}
    </div>
  );
}

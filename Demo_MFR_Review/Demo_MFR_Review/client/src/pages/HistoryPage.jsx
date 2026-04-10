import React, { useState } from "react";
import HistoryStats from "../components/HistoryStats";
import HistoryTable from "../components/HistoryTable";
import DetailModal from "../components/DetailModal";
import { COLORS, fontStack } from "../components/Layout";

export default function HistoryPage({ history, onClearHistory }) {
  const [detailItem, setDetailItem] = useState(null);

  return (
    <div>
      {detailItem && <DetailModal item={detailItem} onClose={() => setDetailItem(null)} />}

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <div style={{ fontSize: 13, color: COLORS.g500, fontFamily: fontStack }}>
          Review history is kept for this browser session so auto-refresh does not re-add parts you already reviewed.
        </div>
        <button
          onClick={onClearHistory}
          disabled={history.length === 0}
          style={{
            background: COLORS.white,
            color: history.length === 0 ? COLORS.g300 : COLORS.red,
            border: `1px solid ${history.length === 0 ? COLORS.g200 : COLORS.redBorder}`,
            borderRadius: 10,
            padding: "10px 16px",
            fontSize: 13,
            fontWeight: 700,
            cursor: history.length === 0 ? "default" : "pointer",
            fontFamily: fontStack,
          }}
        >
          Clear Local Review History
        </button>
      </div>

      <HistoryStats history={history} />
      <HistoryTable history={history} onSelectItem={setDetailItem} />
    </div>
  );
}

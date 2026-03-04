import React from "react";
import { COLORS, monoStack } from "./Layout";

export default function ProgressBar({ current, total }) {
  const progressPct = total > 0 ? ((current + 1) / total) * 100 : 0;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
      <span style={{ fontSize: 13, color: COLORS.g500, fontWeight: 500, whiteSpace: "nowrap" }}>
        Product <strong style={{ color: COLORS.g800 }}>{current + 1}</strong> of <strong style={{ color: COLORS.g800 }}>{total}</strong>
      </span>
      <div style={{ flex: 1, maxWidth: 320, height: 6, background: COLORS.g200, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${progressPct}%`, height: "100%", background: COLORS.blue, borderRadius: 3, transition: "width 0.4s ease" }} />
      </div>
      <span style={{ fontSize: 12, color: COLORS.g400, fontFamily: monoStack }}>{Math.round(progressPct)}%</span>
    </div>
  );
}

import React, { useMemo } from "react";
import { COLORS, monoStack } from "./Layout";

export default function HistoryStats({ history }) {
  const stats = useMemo(() => {
    const total = history.length;
    const accepted = history.filter((h) => h.decision === "accepted").length;
    const rejected = history.filter((h) => h.decision === "rejected").length;
    const skipped = history.filter((h) => h.decision === "skipped").length;
    const acceptedItems = history.filter((h) => h.decision === "accepted");
    const avgConf = acceptedItems.length > 0
      ? acceptedItems.reduce((sum, h) => sum + h.confidence, 0) / acceptedItems.length
      : 0;
    return { total, accepted, rejected, skipped, avgConf: Math.round(avgConf * 10) / 10 };
  }, [history]);

  const cards = [
    { label: "Total Reviewed", value: stats.total, color: COLORS.blue },
    { label: "Accepted", value: `${stats.accepted} (${stats.total ? Math.round((stats.accepted / stats.total) * 100) : 0}%)`, color: COLORS.green },
    { label: "Rejected", value: `${stats.rejected} (${stats.total ? Math.round((stats.rejected / stats.total) * 100) : 0}%)`, color: COLORS.red },
    { label: "Skipped", value: `${stats.skipped} (${stats.total ? Math.round((stats.skipped / stats.total) * 100) : 0}%)`, color: COLORS.amber },
    { label: "Avg. Confidence (Accepted)", value: `${stats.avgConf}%`, color: COLORS.blue },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16, marginBottom: 24 }}>
      {cards.map((s) => (
        <div key={s.label} style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, padding: "18px 20px" }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 0.6, color: COLORS.g400, marginBottom: 6 }}>{s.label}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: s.color, fontFamily: monoStack }}>{s.value}</div>
        </div>
      ))}
    </div>
  );
}

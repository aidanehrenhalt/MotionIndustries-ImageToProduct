import React, { useState } from "react";
import { COLORS, fontStack, monoStack, Icons, confidenceColor, decisionBadge, formatDate } from "./Layout";

export default function DetailModal({ item, onClose }) {
  const [sel, setSel] = useState(0);
  if (!item) return null;

  const b = decisionBadge(item.decision);
  const imgs = item.candidateImages || [];

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 9000, display: "flex", justifyContent: "flex-end" }}>
      <div onClick={onClose} style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.4)" }} />
      <div style={{ position: "relative", width: 580, maxWidth: "95vw", background: COLORS.white, height: "100%", overflowY: "auto", boxShadow: "-8px 0 40px rgba(0,0,0,0.15)", fontFamily: fontStack }}>
        {/* Header */}
        <div style={{ padding: "20px 24px", borderBottom: `1px solid ${COLORS.g200}`, display: "flex", justifyContent: "space-between", alignItems: "center", position: "sticky", top: 0, background: COLORS.white, zIndex: 1 }}>
          <h3 style={{ margin: 0, fontSize: 16, color: COLORS.g900 }}>Review Detail</h3>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: COLORS.g500, padding: 4 }}><Icons.Close /></button>
        </div>

        {/* Content */}
        <div style={{ padding: 24 }}>
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 16 }}>
            <span style={{ fontFamily: monoStack, fontSize: 13, color: COLORS.blue, fontWeight: 600 }}>{item.itemNumber}</span>
            <span style={{ padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 600, background: b.bg, color: b.color, border: `1px solid ${b.border}` }}>{b.label}</span>
          </div>
          <h2 style={{ margin: "0 0 8px", fontSize: 18, color: COLORS.g900 }}>{item.productName}</h2>
          <p style={{ fontSize: 13, color: COLORS.g500, margin: "0 0 20px", lineHeight: 1.6 }}>{item.description}</p>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 24px", marginBottom: 24, fontSize: 13 }}>
            {[["Manufacturer", item.manufacturer], ["Mfr Part #", item.manufacturerPartNumber], ["Part Number", item.partNumber], ["Category", item.category], ["Confidence", `${item.confidence}%`], ["Reviewed", formatDate(item.reviewedAt)]].map(([k, v]) => (
              <div key={k}>
                <div style={{ color: COLORS.g400, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 2 }}>{k}</div>
                <div style={{ color: COLORS.g800, fontWeight: 500 }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Feedback */}
          {(item.feedback || item.feedbackTags?.length > 0) && (
            <div style={{ background: COLORS.g50, borderRadius: 8, padding: 14, marginBottom: 20, border: `1px solid ${COLORS.g200}` }}>
              <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, color: COLORS.g400, marginBottom: 6 }}>Feedback</div>
              {item.feedbackTags?.length > 0 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                  {item.feedbackTags.map((t) => (
                    <span key={t} style={{ background: b.bg, color: b.color, padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600 }}>{t}</span>
                  ))}
                </div>
              )}
              {item.feedback && <p style={{ margin: 0, fontSize: 13, color: COLORS.g700, lineHeight: 1.5 }}>{item.feedback}</p>}
            </div>
          )}

          {/* Candidate images */}
          {imgs.length > 0 && (
            <>
              <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, color: COLORS.g400, marginBottom: 10 }}>Candidate Images</div>
              <img src={imgs[sel]?.url || imgs[0]?.url} alt="" style={{ width: "100%", height: 280, objectFit: "contain", background: COLORS.g100, borderRadius: 8, marginBottom: 12 }} />
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {imgs.map((img, idx) => (
                  <div key={img.id} onClick={() => setSel(idx)} style={{ width: 64, height: 64, borderRadius: 6, overflow: "hidden", cursor: "pointer", border: sel === idx ? `2px solid ${COLORS.blue}` : `2px solid transparent`, position: "relative" }}>
                    <img src={img.thumbnailUrl} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    <span style={{ position: "absolute", bottom: 2, right: 2, fontSize: 9, fontWeight: 700, background: confidenceColor(img.confidence), color: "#fff", padding: "1px 4px", borderRadius: 4 }}>{img.confidence}%</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

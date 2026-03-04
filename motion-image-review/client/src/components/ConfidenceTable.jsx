import React from "react";
import { COLORS, fontStack, monoStack, confidenceColor, confidenceBg, reliabilityColor } from "./Layout";

export default function ConfidenceTable({ images, selectedIndex, onSelectImage }) {
  return (
    <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, overflow: "hidden" }}>
      <div style={{ padding: "14px 20px", borderBottom: `1px solid ${COLORS.g200}` }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: COLORS.g800 }}>Candidate Rankings</h3>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, fontFamily: fontStack }}>
          <thead>
            <tr style={{ background: COLORS.g50 }}>
              {["Rank", "Image", "Confidence", "Text Sim.", "Quality", "Source", "Reliability"].map((h) => (
                <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5, color: COLORS.g400, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {images.map((img, idx) => (
              <tr
                key={img.id}
                onClick={() => onSelectImage(idx)}
                style={{
                  background: selectedIndex === idx ? COLORS.bluePale : "transparent",
                  cursor: "pointer", transition: "background 0.15s ease",
                  borderBottom: `1px solid ${COLORS.g100}`,
                }}
              >
                <td style={{ padding: "10px 14px", fontWeight: 700, color: COLORS.g700 }}>#{img.rank}</td>
                <td style={{ padding: "10px 14px" }}>
                  <img src={img.thumbnailUrl} alt="" style={{ width: 40, height: 40, borderRadius: 6, objectFit: "cover", background: COLORS.g100 }} />
                </td>
                <td style={{ padding: "10px 14px" }}>
                  <span style={{ fontFamily: monoStack, fontWeight: 700, color: confidenceColor(img.confidence), background: confidenceBg(img.confidence), padding: "3px 8px", borderRadius: 6, fontSize: 12 }}>
                    {img.confidence}%
                  </span>
                </td>
                <td style={{ padding: "10px 14px", fontFamily: monoStack, color: COLORS.g700 }}>{img.textSimilarity}</td>
                <td style={{ padding: "10px 14px", fontFamily: monoStack, color: COLORS.g700 }}>{img.imageQuality}/10</td>
                <td style={{ padding: "10px 14px", fontSize: 12, color: COLORS.g500 }}>{img.source}</td>
                <td style={{ padding: "10px 14px" }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: reliabilityColor(img.sourceReliability) }}>{img.sourceReliability}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

import React from "react";
import { COLORS, fontStack, monoStack } from "./Layout";

export default function ProductCard({ product }) {
  return (
    <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <span style={{ fontFamily: monoStack, fontSize: 13, color: COLORS.blue, fontWeight: 600, background: COLORS.bluePale, padding: "4px 10px", borderRadius: 6 }}>{product.itemNumber}</span>
        <span style={{ fontSize: 12, color: COLORS.g400, background: COLORS.g100, padding: "4px 10px", borderRadius: 6 }}>{product.category}</span>
      </div>
      <h2 style={{ margin: "0 0 12px", fontSize: 19, fontWeight: 700, color: COLORS.g900, lineHeight: 1.35 }}>{product.productName}</h2>
      <p style={{ fontSize: 13, color: COLORS.g600, lineHeight: 1.65, margin: "0 0 20px" }}>{product.description}</p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px 24px" }}>
        {[
          ["Part Number", product.partNumber, true],
          ["Manufacturer", product.manufacturer, false],
          ["Mfr Part #", product.manufacturerPartNumber, true],
          ["Category", product.category, false],
        ].map(([label, value, isMono]) => (
          <div key={label}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 0.6, color: COLORS.g400, marginBottom: 3 }}>{label}</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: COLORS.g800, fontFamily: isMono ? monoStack : fontStack }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

import React from "react";
import { COLORS, monoStack, confidenceColor } from "./Layout";

export default function ImageGallery({ images, selectedIndex, onSelectImage }) {
  const selectedImage = images[selectedIndex];

  return (
    <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, padding: 24 }}>
      {/* Main image display */}
      <div style={{ position: "relative", marginBottom: 14 }}>
        <img
          src={selectedImage?.url}
          alt="Candidate product"
          style={{ width: "100%", height: 320, objectFit: "contain", background: COLORS.g50, borderRadius: 10 }}
        />
        <div style={{
          position: "absolute", top: 10, right: 10,
          background: confidenceColor(selectedImage?.confidence),
          color: "#fff", padding: "4px 12px", borderRadius: 20,
          fontSize: 13, fontWeight: 700, fontFamily: monoStack,
        }}>
          {selectedImage?.confidence}%
        </div>
      </div>

      {/* Thumbnail strip */}
      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
        {images.map((img, idx) => (
          <div
            key={img.id}
            onClick={() => onSelectImage(idx)}
            style={{
              width: 72, height: 72, borderRadius: 8, overflow: "hidden",
              cursor: "pointer", flexShrink: 0, position: "relative",
              border: selectedIndex === idx ? `2.5px solid ${COLORS.blue}` : `2px solid ${COLORS.g200}`,
              transition: "border-color 0.15s ease",
            }}
          >
            <img src={img.thumbnailUrl} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            <span style={{
              position: "absolute", bottom: 2, right: 2,
              fontSize: 9, fontWeight: 700, fontFamily: monoStack,
              background: confidenceColor(img.confidence),
              color: "#fff", padding: "1px 5px", borderRadius: 4,
            }}>
              {img.confidence}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

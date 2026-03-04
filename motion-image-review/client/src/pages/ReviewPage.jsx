import React, { useState, useEffect, useCallback } from "react";
import ProductCard from "../components/ProductCard";
import ImageGallery from "../components/ImageGallery";
import ConfidenceTable from "../components/ConfidenceTable";
import ReviewActions from "../components/ReviewActions";
import ProgressBar from "../components/ProgressBar";
import { COLORS, fontStack } from "../components/Layout";

export default function ReviewPage({ queue, onSubmitReview }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedImageIdx, setSelectedImageIdx] = useState(0);
  const [transitioning, setTransitioning] = useState(false);

  const product = queue[currentIndex];
  const images = product?.candidateImages || [];

  const advance = useCallback(() => {
    setTransitioning(true);
    setTimeout(() => {
      setCurrentIndex((i) => Math.min(i + 1, queue.length - 1));
      setSelectedImageIdx(0);
      setTransitioning(false);
    }, 300);
  }, [queue.length]);

  const handleSubmit = useCallback(
    (reviewData) => {
      onSubmitReview({
        itemNumber: product.itemNumber,
        ...reviewData,
        selectedImageId: images[selectedImageIdx]?.id,
      });
      advance();
    },
    [product, images, selectedImageIdx, advance, onSubmitReview]
  );

  // Keyboard shortcuts for image navigation
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "ArrowRight") setSelectedImageIdx((i) => Math.min(i + 1, images.length - 1));
      if (e.key === "ArrowLeft") setSelectedImageIdx((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [images.length]);

  // Empty queue state
  if (!product) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "60vh", fontFamily: fontStack }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🎉</div>
          <h2 style={{ color: COLORS.g800, marginBottom: 8 }}>Queue Complete</h2>
          <p style={{ color: COLORS.g500 }}>All products have been reviewed.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ opacity: transitioning ? 0.4 : 1, transition: "opacity 0.25s ease", fontFamily: fontStack }}>
      <ProgressBar current={currentIndex} total={queue.length} />

      {/* Product info + image gallery side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
        <ProductCard product={product} />
        <ImageGallery images={images} selectedIndex={selectedImageIdx} onSelectImage={setSelectedImageIdx} />
      </div>

      {/* Ranking table */}
      <div style={{ marginBottom: 24 }}>
        <ConfidenceTable images={images} selectedIndex={selectedImageIdx} onSelectImage={setSelectedImageIdx} />
      </div>

      {/* Action buttons */}
      <ReviewActions
        onSubmit={handleSubmit}
        onNextBestImage={() => setSelectedImageIdx((i) => Math.min(i + 1, images.length - 1))}
        canGoNext={selectedImageIdx < images.length - 1}
      />
    </div>
  );
}

import React, { useState, useCallback, useMemo } from "react";
import Layout, { COLORS, fontStack } from "./components/Layout";
import InputPage from "./pages/InputPage";
import ReviewPage from "./pages/ReviewPage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  const [activePage, setActivePage] = useState("input");

  const [rawHeaders, setRawHeaders] = useState([]);
  const [rawRows, setRawRows] = useState([]);
  const [reviewQueue, setReviewQueue] = useState([]);
  const [history, setHistory] = useState([]);

  const reviewedItemNumbers = useMemo(() => {
    return new Set(history.map((entry) => String(entry.itemNumber || "").trim()).filter(Boolean));
  }, [history]);

  const handleLoadExcel = useCallback((headers, rows) => {
    setRawHeaders(headers);
    setRawRows(rows);
  }, []);

  const handleLoadPipelineData = useCallback((products) => {
    setReviewQueue(
      products.filter((product) => {
        const key = String(product.itemNumber || "").trim();
        return key && !reviewedItemNumbers.has(key);
      })
    );
  }, [reviewedItemNumbers]);

  const handleSubmitReview = useCallback((review) => {
    setReviewQueue((prev) => {
      const reviewKey = review.queueKey || review.itemNumber;
      const product = prev.find((item) => (item.queueKey || item.itemNumber) === reviewKey);
      if (!product) return prev;

      const selectedImage = product.candidateImages?.find((img) => img.id === review.selectedImageId) || product.candidateImages?.[0] || null;
      const historyEntry = {
        ...product,
        decision: review.decision,
        confidence: selectedImage?.confidence || 0,
        feedback: review.feedback || "",
        feedbackTags: review.feedbackTags || [],
        reviewedAt: new Date().toISOString(),
        imageUrl: selectedImage?.thumbnailUrl || "",
        selectedImageId: selectedImage?.id || "",
      };

      setHistory((current) => [historyEntry, ...current.filter((entry) => (entry.queueKey || entry.itemNumber) !== reviewKey)]);
      return prev.filter((item) => (item.queueKey || item.itemNumber) !== reviewKey);
    });
  }, []);

  const handleClearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  const pageInfo = {
    input: { title: "Input UI", desc: "Upload the product Excel — this feeds into the web scraper pipeline." },
    review: { title: "Output UI", desc: "Connect a live output folder and review parts built by matching /json metadata to /images candidate files." },
    history: { title: "Review History", desc: "Browse review decisions captured during this browser session." },
  };
  const info = pageInfo[activePage] || pageInfo.input;

  return (
    <Layout activePage={activePage} onNavigate={setActivePage} queueCount={reviewQueue.length} historyCount={history.length}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: COLORS.g900, margin: 0, letterSpacing: -0.5, fontFamily: fontStack }}>{info.title}</h1>
        <p style={{ fontSize: 13, color: COLORS.g400, marginTop: 4, fontFamily: fontStack }}>{info.desc}</p>
      </div>

      <div style={{ display: activePage === "input" ? "block" : "none" }}>
        <InputPage rawHeaders={rawHeaders} rawRows={rawRows} onLoadExcel={handleLoadExcel} />
      </div>
      <div style={{ display: activePage === "review" ? "block" : "none" }}>
        <ReviewPage queue={reviewQueue} onLoadPipelineData={handleLoadPipelineData} onSubmitReview={handleSubmitReview} />
      </div>
      <div style={{ display: activePage === "history" ? "block" : "none" }}>
        <HistoryPage history={history} onClearHistory={handleClearHistory} />
      </div>
    </Layout>
  );
}

import React, { useState, useCallback } from "react";
import Layout, { COLORS, fontStack } from "./components/Layout";
import ReviewPage from "./pages/ReviewPage";
import HistoryPage from "./pages/HistoryPage";
import { generateQueueData, generateHistoryData } from "./data/mockData";

const INITIAL_QUEUE = generateQueueData();
const INITIAL_HISTORY = generateHistoryData();

export default function App() {
  const [activePage, setActivePage] = useState("review");
  const [queue, setQueue] = useState(INITIAL_QUEUE);
  const [history, setHistory] = useState(INITIAL_HISTORY);

  const handleSubmitReview = useCallback(
    (review) => {
      const product = queue.find((p) => p.itemNumber === review.itemNumber);
      if (!product) return;

      const historyEntry = {
        ...product,
        decision: review.decision,
        confidence: product.candidateImages[0]?.confidence || 0,
        feedback: review.feedback || "",
        feedbackTags: review.feedbackTags || [],
        reviewedAt: new Date().toISOString(),
        imageUrl: product.candidateImages[0]?.thumbnailUrl || "",
      };

      setHistory((prev) => [historyEntry, ...prev]);
      setQueue((prev) => prev.filter((p) => p.itemNumber !== review.itemNumber));
    },
    [queue]
  );

  return (
    <Layout
      activePage={activePage}
      onNavigate={setActivePage}
      queueCount={queue.length}
      historyCount={history.length}
    >
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: COLORS.g900, margin: 0, letterSpacing: -0.5, fontFamily: fontStack }}>
          {activePage === "review" ? "Review Queue" : "Review History"}
        </h1>
        <p style={{ fontSize: 13, color: COLORS.g400, marginTop: 4, fontFamily: fontStack }}>
          {activePage === "review"
            ? "Review and approve candidate product images from the AI pipeline."
            : "Browse past review decisions and statistics."}
        </p>
      </div>

      {activePage === "review" ? (
        <ReviewPage queue={queue} onSubmitReview={handleSubmitReview} />
      ) : (
        <HistoryPage history={history} />
      )}
    </Layout>
  );
}

import React, { useState } from "react";
import { COLORS, fontStack, Icons } from "./Layout";
import { REJECT_REASONS, ACCEPT_REASONS } from "../data/mockData";

export default function ReviewActions({ onSubmit, onNextBestImage, canGoNext }) {
  const [showFeedback, setShowFeedback] = useState(null);
  const [feedbackText, setFeedbackText] = useState("");
  const [feedbackTags, setFeedbackTags] = useState([]);

  const toggleTag = (tag) => {
    setFeedbackTags((prev) => prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]);
  };

  const handleDecision = (decision) => {
    if (decision === "skipped") {
      onSubmit({ decision: "skipped", feedback: "", feedbackTags: [] });
      return;
    }
    setShowFeedback(decision);
  };

  const handleFinalSubmit = () => {
    onSubmit({ decision: showFeedback, feedback: feedbackText, feedbackTags });
    setShowFeedback(null);
    setFeedbackText("");
    setFeedbackTags([]);
  };

  const handleCancel = () => {
    setShowFeedback(null);
    setFeedbackText("");
    setFeedbackTags([]);
  };

  return (
    <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, padding: 20 }}>
      {!showFeedback ? (
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <button onClick={() => handleDecision("accepted")} style={{ background: COLORS.green, color: "#fff", border: "none", borderRadius: 10, padding: "12px 28px", fontSize: 15, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", gap: 8, fontFamily: fontStack, boxShadow: `0 2px 8px ${COLORS.green}40` }}>
            <Icons.Check /> Accept
          </button>
          <button onClick={() => handleDecision("rejected")} style={{ background: COLORS.red, color: "#fff", border: "none", borderRadius: 10, padding: "12px 28px", fontSize: 15, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", gap: 8, fontFamily: fontStack, boxShadow: `0 2px 8px ${COLORS.red}40` }}>
            <Icons.X /> Reject
          </button>
          <button onClick={() => handleDecision("skipped")} style={{ background: COLORS.g100, color: COLORS.g600, border: `1px solid ${COLORS.g300}`, borderRadius: 10, padding: "12px 24px", fontSize: 15, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 8, fontFamily: fontStack }}>
            <Icons.Skip /> Skip
          </button>
          <div style={{ flex: 1 }} />
          <button onClick={onNextBestImage} disabled={!canGoNext} style={{ background: "transparent", color: !canGoNext ? COLORS.g300 : COLORS.blue, border: `1px solid ${!canGoNext ? COLORS.g200 : COLORS.blue}`, borderRadius: 10, padding: "12px 20px", fontSize: 13, fontWeight: 600, cursor: !canGoNext ? "default" : "pointer", display: "flex", alignItems: "center", gap: 6, fontFamily: fontStack }}>
            Next Best Image <Icons.Right />
          </button>
        </div>
      ) : (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <span style={{ background: showFeedback === "accepted" ? COLORS.greenBg : COLORS.redBg, color: showFeedback === "accepted" ? COLORS.green : COLORS.red, padding: "4px 12px", borderRadius: 20, fontSize: 13, fontWeight: 700 }}>
              {showFeedback === "accepted" ? "✓ Accepting" : "✗ Rejecting"}
            </span>
            <span style={{ fontSize: 13, color: COLORS.g500 }}>Add optional feedback before submitting</span>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
            {(showFeedback === "rejected" ? REJECT_REASONS : ACCEPT_REASONS).map((tag) => (
              <button key={tag} onClick={() => toggleTag(tag)} style={{ padding: "6px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: fontStack, transition: "all 0.15s ease", background: feedbackTags.includes(tag) ? (showFeedback === "accepted" ? COLORS.green : COLORS.red) : COLORS.g100, color: feedbackTags.includes(tag) ? "#fff" : COLORS.g600, border: feedbackTags.includes(tag) ? "1px solid transparent" : `1px solid ${COLORS.g300}` }}>
                {tag}
              </button>
            ))}
          </div>
          <textarea value={feedbackText} onChange={(e) => setFeedbackText(e.target.value)} placeholder="Additional notes (optional)..." style={{ width: "100%", minHeight: 70, borderRadius: 8, border: `1px solid ${COLORS.g200}`, padding: 12, fontSize: 13, fontFamily: fontStack, resize: "vertical", outline: "none", boxSizing: "border-box" }} />
          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
            <button onClick={handleFinalSubmit} style={{ background: showFeedback === "accepted" ? COLORS.green : COLORS.red, color: "#fff", border: "none", borderRadius: 10, padding: "10px 24px", fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: fontStack }}>Submit & Next</button>
            <button onClick={handleCancel} style={{ background: COLORS.g100, color: COLORS.g600, border: `1px solid ${COLORS.g300}`, borderRadius: 10, padding: "10px 20px", fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: fontStack }}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}

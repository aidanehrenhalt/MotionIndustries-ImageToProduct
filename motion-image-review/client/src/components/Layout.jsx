import React from "react";

// Shared color palette
export const COLORS = {
  navy: "#1a1f36",
  navyHover: "#2f365a",
  blue: "#003DA5",
  bluePale: "#e8f0fe",
  green: "#0d9f6e",
  greenBg: "#ecfdf5",
  greenBorder: "#a7f3d0",
  red: "#dc2626",
  redBg: "#fef2f2",
  redBorder: "#fecaca",
  amber: "#d97706",
  amberBg: "#fffbeb",
  amberBorder: "#fde68a",
  g50: "#f9fafb",
  g100: "#f3f4f6",
  g200: "#e5e7eb",
  g300: "#d1d5db",
  g400: "#9ca3af",
  g500: "#6b7280",
  g600: "#4b5563",
  g700: "#374151",
  g800: "#1f2937",
  g900: "#111827",
  white: "#ffffff",
};

// Shared font stacks
export const fontStack = `'DM Sans', 'Segoe UI', system-ui, sans-serif`;
export const monoStack = `'DM Mono', 'SF Mono', 'Consolas', monospace`;

// Shared SVG icons
export const Icons = {
  Grid: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
    </svg>
  ),
  Clock: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
    </svg>
  ),
  Check: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  X: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
  Skip: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="5 4 15 12 5 20 5 4" /><line x1="19" y1="5" x2="19" y2="19" />
    </svg>
  ),
  Right: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  ),
  Search: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  ),
  Close: () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
  ArrowUp: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="19" x2="12" y2="5" /><polyline points="5 12 12 5 19 12" />
    </svg>
  ),
};

// Shared helper functions
export function confidenceColor(score) {
  if (score >= 90) return COLORS.green;
  if (score >= 70) return COLORS.amber;
  return COLORS.red;
}

export function confidenceBg(score) {
  if (score >= 90) return COLORS.greenBg;
  if (score >= 70) return COLORS.amberBg;
  return COLORS.redBg;
}

export function reliabilityColor(r) {
  if (r === "High") return COLORS.green;
  if (r === "Medium") return COLORS.amber;
  return COLORS.red;
}

export function decisionBadge(decision) {
  const map = {
    accepted: { bg: COLORS.greenBg, color: COLORS.green, border: COLORS.greenBorder, label: "✓ Accepted" },
    rejected: { bg: COLORS.redBg, color: COLORS.red, border: COLORS.redBorder, label: "✗ Rejected" },
    skipped: { bg: COLORS.amberBg, color: COLORS.amber, border: COLORS.amberBorder, label: "⏭ Skipped" },
  };
  return map[decision] || map.skipped;
}

export function formatDate(iso) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

// The Layout shell: sidebar + content area
export default function Layout({ activePage, onNavigate, queueCount, historyCount, children }) {
  const navItems = [
    { key: "review", label: "Review Queue", icon: <Icons.Grid />, count: queueCount },
    { key: "history", label: "History", icon: <Icons.Clock />, count: historyCount },
  ];

  return (
    <div style={{ display: "flex", minHeight: "100vh", fontFamily: fontStack, background: COLORS.g100 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; }
        body { margin: 0; background: ${COLORS.g100}; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.g300}; border-radius: 3px; }
      `}</style>

      {/* Sidebar */}
      <div style={{ width: 240, background: COLORS.navy, color: "#fff", display: "flex", flexDirection: "column", position: "fixed", top: 0, left: 0, bottom: 0, zIndex: 100 }}>
        <div style={{ padding: "24px 20px 20px", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: COLORS.blue, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 800 }}>M</div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700 }}>Motion Industries</div>
              <div style={{ fontSize: 10, opacity: 0.5, letterSpacing: 0.5, textTransform: "uppercase" }}>Image Review</div>
            </div>
          </div>
        </div>
        <nav style={{ padding: "16px 12px", flex: 1 }}>
          {navItems.map((item) => (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 10, padding: "11px 14px",
                borderRadius: 8, border: "none", cursor: "pointer", fontFamily: fontStack, fontSize: 13,
                fontWeight: activePage === item.key ? 600 : 500, marginBottom: 4, transition: "all 0.15s ease",
                background: activePage === item.key ? COLORS.navyHover : "transparent",
                color: activePage === item.key ? "#fff" : "rgba(255,255,255,0.55)",
              }}
            >
              {item.icon}
              <span style={{ flex: 1, textAlign: "left" }}>{item.label}</span>
              <span style={{ fontSize: 11, fontFamily: monoStack, fontWeight: 600, background: activePage === item.key ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.06)", padding: "2px 8px", borderRadius: 10 }}>
                {item.count}
              </span>
            </button>
          ))}
        </nav>
        <div style={{ padding: "16px 20px", borderTop: "1px solid rgba(255,255,255,0.08)", fontSize: 11, opacity: 0.35 }}>
          v1.0 · Image-to-Product AI
        </div>
      </div>

      {/* Main content area */}
      <div style={{ marginLeft: 240, flex: 1, padding: "24px 32px", maxWidth: 1280 }}>
        {children}
      </div>
    </div>
  );
}

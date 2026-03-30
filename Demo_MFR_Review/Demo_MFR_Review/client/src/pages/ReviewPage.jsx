import React, { useState, useEffect, useCallback, useRef } from "react";
import ProductCard from "../components/ProductCard";
import ImageGallery from "../components/ImageGallery";
import ConfidenceTable from "../components/ConfidenceTable";
import ReviewActions from "../components/ReviewActions";
// import ProgressBar from "../components/ProgressBar";
import { COLORS, fontStack, monoStack, Icons, formatDate } from "../components/Layout";
import { parsePipelineFolderFiles, readFilesFromDirectoryHandle } from "../utils/pipelineFolderParser";

function ConnectionCard({ onConnectLiveFolder, error }) {
  return (
    <div style={{ fontFamily: fontStack }}>
      <div style={{ background: COLORS.white, borderRadius: 16, border: `1px solid ${COLORS.g200}`, padding: 28 }}>
        <div style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: COLORS.g900, marginBottom: 8 }}>Connect the live output folder</div>
          <div style={{ fontSize: 13, color: COLORS.g500, lineHeight: 1.7 }}>
            Please attach the root folder that contains two subfolders named <strong>json</strong> and <strong>images</strong>.
            The app reads every part JSON from <strong>/json</strong>, reads every candidate image from <strong>/images</strong>, matches them together, and refreshes every 15 seconds.
          </div>
        </div>

        <button
          onClick={onConnectLiveFolder}
          style={{ background: COLORS.blue, color: "#fff", border: "none", borderRadius: 14, padding: "18px 16px", fontSize: 14, fontWeight: 800, cursor: "pointer", fontFamily: fontStack, textAlign: "left", width: "100%" }}
        >
          <div style={{ marginBottom: 6 }}>Connect Live Folder</div>
          <div style={{ fontSize: 12, fontWeight: 500, lineHeight: 1.5, opacity: 0.92 }}>Choose the root output folder once. The page will rescan it automatically every 15 seconds.</div>
        </button>

        {error && <div style={{ marginTop: 16, color: COLORS.red, fontSize: 13 }}>{error}</div>}
      </div>
    </div>
  );
}

export default function ReviewPage({ queue, onLoadPipelineData, onSubmitReview }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedImageIdx, setSelectedImageIdx] = useState(0);
  const [connection, setConnection] = useState(null);
  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastSyncedAt, setLastSyncedAt] = useState(null);
  const [lastScanProductCount, setLastScanProductCount] = useState(0);

  const directoryHandleRef = useRef(null);
  const urlCacheRef = useRef(new Map());

  const product = queue[currentIndex] || queue[0];
  const images = product?.candidateImages || [];
  const hasImages = images.length > 0;

  const makeObjectUrl = useCallback((file) => {
    const relativePath = file.relativePath || file.webkitRelativePath || file.name;
    const cacheKey = `${relativePath}::${file.size}::${file.lastModified}`;
    const cachedUrl = urlCacheRef.current.get(cacheKey);
    if (cachedUrl) return cachedUrl;
    const objectUrl = URL.createObjectURL(file);
    urlCacheRef.current.set(cacheKey, objectUrl);
    return objectUrl;
  }, []);

  useEffect(() => {
    return () => {
      urlCacheRef.current.forEach((url) => URL.revokeObjectURL(url));
      urlCacheRef.current.clear();
    };
  }, []);

  useEffect(() => {
    if (queue.length === 0) {
      setCurrentIndex(0);
      setSelectedImageIdx(0);
      return;
    }
    if (currentIndex > queue.length - 1) setCurrentIndex(queue.length - 1);
    if (selectedImageIdx > (queue[Math.min(currentIndex, queue.length - 1)]?.candidateImages?.length || 1) - 1) setSelectedImageIdx(0);
  }, [queue, currentIndex, selectedImageIdx]);

  const syncProducts = useCallback((products, sourceLabel) => {
    onLoadPipelineData(products);
    setLastScanProductCount(products.length);
    setLastSyncedAt(new Date().toISOString());
    setSyncError(null);
    setConnection((current) => (current ? { ...current, label: sourceLabel || current.label } : current));
  }, [onLoadPipelineData]);

  const refreshFromLiveFolder = useCallback(async () => {
    if (!directoryHandleRef.current) return;
    setSyncing(true);
    try {
      const files = await readFilesFromDirectoryHandle(directoryHandleRef.current);
      const products = await parsePipelineFolderFiles(files, makeObjectUrl);
      syncProducts(products, directoryHandleRef.current.name || "Live output folder");
    } catch (err) {
      setSyncError(err.message || "Failed to scan the connected folder.");
    } finally {
      setSyncing(false);
    }
  }, [makeObjectUrl, syncProducts]);

  const handleConnectLiveFolder = useCallback(async () => {
    if (!window.showDirectoryPicker) {
      setSyncError("This browser does not support live folder access. Use Chrome or Edge.");
      return;
    }
    try {
      const handle = await window.showDirectoryPicker();
      directoryHandleRef.current = handle;
      setConnection({ type: "live-folder", label: handle.name || "Live output folder" });
      setCurrentIndex(0);
      setSelectedImageIdx(0);
      await refreshFromLiveFolder();
    } catch (err) {
      if (err?.name !== "AbortError") setSyncError(err.message || "Failed to connect the folder.");
    }
  }, [refreshFromLiveFolder]);

  const handleManualRefresh = useCallback(async () => {
    await refreshFromLiveFolder();
  }, [refreshFromLiveFolder]);

  useEffect(() => {
    if (!connection || !autoRefresh) return undefined;
    const timer = window.setInterval(() => {
      refreshFromLiveFolder();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [connection, autoRefresh, refreshFromLiveFolder]);

  useEffect(() => {
    const handler = (event) => {
      const tag = event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (event.key === "ArrowRight" && hasImages) setSelectedImageIdx((index) => Math.min(index + 1, images.length - 1));
      if (event.key === "ArrowLeft" && hasImages) setSelectedImageIdx((index) => Math.max(index - 1, 0));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [images.length, hasImages]);

  const handleSubmit = useCallback((reviewData) => {
    if (!product) return;
    onSubmitReview({ queueKey: product.queueKey, itemNumber: product.itemNumber, ...reviewData, selectedImageId: images[selectedImageIdx]?.id });
    setSelectedImageIdx(0);
  }, [product, images, selectedImageIdx, onSubmitReview]);

  const disconnectSource = useCallback(() => {
    directoryHandleRef.current = null;
    setConnection(null);
    setCurrentIndex(0);
    setSelectedImageIdx(0);
    setLastSyncedAt(null);
    setLastScanProductCount(0);
    setSyncError(null);
    onLoadPipelineData([]);
  }, [onLoadPipelineData]);

  if (!connection) {
    return <ConnectionCard onConnectLiveFolder={handleConnectLiveFolder} error={syncError} />;
  }

  if (queue.length === 0) {
    return (
      <div style={{ fontFamily: fontStack }}>
        <div style={{ background: COLORS.white, borderRadius: 16, border: `1px solid ${COLORS.g200}`, padding: 24, marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 18, fontWeight: 800, color: COLORS.g900, marginBottom: 6 }}>{connection.label}</div>
              <div style={{ fontSize: 13, color: COLORS.g500, lineHeight: 1.6 }}>
                Source type: <strong>Live folder</strong>
                {lastSyncedAt && <> · Last synced {formatDate(lastSyncedAt)}</>}
                {lastScanProductCount > 0 && <> · Parsed {lastScanProductCount} parts from /json and /images</>}
              </div>
            </div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <button onClick={handleManualRefresh} style={{ background: COLORS.blue, color: "#fff", border: "none", borderRadius: 10, padding: "10px 16px", fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: fontStack }}>
                {syncing ? "Refreshing..." : "Refresh Now"}
              </button>
              <button onClick={disconnectSource} style={{ background: COLORS.g100, color: COLORS.g700, border: `1px solid ${COLORS.g300}`, borderRadius: 10, padding: "10px 16px", fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: fontStack }}>
                Disconnect
              </button>
            </div>
          </div>
        </div>

        <div style={{ background: COLORS.white, borderRadius: 16, border: `1px solid ${COLORS.g200}`, padding: "64px 24px", textAlign: "center" }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>{syncing ? "⏳" : "🎉"}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: COLORS.g900, marginBottom: 10 }}>
            {syncing ? "Scanning json and images..." : "No pending products right now"}
          </div>
          <div style={{ fontSize: 14, color: COLORS.g500, lineHeight: 1.7, maxWidth: 680, margin: "0 auto" }}>
            The page is connected and re-checks the root folder every 15 seconds. It reads all part metadata from <strong>/json</strong>, all candidate images from <strong>/images</strong>, and keeps already reviewed item numbers out of the queue for this session.
          </div>
          <label style={{ marginTop: 18, display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13, color: COLORS.g600 }}>
            <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
            Auto-refresh every 15 seconds
          </label>
          {syncError && <div style={{ marginTop: 16, color: COLORS.red, fontSize: 13 }}>{syncError}</div>}
        </div>
      </div>
    );
  }

  return (
    <div style={{ fontFamily: fontStack }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, gap: 12, flexWrap: "wrap" }}>
        {/* <ProgressBar current={currentIndex} total={queue.length} /> */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: COLORS.g400 }}>
            <Icons.Table />
            <span style={{ fontFamily: monoStack }}>{connection.label}</span>
            {lastSyncedAt && <span>· {formatDate(lastSyncedAt)}</span>}
          </div>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: COLORS.g500 }}>
            <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
            Auto-refresh
          </label>
          <button onClick={handleManualRefresh} style={{ background: COLORS.white, color: COLORS.blue, border: `1px solid ${COLORS.blue}`, borderRadius: 8, padding: "8px 12px", fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: fontStack }}>
            {syncing ? "Refreshing..." : "Refresh"}
          </button>
          <button onClick={disconnectSource} style={{ background: COLORS.g100, color: COLORS.g700, border: `1px solid ${COLORS.g300}`, borderRadius: 8, padding: "8px 12px", fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: fontStack }}>
            Disconnect
          </button>
        </div>
      </div>
      {syncError && <div style={{ marginBottom: 16, color: COLORS.red, fontSize: 13 }}>{syncError}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
        <ProductCard product={product} />
        {hasImages ? (
          <ImageGallery images={images} selectedIndex={selectedImageIdx} onSelectImage={setSelectedImageIdx} />
        ) : (
          <div style={{ background: COLORS.amberBg, border: `1px solid ${COLORS.amberBorder}`, borderRadius: 12, padding: "40px 24px", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>🖼️</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: COLORS.amber, marginBottom: 6 }}>No matched images for this part</div>
            <div style={{ fontSize: 12, color: COLORS.g500 }}>The JSON was parsed correctly, but no image filename in /images matched this part yet.</div>
          </div>
        )}
      </div>

      {hasImages && (
        <div style={{ marginBottom: 24 }}>
          <ConfidenceTable images={images} selectedIndex={selectedImageIdx} onSelectImage={setSelectedImageIdx} />
        </div>
      )}

      {hasImages ? (
        <ReviewActions onSubmit={handleSubmit} onNextBestImage={() => setSelectedImageIdx((index) => Math.min(index + 1, images.length - 1))} canGoNext={selectedImageIdx < images.length - 1} />
      ) : (
        <div style={{ background: COLORS.white, borderRadius: 12, border: `1px solid ${COLORS.g200}`, padding: 20, display: "flex", gap: 12 }}>
          <button onClick={() => handleSubmit({ decision: "skipped", feedback: "No matching image found in /images", feedbackTags: [] })} style={{ background: COLORS.g100, color: COLORS.g600, border: `1px solid ${COLORS.g300}`, borderRadius: 10, padding: "12px 24px", fontSize: 15, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 8, fontFamily: fontStack }}>
            <Icons.Skip /> Skip — No Matched Image
          </button>
        </div>
      )}
    </div>
  );
}

(() => {
  const app = document.getElementById("review-app");
  if (!app) return;

  const REVIEW_STORAGE_KEY = "demo-mfr-review-decisions";

  const state = {
    queue: [],
    selectedProductIndex: 0,
    selectedImageIndex: 0,
    decisions: loadStoredDecisions(),
    generatedAt: "",
    status: "",
  };

  function loadStoredDecisions() {
    try {
      return JSON.parse(window.localStorage.getItem(REVIEW_STORAGE_KEY) || "{}");
    } catch {
      return {};
    }
  }

  function persistDecisions() {
    window.localStorage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(state.decisions));
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));
  }

  function formatPct(value, fallback = "") {
    if (value === "" || value == null || Number.isNaN(Number(value))) return fallback;
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function reviewedCount() {
    return Object.keys(state.decisions).length;
  }

  function pendingQueue() {
    return state.queue.filter((product) => !state.decisions[product.queueKey]);
  }

  function selectedProduct() {
    const queue = pendingQueue();
    if (queue.length === 0) return null;
    if (state.selectedProductIndex > queue.length - 1) state.selectedProductIndex = queue.length - 1;
    return queue[state.selectedProductIndex] || queue[0];
  }

  function selectedImage(product) {
    if (!product || !product.candidateImages?.length) return null;
    if (state.selectedImageIndex > product.candidateImages.length - 1) state.selectedImageIndex = 0;
    return product.candidateImages[state.selectedImageIndex] || product.candidateImages[0];
  }

  function downloadJson(filename, data) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function recordDecision(decision) {
    const product = selectedProduct();
    if (!product) return;
    const image = selectedImage(product);
    const feedbackEl = document.getElementById("review-feedback");
    const feedback = feedbackEl ? feedbackEl.value.trim() : "";
    state.decisions[product.queueKey] = {
      product_id: product.productId,
      queue_key: product.queueKey,
      reviewed_at: new Date().toISOString(),
      decision,
      selected_image_id: image?.id || "",
      selected_image_filename: image?.fileName || "",
      selected_image_rank: image?.rank || 0,
      selected_image_score: image?.finalScore || 0,
      feedback,
      source_json_file: product.jsonFile,
      candidate_images_summary: (product.candidateImages || []).map((candidate) => ({
        file_name: candidate.fileName,
        rank: candidate.rank,
        final_score: candidate.finalScore,
      })),
    };
    persistDecisions();
    state.selectedImageIndex = 0;
    state.status = `Saved ${decision} decision for ${product.productId}.`;
    render();
  }

  function exportReviews() {
    downloadJson("demo_mfr_review_results.json", {
      generated_at: new Date().toISOString(),
      source_generated_at: state.generatedAt,
      review_count: reviewedCount(),
      reviews: Object.values(state.decisions),
    });
    state.status = "Exported manual review JSON.";
    render();
  }

  function clearReviews() {
    state.decisions = {};
    persistDecisions();
    state.selectedProductIndex = 0;
    state.selectedImageIndex = 0;
    state.status = "Cleared local review decisions.";
    render();
  }

  function renderQueuePanel(queue) {
    const pending = pendingQueue();
    const items = pending.map((product, index) => {
      const activeClass = index === state.selectedProductIndex ? " active" : "";
      return `
        <button class="review-product-button${activeClass}" data-product-index="${index}">
          <div><strong>${escapeHtml(product.productId)}</strong></div>
          <div>${escapeHtml(product.productName)}</div>
          <small>${escapeHtml(product.manufacturer || "Unknown manufacturer")} · ${product.matchedImageCount || 0} candidate image(s)</small>
        </button>
      `;
    }).join("");

    return `
      <article class="review-panel">
        <div class="review-header">
          <div>
            <p class="eyebrow">Queue</p>
            <h2>Pending review products</h2>
          </div>
          <div class="muted">${pending.length} pending · ${reviewedCount()} reviewed</div>
        </div>
        <div class="review-summary">
          <div class="stat-card"><span class="metric-label">Queue Size</span><strong>${pending.length}</strong></div>
          <div class="stat-card"><span class="metric-label">Reviewed</span><strong>${reviewedCount()}</strong></div>
        </div>
        ${pending.length ? `<div class="review-product-list">${items}</div>` : `<div class="empty-state">No pending products remain. Export the review JSON or clear local review state to start over.</div>`}
      </article>
    `;
  }

  function renderDetailPanel(product) {
    if (!product) {
      return `
        <aside class="review-panel">
          <p class="eyebrow">Selected Product</p>
          <div class="empty-state">The queue is complete. Export the review decisions JSON to keep a copy of the manual review results.</div>
          <div class="review-actions">
            <button class="action-button action-export" data-action="export">Export Review JSON</button>
            <button class="action-button action-skip" data-action="clear">Clear Local Reviews</button>
          </div>
        </aside>
      `;
    }

    const image = selectedImage(product);
    const candidateRows = (product.candidateImages || []).map((candidate, index) => {
      const activeClass = index === state.selectedImageIndex ? " active" : "";
      return `
        <tr class="candidate-row${activeClass}" data-image-index="${index}">
          <td>${candidate.rank || ""}</td>
          <td>${escapeHtml(candidate.fileName)}</td>
          <td>${escapeHtml(candidate.finalScorePct || formatPct(candidate.finalScore))}</td>
          <td>${escapeHtml(candidate.aiConfidencePct || formatPct(candidate.aiConfidence))}</td>
          <td>${escapeHtml(candidate.textScorePct || formatPct(candidate.textScore))}</td>
        </tr>
      `;
    }).join("");

    const specs = (product.highlights || []).slice(0, 4).map((item) => `
      <div class="metric">
        <span class="metric-label">${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(item.value)}</strong>
      </div>
    `).join("");

    return `
      <aside class="review-panel">
        <p class="eyebrow">Selected Product</p>
        <div class="review-grid">
          <div>
            <h2>${escapeHtml(product.productId)} · ${escapeHtml(product.productName)}</h2>
            <p class="muted">${escapeHtml(product.description || "No description available.")}</p>
            <div class="review-metrics">
              <div class="metric"><span class="metric-label">Manufacturer</span><strong>${escapeHtml(product.manufacturer || "Unknown")}</strong></div>
              <div class="metric"><span class="metric-label">Part Number</span><strong>${escapeHtml(product.partNumber || product.productId)}</strong></div>
              <div class="metric"><span class="metric-label">Category</span><strong>${escapeHtml(product.category || "N/A")}</strong></div>
              <div class="metric"><span class="metric-label">JSON File</span><strong>${escapeHtml(product.jsonFile || "N/A")}</strong></div>
            </div>
            ${specs ? `<div class="review-metrics">${specs}</div>` : ""}
          </div>
          ${image ? `
            <div class="review-image-card">
              <div class="review-image-frame">
                <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.fileName)}">
              </div>
            </div>
            <div class="review-metrics">
              <div class="metric"><span class="metric-label">Final Score</span><strong>${escapeHtml(image.finalScorePct || formatPct(image.finalScore))}</strong></div>
              <div class="metric"><span class="metric-label">AI Confidence</span><strong>${escapeHtml(image.aiConfidencePct || formatPct(image.aiConfidence))}</strong></div>
              <div class="metric"><span class="metric-label">Text Score</span><strong>${escapeHtml(image.textScorePct || formatPct(image.textScore))}</strong></div>
              <div class="metric"><span class="metric-label">Source</span><strong>${escapeHtml(image.sourceName || "Unknown")}</strong></div>
            </div>
          ` : `<div class="empty-state">No candidate images were available for this product.</div>`}
          <div>
            <table class="candidate-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Image</th>
                  <th>Final</th>
                  <th>AI</th>
                  <th>Text</th>
                </tr>
              </thead>
              <tbody>${candidateRows}</tbody>
            </table>
          </div>
          <div>
            <label class="muted" for="review-feedback">Reviewer notes</label>
            <textarea id="review-feedback" class="review-feedback" placeholder="Add manual review notes or rationale here."></textarea>
            <div class="review-actions">
              <button class="action-button action-approve" data-action="approve">Approve</button>
              <button class="action-button action-reject" data-action="reject">Reject</button>
              <button class="action-button action-skip" data-action="skip">Skip</button>
              <button class="action-button action-export" data-action="export">Export Review JSON</button>
            </div>
            <div class="review-status">${escapeHtml(state.status || `Dataset generated ${state.generatedAt || "recently"}.`)}</div>
          </div>
        </div>
      </aside>
    `;
  }

  function attachHandlers() {
    app.querySelectorAll("[data-product-index]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedProductIndex = Number(button.dataset.productIndex || 0);
        state.selectedImageIndex = 0;
        state.status = "";
        render();
      });
    });

    app.querySelectorAll("[data-image-index]").forEach((row) => {
      row.addEventListener("click", () => {
        state.selectedImageIndex = Number(row.dataset.imageIndex || 0);
        render();
      });
    });

    app.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.action;
        if (action === "export") exportReviews();
        if (action === "clear") clearReviews();
        if (action === "approve") recordDecision("approved");
        if (action === "reject") recordDecision("rejected");
        if (action === "skip") recordDecision("skipped");
      });
    });
  }

  function render() {
    app.innerHTML = `${renderQueuePanel(state.queue)}${renderDetailPanel(selectedProduct())}`;
    const notes = document.getElementById("review-feedback");
    const product = selectedProduct();
    if (notes && product && state.decisions[product.queueKey]) {
      notes.value = state.decisions[product.queueKey].feedback || "";
    }
    attachHandlers();
  }

  fetch("assets/data/review_queue.json")
    .then((response) => {
      if (!response.ok) throw new Error(`Failed to load review dataset (${response.status})`);
      return response.json();
    })
    .then((payload) => {
      state.queue = payload.products || [];
      state.generatedAt = payload.generatedAt || "";
      render();
    })
    .catch((error) => {
      app.innerHTML = `
        <article class="review-panel">
          <p class="eyebrow">Review Queue</p>
          <div class="empty-state">
            ${escapeHtml(error.message)}. Run the demo pipeline first so it can generate <code>assets/data/review_queue.json</code>.
          </div>
        </article>
        <aside class="review-panel">
          <p class="eyebrow">Next Step</p>
          <div class="empty-state">Expected flow: rebuild the demo site if needed, run <code>./run_pipeline_b.sh</code>, then reload this page.</div>
        </aside>
      `;
    });
})();

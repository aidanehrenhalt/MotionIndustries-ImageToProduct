# Demo_MFR_Review Pipeline Integration

Working checklist for merging `Demo_MFR_Review` into `Demo_MFR_Site` and connecting the review frontend to the local demo pipeline.

This document is intentionally scoped to the first implementation target:

- the review frontend works reliably against the demo pipeline's `output/` directory
- the review queue is built from pipeline artifacts instead of isolated frontend-only assumptions
- manual review results can later be written back as JSON without redesigning the data flow

---

## Current State Summary

### What already fits

- The current review frontend already expects a split output layout with:
  - `output/json/`
  - `output/images/`
- The main pipeline documentation already describes that same layout for non-MinIO runs.
- The frontend parser is tolerant of JSON field aliases and filename-based image matching.

### What is not yet integrated

- `Demo_MFR_Review` is currently a standalone frontend, not part of `Demo_MFR_Site`.
- The current review page uses browser folder access instead of a local site/backend integration.
- Review decisions are only kept in browser memory.
- The frontend does not currently use `output/rankings.csv` as an input source.
- The frontend does not currently write review results back to disk.

### Recommended integration direction

- Keep `Demo_MFR_Site` as the local host for the pipeline and review UI.
- Treat `output/` as the shared contract between pipeline and review frontend.
- Promote `output/rankings.csv` to the primary queue source.
- Use `output/json/` as metadata enrichment.
- Use `output/images/` as the local image source.
- Add `output/reviews/` for manual review results.

---

## Target Architecture

### Phase 1 target

The first usable merged flow should be:

1. Demo pipeline runs.
2. Pipeline writes artifacts into `output/`.
3. Review frontend loads reviewable products from `output/`.
4. Reviewer selects an image / decision in the UI.
5. Review result can be serialized into JSON.

### Desired output contract

```text
output/
  rankings.csv
  json/
    <product_id>.json
  images/
    <image files>
  reviews/
    <product_id>.review.json
```

### Source-of-truth by artifact

- `output/rankings.csv`
  - primary queue source
  - candidate image ranking
  - score display fields
- `output/json/*.json`
  - product metadata enrichment
  - fallback identifiers
  - optional explicit image references
- `output/images/*`
  - image rendering source
- `output/reviews/*.json`
  - manual review output

---

## Integration Goals

### Goal 1: Basic frontend review integration

The frontend review site should work alongside the demo pipeline by reading pipeline outputs directly and rendering a stable review queue.

### Goal 2: Ranking CSV integration

The frontend should no longer depend only on `output/json` and `output/images`. It should also ingest `output/rankings.csv`, and preferably treat it as the primary review queue source.

### Goal 3: Manual review JSON export

The review flow should persist manual review decisions as JSON files that can be reused by later pipeline stages or audit workflows.

---

## Phase 1 Scope

Focus on the minimum viable integration first.

### In scope

- Merge planning for `Demo_MFR_Review` into `Demo_MFR_Site`
- Read queue data from `output/`
- Join ranking CSV rows with JSON metadata and image files
- Render candidate images and review controls
- Define review JSON output schema
- Prepare the code structure for persisted review results

### Out of scope for the first pass

- Full production deployment concerns
- Elasticsearch-backed review loading
- MinIO-backed review image retrieval
- Authentication / reviewer identity management
- Review analytics beyond current history
- Auto-writeback into rankings CSV

---

## Main Design Decisions

### 1. Use `rankings.csv` as the primary queue source

Reason:

- It already represents ranked candidate images.
- It includes score-oriented fields that the review UI should surface.
- It is a better queue source than inferring ranking only from image filenames.

### 2. Keep JSON as enrichment, not the primary queue builder

Reason:

- JSON files are useful for names, descriptions, manufacturer info, and extra attributes.
- Queue construction should not fail just because a JSON file is missing or incomplete.

### 3. Keep images sourced from `output/images/`

Reason:

- This matches the current local frontend model.
- It avoids mixing review logic with MinIO concerns during the first integration pass.

### 4. Write review results to new JSON files, not back into source artifacts

Reason:

- Avoid mutating `rankings.csv`.
- Avoid mutating pipeline output JSON.
- Preserve a clear boundary between generated data and human review data.

---

## Expected Review Data Model

Each UI review item should normalize into a shape like:

```json
{
  "productId": "MB2-10",
  "queueKey": "MB2-10",
  "productName": "Stainless Steel Set Screw Locking Bearing Insert, MB200 Series",
  "manufacturer": "AMI Bearings Inc.",
  "manufacturerPartNumber": "MB2-10",
  "partNumber": "MB2-10",
  "description": "...",
  "jsonFile": "MB2-10.json",
  "candidateImages": [
    {
      "id": "MB2-10-mb2-10-1.jpg",
      "fileName": "mb2-10-1.jpg",
      "rank": 1,
      "finalScore": 0.626959,
      "finalScorePct": "62.7%",
      "aiConfidence": 0.9464,
      "textScore": 0.058824,
      "sourceName": "Manufacturer Site / AMI BEARINGS INC",
      "imagePath": "output/images/mb2-10-1.jpg"
    }
  ]
}
```

This does not need to match the final implementation exactly, but the frontend should normalize all inputs into one stable UI model.

---

## Expected Review Output Schema

Recommended first-pass file:

`output/reviews/<product_id>.review.json`

Suggested schema:

```json
{
  "product_id": "MB2-10",
  "queue_key": "MB2-10",
  "reviewed_at": "2026-03-30T12:00:00Z",
  "decision": "approved",
  "selected_image_id": "MB2-10-mb2-10-1.jpg",
  "selected_image_filename": "mb2-10-1.jpg",
  "selected_image_rank": 1,
  "selected_image_score": 0.626959,
  "feedback": "Primary catalog image is correct.",
  "feedback_tags": [],
  "source_json_file": "MB2-10.json",
  "source_ranking_rows": [
    {
      "image_filename": "mb2-10-1.jpg",
      "image_rank": 1,
      "final_score": 0.626959
    }
  ]
}
```

### Notes

- Keep the saved review file additive and self-contained.
- Include enough source data to audit the decision later.
- Do not require the original frontend session state to reconstruct what happened.

---

## Phase 1 Checklist

### A. Confirm the output contract

- [ ] Confirm the canonical product key used across:
  - `rankings.csv`
  - JSON filenames
  - JSON metadata fields
  - image filenames
- [ ] Confirm whether `motion_product_id` is the correct join key for review items.
- [ ] Confirm whether `output/images/` uses a flat layout or nested product subfolders.
- [ ] Confirm whether image filenames in `rankings.csv` are sufficient to resolve local files.
- [ ] Confirm whether every reviewable product is expected to have:
  - a rankings row
  - a JSON file
  - at least one local image

### B. Replace the frontend-only queue builder

- [ ] Refactor the current parser so queue construction starts from `output/rankings.csv`.
- [ ] Group CSV rows into one review item per product.
- [ ] Sort each product's candidate images by rank or final score.
- [ ] Preserve score fields needed by the UI:
  - `image_rank`
  - `final_score`
  - `final_score_pct`
  - `ai_confidence`
  - `text_score`
  - `source_name`
- [ ] Join each grouped product with matching JSON metadata from `output/json/`.
- [ ] Resolve each candidate image against `output/images/`.
- [ ] Keep the queue resilient when JSON is missing.
- [ ] Keep the queue resilient when an image file is missing.

### C. Adapt the review UI to the normalized queue

- [ ] Update the review page to load pipeline-backed queue data instead of only folder-parser output.
- [ ] Update the product summary card to show ranking-derived fields where useful.
- [ ] Update the image table/gallery to show the score fields from the CSV.
- [ ] Keep current behavior for:
  - selecting a candidate image
  - submitting a review decision
  - skipping when no valid image exists
- [ ] Preserve the one-product-at-a-time workflow.

### D. Add review export design

- [ ] Finalize the `output/reviews/<product_id>.review.json` schema.
- [ ] Decide whether Phase 1 review export is:
  - browser download only
  - local backend write
- [ ] Prefer backend write for the merged local site.
- [ ] Ensure saved review JSON includes:
  - product id
  - decision
  - selected image
  - selected rank/score
  - timestamp
  - feedback

### E. Prepare merge into `Demo_MFR_Site`

- [ ] Decide whether `Demo_MFR_Review` is merged as:
  - a route/page inside the existing site
  - a feature module inside the same frontend app
  - a temporary embedded standalone app
- [ ] Identify frontend files that can be reused unchanged.
- [ ] Identify parser/data-loading code that should be replaced instead of carried over.
- [ ] Keep the review UI visually separate from pipeline generation logic.

---

## Recommended Implementation Order

### Step 1: Normalize queue loading

Build a loader that reads:

- `output/rankings.csv`
- `output/json/`
- `output/images/`

And returns one normalized review queue structure for the frontend.

This is the highest-value first step because everything else depends on a stable queue model.

### Step 2: Update frontend components to consume normalized ranking data

Once queue normalization works, wire the current review UI to it with minimal visual changes.

### Step 3: Add persisted review JSON output

After the review queue is stable, implement JSON writeback for manual review results.

### Step 4: Move from folder-picker mode to local-site integration

Once the merged site owns the data-loading path, replace browser folder access with a proper local integration path.

---

## Recommended Code Changes

### Existing frontend files likely to change

- `Demo_MFR_Review/Demo_MFR_Review/client/src/pages/ReviewPage.jsx`
- `Demo_MFR_Review/Demo_MFR_Review/client/src/App.jsx`
- `Demo_MFR_Review/Demo_MFR_Review/client/src/components/ProductCard.jsx`
- `Demo_MFR_Review/Demo_MFR_Review/client/src/components/ConfidenceTable.jsx`
- `Demo_MFR_Review/Demo_MFR_Review/client/src/utils/pipelineFolderParser.js`

### Recommended new loader modules

- `rankingsCsvParser.js`
- `reviewQueueNormalizer.js`
- `reviewExport.js`

These can live beside the current parser utilities or replace them gradually.

---

## Risks and Merge Concerns

### Low-risk areas

- Reusing current review UI components
- Reusing current product/image selection flow
- Reusing the current JSON metadata display model

### Medium-risk areas

- Joining product IDs consistently across CSV, JSON, and image filenames
- Handling missing or stale files in `output/`
- Preserving queue stability during auto-refresh

### Higher-risk areas

- If `Demo_MFR_Site` has a conflicting frontend architecture
- If `rankings.csv` schema is still changing
- If the pipeline sometimes outputs MinIO-only paths instead of local image files

---

## Questions To Resolve Before Full Implementation

These are not blockers for starting, but they should be answered during the first build phase.

- [ ] What exact field in `rankings.csv` should be the canonical review product key?
- [ ] Should already-reviewed products be excluded by checking for `output/reviews/*.json`?
- [ ] Should review JSON overwrite the prior review for a product, or append history?
- [ ] Should the review UI remain auto-refreshing once JSON writeback is added?
- [ ] Will the merged local site have a backend capable of writing files directly?

---

## Immediate Next Step

Implement the queue-loading layer first.

Definition of done for that step:

- the frontend can build a review queue from `output/rankings.csv`
- each review item is enriched from `output/json/` when available
- candidate images resolve from `output/images/`
- the current review page can render those results without relying solely on filename heuristics

Once that is in place, the next step is review JSON persistence.

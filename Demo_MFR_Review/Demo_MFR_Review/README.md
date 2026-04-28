# Motion Industries — Image-to-Product Review Dashboard

## What changed

This version keeps the original 3-page frontend, but the **Output UI** now supports only one review-data source:

- **Connect Live Folder**

The live folder must contain exactly two pipeline subfolders:

```text
ROOT/
  json/
    12345.json
    67890.json
  images/
    12345_1.png
    12345_2.png
    67890_1.png
```

## How the Output UI works now

When you connect the root folder, the frontend:

1. reads every JSON file inside **/json**
2. reads every image file inside **/images**
3. builds one review product per JSON file / part
4. shows only **one pending part at a time** in the Output UI
5. tries to match images to parts using the values it can find in both places:
   - item number
   - part number
   - manufacturer part number
   - JSON filename
   - image filename
   - explicit image filenames embedded in the JSON, if present
6. refreshes the root folder every **15 seconds**

## Matching logic

The new parser is designed for a split-output layout where metadata and images are stored separately.

It matches images to a JSON part using a scoring system based on shared identifiers, for example:

- `json/MI-123456.json` ↔ `images/MI-123456_1.png`
- JSON field `itemNumber = "123456"` ↔ image filename `123456-image-2.png`
- JSON image reference `"filename": "result_a.png"` ↔ `images/result_a.png`

If a JSON file loads but no image matches it yet, the product still appears in the queue with a **No matched images** state so the reviewer can skip it.

## Important limitation

This is still a **frontend-only** folder connection.

That means:
- it works for local testing in Chrome / Edge using the browser's folder access API
- it does **not** let a deployed website read files directly from a server disk on its own
- for a true deployed live server connection, your backend still needs to expose those files through an API or another server-side integration

## Browser support

Use:
- Google Chrome
- Microsoft Edge

## How to run

```bash
cd client
npm install
npm start
```

## Main frontend files changed

### `client/src/pages/ReviewPage.jsx`
- removed all non-live source options
- the Output UI now only supports **Connect Live Folder**
- updated all messaging to the `json/ + images/` layout
- keeps auto-refresh every 15 seconds

### `client/src/utils/pipelineFolderParser.js`
- rewritten for the new split-folder output format
- reads metadata only from `/json`
- reads candidate images only from `/images`
- matches images to JSON parts using filename and identifier heuristics

### `client/src/components/ProductCard.jsx`
- now shows the JSON file name and matched image count

## In plain English

The review page now does one thing only:

- open the root output folder
- read the part details from **json**
- read the candidate images from **images**
- match them together
- keep refreshing automatically


## Latest behavior fixes

- the Output UI now keeps the connected live folder when you switch between the 3 frontend tabs
- the parser now treats each JSON file as its own review item so multiple JSON files do not get merged together into one screen
- the reviewer still works through the queue one part at a time

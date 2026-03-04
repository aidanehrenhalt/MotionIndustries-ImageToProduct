# Motion Industries — Image Review Dashboard

## How to Run

1. Install Node.js from https://nodejs.org (LTS version)
2. Open a terminal in this folder
3. Run:

```
cd client
npm install
npm start
```

4. Browser opens to http://localhost:3000 automatically

## File Structure

```
client/
├── public/index.html              ← HTML shell
├── package.json                   ← Dependencies
└── src/
    ├── index.jsx                  ← React entry point
    ├── App.jsx                    ← Root component, global state
    ├── data/
    │   └── mockData.js            ← Sample product data
    ├── components/
    │   ├── Layout.jsx             ← Sidebar, colors, icons, helpers
    │   ├── ProductCard.jsx        ← Product metadata card
    │   ├── ImageGallery.jsx       ← Main image + thumbnails
    │   ├── ConfidenceTable.jsx    ← Ranking table
    │   ├── ReviewActions.jsx      ← Accept/Reject/Skip + feedback
    │   ├── ProgressBar.jsx        ← Queue progress indicator
    │   ├── HistoryStats.jsx       ← Summary stat cards
    │   ├── HistoryTable.jsx       ← Searchable history table
    │   └── DetailModal.jsx        ← Slide-out review detail panel
    └── pages/
        ├── ReviewPage.jsx         ← Review queue workflow
        └── HistoryPage.jsx        ← History + stats page
```

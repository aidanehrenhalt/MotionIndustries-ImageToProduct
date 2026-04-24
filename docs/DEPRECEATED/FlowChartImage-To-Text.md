# Image-to-Product Pipeline Flow

## Overview
This document translates the flowchart for the Motion Industries Image-to-Product pipeline into text.

## Pipeline Steps

1. **User provides product input**
   - The user uploads or supplies a **Product Excel** file.
   - This file is sent into the **Input UI**.

2. **Input UI processes product data**
   - The **Input UI** extracts or passes along the **Product Name & Description**.
   - The **Input UI** also provides:
     - **Product Labels (PGC1)**
     - **Text Description**

3. **Web Scraper gathers candidate content**
   - The **Product Name & Description** are sent from the **Input UI** to the **Web Scraper**.
   - The **Web Scraper** returns:
     - **Web Images**
     - **Website Text**

4. **Image classification stage**
   - The **Image Classifier** receives:
     - **Web Images** from the **Web Scraper**
     - **Product Labels (PGC1)** from the **Input UI**
     - **Text Description** from the **Input UI**
   - The **Image Classifier** produces **Assigned Classes**.

5. **Text analysis stage**
   - The **Text Analysis Model** receives:
     - **Website Text** from the **Web Scraper**
     - **Text Description** from the **Input UI**
   - The **Text Analysis Model** produces **Similarity Scores**.

6. **Ranking stage**
   - The **Ranker** combines:
     - **Assigned Classes** from the **Image Classifier**
     - **Similarity Scores** from the **Text Analysis Model**
   - The **Ranker** outputs:
     - **Top Ranked Images**
     - **Metrics**

7. **Output review stage**
   - The **Output UI** receives:
     - **Top Ranked Images** from the **Ranker**
     - **Metrics** from the **Ranker**
   - The **Output UI** presents the results for review.

8. **User decision loop**
   - The **User** reviews results in the **Output UI**.
   - The user can:
     - move to the **Next Product Image / Next Best Image**
     - **Approve** or **Reject** results
   - The **Approve/Reject** decision is fed back into the **Output UI** workflow.

## Condensed Flow

```text
User
  -> Product Excel
  -> Input UI

Input UI
  -> Product Name & Description -> Web Scraper
  -> Product Labels (PGC1) -> Image Classifier
  -> Text Description -> Image Classifier
  -> Text Description -> Text Analysis Model

Web Scraper
  -> Web Images -> Image Classifier
  -> Website Text -> Text Analysis Model

Image Classifier
  -> Assigned Classes -> Ranker

Text Analysis Model
  -> Similarity Scores -> Ranker

Ranker
  -> Top Ranked Images -> Output UI
  -> Metrics -> Output UI

Output UI
  <-> User review
  <- Approve/Reject
  -> Next Product Image / Next Best Image
```

## Functional Interpretation
- The system starts with structured catalog input from the user.
- Product metadata is used to scrape candidate images and related website text.
- Images are classified using both the scraped images and product context.
- Text is analyzed to measure how well scraped website text matches the product description.
- A ranker combines image classification output and text similarity output to score candidate images.
- The output interface presents the best-ranked images and supporting metrics for human review.
- The user then approves, rejects, or moves to the next best candidate image.

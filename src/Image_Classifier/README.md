# Image Classifier

A CNN-based image classifier that predicts the product category class for a given product image. Loads a pretrained model (`trained_model.pth`) and runs inference on a single image pulled from the product catalog.

## Requirements

```bash
pip install torch torchvision pandas pillow openpyxl
```

## File Dependencies

| File | Description |
|------|-------------|
| `trained_model.pth` | Pretrained CNN weights |
| `cleaned_product_list.xlsx` | Product catalog with `PrimaryImageFilename` column |
| Image dataset directory | Folder containing the actual product images |

## Model Architecture

A 5-block CNN followed by a fully connected classifier, built with `torch.nn.Sequential`.

| Layer | Details |
|-------|---------|
| Conv Block 1 | Conv2d(3→25) + BatchNorm + ReLU + MaxPool |
| Conv Block 2 | Conv2d(25→50) + BatchNorm + ReLU + Dropout(0.2) + MaxPool |
| Conv Block 3 | Conv2d(50→75) + BatchNorm + ReLU + MaxPool |
| Conv Block 4 | Conv2d(75→75, stride=2) + BatchNorm + ReLU + MaxPool |
| Conv Block 5 | Conv2d(75→75, stride=2) + BatchNorm + ReLU + Dropout(0.2) + MaxPool |
| FC Head | Flatten → Linear(1200→512) → ReLU → Dropout(0.3) → Linear(512→8) |

- **Input size:** 500 × 500 × 3 (RGB)
- **Output:** 8 class logits (`N_CLASSES = 8`)
- **Inference device:** CPU

## Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `N_CLASSES` | `8` | Number of output classes |
| `IMG_H` / `IMG_W` | `500` | Input image dimensions (px) |
| `IMG_C` | `3` | Number of color channels (RGB) |
| `model_path` | `trained_model.pth` | Path to saved model weights |
| `Excel_path` | `cleaned_product_list.xlsx` | Path to product catalog |
| `img_folder` | `/home/hice1/rlopez76/scratch/motion_dataset` | Root directory for images |

## Preprocessing

Images are resized to 500×500 and converted to a normalized tensor:

```python
transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
])
```

## Usage

Update the hardcoded paths and row index at the bottom of the script as needed, then run:

```bash
python img_classifier.py
```

The script will print the predicted class index for the selected product image:

```
Predicted class index: 3
```

## Notes

- Currently runs inference on a single hardcoded row (`df['PrimaryImageFilename'][3000]`). Update the index to target a different product.
- `img_folder` is hardcoded to a specific scratch directory — update this path for your environment.
- The model runs on CPU (`map_location=torch.device('cpu')`). For GPU inference, update the `map_location` argument.
- Class index-to-label mapping is defined in `pgc_mapping.py`.
- For batch inference across scraped JSON records, see `classify_json_images.py`.

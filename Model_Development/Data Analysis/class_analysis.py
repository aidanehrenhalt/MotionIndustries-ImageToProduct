import os
import pandas as pd
from PIL import Image

# Load the excel file
file_path = 'GT Capstone Image Mapping.xlsx'
df = pd.read_excel(file_path)

# Determining Values in PGC Description Column
labels = df['PGC1 Description'].unique()
#print("Values of PGC Description", labels)
print("Number of unique PGC1 Descriptions in GT Capstone Image Mapping.xlsx", len(labels))

# Determining Values in PGC Column
pgc_numbers = df['PGC1'].unique()
#print("Values of PGC", pgc_numbers)
print("Number of unique PGC1s in GT Capstone Image Mapping.xlsx", len(pgc_numbers))

# Print the number of elements that belong to each PGC number
print("Number of elements per PGC1:")
print(df['PGC1'].value_counts())

# Calculate min and max width and height of images in cleaned_product_list.xlsx
cleaned_df = pd.read_excel('cleaned_product_list.xlsx')
image_dir = '/home/hice1/rlopez76/scratch/motion_dataset'

min_width, max_width = float('inf'), 0
min_height, max_height = float('inf'), 0
sum_width, sum_height = 0, 0

print("\nCalculating image dimensions for cleaned_product_list.xlsx...")
valid_images = 0
for _, row in cleaned_df.iterrows():
    img_filename = str(row['PrimaryImageFilename'])
    img_path = os.path.join(image_dir, img_filename)
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w < min_width: min_width = w
                if w > max_width: max_width = w
                if h < min_height: min_height = h
                if h > max_height: max_height = h
                sum_width += w
                sum_height += h
                valid_images += 1
        except Exception:
            pass

if valid_images > 0:
    avg_width = sum_width / valid_images
    avg_height = sum_height / valid_images
    print(f"Processed {valid_images} images.")
    print(f"Minimum width: {min_width}")
    print(f"Maximum width: {max_width}")
    print(f"Average width: {avg_width:.2f}")
    print(f"Minimum height: {min_height}")
    print(f"Maximum height: {max_height}")
    print(f"Average height: {avg_height:.2f}")
else:
    print("No valid images found.")
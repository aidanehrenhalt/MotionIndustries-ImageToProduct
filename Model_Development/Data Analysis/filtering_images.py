import pandas as pd
import os

def filter_existing_images(excel_path, image_folder, column_name="PrimaryImageFilename"):
    """
    Reads an Excel file and removes rows where the image file (found in image_folder) 
    does not exist on disk.
    
    Parameters:
    - excel_path: Path to the .xlsx file.
    - image_folder: The directory path where the images are actually stored.
    - column_name: The name of the column containing the image filenames.
    """
    
    # 1. Load the Excel file into a DataFrame
    df = pd.read_excel(excel_path)
    
    # 2. Define a helper to check the full path
    def check_file(filename):
        # Handle potential empty/NaN cells
        if pd.isna(filename):
            return False
        
        # Combine folder path with the filename from the cell
        full_path = os.path.join(image_folder, str(filename))
        
        # Return True if it exists and is a file
        return os.path.isfile(full_path)

    # 3. Create a mask and filter the DataFrame
    # This checks existence for every row
    keep_mask = df[column_name].apply(check_file)
    df_cleaned = df[keep_mask].copy()
    
    # Summary of changes
    original_count = len(df)
    final_count = len(df_cleaned)
    print(f"Processing complete.")
    print(f"Original rows: {original_count}")
    print(f"Rows kept:     {final_count}")
    print(f"Rows deleted:  {original_count - final_count}")
    
    return df_cleaned

# --- Example Usage ---
cleaned_df = filter_existing_images(
    excel_path="GT Capstone Image Mapping.xlsx", 
    image_folder="/home/hice1/rlopez76/scratch/motion_dataset",
    column_name="PrimaryImageFilename"
)

# Optional: Save the cleaned version back to a new Excel file
cleaned_df.to_excel("cleaned_product_list.xlsx", index=False)
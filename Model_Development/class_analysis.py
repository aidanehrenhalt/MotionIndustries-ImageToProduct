import pandas as pd

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
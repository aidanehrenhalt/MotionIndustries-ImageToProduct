import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from IPython.display import display

from torchvision import transforms

N_CLASSES = 8
KERNEL_SIZE = 3
#FLATTENED_IMG_SIZE = IMG_H * IMG_W * IMG_C

model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(25),
        nn.ReLU(),        
        nn.MaxPool2d(2, stride = 2),
        nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2, stride = 2),
        nn.Conv2d(in_channels=50, out_channels=75, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),
        nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),
        nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2, stride = 2),
        nn.Flatten(),
        nn.Linear(1200, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES),
        )

model_path = 'trained_model.pth'
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
model.eval()

IMG_H = 500
IMG_W = 500
IMG_C = 3

# Transform to resize images
preprocess = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

Excel_path = 'cleaned_product_list.xlsx'

df = pd.read_excel(Excel_path)

img_folder = '/home/hice1/rlopez76/scratch/motion_dataset'
img_name = str(df['PrimaryImageFilename'][3000])

img_name = os.path.join(img_folder, img_name)

img = Image.open(img_name).convert('RGB')

img_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    
_, predicted = torch.max(output.data, 1)
print(f"Predicted class index: {predicted.item()}")

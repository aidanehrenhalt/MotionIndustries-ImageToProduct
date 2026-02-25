import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from torchvision import transforms

IMG_H = 224
IMG_W = 224
IMG_C = 3

# Transform to resize images
motion_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

# Custom Dataset class to read the data from the excel file and image directory
class MotionDataset(Dataset):
    def __init__(self, xlsx_file, image_dir, transform=None):
        """
        Args:
            xlsx_file (string): Path to the excel file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_excel(xlsx_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir,
                                str(self.data_info.iloc[idx]['PrimaryImageFilename']))
        image = Image.open(img_name).convert('RGB')

        # Assuming 'PGC' is the target label
        label = self.data_info.iloc[idx]['PGC']

        if self.transform:
            image = self.transform(image)

        return image, label

dataset = MotionDataset(xlsx_file='GT Capstone Image Mapping.xlsx', image_dir='/home/hice1/rlopez76/scratch/motion_dataset', transform=motion_transform)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32) # need to look into other parameters
test_loader = DataLoader(test_dataset, batch_size=32)

train_N = len(train_loader.dataset)
test_N = len(test_loader.dataset)

N_CLASSES = dataset.data_info['PGC'].nunique()
KERNEL_SIZE = 3
FLATTENED_IMG_SIZE = IMG_H * IMG_W * IMG_C

model = nn.Sequential(
    #First Convolution
    nn.Conv2d(IMG_C, 25, KERNEL_SIZE, stride=1, padding=1),
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride = 2),
    #Second Convolution
    nn.Conv2d(25, 50, KERNEL_SIZE, stride=1, padding=1),
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.MaxPool2d(2, stride=2),
    #Third Convolution
    nn.Conv2d(50, 75, KERNEL_SIZE, stride=1, padding=1),
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    #Flatten to dense
    nn.Flatten(),
    nn.Linear(FLATTENED_IMG_SIZE, 512),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, N_CLASSES)
)

model = torch.compile(model.to("cuda"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim = 1, keepdim = True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    acc = correct / N
    return acc

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in training_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_fn(output, y)
        batch_loss.backward()
        optimizer.step()
        
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)

        print("Training Loss", loss)
        print("Training Accuracy", accuracy)

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            loss += loss_fn(output, y).item()
            accuracy += get_batch_accuracy(output, y, test_N)
    
    print("Testing Loss", loss)
    print("Testing Accuracy", accuracy)

# %%time
EPOCHS = 20
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}")
    train()
    validate()
    print("\n")
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "five_conv_model_focal_loss.tar"

IMG_H = 500
IMG_W = 500
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

        # 'PGC1' is the target label
        label = self.data_info.iloc[idx]['PGC1']-1

        if self.transform:
            image = self.transform(image)

        return image, label

dataset = MotionDataset(xlsx_file='cleaned_product_list.xlsx', image_dir='/home/hice1/rlopez76/scratch/motion_dataset', transform=motion_transform)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32) # need to look into other parameters
test_loader = DataLoader(test_dataset, batch_size=32)

train_N = len(train_loader.dataset)
test_N = len(test_loader.dataset)

N_CLASSES = 8
KERNEL_SIZE = 3
FLATTENED_IMG_SIZE = IMG_H * IMG_W * IMG_C

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
        nn.Linear(512, N_CLASSES),
        )

model = model.to(device)
# loss_fn = nn.CrossEntropyLoss()
test_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }, save_path)
    
def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint.get('train_loss', None)
    test_loss = checkpoint.get('test_loss', None)
    train_acc = checkpoint.get('train_acc', None)
    test_acc = checkpoint.get('test_acc', None)
    
    return model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim = 1, keepdim = True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    acc = correct / N
    return acc

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)

        y_one_hot = F.one_hot(y, num_classes=N_CLASSES).float()

        optimizer.zero_grad()
        #batch_loss = loss_fn(output, y)
        batch_loss = sigmoid_focal_loss(output, y_one_hot, alpha=0.25, gamma=2.0, reduction='mean')
        batch_loss.backward()
        optimizer.step()
        
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)

    print("Training Loss", loss)
    print("Training Accuracy", accuracy)
    return loss, accuracy

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss += test_loss_fn(output, y).item()
            accuracy += get_batch_accuracy(output, y, test_N)
            
    print("Testing Loss", loss)
    print("Testing Accuracy", accuracy)
            
    return loss, accuracy


os.makedirs("Model_Checkpoints", exist_ok=True)

if (os.path.isfile("Model_Checkpoints/" + model_name)):
    model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc = load_checkpoint(model, optimizer, "Model_Checkpoints/" + model_name)
else:
    epoch = 0
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

N_EPOCHS = 50
while epoch < N_EPOCHS:
    epoch += 1
    print(f"Epoch {epoch}")
    current_train_loss, current_train_acc = train()
    current_test_loss, current_test_acc = validate()
    train_loss.append(current_train_loss)
    train_acc.append(current_train_acc)
    test_loss.append(current_test_loss)
    test_acc.append(current_test_acc)
    save_checkpoint(model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc, "Model_Checkpoints/" + model_name)
    print("\n")
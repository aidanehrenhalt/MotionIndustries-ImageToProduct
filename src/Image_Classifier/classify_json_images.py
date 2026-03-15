import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

N_CLASSES = 8

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(25),
    nn.ReLU(),        
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(in_channels=50, out_channels=75, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.MaxPool2d(2, stride=2),
    nn.Flatten(),
    nn.Linear(1200, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, N_CLASSES),
)

def main():
    project_root = '/home/hice1/rlopez76/Senior_Design/MotionIndustries-ImageToProduct'
    model_path = os.path.join(project_root, 'src', 'Image_Classifier', 'trained_model.pth')
    
    # Load the trained model
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    IMG_H = 500
    IMG_W = 500

    # Transform to resize images and convert to tensor
    preprocess = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
    ])

    json_dir = os.path.join(project_root, 'output', 'json')
    
    # Iterate over all JSON files in output/json
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(json_dir, filename)
        
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON from {filename}")
                continue
            
        candidate_images = data.get('candidate_images', [])
        # If no candidate images, do not modify the JSON file
        if not candidate_images:
            continue
            
        modified = False
        for img_info in candidate_images:
            local_path = img_info.get('local_path')
            if not local_path:
                continue
                
            full_img_path = os.path.join(project_root, local_path)
            if not os.path.exists(full_img_path):
                print(f"Image not found: {full_img_path}")
                continue
                
            try:
                # Open image, convert to RGB, preprocess and unsqueeze to add batch dimension
                img = Image.open(full_img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                
                # Predict class
                with torch.no_grad():
                    output = model(img_tensor)
                    
                _, predicted = torch.max(output.data, 1)
                
                # Add predicted_class attribute
                img_info['predicted_class'] = predicted.item()
                modified = True
            except Exception as e:
                print(f"Error processing image {full_img_path}: {e}")
                
        # If any candidate image was successfully classified, save the modified JSON
        if modified:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Updated {filename}")

if __name__ == '__main__':
    main()

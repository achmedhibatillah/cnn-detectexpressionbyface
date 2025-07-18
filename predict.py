from PIL import Image
import torch
import json
import torchvision.transforms as transforms
import torch.nn as nn
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), # 48 -> 46
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), # 46 -> 44
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 44 -> 22

            nn.Dropout(0.25), # Early dropout

            nn.Conv2d(32, 64, kernel_size=3), # 22 -> 20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # 20 -> 18
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 18 -> 9

            nn.Conv2d(64, 128, kernel_size=3), # 9 -> 7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 7 -> 3

            nn.AdaptiveAvgPool2d((1, 1)) # output: (batch, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), # (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
base_dir = os.path.dirname(__file__) # lokasi detect.py

# Load class mapping dari JSON
with open(os.path.join(base_dir, "classes.json")) as f:
    classes = json.load(f)


# Load model dan class
model = CNN().to(device)
model.load_state_dict(torch.load(os.path.join(base_dir, 'model_cnn.pth')))
model.eval()

classes = torch.load(os.path.join(base_dir, 'classes.pth'))

# Load gambar
img_path = os.path.join(base_dir, 'data/detect/image.png')
img = Image.open(img_path).convert("L") # grayscale

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_tensor = transform(img).unsqueeze(0).to(device)

# Prediksi
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    label = classes[predicted.item()]

print(f"Ekspresi terdeteksi: {label}")

# Visualisasi matplotlib
import matplotlib.pyplot as plt

conv1 = model.features[0] # asumsi ini Conv2d pertama
weights = conv1.weight.detach().cpu() # shape: (32, 1, 3, 3)

fig, axs = plt.subplots(4, 8, figsize=(12, 6))
axs = axs.flatten()

for i in range(32):
    axs[i].imshow(weights[i][0], cmap='gray')
    axs[i].set_title(f'Filter {i}')
    axs[i].axis('off')

plt.tight_layout()
plt.show()

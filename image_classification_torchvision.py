# image_classification_torchvision.py

# This script performs image classification using PyTorch and torchvision.
# It loads a pretrained model, prepares image transforms, and classifies an input image.
# Common in computer vision tasks like object detection, classification, etc.

import torch  # PyTorch is an open-source deep learning library
import torchvision.transforms as transforms  # For preprocessing images
from torchvision import models  # Includes pretrained models like ResNet, VGG
from PIL import Image  # To load and process images

# Load a sample image (change the filename to your own image if needed)
image_path = "cat.jpg"
img = Image.open(image_path).convert("RGB")  # Load and convert the image to RGB format

# Define the standard image preprocessing steps
# Resize, center-crop, convert to tensor, normalize to ImageNet stats
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values used during ImageNet training
        std=[0.229, 0.224, 0.225]    # Standard deviations used during ImageNet training
    )
])

# Apply the transformation
img_t = transform(img)
img_t = img_t.unsqueeze(0)  # Add batch dimension (N, C, H, W) for model input

# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode (important to disable dropout/batchnorm updates)

# Perform inference (disable gradient computation for speed/memory)
with torch.no_grad():
    output = model(img_t)  # Forward pass

# Get the predicted class index
display_idx = output.argmax(dim=1).item()  # Get the index of the highest score
print(f"Predicted class index: {display_idx}")

# Optionally load ImageNet class labels to make this more human-friendly
# Downloadable from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
    print(f"Predicted class label: {labels[display_idx]}")

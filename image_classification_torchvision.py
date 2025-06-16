import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True).to(device)
model.eval()

url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
image = Image.open(requests.get(url, stream=True).raw)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.DEFAULT
labels = weights.meta["categories"]
predicted = output[0].softmax(0).argmax().item()

print("Predicted:", labels[predicted])

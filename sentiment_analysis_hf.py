from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    classifier = pipeline("sentiment-analysis", device=device)
else:
    classifier = pipeline("sentiment-analysis")  # Let Hugging Face choose CPU internally

results = classifier([
    "I love this product! It's amazing!",
    "This is the worst experience I've ever had."
])
for result in results:
    print(result)

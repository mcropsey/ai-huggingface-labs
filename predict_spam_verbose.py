import torch
from torch import nn
import numpy as np

# Feature labels (same order as training)
feature_names = [
    "num_links", "num_images", "has_spam_words", "all_caps_words",
    "num_exclamations", "contains_offer", "reply_requested", "sender_blacklisted",
    "html_content", "short_subject"
]

# Define model
def build_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(torch.load("spam_classifier.pth", map_location=device))
model.eval()
print(f"📥 Loaded trained spam classifier on {device}\n")

# Sample input emails (10 features each)
# Format: [links, images, spam_words, caps, exclamations, offer, reply_req, blacklisted, html, short_subj]
test_data = np.array([
    [0.9, 0.1, 0.8, 0.2, 0.5, 0.9, 0.1, 0.8, 0.2, 0.3],  # likely spam
    [0.1, 0.5, 0.1, 0.2, 0.3, 0.0, 0.2, 0.1, 0.4, 0.6],  # likely not spam
    [0.7, 0.2, 0.7, 0.6, 0.5, 0.7, 0.5, 0.6, 0.3, 0.2],  # borderline spam
])

inputs = torch.tensor(test_data, dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    probs = model(inputs)
    preds = (probs > 0.5).int()

# Output results
for i, (row, prob, pred) in enumerate(zip(test_data, probs, preds)):
    print(f"\n📨 Email {i+1}:")
    for fname, fval in zip(feature_names, row):
        print(f"  {fname:<20}: {fval:.2f}")
    print(f"🧠 Spam Probability : {prob.item():.4f}")
    print(f"🔍 Predicted Class  : {'SPAM' if pred.item() == 1 else 'NOT SPAM'}")



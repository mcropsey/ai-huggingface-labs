import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Selects GPU (if available) or falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📡 Training on device: {device}")

# 📊 Define 10 simulated email features.
# These are just example features that might affect spam detection
feature_names = [
    "num_links", "num_images", "has_spam_words", "all_caps_words",
    "num_exclamations", "contains_offer", "reply_requested", "sender_blacklisted",
    "html_content", "short_subject"
]

# 📨 Simulate 1000 fake emails, each with 10 random features (values from 0 to 1)
# This simulates our "email dataset"
print("📨 Generating synthetic spam dataset...")
X = torch.rand((1000, 10))  # 1000 emails × 10 features

# 🧠 Define a simple "spam score" rule:
# If the sum of some spammy features exceeds a threshold → it's spam (label 1)
# Specifically: num_links + has_spam_words + contains_offer + sender_blacklisted > 2
spam_score = X[:, 0] + X[:, 2] + X[:, 5] + X[:, 7]
y = (spam_score > 2).float()  # If score > 2, label as spam (1), else not spam (0)

# Package data for training
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 🤖 Define a simple neural network:
# - 10 input features → 32 neurons → 1 output (spam probability)
# - Sigmoid activation squashes output to range [0, 1] (good for binary prediction)
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
).to(device)

# Use Binary Cross Entropy Loss since we’re doing binary classification
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 🚀 Train the model for 5 epochs (loops over all emails 5 times)
print("🚀 Starting training...\n")
for epoch in range(5):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)  # match output shape
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print out training progress
    loss_val = loss.item()
    print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}")
    if loss_val > 0.5:
        print("🧠 Still learning basic spam patterns...\n")
    elif loss_val > 0.3:
        print("🔍 Getting better at spotting spam...\n")
    else:
        print("✅ Model is accurately predicting spam!\n")

# 💾 Save the trained model so it can be reused later
torch.save(model.state_dict(), "spam_classifier.pth")
print("✅ Model saved as 'spam_classifier.pth'")
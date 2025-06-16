import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📡 Training on device: {device}")

# Feature meanings (for simulation purposes)
feature_names = [
    "num_links", "num_images", "has_spam_words", "all_caps_words",
    "num_exclamations", "contains_offer", "reply_requested", "sender_blacklisted",
    "html_content", "short_subject"
]

# Create fake dataset
print("📨 Generating synthetic spam dataset...")
X = torch.rand((1000, 10))  # 1000 emails with 10 features

# Rule to simulate spam:
# If num_links + has_spam_words + contains_offer + sender_blacklisted > 2 → likely spam
spam_score = X[:, 0] + X[:, 2] + X[:, 5] + X[:, 7]
y = (spam_score > 2).float()  # Label as spam (1) if "spammy" score is high

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
).to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
print("🚀 Starting training...\n")
for epoch in range(5):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_val = loss.item()
    print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}")
    if loss_val > 0.5:
        print("🧠 Still learning basic spam patterns...\n")
    elif loss_val > 0.3:
        print("🔍 Getting better at spotting spam...\n")
    else:
        print("✅ Model is accurately predicting spam!\n")

# Save the model
torch.save(model.state_dict(), "spam_classifier.pth")
print("✅ Model saved as 'spam_classifier.pth'")

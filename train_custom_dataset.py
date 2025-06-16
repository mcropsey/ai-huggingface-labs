# train_custom_dataset.py

# This script demonstrates how to train a PyTorch model on a custom dataset using a DataLoader.
# It includes dataset loading, model definition, training loop, and loss computation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Sample custom dataset using PyTorch's Dataset class
class MyDataset(Dataset):
    def __init__(self):
        # Toy dataset: 100 samples with 10 features each
        self.data = torch.randn(100, 10)  # Random input features
        self.labels = torch.randint(0, 2, (100,))  # Binary classification labels (0 or 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 16)  # Input layer to hidden layer
        self.relu = nn.ReLU()         # Non-linear activation function
        self.fc2 = nn.Linear(16, 2)   # Hidden layer to output layer (2 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate dataset and wrap it in a DataLoader for batching
train_dataset = MyDataset()
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Train for 5 epochs
    total_loss = 0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()            # Clear previous gradients
        outputs = model(batch_data)      # Forward pass
        loss = criterion(outputs, batch_labels)  # Compute loss
        loss.backward()                  # Backpropagation
        optimizer.step()                 # Update weights
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Summary:
# - This demonstrates supervised learning with PyTorch
# - Model learns to classify synthetic data into 2 categories
# - `CrossEntropyLoss` + softmax is common for classification tasks

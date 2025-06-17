"""
What is the model learning to do?

It’s learning to map random 10-number inputs to a classification decision (label 0 or 1).


“Given these 10 numbers, is this sample more like a class 0 or a class 1?”

At first, the model is just guessing. But as it sees more data (through batches and epochs), it adjusts itself to make better predictions.


"""
# train_custom_dataset_verbose.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Custom Dataset Creation
# -----------------------
class MyDataset(Dataset):
    def __init__(self):
        print("Initializing custom dataset with 100 samples and 10 features each.")
        self.data = torch.randn(100, 10)  # Random input features
        self.labels = torch.randint(0, 2, (100,))  # Binary classification labels (0 or 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -----------------------
# Neural Network Definition
# -----------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        print("Building the neural network model...")
        self.fc1 = nn.Linear(10, 16)  # Input layer to hidden layer
        self.relu = nn.ReLU()         # Activation
        self.fc2 = nn.Linear(16, 2)   # Hidden layer to output layer (2 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# -----------------------
# Setup
# -----------------------
print("Loading dataset and preparing data loader...")
train_dataset = MyDataset()
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print("Instantiating the model, loss function, and optimizer...")
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Training Loop
# -----------------------
print("\nStarting training...\n")
for epoch in range(5):  # Train for 5 epochs
    total_loss = 0
    print(f"Epoch {epoch+1} started...")
    
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"  Batch {i+1}:")
        print(f"    Input shape: {batch_data.shape}")
        print(f"    Labels: {batch_labels.tolist()}")

        optimizer.zero_grad()  # Reset gradients
        outputs = model(batch_data)  # Forward pass
        print(f"    Model outputs (logits): {outputs.detach().numpy()}")

        loss = criterion(outputs, batch_labels)  # Loss calculation
        print(f"    Loss for this batch: {loss.item():.4f}")
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1} complete. Total Loss: {total_loss:.4f}\n")

print("Training finished!")
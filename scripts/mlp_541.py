import numpy as np
from pathlib import Path
from src.data.synthetic import _sigmoid
import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 64
EPOCHS=500
LEARNING_RATE=0.001
RANDOM_SEED=42

data_dir= Path("data/raw")
X_file = data_dir / "synth_X_e4.csv"
y_file = data_dir / "synth_y_e4.csv"


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


#dtaa laoding
X=np.loadtxt(X_file, delimiter=",").astype(np.float32)
y=np.loadtxt(y_file, delimiter=",").astype(np.float32)

#train/test split
n=len(X)
idx=np.random.permutation(n)
split=int(0.8*n)


train_idx = idx[:split]
test_idx = idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Convert to torch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).unsqueeze(1)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

#MLP model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(5,16)
        self.Relu=nn.ReLU()
        self.fc2= nn.Linear(16,1)
        #self.fc3= nn.Linear(8,1)

    def forward(self, x):
        
        h1=self.Relu(self.fc1(x))

        out=self.fc2(h1)

        return out
    
model=MLP()

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#trainning loop
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss = {loss.item():.4f}")


#evaluation
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = (logits > 0).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total

#accuracy calculation
train_acc=accuracy(model, train_loader)
test_acc=accuracy(model, test_loader)

#results
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
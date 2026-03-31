import numpy as np
from pathlib import Path
from src.data.synthetic import _sigmoid
import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, TensorDataset
from scripts.evaluation import evaluate_binary_classifier, compute_margins, evaluate_with_logit_noise
import json
import datetime
import pandas as pd


BATCH_SIZE = 64
EPOCHS=500
LEARNING_RATE=0.001
RANDOM_SEED=42

# Load clean dataset
df = pd.read_csv("data/raw/4_proteins_clean.csv")

# Features (X)
X_file = df[["AGR2", "KRT19", "MUC16", "RRM2"]].values

# Labels (y)
y_file = df["cancer"].values


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
        self.fc1= nn.Linear(4,8)
        self.Relu=nn.ReLU()
        self.fc2= nn.Linear(8,1)
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

trained_model=model
#function to return trained model
def return_trained_model():
    
    return model

def prompt_and_save_model(model, metrics, model_name):

    answer = input("\nSave this model for HIL testing? (y/n): ").strip().lower()
    if answer not in ("y", "yes"):
        print("Model not saved.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = Path("models/trained") / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    #Save model weights
    torch.save(model.state_dict(), model_dir / "weights.pt")

    # Save metadata 
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics,
        "architecture": "5 → 16 → 1 ReLU",
        "decision_threshold": 0.0,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {model_dir}")

#results
print("\n=== Test set evaluation ===")
metrics = evaluate_binary_classifier(model, X_test, y_test, threshold=0.0)
#decide if model to be saved



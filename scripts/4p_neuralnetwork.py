from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler# Load data
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# Load clean dataset
df = pd.read_csv("data/raw/10_proteins_dataset.csv")

X = df.drop(columns=["cancer"]).values
y = df["cancer"].values.reshape(-1,1)

# Optional checks
print(df.head())
print(df.dtypes)
print(X.shape, y.shape)


#data_dir= Path("data/raw")
#X = data_dir / "synth_X_delta_0.8t.csv"
#y = data_dir / "synth_y_delta_0.8t.csv"



scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

class CancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
           
            nn.Linear(32,1 )
        )
    
    def forward(self, x):
        return self.model(x)

model = CancerModel()


pos_weight = torch.tensor([1.5])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)

    predicted = (probs > 0.5).float()
    accuracy = (predicted == y_test).float().mean()

print("Accuracy:", accuracy.item())

auc = roc_auc_score(y_test.numpy(), probs.numpy())
print("AUC:", auc)

predicted = (probs > 0.5).float()
print(predicted.sum())   # how many predicted as cancer

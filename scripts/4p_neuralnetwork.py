from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler# Load data
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# Load clean dataset
df = pd.read_csv("data/raw/4_proteins_clean.csv")

# Features (X)
X = df[["AGR2", "KRT19", "MUC16", "RRM2"]].values

# Labels (y)
y = df["cancer"].values

y = y.reshape(-1, 1)

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
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8,1 ),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

model = CancerModel()

criterion = nn.BCELoss()          # binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
EPOCHS = 500

for epoch in range(EPOCHS):
   
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    preds = model(X_test)
    predicted = (preds > 0.5).float()
    accuracy = (predicted == y_test).float().mean()

print("Accuracy:", accuracy.item())

with torch.no_grad():
    preds = model(X_test).numpy()

auc = roc_auc_score(y_test.numpy(), preds)
print("AUC:", auc)

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler# Load data
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# -----------------------------
# 1. LOAD RAW DATA (NO PRE-FILTERED FILES)
# -----------------------------
df = pd.read_csv("data/raw/final_olink_data_synthetic.csv")
meta = pd.read_csv("data/raw/combined_metadata_synthetic.csv")

# -----------------------------
# 2. PIVOT TO WIDE FORMAT
# -----------------------------
df_wide = df.pivot(index="Sample", columns="Assay", values="NPX")

# Merge labels
df_final = df_wide.merge(meta[["Sample", "Cancer"]], on="Sample")

# Convert labels to 0/1
df_final["Cancer"] = df_final["Cancer"].map({"Yes": 1, "No": 0})

# -----------------------------
# 3. SPLIT FIRST (CRITICAL)
# -----------------------------
train_df, test_df = train_test_split(
    df_final, test_size=0.2, random_state=42
)

# -----------------------------
# 4. FEATURE SELECTION (TRAIN ONLY)
# -----------------------------
scores = {}

for col in df_wide.columns:
    cancer_mean = train_df[train_df["Cancer"] == 1][col].mean()
    non_mean = train_df[train_df["Cancer"] == 0][col].mean()
    scores[col] = abs(cancer_mean - non_mean)

# Top 10 proteins
top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
top10_cols = [x[0] for x in top10]

print("Selected proteins:", top10_cols)

# -----------------------------
# 5. BUILD TRAIN / TEST SETS
# -----------------------------
X_train = train_df[top10_cols].values
y_train = train_df["Cancer"].values.reshape(-1, 1)

X_test = test_df[top10_cols].values
y_test = test_df["Cancer"].values.reshape(-1, 1)

print(X_train.shape, y_train.shape)

# -----------------------------
# 6. SCALE (TRAIN ONLY)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 7. CONVERT TO TORCH
# -----------------------------
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

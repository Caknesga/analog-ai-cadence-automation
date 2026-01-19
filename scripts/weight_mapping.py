import numpy as np
from pathlib import Path
from src.data.synthetic import _sigmoid
import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, TensorDataset


WEIGHTS_FILE = Path("models/trained/mlp_5-16-1_relu_test_20260119_114347/weights.pt")

def load_model_weights(model: nn.Module, weights_path: Path):
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    return model

print(f"Loading weights from {WEIGHTS_FILE}")

#MLP model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(5,8)
        self.Relu=nn.ReLU()
        self.fc2= nn.Linear(8,1)
        #self.fc3= nn.Linear(8,1)

    def forward(self, x):
        
        h1=self.Relu(self.fc1(x))

        out=self.fc2(h1)

        return out
    
model=MLP()

load_model_weights(model, WEIGHTS_FILE)

model.eval()

with torch.no_grad():
    print("\n=== Layer fc1 (5 → 8) ===")
    print("Weights (shape [8, 5]):")
    print(model.fc1.weight.detach().cpu().numpy())

    print("\nBiases (shape [8]):")
    print(model.fc1.bias.detach().cpu().numpy())

    print("\n=== Layer fc2 (8 → 1) ===")
    print("Weights (shape [1, 8]):")
    print(model.fc2.weight.detach().cpu().numpy())

    print("\nBiases (shape [1]):")
    print(model.fc2.bias.detach().cpu().numpy())
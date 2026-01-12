
import torch
import numpy as np

#evaluation
def evaluate_binary_classifier(model, X, y, threshold=0.0):

    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = (logits > threshold).int()

    y_true = y.int()

    TP = ((preds == 1) & (y_true == 1)).sum().item()
    TN = ((preds == 0) & (y_true == 0)).sum().item()
    FP = ((preds == 1) & (y_true == 0)).sum().item()
    FN = ((preds == 0) & (y_true == 1)).sum().item()

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    accuracy    = (TP + TN) / (TP + TN + FP + FN)

    print("Confusion Matrix:")
    print(f"           Pred 0    Pred 1")
    print(f"True 0      {TN:5d}     {FP:5d}")
    print(f"True 1      {FN:5d}     {TP:5d}")
    print()
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Sensitivity : {sensitivity:.3f}  (TPR)")
    print(f"Specificity : {specificity:.3f}  (TNR)")

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def compute_margins(model, X, threshold=0.0):
    model.eval()
    with torch.no_grad():
        logits = model(X).squeeze(1)
    margins = torch.abs(logits - threshold)
    return margins.cpu().numpy()

def evaluate_with_logit_noise(model, X, y, noise_std, threshold=0.0, n_mc=20):
    accs = []
    sens = []
    spec = []

    for _ in range(n_mc):
        model.eval()
        with torch.no_grad():
            logits = model(X).squeeze(1)
            noise = torch.randn_like(logits) * noise_std
            noisy_logits = logits + noise
            preds = (noisy_logits > threshold).int()

        y_true = y.int().squeeze(1)

        TP = ((preds == 1) & (y_true == 1)).sum().item()
        TN = ((preds == 0) & (y_true == 0)).sum().item()
        FP = ((preds == 1) & (y_true == 0)).sum().item()
        FN = ((preds == 0) & (y_true == 1)).sum().item()

        accs.append((TP + TN) / (TP + TN + FP + FN))
        sens.append(TP / (TP + FN + 1e-8))
        spec.append(TN / (TN + FP + 1e-8))

    return {
        "accuracy": np.mean(accs),
        "sensitivity": np.mean(sens),
        "specificity": np.mean(spec),
    }

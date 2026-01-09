# src/data/synthetic.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SyntheticConfig:
    n_samples: int = 5000
    n_features: int = 5
    cancer_rate: float = 0.15          # class imbalance: P(y=1)
    feature_noise_std: float = 0.10    # measurement noise on features
    label_noise_rate: float = 0.02     # flip labels (annotation noise)
    nonlinear: bool = True             # enable some nonlinearity
    random_seed: int = 42


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_medical_data(cfg: SyntheticConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: shape (n_samples, 5)
        y: shape (n_samples,) in {0,1}
    """
    rng = np.random.default_rng(cfg.random_seed)

    # Base latent factors (correlated-ish features)
    # Think: biomarkers with correlations + different scales
    cov = np.array([
        [1.0, 0.4, 0.2, 0.0, 0.1],
        [0.4, 1.0, 0.3, 0.1, 0.0],
        [0.2, 0.3, 1.0, 0.2, 0.1],
        [0.0, 0.1, 0.2, 1.0, 0.5],
        [0.1, 0.0, 0.1, 0.5, 1.0],
    ])
    L = np.linalg.cholesky(cov)

    Z = rng.normal(size=(cfg.n_samples, cfg.n_features))
    X = Z @ L.T

    # Feature scaling differences (common in medical features)
    scales = np.array([1.0, 0.6, 1.8, 0.9, 1.3])
    X = X * scales

    # Add measurement noise
    X = X + rng.normal(0.0, cfg.feature_noise_std, size=X.shape)

    # Build a ground-truth decision function
    # Linear term
    w = np.array([1.2, -0.9, 0.7, 0.4, -0.6])
    z = X @ w

    # Add nonlinearity to force benefit from an MLP (optional)
    if cfg.nonlinear:
        # e.g., interaction + saturation-like effects
        z = z + 0.8 * np.tanh(0.9 * X[:, 0] * X[:, 2]) - 0.6 * (X[:, 3] ** 2)

    # Choose bias to roughly hit desired cancer_rate
    # We solve for b so that mean(sigmoid(z + b)) ~= cancer_rate using binary search.
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = _sigmoid(z + mid).mean()
        if p > cfg.cancer_rate:
            hi = mid
        else:
            lo = mid
    b = (lo + hi) / 2.0

    probs = _sigmoid(z + b)
    y = (rng.uniform(size=cfg.n_samples) < probs).astype(np.int64)

    # Label noise (annotation errors)
    if cfg.label_noise_rate > 0.0:
        flip = rng.uniform(size=cfg.n_samples) < cfg.label_noise_rate
        y[flip] = 1 - y[flip]

    return X.astype(np.float32), y.astype(np.int64)


def save_dataset(
    out_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_seed: int = 123,
) -> None:
    """
    Saves:
      - data/processed/synth_X.csv
      - data/processed/synth_y.csv
      - data/splits/train_idx.npy, val_idx.npy, test_idx.npy
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    # Save CSVs (simple + tool-friendly)
    np.savetxt(out_dir / "synth_X.csv", X, delimiter=",")
    np.savetxt(out_dir / "synth_y.csv", y.reshape(-1, 1), delimiter=",", fmt="%d")

    # Indices split
    rng = np.random.default_rng(random_seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    np.save(out_dir / "splits" / "train_idx.npy", train_idx)
    np.save(out_dir / "splits" / "val_idx.npy", val_idx)
    np.save(out_dir / "splits" / "test_idx.npy", test_idx)

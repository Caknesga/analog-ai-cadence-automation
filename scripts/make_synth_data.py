# scripts/make_synth_data.py
from pathlib import Path
from src.data.synthetic import SyntheticConfig, generate_synthetic_medical_data, save_dataset

def main():
    cfg = SyntheticConfig(
        n_samples=8000,
        n_features=5,
        cancer_rate=0.12,          # imbalanced
        feature_noise_std=0.12,    # noisy measurements
        label_noise_rate=0.02,
        nonlinear=True,
        random_seed=42,
    )

    X, y = generate_synthetic_medical_data(cfg)

    out_dir = Path("data/processed")
    save_dataset(out_dir=out_dir, X=X, y=y, train_frac=0.70, val_frac=0.15, random_seed=123)

    print("Saved:")
    print(f"  {out_dir / 'synth_X.csv'}  shape={X.shape}")
    print(f"  {out_dir / 'synth_y.csv'}  shape={y.shape}")
    print(f"  {out_dir / 'splits'}")

if __name__ == "__main__":
    main()

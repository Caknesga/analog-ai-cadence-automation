import numpy as np
from pathlib import Path
from src.data.synthetic import _sigmoid, SyntheticConfig, generate_synthetic_medical_data, save_dataset


N_SAMPLES = 1_000
N_FEATURES = 5
RANDOM_SEED = 42

out_dir = Path("data/raw")
X_file = out_dir / "synth_X_delta_0.8t.csv"
y_file = out_dir / "synth_y_delta_0.8t.csv"

#data generation function
def generate_data(n_sample:int):
    range=np.random.default_rng(RANDOM_SEED)

    X=range.uniform(0.0,1.0,size=(n_sample,N_FEATURES))

      # Cancer rule (ground truth)
  
    cancer=(-1.1 + 2.5*X[:,0] - 0.8*X[:,1] + 0.9*X[:,2]
             + 0.4*X[:,3] - 0.6*X[:,4]+
             1.2*X[:,0]*X[:,1]-0.9*X[:,2]*X[:,4]+0.7*X[:,3]*X[:,2]+0.8*np.tanh(2*(X[:,2]-0.5))-0.7*(X[:,3]-0.5)**2)
    

    delta = 0.8   # width of uncertain band (tune this)
    tau   = 0.5   # softness inside band

    y = np.zeros(len(cancer), dtype=int)

    # Clearly cancer
    y[cancer >  delta] = 1

    # Clearly non-cancer
    y[cancer < -delta] = 0

    # Uncertain region
    mask = np.abs(cancer) <= delta
    probs = 1.0 / (1.0 + np.exp(-cancer[mask] / tau))
    y[mask] = (np.random.rand(mask.sum()) < probs).astype(int)

   # y=(cancer>0).astype(int)
    # Convert to probabilities?, NOT SURE IF NEEDED
    probs=_sigmoid(cancer)
    #y = (np.random.rand(len(probs)) < probs).astype(int)
    """
        # Cancer rule (ground truth)
    cancer = (
        ((X[:, 0] > 0.7) & (X[:, 1] > 0.7))  # protein 1 AND 2
       | (X[:, 2] > 0.8)                   # protein 3 alone
    )

    y = cancer.astype(int)
    """
    return X, y

#save csv function
def save_csv(X, y):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(X_file, X, delimiter=",", fmt="%.5f")
    np.savetxt(y_file, y.reshape(-1, 1), delimiter=",", fmt="%d")

    print("Saved files:")
    print(f"  {X_file}")
    print(f"  {y_file}")
    print()
    print("Dataset summary:")
    print(f"  Samples      : {len(y)}")
    print(f"  Cancer rate  : {y.mean():.3f}")


# Main
if __name__ == "__main__":
    X, y = generate_data(N_SAMPLES)
    save_csv(X, y)
  
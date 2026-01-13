import subprocess
import os

env = os.environ.copy()
env["VIN1"] = "0.5"

subprocess.run(
    ["./scripts/run_ocean.sh"],
    env=env,
    check=True
)
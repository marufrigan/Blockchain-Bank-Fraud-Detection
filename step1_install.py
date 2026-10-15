import subprocess, sys

packages = [
    "pandas", "numpy", "scikit-learn", "xgboost",
    "lightgbm", "imbalanced-learn", "plotly", "kaleido", "networkx", "web3"
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("All libraries installed successfully!")

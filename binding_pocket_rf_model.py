import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import OneHotEncoder

print("Loading data...")
df = pd.read_csv("train_features.csv")
print(f"Loaded {len(df)} residues.")

print("Computing derived features...")
df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
df["xy"] = df["x"] * df["y"]
df["yz"] = df["y"] * df["z"]
df["zx"] = df["z"] * df["x"]
df["is_surface"] = (df["r"] > df["r"].median()).astype(int)
print("Derived features added.")

print("Encoding residue names...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
resname_encoded = encoder.fit_transform(df[["resname"]])
print(f"Encoded {resname_encoded.shape[1]} residue types.")

print("Preparing feature matrix...")
X_basic = df[["x", "y", "z", "hydrophobicity", "r", "xy", "yz", "zx", "is_surface"]].values
X = np.hstack([X_basic, resname_encoded])
y = df["label"].values
print(f"Feature matrix shape: {X.shape}")

print("Splitting train and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(y_train)}, Validation size: {len(y_val)}")

print("Training Random Forest (100 trees)...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Model trained.")

print("Evaluating performance...")
y_pred = model.predict(X_val)
iou = jaccard_score(y_val, y_pred)
print(f"IOU (Jaccard Index) with enhancements: {iou:.3f}")

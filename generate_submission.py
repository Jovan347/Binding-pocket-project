import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Load training data
print("Loading training features...")
df = pd.read_csv("train_features.csv")
print(f"Loaded {len(df)} training residues.")

# Feature engineering
print("Computing derived features...")
df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
df["xy"] = df["x"] * df["y"]
df["yz"] = df["y"] * df["z"]
df["zx"] = df["z"] * df["x"]
df["is_surface"] = (df["r"] > df["r"].median()).astype(int)
print("Derived features added.")

# Encode residue names
print("Encoding residue names...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
resname_encoded = encoder.fit_transform(df[["resname"]])
print(f"Encoded {resname_encoded.shape[1]} residue types.")

# Prepare features and labels
print("Preparing training feature matrix...")
X_basic = df[["x", "y", "z", "hydrophobicity", "r", "xy", "yz", "zx", "is_surface"]].values
X = np.hstack([X_basic, resname_encoded])
y = df["label"].values
print(f"Training feature matrix shape: {X.shape}")

# Train model
print("Training final Random Forest model on full training data...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
print("Model trained.")

# Load test features
print("Loading test features...")
test_df = pd.read_csv("test_features.csv")
print(f"Loaded {len(test_df)} test residues.")

# Apply same feature engineering to test set
print("Computing derived features on test data...")
test_df["r"] = np.sqrt(test_df["x"]**2 + test_df["y"]**2 + test_df["z"]**2)
test_df["xy"] = test_df["x"] * test_df["y"]
test_df["yz"] = test_df["y"] * test_df["z"]
test_df["zx"] = test_df["z"] * test_df["x"]
test_df["is_surface"] = (test_df["r"] > df["r"].median()).astype(int)  # use training median!
print("Derived features added to test set.")

print("Encoding test residue names...")
test_resname_encoded = encoder.transform(test_df[["resname"]])
print(f"Encoded test residue types.")

# Prepare test feature matrix
print("Preparing test feature matrix...")
X_test_basic = test_df[["x", "y", "z", "hydrophobicity", "r", "xy", "yz", "zx", "is_surface"]].values
X_test = np.hstack([X_test_basic, test_resname_encoded])
print(f"Test feature matrix shape: {X_test.shape}")

# Predict
print("Predicting binding site residues...")
y_pred = model.predict(X_test)
test_df["label"] = y_pred
print("Predictions complete.")

# Format and save submission
print("Formatting predictions for submission...")
submission_dict = {}

for _, row in test_df[test_df["label"] == 1].iterrows():
    pid = row["pdb_id"]
    res = row["res_id"]
    if pid not in submission_dict:
        submission_dict[pid] = []
    submission_dict[pid].append(res)

submission_rows = [{"id": pid, "prediction": " ".join(res_list)} for pid, res_list in submission_dict.items()]
submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv("submission1.csv", index=False)
print("submission1.csv created successfully!")

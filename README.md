# Binding Pocket Prediction Project

This repository contains a machine-learning pipeline for predicting protein binding pocket residues from PDB protein structures. The project extracts residue-level structural and biochemical features from protein files, trains a Random Forest classifier, evaluates it using the Jaccard Index / Intersection over Union (IoU), and generates a submission file for test proteins.

The main idea is to treat each amino acid residue as one training example. For every residue, the pipeline extracts its C-alpha coordinates, residue type, hydrophobicity score, and several derived geometric features. A Random Forest model then predicts whether each residue belongs to a binding pocket.

## Project Overview

Binding pockets are regions of a protein that can interact with ligands or other molecules. Predicting these regions is important in structural bioinformatics, drug discovery, and protein-function analysis.

This project implements a simple but complete residue-level prediction workflow:

1. Read protein structures from `.pdb` files.
2. Extract residue-level features from each protein.
3. Match training residues with known binding-pocket labels.
4. Train a Random Forest classifier.
5. Evaluate performance using IoU / Jaccard score.
6. Apply the trained model to test proteins.
7. Generate a Kaggle-style submission file.

## Repository Structure

```text
.
├── main.py                     # Extracts training features from train PDB files
├── test_features.py            # Extracts test features from test PDB files
├── binding_pocket_rf_model.py  # Trains and validates the Random Forest model
├── generate_submission.py      # Trains on all data and creates submission1.csv
├── .gitignore
└── data/                       # Not committed; 
```

The `data/` folder is ignored by Git because it contains large dataset files.

## Expected Data Layout

The scripts expect the following local folder structure:

```text
data/
├── train.csv
├── train_data/
│   ├── <protein_id>_protein.pdb
│   └── ...
└── test_data/
    ├── <protein_id>_protein.pdb
    └── ...
```


## Feature Extraction

The feature extraction scripts use `Bio.PDB` to parse protein structures. Only residues with a C-alpha atom (`CA`) are included.

For each residue, the pipeline extracts:

| Feature | Description |
|---|---|
| `pdb_id` | Protein identifier |
| `res_id` | Residue identifier in chain/residue-number format |
| `x`, `y`, `z` | C-alpha atom coordinates |
| `resname` | Amino acid residue name |
| `hydrophobicity` | Kyte-Doolittle-style hydrophobicity value |
| `label` | Binding-pocket label, only for training data |

The model script then adds derived features:

| Feature | Description |
|---|---|
| `r` | Distance of the residue from the coordinate origin |
| `xy`, `yz`, `zx` | Pairwise coordinate interaction features |
| `is_surface` | Binary surface-like feature based on whether `r` is above the training median |
| one-hot residue type | Encoded amino acid identity |

## Model

The project uses a **Random Forest classifier** from scikit-learn.

Current model configuration:

```python
RandomForestClassifier(n_estimators=200, random_state=42)
```

The input features include:

- C-alpha coordinates
- Hydrophobicity
- Derived geometric features
- Surface-like indicator
- One-hot encoded residue type

This makes the model simple, interpretable, and easy to run without deep-learning infrastructure.

## Evaluation

The validation script splits the extracted training features into train and validation sets:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

Performance is evaluated using the **Jaccard Index**, also called **Intersection over Union (IoU)**:

```text
IoU = intersection of predicted and true binding residues / union of predicted and true binding residues
```

The script reports:

```text
IOU (Jaccard Index) with enhancements: <score>
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn biopython
```

## How to Run

### 1. Extract training features

Place the training files under `data/train_data/` and the labels in `data/train.csv`, then run:

```bash
python main.py
```

This creates:

```text
train_features.csv
```

### 2. Train and validate the model

```bash
python binding_pocket_rf_model.py
```

This trains the Random Forest model on the extracted residue features and evaluates it on a validation split using IoU / Jaccard score.

### 3. Extract test features

Place the test PDB files under `data/test_data/`, then run:

```bash
python test_features.py
```

This creates:

```text
test_features.csv
```

### 4. Generate the final submission

```bash
python generate_submission.py
```

This trains the Random Forest model on the full training feature set, predicts binding-pocket residues for the test proteins, and creates:

```text
submission1.csv
```

## Output Files

The pipeline generates the following CSV files:

| File | Description |
|---|---|
| `train_features.csv` | Extracted residue-level training features and labels |
| `test_features.csv` | Extracted residue-level test features |
| `submission1.csv` | Final predicted binding residues for each test protein |

These generated CSV files are ignored by Git.

## Submission Format

The final submission file has the following format:

```text
id,prediction
protein_1,A_45 A_46 B_102
protein_2,C_12 C_13
...
```

Each row contains a protein ID and a space-separated list of residues predicted to belong to the binding pocket.

## Notes

This project is intentionally lightweight. It does not use a neural network or external structural descriptors, but instead builds a classical machine-learning baseline using residue coordinates, amino acid identity, and hydrophobicity.

Possible future improvements include:

- Adding distance-to-neighbor features
- Adding residue solvent-accessibility features
- Using class balancing for binding/non-binding residues
- Predicting probabilities and tuning a decision threshold instead of using hard class predictions
- Grouping validation by protein instead of splitting individual residues randomly
- Testing gradient boosting or graph neural network models

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Biopython
- PDB protein structure files

## License

No license.

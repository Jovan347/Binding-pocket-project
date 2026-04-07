import os
from Bio.PDB import PDBParser
import pandas as pd

# Hydrophobicity dictionary
hydrophobicity = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# Load labels and initialize parser
parser = PDBParser(QUIET=True)
train_df = pd.read_csv("data/train.csv")
feature_rows = []

# Main loop through all PDBs in training set
for i, pdb_id in enumerate(train_df["id"].unique()):
    if i % 100 == 0:
        print(f"Processing file {i}/{len(train_df)}: {pdb_id}")

    try:
        pdb_file = f"data/train_data/{pdb_id}_protein.pdb"
        structure = parser.get_structure(str(pdb_id), pdb_file)
        label_str = train_df.loc[train_df['id'] == pdb_id, 'resid'].values[0]
        binding_residues = set(label_str.split())

        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' not in residue:
                        continue
                    resname = residue.get_resname()
                    if resname not in hydrophobicity:
                        continue
                    chain_id = chain.id
                    resnum = residue.id[1]
                    res_id = f"{chain_id}_{resnum}"
                    x, y, z = residue['CA'].coord
                    hydro = hydrophobicity[resname]
                    label = 1 if res_id in binding_residues else 0
                    feature_rows.append([pdb_id, res_id, x, y, z, resname, hydro, label])
    except Exception as e:
        print(f"Skipped {pdb_id} due to error: {e}")

# Save final dataset
df = pd.DataFrame(feature_rows, columns=["pdb_id", "res_id", "x", "y", "z", "resname", "hydrophobicity", "label"])
df.to_csv("train_features.csv", index=False)
print("Saved train_features.csv with", len(df), "residues.")

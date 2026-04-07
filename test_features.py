import os
from Bio.PDB import PDBParser
import pandas as pd
import numpy as np

# Hydrophobicity dictionary
hydrophobicity = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# Initialize
parser = PDBParser(QUIET=True)
test_folder = "data/test_data"
feature_rows = []

# Loop through all test PDB files
for i, filename in enumerate(os.listdir(test_folder)):
    if not filename.endswith(".pdb"):
        continue

    pdb_id = filename.replace("_protein.pdb", "")
    if i % 100 == 0:
        print(f"Processing {i}: {filename}")

    try:
        structure = parser.get_structure(pdb_id, os.path.join(test_folder, filename))
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
                    feature_rows.append([pdb_id, res_id, x, y, z, resname, hydro])
    except Exception as e:
        print(f"Skipped {pdb_id} due to error: {e}")

# Save to CSV
df = pd.DataFrame(feature_rows, columns=["pdb_id", "res_id", "x", "y", "z", "resname", "hydrophobicity"])
df.to_csv("test_features.csv", index=False)
print("Saved test_features.csv with", len(df), "residues.")

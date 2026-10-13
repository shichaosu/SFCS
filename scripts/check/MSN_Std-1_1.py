# Quality-control and data-conversion helper for the SFCS workflow.
# Update the path variables below before running this script.
import os
import numpy as np

base_path = "data/data_use/aparc/MSN/"


def normalize_off_diag(mat):
    if mat.shape[0] != mat.shape[1]:
        return mat
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag_vals = mat[mask]
    min_val = off_diag_vals.min()
    max_val = off_diag_vals.max()
    if max_val > min_val:
        norm_vals = 2 * (off_diag_vals - min_val) / (max_val - min_val) - 1
    else:
        norm_vals = np.zeros_like(off_diag_vals)
    mat_norm = np.zeros_like(mat)
    mat_norm[mask] = norm_vals
    return mat_norm


for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".npy"):
            full_path = os.path.join(root, file)
            try:
                mat = np.load(full_path)
                mat_norm = normalize_off_diag(mat)
                np.save(full_path, mat_norm)
                print(f"Processing message.")
            except Exception as e:
                print(f"Processing message.")

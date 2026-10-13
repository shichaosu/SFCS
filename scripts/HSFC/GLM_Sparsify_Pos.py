#!/usr/bin/env python3
# GLM-based structure-function constraint estimation script.
# Update the path variables below before running this script.
"""Script documentation for this SFCS analysis step."""
import os
import numpy as np
from tqdm import tqdm
import csv
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

base_dir = "data/data_use/aparc"
sc_dir = os.path.join(base_dir, "SC_Std0_1")
msn_dir = os.path.join(base_dir, "MSN")
hoi_dir = os.path.join(base_dir, "HOI_GSR_Z_Sparsification_Pos/20")  # _Pos or _Neg
sparsification_dir = "data/data_use/aparc/Matrix_Sparsification_Pos/20"  # _Pos or _Neg

out_base = os.path.join(base_dir, "GLM_Sparsify_Pos")  # _Pos or _Neg
out_coef_dir = os.path.join(out_base, "20Coefficient")
out_recon_dir = os.path.join(out_base, "20reconstruct")
out_metrics_dir = os.path.join(out_base, "20MSE")
os.makedirs(out_coef_dir, exist_ok=True)
os.makedirs(out_recon_dir, exist_ok=True)
os.makedirs(out_metrics_dir, exist_ok=True)


def list_subdirs(path):
    return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]


sc_subdirs = list_subdirs(sc_dir)
msn_subdirs = list_subdirs(msn_dir)

if len(sc_subdirs) == 0:
    raise RuntimeError(f"SC input folder is missing or invalid.")
if len(msn_subdirs) == 0:
    raise RuntimeError(f"MSN input folder is missing or invalid.")

print("SC components (count={}): {}".format(len(sc_subdirs), sc_subdirs))
print("MSN components (count={}): {}".format(len(msn_subdirs), msn_subdirs))
comp_names = sc_subdirs + msn_subdirs
if len(comp_names) != 23:
    raise RuntimeError(f"SC input folder is missing or invalid.")

compnames_global_path = os.path.join(out_base, "compnames.txt")
with open(compnames_global_path, "w") as fh:
    fh.write("intercept\n")
    for cname in comp_names:
        fh.write(f"{cname }\n")
print(f"Saved global comp names to {compnames_global_path }")
ref_sc_subdir = os.path.join(sc_dir, sc_subdirs[0])
individual_files = sorted([f for f in os.listdir(ref_sc_subdir) if f.endswith(".npy")])
if not individual_files:
    raise RuntimeError(f"Processing message.")
print(f"Found {len (individual_files )} individuals (based on {ref_sc_subdir }).")

n_nodes = 68
triu_idx = np.triu_indices(n_nodes, k=1)
n_samples = triu_idx[0].shape[0]  # 2278
glm = LinearRegression(fit_intercept=False)

for fname in tqdm(individual_files, desc="Individuals"):
    try:
        comps = []
        for sub in sc_subdirs:
            fpath = os.path.join(sc_dir, sub, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Processing message.")
            arr = np.load(fpath)
            if arr.shape != (n_nodes, n_nodes):
                raise ValueError(f"Input shape does not match the expected dimensions.")
            comps.append(arr)
        for sub in msn_subdirs:
            fpath = os.path.join(msn_dir, sub, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Processing message.")
            arr = np.load(fpath)
            if arr.shape != (n_nodes, n_nodes):
                raise ValueError(f"Input shape does not match the expected dimensions.")
            comps.append(arr)
        if len(comps) != 23:
            raise RuntimeError(f"Processing message.")
        hoi_path = os.path.join(hoi_dir, fname)
        if not os.path.exists(hoi_path):
            raise FileNotFoundError(f"HOI input file is missing or invalid.")
        hoi = np.load(hoi_path)
        if hoi.shape != (n_nodes, n_nodes, n_nodes):
            raise ValueError(f"HOI input file is missing or invalid.")
        sparsification_path = os.path.join(sparsification_dir, fname)
        if not os.path.exists(sparsification_path):
            raise FileNotFoundError(f"Sparsification input file is missing or invalid.")
        sparsification_matrix = np.load(sparsification_path)
        if sparsification_matrix.shape != (n_nodes, n_nodes, n_nodes):
            raise ValueError(f"Sparsification input file is missing or invalid.")
        n_params = 24  # 1 intercept + 23 comps
        coefs_all = np.zeros((n_nodes, n_params), dtype=np.float64)
        hoi_recon = np.zeros_like(hoi, dtype=np.float64)
        metrics = np.zeros((n_nodes, 2), dtype=np.float64)
        slice_variances = np.zeros(n_nodes, dtype=np.float64)

        for k in range(n_nodes):
            mask = sparsification_matrix[:, :, k]

            X_base = np.zeros((n_samples, 23), dtype=np.float64)
            for idx, comp in enumerate(comps):
                comp_filtered = comp * mask
                col = comp_filtered[triu_idx]
                col = np.nan_to_num(col, nan=0.0)
                X_base[:, idx] = col

            X_design = np.concatenate(
                [np.ones((n_samples, 1), dtype=np.float64), X_base], axis=1
            )

            y_full = hoi[:, :, k]
            y = y_full[triu_idx]
            y = np.nan_to_num(y, nan=0.0)

            glm.fit(X_design, y)
            params = glm.coef_.flatten()
            coefs_all[k, :] = params

            y_hat = X_design @ params
            mse = np.mean((y - y_hat) ** 2)
            metrics[k, 0] = mse
            pearson_val, _ = pearsonr(y, y_hat)
            if np.isnan(pearson_val):
                pearson_val = 0.0
            metrics[k, 1] = pearson_val

            slice_variances[k] = np.var(y)
            recon_slice = np.zeros((n_nodes, n_nodes), dtype=np.float64)
            recon_slice[triu_idx] = y_hat
            recon_slice = recon_slice + recon_slice.T
            diag_vals = np.diag(hoi[:, :, k])
            recon_slice[np.diag_indices(n_nodes)] = diag_vals
            hoi_recon[:, :, k] = recon_slice
        pearson_overall, _ = pearsonr(hoi.flatten(), hoi_recon.flatten())
        out_coef_path = os.path.join(out_coef_dir, fname)
        np.save(out_coef_path, coefs_all)

        out_recon_path = os.path.join(out_recon_dir, fname)
        np.save(out_recon_path, hoi_recon)
        metrics_npy_path = os.path.join(out_metrics_dir, f"{fname }_metrics.npy")
        np.save(metrics_npy_path, metrics)
        metrics_csv_path = os.path.join(
            out_metrics_dir, f"{os .path .splitext (fname )[0 ]}_metrics.csv"
        )
        with open(metrics_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["slice_index", "MSE", "Pearson_r"])
            for k in range(n_nodes):
                writer.writerow([k, float(metrics[k, 0]), float(metrics[k, 1])])
        mean_slice_pearson = np.mean(metrics[:, 1])
        total_var = np.sum(slice_variances)
        if total_var > 0:
            weighted_pearson_by_variance = (
                np.sum(metrics[:, 1] * slice_variances) / total_var
            )
        else:
            weighted_pearson_by_variance = mean_slice_pearson

        overall_MSE = float(np.mean(metrics[:, 0]))
        print(
            f"Finished {fname }: mean_slice_pearson={mean_slice_pearson :.4f}, "
            f"weighted_pearson_by_variance={weighted_pearson_by_variance :.4f}, "
            f"overall_MSE={overall_MSE :.6f}, pearson_overall={pearson_overall :.4f}"
        )
        summary_csv = os.path.join(out_metrics_dir, "summary_metrics.csv")
        if not os.path.exists(summary_csv):
            with open(summary_csv, "w", newline="") as scsv:
                w = csv.writer(scsv)
                w.writerow(
                    [
                        "filename",
                        "mean_slice_pearson",
                        "weighted_pearson_by_variance",
                        "overall_MSE",
                        "pearson_overall",
                    ]
                )
        with open(summary_csv, "a", newline="") as scsv:
            w = csv.writer(scsv)
            w.writerow(
                [
                    fname,
                    f"{mean_slice_pearson :.6f}",
                    f"{weighted_pearson_by_variance :.6f}",
                    f"{overall_MSE :.8f}",
                    f"{pearson_overall :.6f}",
                ]
            )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Skipping {fname } due to error: {e }")
        continue

print("All done.")

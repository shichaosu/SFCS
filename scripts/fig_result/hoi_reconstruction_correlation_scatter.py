#!/usr/bin/env python3
# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def get_unique_triple_mask(n_nodes: int) -> np.ndarray:
    """Script documentation for this SFCS analysis step."""
    mask = np.zeros((n_nodes, n_nodes, n_nodes), dtype=bool)
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            for k in range(j, n_nodes):
                mask[i, j, k] = True
    return mask


def list_common_npy_files(directories: Iterable[Path]) -> List[str]:
    """Script documentation for this SFCS analysis step."""
    npy_sets = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Processing message.")
        npy_sets.append({f for f in os.listdir(directory) if f.endswith(".npy")})
    common = set.intersection(*npy_sets)
    return sorted(common)


def load_matrix(file_name: str, directory: Path) -> np.ndarray:
    """Script documentation for this SFCS analysis step."""
    file_path = directory / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Processing message.")
    return np.load(file_path)


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Script documentation for this SFCS analysis step."""
    if x.size < 2:
        return np.nan
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    denom = np.sqrt(np.sum(x_diff**2) * np.sum(y_diff**2))
    if denom == 0:
        return np.nan
    return float(np.dot(x_diff, y_diff) / denom)


def compute_correlations(
    subjects: List[str],
    recon_dir: Path,
    real_dir: Path,
    mask: np.ndarray,
    remove_double_zeros_in_cross: bool,
) -> Dict[str, Dict[str, float]]:
    """Script documentation for this SFCS analysis step."""
    recon_vectors = {}
    per_subject_corr = {}

    for subject in subjects:
        recon_matrix = load_matrix(subject, recon_dir)
        real_matrix = load_matrix(subject, real_dir)

        if recon_matrix.shape != real_matrix.shape:
            raise ValueError(f"Processing message.")

        recon_vals = recon_matrix[mask]
        real_vals = real_matrix[mask]
        recon_vectors[subject] = recon_vals
        valid_mask = real_vals != 0
        recon_filtered = recon_vals[valid_mask]
        real_filtered = real_vals[valid_mask]
        per_subject_corr[subject] = safe_pearsonr(recon_filtered, real_filtered)

    max_cross_corr = {}
    for subject_i in subjects:
        rec_i = recon_vectors[subject_i]
        best_corr = -np.inf
        for subject_j in subjects:
            if subject_i == subject_j:
                continue

            rec_j = recon_vectors[subject_j]
            if remove_double_zeros_in_cross:
                nonzero_mask = ~((rec_i == 0) & (rec_j == 0))
                vec_i = rec_i[nonzero_mask]
                vec_j = rec_j[nonzero_mask]
            else:
                vec_i = rec_i
                vec_j = rec_j

            corr = safe_pearsonr(vec_i, vec_j)
            if np.isnan(corr):
                continue
            if corr > best_corr:
                best_corr = corr

        max_cross_corr[subject_i] = best_corr if best_corr != -np.inf else np.nan

    return {
        "within": per_subject_corr,
        "between": max_cross_corr,
    }


def main() -> None:
    REMOVE_DOUBLE_ZEROS_IN_CROSS = True

    recon_neg_dir = Path("data/data_use/Final_data/GLM_Sparsify_Neg/20recons/CN")
    recon_pos_dir = Path("data/data_use/Final_data/GLM_Sparsify_Pos/20recons/CN")
    real_neg_dir = Path(
        "data/data_use/Final_data/GLM_Sparsify_Neg/HOI_GSR_Z_Sparsification_20/CN"
    )
    real_pos_dir = Path(
        "data/data_use/Final_data/GLM_Sparsify_Pos/HOI_GSR_Z_Sparsification_20/CN"
    )

    neg_subjects = list_common_npy_files([recon_neg_dir, real_neg_dir])
    pos_subjects = list_common_npy_files([recon_pos_dir, real_pos_dir])
    if not neg_subjects and not pos_subjects:
        raise RuntimeError("Processing message.")

    print(f"Processing message." f"Processing message.")

    mask = get_unique_triple_mask(68)

    print("Processing message.")
    neg_results = compute_correlations(
        neg_subjects,
        recon_neg_dir,
        real_neg_dir,
        mask,
        REMOVE_DOUBLE_ZEROS_IN_CROSS,
    )
    pos_results = compute_correlations(
        pos_subjects,
        recon_pos_dir,
        real_pos_dir,
        mask,
        REMOVE_DOUBLE_ZEROS_IN_CROSS,
    )

    for label, results in [("neg", neg_results), ("pos", pos_results)]:
        sorted_corrs = sorted(
            [(k, v) for k, v in results["within"].items() if not np.isnan(v)],
            key=lambda item: item[1],
            reverse=True,
        )
        print("\n" + "=" * 50)
        print(f"Processing message.")
        for rank, (subj, val) in enumerate(sorted_corrs[:5], start=1):
            print(f"Top {rank }: {subj } (Pearson r = {val :.4f})")
        print("=" * 50 + "\n")

    print(f"Processing message.")
    plot_data = {
        "neg": {
            "color": "tab:blue",
            "legend_label": "Synergy",
            "results": neg_results,
            "subjects": neg_subjects,
        },
        "pos": {
            "color": "tab:orange",
            "legend_label": "Redundancy",
            "results": pos_results,
            "subjects": pos_subjects,
        },
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    all_vals = []
    has_points = False

    for label, config in plot_data.items():
        x_vals = []
        y_vals = []
        results = config["results"]

        for subject in config["subjects"]:
            x = results["within"][subject]
            y = results["between"][subject]
            if np.isnan(x) or np.isnan(y):
                continue
            x_vals.append(x)
            y_vals.append(y)

        if not x_vals:
            continue

        has_points = True
        all_vals.extend(x_vals)
        all_vals.extend(y_vals)
        ax.scatter(
            x_vals,
            y_vals,
            c=config["color"],
            edgecolors="k",
            alpha=0.75,
            label=f'{config ["legend_label"]} (n={len (x_vals )})',
        )

    if not has_points:
        raise RuntimeError("Processing message.")

    all_vals = np.array(all_vals)
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.05
    line_min = min_val - padding
    line_max = max_val + padding
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        color="tab:gray",
        label="y = x",
    )
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("Within-subject", fontsize=20)
    ax.set_ylabel("Maximum between-subject", fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    save_path = "data/visulization/figure_result/20/CN_hoi_reconstruction_correlation_scatter.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=200)
    print(f"Processing message." "Processing message." "Processing message.")


if __name__ == "__main__":
    main()

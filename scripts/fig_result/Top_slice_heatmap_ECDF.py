# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

recon_neg_dir = Path("data/data_use/aparc/GLM_Sparsify_Neg/20reconstruct")
recon_pos_dir = Path("data/data_use/aparc/GLM_Sparsify_Pos/20reconstruct")
real_neg_dir = Path("data/data_use/aparc/HOI_GSR_Z_Sparsification_Neg/20")
real_pos_dir = Path("data/data_use/aparc/HOI_GSR_Z_Sparsification_Pos/20")

heatmap_dir = Path("data/visulization/figure_result/20/slice_ECDF/")
heatmap_dir.mkdir(parents=True, exist_ok=True)


def list_shared_files(*dirs: Path) -> List[str]:
    file_sets = [set(p.name for p in d.glob("*.npy")) for d in dirs]
    shared = set.intersection(*file_sets)
    return sorted(shared)


def soft_red_blue_cmap():

    return cm.get_cmap("viridis", 256)


def load_and_sum(dir_a: Path, dir_b: Path, filename: str) -> np.ndarray:
    return np.load(dir_a / filename) + np.load(dir_b / filename)


def compute_correlations(subjects: List[str]) -> List[Tuple[str, int, float]]:
    records: List[Tuple[str, int, float]] = []
    for subj in subjects:
        recon = load_and_sum(recon_neg_dir, recon_pos_dir, subj)
        real = load_and_sum(real_neg_dir, real_pos_dir, subj)

        if recon.shape != real.shape or recon.ndim != 3:
            raise ValueError(f"Input shape does not match the expected dimensions.")

        num_slices = recon.shape[2]
        for k in range(num_slices):
            recon_slice = recon[:, :, k]
            real_slice = real[:, :, k]

            mask = ~((np.isclose(recon_slice, 0) & np.isclose(real_slice, 0)))
            if not np.any(mask):
                continue

            x = recon_slice[mask].ravel()
            y = real_slice[mask].ravel()

            if x.size < 2 or y.size < 2:
                continue
            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                continue

            r, _ = pearsonr(x, y)
            records.append((subj, k, r))
    return records


def build_combined_matrix(
    recon_slice: np.ndarray, real_slice: np.ndarray
) -> np.ndarray:
    if recon_slice.shape != real_slice.shape:
        raise ValueError("Processing message.")
    n = recon_slice.shape[0]
    combined = np.zeros((n, n), dtype=recon_slice.dtype)
    upper = np.triu_indices(n, k=1)
    lower = np.tril_indices(n, k=-1)
    combined[upper] = recon_slice[upper]
    combined[lower] = real_slice[lower]
    np.fill_diagonal(combined, 0)
    return combined


def zscore_matrix(mat: np.ndarray) -> np.ndarray:
    """Script documentation for this SFCS analysis step."""
    arr = mat.astype(float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if np.isclose(std, 0) or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - mean) / std


def ecdf_map(mat: np.ndarray) -> np.ndarray:
    """Script documentation for this SFCS analysis step."""
    arr = mat.astype(float)
    flat = arr.ravel()
    mask = ~np.isnan(flat)
    vals = flat[mask]
    n = vals.size
    if n == 0:
        return np.zeros_like(arr)
    ranks = rankdata(vals, method="average")
    pct = ranks / (n + 1.0)
    mapped = pct * 2.0 - 1.0

    out_flat = np.full(flat.shape, np.nan, dtype=float)
    out_flat[mask] = mapped
    return out_flat.reshape(arr.shape)


def plot_heatmap(matrix: np.ndarray, title: str, save_path: Path) -> None:
    vmax = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 1
    vmin = -vmax
    cmap = soft_red_blue_cmap()
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Combined value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main(top_n: int):
    subjects = list_shared_files(
        recon_neg_dir, recon_pos_dir, real_neg_dir, real_pos_dir
    )
    if not subjects:
        raise RuntimeError("Processing message.")

    records = compute_correlations(subjects)
    if not records:
        raise RuntimeError("Processing message.")

    sorted_records = sorted(records, key=lambda tup: tup[2], reverse=True)
    top_n = max(1, min(top_n, len(sorted_records)))
    top_records = sorted_records[:top_n]

    print(f"Processing message.")
    for idx, (subj, slice_idx, corr) in enumerate(top_records, start=1):
        print(f"  {idx }. {subj } | slice={slice_idx } | r={corr :.6f}")
        recon_slice = load_and_sum(recon_neg_dir, recon_pos_dir, subj)[:, :, slice_idx]
        real_slice = load_and_sum(real_neg_dir, real_pos_dir, subj)[:, :, slice_idx]
        recon_slice_ecdf = ecdf_map(recon_slice)
        real_slice_ecdf = ecdf_map(real_slice)

        combined = build_combined_matrix(recon_slice_ecdf, real_slice_ecdf)
        heatmap_path = (
            heatmap_dir / f"{subj }_slice{slice_idx }_r{corr :.4f}_top{idx }.png"
        )
        title = f"{subj } slice {slice_idx } r={corr :.4f} (top {idx })"
        plot_heatmap(combined, title, heatmap_path)
        print(f"Processing message.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing message.")
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Processing message.",
    )
    args = parser.parse_args()
    main(args.top)

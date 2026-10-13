# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from matplotlib import font_manager

recon_neg_dir = Path("data/data_use/aparc/GLM_Sparsify_Neg/20reconstruct")
recon_pos_dir = Path("data/data_use/aparc/GLM_Sparsify_Pos/20reconstruct")
real_neg_dir = Path("data/data_use/aparc/HOI_GSR_Z_Sparsification_Neg/20")
real_pos_dir = Path("data/data_use/aparc/HOI_GSR_Z_Sparsification_Pos/20")

heatmap_dir = Path("data/visulization/figure_result/20/slice_ECDF0/")
heatmap_dir.mkdir(parents=True, exist_ok=True)
atlas_path = Path("data/atlas/aparc.annot.xlsx")


def get_roi_names(excel_path: Path) -> List[str]:
    """Script documentation for this SFCS analysis step."""
    try:
        df = pd.read_excel(excel_path, sheet_name="name")
        if "Abbreviation" in df.columns:
            base_names = df["Abbreviation"].astype(str).values[:34]
        else:
            base_names = df.iloc[:34, 1].astype(str).values

        if len(base_names) < 34:
            print(f"Processing message.")
        left_names = [f"L_{name }" for name in base_names]
        right_names = [f"R_{name }" for name in base_names]

        full_names = left_names + right_names
        return full_names

    except Exception as e:
        print(f"Processing message.")
        return [f"Slice_{i +1 }" for i in range(68)]


def list_shared_files(*dirs: Path) -> List[str]:
    file_sets = [set(p.name for p in d.glob("*.npy")) for d in dirs]
    shared = set.intersection(*file_sets)
    return sorted(shared)


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


def ecdf_map(mat: np.ndarray) -> np.ndarray:
    """Script documentation for this SFCS analysis step."""
    arr = mat.astype(float)
    flat = arr.ravel()
    out_flat = np.zeros_like(flat)
    nan_mask = np.isnan(flat)
    out_flat[nan_mask] = np.nan
    valid_data = flat[~nan_mask]

    if valid_data.size == 0:
        return out_flat.reshape(mat.shape)
    p_high = np.percentile(valid_data, 80)
    p_low = np.percentile(valid_data, 20)
    pos_mask = flat > p_high
    if np.any(pos_mask):
        pos_vals = flat[pos_mask]
        n_pos = pos_vals.size
        ranks = rankdata(pos_vals, method="average")
        pct = ranks / (n_pos + 1.0)
        out_flat[pos_mask] = pct
    neg_mask = flat < p_low
    if np.any(neg_mask):
        neg_vals = flat[neg_mask]
        n_neg = neg_vals.size
        ranks = rankdata(neg_vals, method="average")
        pct = ranks / (n_neg + 1.0)
        out_flat[neg_mask] = pct - 1.0

    return out_flat.reshape(mat.shape)


def plot_heatmap(matrix: np.ndarray, title: str, save_path: Path) -> None:
    vmax = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 1
    vmin = -vmax
    cmap = plt.get_cmap("RdYlBu_r")
    cmap.set_bad(color="white")

    plt.figure(figsize=(8, 7))
    rows, cols = np.indices(matrix.shape)
    thickness = 1
    diag_mask = np.abs(rows - cols) <= thickness
    masked_matrix = np.ma.masked_where(diag_mask, matrix)
    plt.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    default_title_size = plt.rcParams.get(
        "axes.titlesize", plt.rcParams.get("font.size", 12)
    )
    if isinstance(default_title_size, str):
        default_title_size = font_manager.FontProperties(
            size=default_title_size
        ).get_size_in_points()
    plt.title(title, fontsize=default_title_size * 3, pad=16)
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar(label="Combined value")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(top_n: int):
    subjects = list_shared_files(
        recon_neg_dir, recon_pos_dir, real_neg_dir, real_pos_dir
    )
    if not subjects:
        raise RuntimeError("Processing message.")
    roi_names = get_roi_names(atlas_path)
    records = compute_correlations(subjects)
    if not records:
        raise RuntimeError("Processing message.")
    from collections import defaultdict

    subj_rs = defaultdict(list)
    for subj, slice_idx, r in records:
        subj_rs[subj].append(r)

    mean_rs = {s: float(np.mean(rs)) for s, rs in subj_rs.items() if len(rs) > 0}
    if not mean_rs:
        raise RuntimeError("Processing message.")

    best_subj = max(mean_rs.items(), key=lambda kv: kv[1])[0]
    best_mean = mean_rs[best_subj]
    print(f"Processing message.")
    subj_dir = heatmap_dir / best_subj
    subj_dir.mkdir(parents=True, exist_ok=True)

    slice_r_map = {k: r for (s, k, r) in records if s == best_subj}

    recon = load_and_sum(recon_neg_dir, recon_pos_dir, best_subj)
    real = load_and_sum(real_neg_dir, real_pos_dir, best_subj)
    if recon.shape != real.shape or recon.ndim != 3:
        raise ValueError(f"Input shape does not match the expected dimensions.")

    num_slices = recon.shape[2]
    for k in range(num_slices):
        recon_slice = recon[:, :, k]
        real_slice = real[:, :, k]
        corr = slice_r_map.get(k, None)
        if corr is None:
            mask = ~((np.isclose(recon_slice, 0) & np.isclose(real_slice, 0)))
            if np.any(mask):
                x = recon_slice[mask].ravel()
                y = real_slice[mask].ravel()
                if (
                    x.size >= 2
                    and y.size >= 2
                    and not (np.allclose(x, x[0]) or np.allclose(y, y[0]))
                ):
                    corr, _ = pearsonr(x, y)
                else:
                    corr = float("nan")
            else:
                corr = float("nan")
        recon_slice_ecdf = ecdf_map(recon_slice)
        real_slice_ecdf = ecdf_map(real_slice)
        combined = build_combined_matrix(recon_slice_ecdf, real_slice_ecdf)
        if k < len(roi_names):
            current_roi_name = roi_names[k]
        else:
            current_roi_name = f"Slice_{k +1 }"
        title = (
            f"{current_roi_name }   r={corr :.2f}"
            if corr is not None
            else f"{current_roi_name }   r=NaN"
        )
        heatmap_path = (
            subj_dir
            / f"{best_subj }_slice{k }_r{(corr if corr is not None else np .nan ):.2f}.png"
        )

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

# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

RECON_DIRS = [
    "data/data_use/aparc/GLM_Sparsify_Neg/20reconstruct",
    "data/data_use/aparc/GLM_Sparsify_Pos/20reconstruct",
]
ORIG_DIRS = [
    "data/data_use/aparc/HOI_GSR_Z_Sparsification_Neg/20",
    "data/data_use/aparc/HOI_GSR_Z_Sparsification_Pos/20",
]
OUTPUT_PNG_PATH = (
    "data/visulization/figure_result/20/pearsonr_correlation_scatter_boxplot_sorted.png"
)
LABEL_EXCEL_PATH = "data/atlas/aparc.annot.xlsx"
# ===========================================


def list_npy_files(base_dir):
    pattern = os.path.join(base_dir, "*.npy")
    return {os.path.basename(p): p for p in glob.glob(pattern)}


def intersect_file_names(dirs):
    sets = []
    for d in dirs:
        files = list_npy_files(d)
        sets.append(set(files.keys()))
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def load_and_sum(basename, dirs):
    summed = None
    for d in dirs:
        path = os.path.join(d, basename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path } does not exist")
        data = np.load(path)
        if summed is None:
            summed = data.astype(np.float64)
        else:
            summed = summed + data
    return summed


def compute_slice_correlations(recon_array, orig_array):
    if recon_array.shape != orig_array.shape:
        raise ValueError(
            f"Shape mismatch: {recon_array .shape } vs {orig_array .shape }"
        )
    if recon_array.shape[2] != 68:
        raise ValueError(
            f"Expected 68 slices in third dimension; got {recon_array .shape [2 ]}"
        )

    correlations = []
    for slice_idx in range(68):
        recon_slice = recon_array[:, :, slice_idx].ravel()
        orig_slice = orig_array[:, :, slice_idx].ravel()
        # mask = (recon_slice != 0) | (orig_slice != 0)
        mask = orig_slice != 0
        # =========================================

        if np.count_nonzero(mask) < 2:
            correlations.append(np.nan)
            continue
        corr, _ = pearsonr(recon_slice[mask], orig_slice[mask])
        correlations.append(corr)
    return correlations


def load_labels_from_excel(path):
    if not os.path.exists(path):
        print(f"Warning: Label file not found at {path }. Using numeric labels.")
        return [str(i) for i in range(1, 35)]

    try:
        df = pd.read_excel(path, sheet_name="name")
        if "Abbreviation" not in df.columns:
            raise ValueError("Column 'Abbreviation' not found in sheet 'name'")

        labels = df["Abbreviation"].iloc[0:34].tolist()
        labels = [str(l) if pd.notna(l) else "" for l in labels]

        if len(labels) < 34:
            labels += [str(i) for i in range(len(labels) + 1, 35)]

        return labels
    except Exception as e:
        print(f"Error reading labels: {e }")
        return [str(i) for i in range(1, 35)]


def set_box_style_black(bp):
    plt.setp(bp["boxes"], color="black", linewidth=1.5)
    plt.setp(bp["whiskers"], color="black", linewidth=1.5)
    plt.setp(bp["caps"], color="black", linewidth=1.5)
    plt.setp(bp["medians"], color="black", linewidth=1.5)


def main():
    common_recon = intersect_file_names(RECON_DIRS)
    common_orig = intersect_file_names(ORIG_DIRS)
    common_subjects = sorted(set(common_recon) & set(common_orig))

    if not common_subjects:
        raise RuntimeError("No matching .npy files found.")

    print(f"Found {len (common_subjects )} matched subjects.")

    all_correlations = []
    for filename in tqdm(common_subjects, desc="Processing subjects"):
        try:
            recon_sum = load_and_sum(filename, RECON_DIRS)
            orig_sum = load_and_sum(filename, ORIG_DIRS)
        except Exception as exc:
            print(f"Skipping {filename }: {exc }")
            continue
        correlations = compute_slice_correlations(recon_sum, orig_sum)
        all_correlations.append(correlations)

    all_correlations = np.array(all_correlations)
    if all_correlations.size == 0:
        raise RuntimeError("No valid correlation data generated.")
    data_left = all_correlations[:, 0:34]
    data_right = all_correlations[:, 34:68]
    labels = load_labels_from_excel(LABEL_EXCEL_PATH)
    labels = np.array(labels)
    print("Sorting ROIs based on average median correlation...")
    median_left = np.nanmedian(data_left, axis=0)  # shape (34,)
    median_right = np.nanmedian(data_right, axis=0)  # shape (34,)
    avg_medians = (median_left + median_right) / 2
    sorted_indices = np.argsort(avg_medians)[::-1]
    data_left_sorted = data_left[:, sorted_indices]
    data_right_sorted = data_right[:, sorted_indices]
    labels_sorted = labels[sorted_indices]
    # ====================================================
    plt.figure(figsize=(20, 10))

    x_positions = np.arange(1, 35)
    box_width = 0.35
    offset = 0.2
    jitter_strength = 0.04
    color_left_scatter = "tab:blue"
    color_right_scatter = "tab:orange"

    for i in range(34):
        y_l = data_left_sorted[:, i]
        mask_l = ~np.isnan(y_l)
        y_l = y_l[mask_l]
        x_l = np.random.normal(x_positions[i] - offset, jitter_strength, size=len(y_l))
        plt.scatter(x_l, y_l, s=15, alpha=0.5, color=color_left_scatter, zorder=1)
        y_r = data_right_sorted[:, i]
        mask_r = ~np.isnan(y_r)
        y_r = y_r[mask_r]
        x_r = np.random.normal(x_positions[i] + offset, jitter_strength, size=len(y_r))
        plt.scatter(x_r, y_r, s=15, alpha=0.5, color=color_right_scatter, zorder=1)
    bp_left = plt.boxplot(
        data_left_sorted,
        positions=x_positions - offset,
        widths=box_width,
        showfliers=False,
        patch_artist=False,
        zorder=10,
    )
    bp_right = plt.boxplot(
        data_right_sorted,
        positions=x_positions + offset,
        widths=box_width,
        showfliers=False,
        patch_artist=False,
        zorder=10,
    )

    set_box_style_black(bp_left)
    set_box_style_black(bp_right)
    plt.xlabel("ROIs (Hemisphere Pairs)", fontsize=24, fontweight="bold")
    plt.ylabel("Pearson Correlation", fontsize=24, fontweight="bold")
    plt.xticks(x_positions, labels_sorted, rotation=90, fontsize=22)
    plt.yticks(fontsize=18)

    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.ylim(0.6, 1.05)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_left_scatter,
            markersize=10,
            label="Left Hemisphere",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_right_scatter,
            markersize=10,
            label="Right Hemisphere",
        ),
        Line2D([0], [0], color="black", lw=2, label="Statistical Distribution"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    os.makedirs(os.path.dirname(OUTPUT_PNG_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PNG_PATH, dpi=300)
    plt.close()
    print(f"Sorted scatter + boxplot saved to {OUTPUT_PNG_PATH }")


if __name__ == "__main__":
    main()

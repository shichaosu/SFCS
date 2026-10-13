# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
base_dir = "data/data_use/aparc"
txt_dir = "data/visulization/Coefficient"
path_pos = os.path.join(
    base_dir, "GLM_Sparsify_Pos/average/20_absolute/CN_average_matrix.npy"
)
path_neg = os.path.join(
    base_dir, "GLM_Sparsify_Neg/average/20_absolute/CN_average_matrix.npy"
)
path_names = os.path.join(txt_dir, "compnames.txt")

save_dir = "data/visulization/figure_result/20"
save_name = "CN_Heatmap_Redundancy_vs_Synergy_Abs.png"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print("Loading matrix data...")
try:
    raw_pos = np.load(path_pos)
    raw_neg = np.load(path_neg)
    matrix_pos = np.abs(raw_pos[:, 1:])
    matrix_neg = np.abs(raw_neg[:, 1:])
except FileNotFoundError:
    print("Generating mock data...")
    raw_pos = np.random.randn(68, 24)
    raw_neg = np.random.randn(68, 24)
    matrix_pos = np.abs(raw_pos[:, 1:])
    matrix_neg = np.abs(raw_neg[:, 1:])
print("Loading component names...")
try:
    with open(path_names, "r") as f:
        all_names = [line.strip() for line in f.readlines()]
    if len(all_names) > 1:
        comp_names = all_names[1:]
    else:
        comp_names = all_names
except FileNotFoundError:
    print("Generating mock names...")
    comp_names = [f"Comp_{i }" for i in range(1, 24)]
if len(comp_names) != matrix_pos.shape[1]:
    comp_names = comp_names[: matrix_pos.shape[1]]

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 28
plt.rcParams["axes.labelsize"] = 22
idx_lh_end = 34
idx_split_col = 9
global_max = max(np.max(matrix_pos), np.max(matrix_neg))
vmin, vmax = 0, global_max

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(22, 10), constrained_layout=True)


def plot_single_heatmap(ax, data, title, show_y_labels=True, add_cbar=False):
    heatmap = sns.heatmap(
        data,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        cbar=add_cbar,
        cbar_kws={"shrink": 0.9, "ticks": []} if add_cbar else None,
        xticklabels=comp_names,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=32, fontweight="bold", pad=45)
    total_cols = data.shape[1]
    x_pos_comm = (idx_split_col / 2) / total_cols
    x_pos_morph = (idx_split_col + (total_cols - idx_split_col) / 2) / total_cols
    ax.text(
        x_pos_comm,
        1.02,
        "Communication Models",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=26,
    )
    ax.text(
        x_pos_morph,
        1.02,
        "Morphological Similarity",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=26,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)
    line_color = "black"
    line_width = 1.5
    ax.hlines(
        [idx_lh_end],
        *ax.get_xlim(),
        colors=line_color,
        linestyles="-",
        linewidth=line_width,
    )
    ax.vlines(
        [idx_split_col],
        *ax.get_ylim(),
        colors=line_color,
        linestyles="-",
        linewidth=line_width,
    )
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    if show_y_labels:
        pos_lh = idx_lh_end / 2
        pos_rh = idx_lh_end + (68 - idx_lh_end) / 2
        ax.text(
            -0.02,
            pos_lh / 68,
            "Right Hemisphere",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=24,
        )
        ax.text(
            -0.02,
            pos_rh / 68,
            "Left Hemisphere",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=24,
        )
    if add_cbar:
        cbar = heatmap.collections[0].colorbar
        cax = cbar.ax
        cax.text(
            -0.7,
            0.02,
            "Weak",
            transform=cax.transAxes,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=15,
        )
        cax.text(
            -0.7,
            0.98,
            "Strong",
            transform=cax.transAxes,
            rotation=90,
            ha="center",
            va="top",
            fontsize=15,
        )

        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------


plot_single_heatmap(
    axes[0], matrix_pos, "Redundancy", show_y_labels=True, add_cbar=False
)
plot_single_heatmap(axes[1], matrix_neg, "Synergy", show_y_labels=False, add_cbar=True)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
full_save_path = os.path.join(save_dir, save_name)
plt.savefig(full_save_path, dpi=300, bbox_inches="tight")

print(f"Figure saved to: {full_save_path }")
plt.show()

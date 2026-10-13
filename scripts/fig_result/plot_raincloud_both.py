# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.colors as mcolors

FILE_NEG = (
    "data/data_use/aparc/GLM_Sparsify_Neg/average/20_absolute/CN_average_matrix.npy"
)
FILE_POS = (
    "data/data_use/aparc/GLM_Sparsify_Pos/average/20_absolute/CN_average_matrix.npy"
)
NAME_FILE = "data/visulization/Coefficient/compnames.txt"
OUTPUT_DIR = "data/visulization/figure_result/20"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Processing message.")
with open(NAME_FILE, "r") as f:
    all_names = [line.strip() for line in f.readlines()]
comp_names = all_names[1:]

mat_neg = np.load(FILE_NEG)[:, 1:]
mat_pos = np.load(FILE_POS)[:, 1:]
COLOR_NEG = "#3C5488"
COLOR_POS = "#E64B35"


def get_darker_color(hex_color, factor=1):
    """Script documentation for this SFCS analysis step."""
    rgb = mcolors.to_rgb(hex_color)
    new_rgb = [max(0, min(1, x * factor)) for x in rgb]
    return mcolors.to_hex(new_rgb)


COLOR_NEG_DARK = get_darker_color(COLOR_NEG, factor=0.8)
COLOR_POS_DARK = get_darker_color(COLOR_POS, factor=0.8)
OFFSET_CENTER = 0.12
WIDTH_CLOUD = 0.25
WIDTH_BOX = 0.12
JITTER_WIDTH = 0.08


def draw_half_violin(ax, data, pos, color, direction, width):
    """Script documentation for this SFCS analysis step."""
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return

    kde = stats.gaussian_kde(data)
    y_min, y_max = data.min(), data.max()
    y_grid = np.linspace(
        y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1, 100
    )
    z = kde(y_grid)
    z = z / z.max() * width

    x_curve = pos + (z * direction)
    ax.fill_betweenx(y_grid, pos, x_curve, facecolor=color, alpha=0.6, edgecolor=None)


def add_rain_box(ax, data, pos, scatter_color, box_edge_color):
    """Script documentation for this SFCS analysis step."""
    jitter = np.random.uniform(-JITTER_WIDTH / 2, JITTER_WIDTH / 2, size=len(data))
    x_scatter = pos + jitter
    ax.scatter(x_scatter, data, s=3, color=scatter_color, alpha=0.5, zorder=1)
    bp = ax.boxplot(
        [data],
        positions=[pos],
        widths=WIDTH_BOX,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        zorder=10,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_linewidth(1.5)
        patch.set_edgecolor(box_edge_color)
    for element in ["whiskers", "caps"]:
        for patch in bp[element]:
            patch.set_color(box_edge_color)
            patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)


print("Processing message.")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

f, ax = plt.subplots(figsize=(24, 12))

for i, name in enumerate(comp_names):
    d_neg = mat_neg[:, i]
    d_pos = mat_pos[:, i]
    pos_pos_anchor = i - OFFSET_CENTER
    draw_half_violin(
        ax,
        d_pos,
        pos=pos_pos_anchor - 0.02,
        color=COLOR_POS,
        direction=-1,
        width=WIDTH_CLOUD,
    )
    add_rain_box(
        ax,
        d_pos,
        pos=pos_pos_anchor,
        scatter_color=COLOR_POS,
        box_edge_color=COLOR_POS_DARK,
    )
    pos_neg_anchor = i + OFFSET_CENTER
    draw_half_violin(
        ax,
        d_neg,
        pos=pos_neg_anchor + 0.02,
        color=COLOR_NEG,
        direction=1,
        width=WIDTH_CLOUD,
    )
    add_rain_box(
        ax,
        d_neg,
        pos=pos_neg_anchor,
        scatter_color=COLOR_NEG,
        box_edge_color=COLOR_NEG_DARK,
    )

y_max = max(mat_neg.max(), mat_pos.max())
top_margin = y_max * 1.25
ax.set_ylim(0, top_margin)

split_index = 8.5
ax.axvline(
    x=split_index, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, ymax=0.9
)

text_y = top_margin * 0.92
ax.text(
    x=4,
    y=text_y,
    s="Communication Models",
    ha="center",
    va="center",
    fontsize=36,
    fontweight="bold",
    color="black",
)
ax.text(
    x=16,
    y=text_y,
    s="Morphological Similarity",
    ha="center",
    va="center",
    fontsize=36,
    fontweight="bold",
    color="black",
)

ax.set_xticks(range(len(comp_names)))
ax.set_xticklabels(comp_names, rotation=45, ha="right", fontsize=24)
ax.tick_params(
    axis="x", which="major", direction="out", length=6, width=1.5, colors="black"
)
ax.set_xlabel("", fontsize=14)

ax.set_ylabel(r"Constraint Strength", fontsize=34, fontweight="bold", color="black")
ax.tick_params(
    axis="y", which="major", direction="out", length=6, width=1.5, colors="black"
)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(
        facecolor=COLOR_NEG, edgecolor=COLOR_NEG_DARK, linewidth=1.5, label="Synergy"
    ),
    Patch(
        facecolor=COLOR_POS, edgecolor=COLOR_POS_DARK, linewidth=1.5, label="Redundancy"
    ),
]

legend = ax.legend(
    handles=legend_elements,
    title="Constraint Type",
    title_fontsize="30",
    fontsize="28",
    loc="upper left",
    bbox_to_anchor=(
        min(
            (split_index - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
            + 0.02,
            0.98,
        ),
        0.88,
    ),
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1,
    labelcolor="black",
)
plt.setp(legend.get_title(), color="black")

sns.despine(bottom=False, left=False, ax=ax)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("black")
    ax.spines[spine].set_linewidth(1.5)

save_name = "CN_Structural_Contribution_DarkerBox.png"
save_path = os.path.join(OUTPUT_DIR, save_name)

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(f)

print(f"Processing message.")

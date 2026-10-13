# Visualization helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle


FILE_PATH = r"data/CN_performance_BrainGNN.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = r"data/fig"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "CN_grouped_dumbbell_plot.png")

COLOR_REDUNDANCY = "#E64B35"
COLOR_SYNERGY = "#4DBBD5"
GROUP_ORDER = ["Global Staging", "Executive functions", "Basic cognitive"]
UNGROUPED_LABEL = "Ungrouped"
GROUP_COLORS = {
    "Global Staging": "#F6C445",
    "Executive functions": "#58B368",
    "Basic cognitive": "#9B7EDE",
    UNGROUPED_LABEL: "#D9D9D9",
}


def find_sheet_name(file_path: str, target_name: str) -> str:
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        if sheet_name.lower() == target_name.lower():
            return sheet_name
    raise ValueError(f"Worksheet named '{target_name }' not found in {file_path }.")


def load_and_sort_data(file_path: str, target_sheet: str) -> pd.DataFrame:
    sheet_name = find_sheet_name(file_path, target_sheet)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    required_columns = ["Metric", "Pos_r", "Pos_p", "Neg_r", "Neg_p", "Group"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns }")

    df = df[required_columns].copy()
    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Group"] = df["Group"].fillna("").astype(str).str.strip()
    df["DisplayGroup"] = df["Group"].replace("", UNGROUPED_LABEL)
    df["OriginalOrder"] = range(len(df))
    df["Delta_r"] = (df["Pos_r"] - df["Neg_r"]).abs()
    df["Highlighted"] = df["Delta_r"] >= 0.05

    df = df.sort_values(
        by=["Delta_r", "OriginalOrder"], ascending=[False, True]
    ).reset_index(drop=True)
    return df


def build_legends(ax: plt.Axes, separator_y: float) -> None:
    separator_axes_y = ax.transAxes.inverted().transform(
        ax.transData.transform((0, separator_y))
    )[1]
    legend_left = 0.74

    top_handles = [
        mlines.Line2D(
            [], [], color=COLOR_REDUNDANCY, linewidth=3.2, label="Redundancy-dominant"
        ),
        mlines.Line2D(
            [], [], color=COLOR_SYNERGY, linewidth=3.2, label="Synergy-dominant"
        ),
        Patch(
            facecolor=GROUP_COLORS["Global Staging"],
            edgecolor="none",
            label="Global Staging",
        ),
        Patch(
            facecolor=GROUP_COLORS["Executive functions"],
            edgecolor="none",
            label="Executive functions",
        ),
        Patch(
            facecolor=GROUP_COLORS["Basic cognitive"],
            edgecolor="none",
            label="Basic cognitive",
        ),
    ]
    bottom_handles = [
        mlines.Line2D(
            [],
            [],
            color=COLOR_REDUNDANCY,
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=COLOR_REDUNDANCY,
            markeredgecolor=COLOR_REDUNDANCY,
            label="Redundancy",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLOR_SYNERGY,
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=COLOR_SYNERGY,
            markeredgecolor=COLOR_SYNERGY,
            label="Synergy",
        ),
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor="#444444",
            markeredgecolor="#444444",
            label="p < 0.05",
        ),
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=1.8,
            label="p >= 0.05",
        ),
    ]

    top_legend = ax.legend(
        handles=top_handles,
        loc="lower left",
        bbox_to_anchor=(legend_left, min(0.94, separator_axes_y + 0.03)),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=11,
        borderaxespad=0.25,
        handlelength=1.8,
        labelspacing=0.65,
        facecolor="white",
        edgecolor="white",
    )
    ax.add_artist(top_legend)

    ax.legend(
        handles=bottom_handles,
        loc="lower left",
        bbox_to_anchor=(legend_left, 0.03),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=11,
        borderaxespad=0.3,
        handlelength=2.0,
        labelspacing=0.8,
        facecolor="white",
        edgecolor="white",
    )


def plot_dumbbell(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig_height = max(7.5, 0.56 * len(df) + 2.6)
    fig, (ax_labels, ax) = plt.subplots(
        1,
        2,
        figsize=(11.8, fig_height),
        gridspec_kw={"width_ratios": [2.4, 5.2], "wspace": 0.0},
        sharey=True,
    )

    highlight_count = int(df["Highlighted"].sum())
    section_gap = 1.0
    y_positions = []
    for idx in range(len(df)):
        y = idx if idx < highlight_count else idx + section_gap
        y_positions.append(y)
    separator_y = highlight_count - 0.5 + section_gap / 2

    for y, row in zip(y_positions, df.itertuples(index=False)):
        is_highlighted = row.Highlighted
        if is_highlighted:
            line_color = COLOR_REDUNDANCY if row.Pos_r >= row.Neg_r else COLOR_SYNERGY
            line_alpha = 1.0
            line_width = 4.0
        else:
            line_color = "#B8B8B8"
            line_alpha = 0.35
            line_width = 1.8

        ax.hlines(
            y=y,
            xmin=min(row.Pos_r, row.Neg_r),
            xmax=max(row.Pos_r, row.Neg_r),
            color=line_color,
            alpha=line_alpha,
            linewidth=line_width,
            zorder=1,
        )

    for y, row in zip(y_positions, df.itertuples(index=False)):
        pos_significant = row.Pos_p < 0.05
        neg_significant = row.Neg_p < 0.05

        ax.scatter(
            row.Pos_r,
            y,
            s=120,
            facecolor=COLOR_REDUNDANCY if pos_significant else "white",
            edgecolor=COLOR_REDUNDANCY,
            linewidth=2,
            alpha=1.0,
            zorder=3,
        )
        ax.scatter(
            row.Neg_r,
            y,
            s=120,
            facecolor=COLOR_SYNERGY if neg_significant else "white",
            edgecolor=COLOR_SYNERGY,
            linewidth=2,
            alpha=1.0,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.invert_yaxis()

    x_min = min(df["Pos_r"].min(), df["Neg_r"].min())
    x_max = max(df["Pos_r"].max(), df["Neg_r"].max())
    x_padding_left = 0.0
    x_padding_right = 0.04
    ax.set_xlim(0.0, 0.30)

    ax.set_xlabel("Prediction performance (Pearson r)", fontsize=15)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#CFCFCF")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=13)

    ax.axvline(0, color="#CFCFCF", linewidth=1.2, zorder=2)
    ax.axhline(separator_y, color="black", linewidth=1.4, zorder=2)
    ax.text(
        0.006,
        separator_y - 0.28,
        r"$\Delta r > 0.05$",
        ha="left",
        va="bottom",
        fontsize=18,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.text(
        0.006,
        separator_y + 0.28,
        r"$\Delta r < 0.05$",
        ha="left",
        va="top",
        fontsize=18,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    ax_labels.set_xlim(0, 1)
    ax_labels.set_xticks([])
    ax_labels.set_yticks(y_positions)
    ax_labels.set_yticklabels([])
    ax_labels.tick_params(axis="y", length=0)
    for spine in ax_labels.spines.values():
        spine.set_visible(False)

    for y, row in zip(y_positions, df.itertuples(index=False)):
        strip_color = GROUP_COLORS.get(row.DisplayGroup, GROUP_COLORS[UNGROUPED_LABEL])
        strip_alpha = 1.0 if row.Highlighted else 0.45
        ax_labels.add_patch(
            Rectangle(
                (0.92, y - 0.40),
                0.08,
                0.86,
                transform=ax_labels.transData,
                facecolor=strip_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=strip_alpha,
                clip_on=False,
                zorder=0,
            )
        )
        ax_labels.text(
            0.89,
            y,
            row.Metric,
            va="center",
            ha="right",
            fontsize=14.5,
            color="black",
        )

    ax_labels.axhline(separator_y, color="black", linewidth=1.4, zorder=2)

    build_legends(ax, separator_y)

    plt.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.09)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_and_sort_data(FILE_PATH, SHEET_NAME)
    plot_dumbbell(df, OUTPUT_PATH)
    print(f"Saved figure to: {OUTPUT_PATH }")


if __name__ == "__main__":
    main()

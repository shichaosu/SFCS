# Visualization helper for the SFCS manuscript.
# Update the path variables below before running this script.
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILE_PATH = Path(r"data/CN_performance_BrainGNN.xlsx")
SHEET_NAME = "Sheet4"
OUTPUT_DIR = Path(r"data/fig")
OUTPUT_PATH = OUTPUT_DIR / "CN_Sheet4_Bubble_Comparison.png"

CATEGORY_ORDER = ["PT", "SP"]
CATEGORY_LABELS = {
    "PT": "Patient\nReported",
    "SP": "Informant\nReported",
}
DIMENSION_ORDER = [
    "Total",
    "Memory",
    "Language",
    "Visuospatial",
    "Planning",
    "Organization",
    "Divided attention",
]
DIMENSION_LABELS = {
    "Total": "Total",
    "Memory": "Memory",
    "Language": "Language",
    "Visuospatial": "Visuospatial",
    "Planning": "Planning",
    "Organization": "Organization",
    "Divided attention": "Divided\nattention",
}
DIMENSION_DISPLAY = {
    "Total": "Total",
    "Mem": "Memory",
    "Lang": "Language",
    "Visspat": "Visuospatial",
    "Plan": "Planning",
    "Organ": "Organization",
    "Divatt": "Divided attention",
}


def find_sheet_name(file_path: Path, target_name: str) -> str:
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        if sheet_name.lower() == target_name.lower():
            return sheet_name
    raise ValueError(f"Worksheet named '{target_name }' was not found in {file_path }.")


def parse_metric(metric: str) -> tuple[str, str]:
    text = str(metric).strip()
    match = re.match("Processing message.", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Unable to parse Metric value '{metric }'. Expected format like 'EcogSPDivatt' or 'EcogPtMem'."
        )

    raw_category = match.group("category")
    raw_dimension = match.group("dimension")

    category = "SP" if raw_category.upper() == "SP" else "PT"
    try:
        dimension = DIMENSION_DISPLAY[raw_dimension]
    except KeyError as exc:
        raise ValueError(
            f"Unknown cognitive dimension '{raw_dimension }' in Metric '{metric }'."
        ) from exc

    return category, dimension


def load_data(file_path: Path, target_sheet: str) -> pd.DataFrame:
    sheet_name = find_sheet_name(file_path, target_sheet)
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    required_columns = ["Metric", "R", "P"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {sheet_name }: {missing_columns }"
        )

    df = df[required_columns].copy()
    df[["Category", "Dimension"]] = df["Metric"].apply(
        lambda value: pd.Series(parse_metric(value))
    )
    df["Category"] = pd.Categorical(
        df["Category"], categories=CATEGORY_ORDER, ordered=True
    )
    df["Dimension"] = pd.Categorical(
        df["Dimension"], categories=DIMENSION_ORDER, ordered=True
    )
    df["R_abs"] = df["R"].abs()
    df["neg_log10_p"] = -np.log10(df["P"].clip(lower=np.finfo(float).tiny))

    df = df.sort_values(["Dimension", "Category"]).reset_index(drop=True)
    return df


def scale_bubble_sizes(
    r_values: pd.Series, min_size: float = 35, max_size: float = 720
) -> np.ndarray:
    r_array = np.asarray(r_values, dtype=float)
    if np.allclose(r_array.max(), r_array.min()):
        return np.full_like(r_array, fill_value=(min_size + max_size) / 2, dtype=float)

    normalized = (r_array - r_array.min()) / (r_array.max() - r_array.min())
    return min_size + normalized * (max_size - min_size)


def draw_size_legend(ax: plt.Axes, r_abs_series: pd.Series) -> None:
    legend_levels = np.array([0.05, 0.15, 0.25])
    valid_levels = legend_levels[legend_levels <= r_abs_series.max() + 1e-9]
    if len(valid_levels) == 0:
        valid_levels = np.array([float(r_abs_series.max())])

    source_min = float(r_abs_series.min())
    source_max = float(r_abs_series.max())
    if np.isclose(source_min, source_max):
        legend_sizes = np.full(len(valid_levels), 230.0)
    else:
        normalized = (valid_levels - source_min) / (source_max - source_min)
        legend_sizes = 60 + normalized * (320 - 60)

    y_positions = np.linspace(0.32, 0.08, len(valid_levels))
    circle_x = 0.24
    label_x = 0.43

    for y_pos, size, level in zip(y_positions, legend_sizes, valid_levels):
        ax.scatter(
            circle_x,
            y_pos,
            s=size,
            facecolor="#F4A3A3",
            edgecolor="#8B1E3F",
            linewidth=1.0,
            alpha=0.9,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            label_x,
            y_pos,
            f"R = {level :.2f}",
            fontsize=14,
            color="black",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )


def plot_bubble_comparison(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(8.5, 6.4))
    grid_spec = fig.add_gridspec(1, 2, width_ratios=[4.25, 1.10], wspace=0.08)
    ax = fig.add_subplot(grid_spec[0, 0])
    ax_legend = fig.add_subplot(grid_spec[0, 1])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    x_positions = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    y_positions = {dimension: index for index, dimension in enumerate(DIMENSION_ORDER)}

    x = df["Category"].map(x_positions).to_numpy(dtype=float)
    y = df["Dimension"].map(y_positions).to_numpy(dtype=float)
    sizes = scale_bubble_sizes(df["R_abs"])

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=df["neg_log10_p"],
        cmap="Reds",
        alpha=0.88,
        edgecolor="#8B1E3F",
        linewidth=1.0,
    )

    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(
        [CATEGORY_LABELS[category] for category in CATEGORY_ORDER],
        fontsize=19,
        color="black",
    )
    ax.set_yticks(range(len(DIMENSION_ORDER)))
    ax.set_yticklabels(
        [DIMENSION_LABELS[dimension] for dimension in DIMENSION_ORDER],
        fontsize=18,
        color="black",
    )
    ax.invert_yaxis()

    ax.set_axisbelow(True)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.8, color="#E6D6D8")
    ax.set_xlim(-0.5, len(CATEGORY_ORDER) - 0.5)
    ax.set_ylim(len(DIMENSION_ORDER) - 0.5, -0.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#B38A91")
    ax.spines["bottom"].set_color("#B38A91")
    ax.tick_params(axis="both", colors="black", labelcolor="black")

    cax = ax_legend.inset_axes([0.20, 0.57, 0.46, 0.27])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(r"$-\log_{10}(P)$", fontsize=12, color="black")
    cbar.ax.tick_params(labelsize=10, colors="black")

    draw_size_legend(ax_legend, df["R_abs"])

    fig.subplots_adjust(left=0.20, right=0.95, top=0.96, bottom=0.12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_data(FILE_PATH, SHEET_NAME)
    plot_bubble_comparison(df, OUTPUT_PATH)
    print(f"Bubble comparison plot saved to: {OUTPUT_PATH }")


if __name__ == "__main__":
    main()

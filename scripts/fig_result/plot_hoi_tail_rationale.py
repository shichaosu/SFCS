#!/usr/bin/env python3
# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
"""Plot sample-level HOI distribution figures that justify rank-based 20% tails.

The script reads sample HOI tensors from ``*.npy`` files, extracts unique
triplets with ``i < j < k``, and saves separate figures showing why using the
lowest/highest 20% of values within each sample is more defensible than using
one fixed value threshold for everyone.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_INPUT_DIR = Path("data/data_use/Final_data/HOI/CN")
DEFAULT_OUTPUT_DIR = Path("data/visulization/figure_result/20/sup")

COLOR_SYNERGY = "#2B5C8A"
COLOR_SYNERGY_FILL = "#A9BED7"
COLOR_REDUNDANCY = "#C4443C"
COLOR_REDUNDANCY_FILL = "#E9B0AB"
COLOR_CUTOFF = "#333333"
COLOR_ZERO = "#404040"
COLOR_SAMPLE_CURVE = "#B8B8B8"
COLOR_SEPARATOR = "#D9D9D9"
DISPLAY_ORDER_SEED = 20260424
HEATMAP_BASE_CMAP = "YlGnBu"
HEATMAP_CMAP_START = 0.02
HEATMAP_CMAP_END = 0.98
HEATMAP_REFERENCE_QUANTILE = 0.80
HEATMAP_REFERENCE_COLOR_POSITION = 0.48
HEATMAP_GAMMA_MIN = 0.35
HEATMAP_GAMMA_MAX = 0.75
HEATMAP_COLORBAR_TICK_STEP = 0.2
TICK_LABEL_FONTSIZE = 13
INSET_TICK_LABEL_FONTSIZE = 11.5
AXIS_LABEL_FONTSIZE = 18
LARGE_AXIS_LABEL_FONTSIZE = 22
COLORBAR_LABEL_FONTSIZE = 17
TITLE_FONTSIZE = 19
SUBTITLE_FONTSIZE = 15
ANNOTATION_FONTSIZE = 14
LEGEND_FONTSIZE = 15
HEATMAP_TICK_LABEL_FONTSIZE = 16
HEATMAP_AXIS_LABEL_FONTSIZE = 28
HEATMAP_COLORBAR_LABEL_FONTSIZE = 20
HEATMAP_COLORBAR_LABELPAD = 4
HEATMAP_INSET_TICK_LABEL_FONTSIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot figures supporting within-sample 20%% HOI tail selection."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-regions", type=int, default=68)
    parser.add_argument("--curve-points", type=int, default=501)
    parser.add_argument("--hist-bins", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def build_unique_triplets(n_regions: int) -> np.ndarray:
    return np.array(list(combinations(range(n_regions), 3)), dtype=np.int16)


def extract_unique_triplets(
    path: Path, triplets: np.ndarray, n_regions: int
) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.shape != (n_regions, n_regions, n_regions):
        raise ValueError(
            f"{path } has shape {arr .shape }, expected {(n_regions ,n_regions ,n_regions )}"
        )
    values = arr[triplets[:, 0], triplets[:, 1], triplets[:, 2]].astype(
        np.float32, copy=False
    )
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
    return values


def interpolate_sorted(
    sorted_values: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    if sorted_values.size == 0:
        raise ValueError("Cannot interpolate an empty array.")
    positions = probabilities * (sorted_values.size - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weights = positions - lower
    return sorted_values[lower] * (1.0 - weights) + sorted_values[upper] * weights


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0:
        return 1.0 if np.allclose(x, y) else np.nan
    return float(np.dot(x_centered, y_centered) / denom)


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def format_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            linestyle=(0, (3, 3)),
            linewidth=0.5,
            alpha=0.18,
            color=COLOR_ZERO,
        )


def add_cutoff_lines(ax: plt.Axes, cutoffs: list[float]) -> None:
    for cutoff in cutoffs:
        ax.axvline(
            cutoff, color=COLOR_CUTOFF, linestyle=(0, (4, 2)), linewidth=1.0, zorder=1
        )


def build_heatmap_cmap() -> LinearSegmentedColormap:
    base_cmap = plt.get_cmap(HEATMAP_BASE_CMAP)
    color_positions = np.linspace(HEATMAP_CMAP_START, HEATMAP_CMAP_END, 256)
    color_stops = [(0.0, "#FFFFFF")]
    color_stops.extend(
        (idx / color_positions.size, base_cmap(position))
        for idx, position in enumerate(color_positions, start=1)
    )
    return LinearSegmentedColormap.from_list("hoi_density_data_adaptive", color_stops)


def build_heatmap_norm(display_matrix: np.ndarray, vmax: float) -> PowerNorm:
    positive_values = display_matrix[
        np.isfinite(display_matrix) & (display_matrix > 0.0)
    ]
    if vmax <= 0.0 or positive_values.size == 0:
        return PowerNorm(gamma=1.0, vmin=0.0, vmax=max(vmax, 1.0))

    reference_value = float(np.quantile(positive_values, HEATMAP_REFERENCE_QUANTILE))
    reference_ratio = np.clip(reference_value / vmax, 1e-6, 1.0 - 1e-6)
    gamma = math.log(HEATMAP_REFERENCE_COLOR_POSITION) / math.log(reference_ratio)
    gamma = float(np.clip(gamma, HEATMAP_GAMMA_MIN, HEATMAP_GAMMA_MAX))
    return PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)


def build_colorbar_ticks(vmax: float) -> np.ndarray:
    tick_stop = (
        math.floor(vmax / HEATMAP_COLORBAR_TICK_STEP) * HEATMAP_COLORBAR_TICK_STEP
    )
    return np.arange(
        0.0, tick_stop + HEATMAP_COLORBAR_TICK_STEP * 0.5, HEATMAP_COLORBAR_TICK_STEP
    )


def add_linear_tick_colorbar(
    ax: plt.Axes,
    cmap: LinearSegmentedColormap,
    norm: PowerNorm,
    label_fontsize: float = COLORBAR_LABEL_FONTSIZE,
    tick_labelsize: float = TICK_LABEL_FONTSIZE,
    labelpad: float = 20,
) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.52)
    vmax = float(norm.vmax)
    gradient = np.linspace(0.0, vmax, 512, dtype=np.float32)[:, np.newaxis]
    cax.imshow(
        gradient,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[0.0, 1.0, 0.0, vmax],
        interpolation="nearest",
    )
    cax.set_xticks([])
    cax.set_yticks(build_colorbar_ticks(vmax))
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("left")
    cax.set_ylabel(
        "Relative within-sample density",
        rotation=90,
        labelpad=labelpad,
        fontsize=label_fontsize,
    )
    cax.tick_params(width=0.8, length=4, labelsize=tick_labelsize)
    for spine in cax.spines.values():
        spine.set_linewidth(0.7)


def plot_sample_distribution_heatmap(
    hist_matrix: np.ndarray,
    bin_edges: np.ndarray,
    main_abs_max: float,
    output_stem: Path,
    dpi: int,
) -> None:
    rng = np.random.default_rng(DISPLAY_ORDER_SEED)
    order = rng.permutation(hist_matrix.shape[0])
    display_matrix = np.log1p(hist_matrix[order])
    vmax = float(np.quantile(display_matrix, 0.995))
    heatmap_cmap = build_heatmap_cmap()
    heatmap_norm = build_heatmap_norm(display_matrix, vmax)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    main_idx = np.where(np.abs(bin_centers) <= main_abs_max)[0]
    main_start = int(main_idx[0])
    main_stop = int(main_idx[-1]) + 1

    fig, ax = plt.subplots(figsize=(7.8, 8.7))
    image = ax.imshow(
        display_matrix[:, main_start:main_stop],
        aspect="auto",
        origin="lower",
        extent=[bin_edges[main_start], bin_edges[main_stop], 1, hist_matrix.shape[0]],
        cmap=heatmap_cmap,
        norm=heatmap_norm,
        interpolation="nearest",
    )
    ax.axvline(
        0.0, color=COLOR_CUTOFF, linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.95
    )
    ax.set_xlabel("O-information", fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Samples (random order)", fontsize=HEATMAP_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(-main_abs_max, main_abs_max)
    ax.tick_params(width=0.8, length=4, labelsize=HEATMAP_TICK_LABEL_FONTSIZE)
    # ax.set_title("HOI value distributions across samples", pad=8, fontsize=TITLE_FONTSIZE)

    inset = ax.inset_axes([0.67, 0.74, 0.27, 0.18])
    inset.imshow(
        display_matrix,
        aspect="auto",
        origin="lower",
        extent=[bin_edges[0], bin_edges[-1], 1, hist_matrix.shape[0]],
        cmap=heatmap_cmap,
        norm=heatmap_norm,
        interpolation="nearest",
    )
    inset.axvline(
        0.0, color=COLOR_CUTOFF, linestyle=(0, (4, 2)), linewidth=0.8, alpha=0.95
    )
    inset_x_ticks = [bin_edges[0], 0.0, bin_edges[-1]]
    inset.set_xticks(inset_x_ticks, labels=[f"{tick :.1f}" for tick in inset_x_ticks])
    inset.set_yticks([])
    inset.tick_params(
        labelsize=HEATMAP_INSET_TICK_LABEL_FONTSIZE, width=0.6, length=2.5
    )
    # inset.set_title("Full range", fontsize=SUBTITLE_FONTSIZE, pad=3)
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)

    add_linear_tick_colorbar(
        ax,
        heatmap_cmap,
        heatmap_norm,
        label_fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
        tick_labelsize=HEATMAP_TICK_LABEL_FONTSIZE,
        labelpad=HEATMAP_COLORBAR_LABELPAD,
    )
    fig.tight_layout()
    save_figure(fig, output_stem, dpi)
    plt.close(fig)


def plot_rank_value_curves(
    quantile_curves: np.ndarray,
    percentile_grid: np.ndarray,
    zoom_y_limits: tuple[float, float],
    full_y_limits: tuple[float, float],
    output_stem: Path,
    dpi: int,
) -> None:
    x_values = percentile_grid * 100.0
    curve_median = np.median(quantile_curves, axis=0)
    curve_q25, curve_q75 = np.quantile(quantile_curves, [0.25, 0.75], axis=0)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for curve in quantile_curves:
        ax.plot(x_values, curve, color=COLOR_SAMPLE_CURVE, linewidth=0.55, alpha=0.09)
    ax.fill_between(
        x_values, curve_q25, curve_q75, color=COLOR_SYNERGY_FILL, alpha=0.20
    )
    ax.plot(x_values, curve_median, color=COLOR_SYNERGY, linewidth=2.0)
    ax.axhline(0.0, color=COLOR_ZERO, linestyle=(0, (1.5, 2.2)), linewidth=0.95)
    add_cutoff_lines(ax, [20.0, 80.0])
    ax.set_xlabel(
        "Within-sample rank percentile (%)", fontsize=LARGE_AXIS_LABEL_FONTSIZE
    )
    ax.set_ylabel("O-information", fontsize=LARGE_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(*zoom_y_limits)
    # ax.set_title("Within-sample rank-value curves", pad=8, fontsize=TITLE_FONTSIZE)
    format_axes(ax)

    legend_handles = [
        Line2D([0], [0], color=COLOR_SYNERGY, linewidth=2.0, label="Median"),
        Line2D(
            [0],
            [0],
            color=COLOR_CUTOFF,
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label="20% and 80% cutoffs",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        handlelength=2.2,
        fontsize=LEGEND_FONTSIZE,
    )

    inset = ax.inset_axes([0.63, 0.10, 0.31, 0.31])
    inset.fill_between(
        x_values, curve_q25, curve_q75, color=COLOR_SYNERGY_FILL, alpha=0.20
    )
    inset.plot(x_values, curve_median, color=COLOR_SYNERGY, linewidth=1.2)
    inset.axhline(0.0, color=COLOR_ZERO, linestyle=(0, (1.5, 2.2)), linewidth=0.8)
    add_cutoff_lines(inset, [20.0, 80.0])
    inset.set_xlim(0.0, 100.0)
    inset.set_ylim(*full_y_limits)
    inset.set_xticks([0, 20, 50, 80, 100])
    inset.tick_params(labelsize=INSET_TICK_LABEL_FONTSIZE, width=0.7, length=3)
    inset.spines["top"].set_linewidth(0.7)
    inset.spines["right"].set_linewidth(0.7)
    inset.spines["left"].set_linewidth(0.7)
    inset.spines["bottom"].set_linewidth(0.7)
    # inset.set_title("Full range", fontsize=SUBTITLE_FONTSIZE, pad=3)

    fig.tight_layout()
    save_figure(fig, output_stem, dpi)
    plt.close(fig)


def plot_sample_quantiles(
    q20_values: np.ndarray,
    q80_values: np.ndarray,
    output_stem: Path,
    dpi: int,
) -> None:
    colors = [COLOR_SYNERGY, COLOR_REDUNDANCY]
    rng = np.random.default_rng(42)
    y_abs_limit = max(abs(float(q20_values.min())), abs(float(q80_values.max()))) * 1.08

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    violin = ax.violinplot(
        [q20_values, q80_values],
        positions=[1, 2],
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)

    box = ax.boxplot(
        [q20_values, q80_values],
        positions=[1, 2],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 0.95},
        whiskerprops={"linewidth": 0.95},
        capprops={"linewidth": 0.95},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)

    ax.scatter(
        1 + rng.normal(0, 0.045, size=q20_values.size),
        q20_values,
        s=11,
        alpha=0.10,
        color=colors[0],
    )
    ax.scatter(
        2 + rng.normal(0, 0.045, size=q80_values.size),
        q80_values,
        s=11,
        alpha=0.10,
        color=colors[1],
    )
    ax.axhline(0.0, color=COLOR_ZERO, linestyle=(0, (1.5, 2.2)), linewidth=0.95)
    ax.set_xticks([1, 2], labels=["Sample q20\n(synergy)", "Sample q80\n(redundancy)"])
    ax.set_ylabel("Sample-specific O-information", fontsize=LARGE_AXIS_LABEL_FONTSIZE)
    ax.set_ylim(-y_abs_limit, y_abs_limit)
    # ax.set_title("Sample-specific q20 and q80 thresholds", pad=8, fontsize=TITLE_FONTSIZE)
    format_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_stem, dpi)
    plt.close(fig)


def plot_selected_fraction_comparison(
    frac_neg_fixed: np.ndarray,
    frac_pooled_q20: np.ndarray,
    frac_rank_low: np.ndarray,
    frac_pos_fixed: np.ndarray,
    frac_pooled_q80: np.ndarray,
    frac_rank_high: np.ndarray,
    output_stem: Path,
    dpi: int,
) -> None:
    labels = [
        "HOI < -0.3",
        "<= global q20",
        "Bottom 20%\n(within sample)",
        "HOI > 0.3",
        ">= global q80",
        "Top 20%\n(within sample)",
    ]
    values_list = [
        frac_neg_fixed,
        frac_pooled_q20,
        frac_rank_low,
        frac_pos_fixed,
        frac_pooled_q80,
        frac_rank_high,
    ]
    positions = [1, 2, 3, 5, 6, 7]
    colors = [
        COLOR_SYNERGY,
        COLOR_SYNERGY,
        COLOR_SYNERGY,
        COLOR_REDUNDANCY,
        COLOR_REDUNDANCY,
        COLOR_REDUNDANCY,
    ]
    rng = np.random.default_rng(7)
    y_lower = max(0.10, min(float(np.min(values)) for values in values_list) - 0.03)
    y_upper = min(0.82, max(float(np.max(values)) for values in values_list) + 0.03)

    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    box = ax.boxplot(
        values_list,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 0.95},
        whiskerprops={"linewidth": 0.95},
        capprops={"linewidth": 0.95},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.40)

    for position, values, color in zip(positions, values_list, colors):
        x = position + rng.normal(0, 0.05, size=values.size)
        ax.scatter(x, values, s=9, alpha=0.12, color=color)

    ax.axvline(4.0, color=COLOR_SEPARATOR, linewidth=0.8)
    ax.axhline(0.2, color=COLOR_CUTOFF, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlim(0.35, 7.65)
    ax.set_xticks(positions, labels=labels)
    ax.set_ylabel("Ratio of selected triples", fontsize=LARGE_AXIS_LABEL_FONTSIZE)
    ax.set_ylim(y_lower, y_upper)
    ax.text(
        2.0,
        y_upper - 0.014,
        "Synergy",
        ha="center",
        va="top",
        color=COLOR_SYNERGY,
        fontsize=22,
    )
    ax.text(
        6.0,
        y_upper - 0.014,
        "Redundancy",
        ha="center",
        va="top",
        color=COLOR_REDUNDANCY,
        fontsize=22,
    )
    ax.text(
        7.55,
        0.203,
        "20% target",
        ha="right",
        va="bottom",
        color=COLOR_CUTOFF,
        fontsize=ANNOTATION_FONTSIZE,
    )
    # ax.set_title("Fixed-value thresholds produce unequal selection sizes", pad=8, fontsize=TITLE_FONTSIZE)
    format_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_stem, dpi)
    plt.close(fig)


def plot_cutoff_sensitivity(
    cutoff_grid: np.ndarray,
    synergy_corrs: np.ndarray,
    redundancy_corrs: np.ndarray,
    output_stem: Path,
    dpi: int,
) -> None:
    x_values = cutoff_grid * 100.0
    syn_median = np.median(synergy_corrs, axis=0)
    syn_q25, syn_q75 = np.quantile(synergy_corrs, [0.25, 0.75], axis=0)
    red_median = np.median(redundancy_corrs, axis=0)
    red_q25, red_q75 = np.quantile(redundancy_corrs, [0.25, 0.75], axis=0)

    lower_limit = min(float(np.nanmin(syn_q25)), float(np.nanmin(red_q25)))
    lower_limit = min(lower_limit, 0.98)
    lower_limit = max(0.90, lower_limit - 0.02)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.fill_between(x_values, syn_q25, syn_q75, color=COLOR_SYNERGY_FILL, alpha=0.35)
    ax.plot(
        x_values,
        syn_median,
        color=COLOR_SYNERGY,
        linewidth=2.0,
        marker="o",
        label="Synergy",
    )
    ax.fill_between(x_values, red_q25, red_q75, color=COLOR_REDUNDANCY_FILL, alpha=0.33)
    ax.plot(
        x_values,
        red_median,
        color=COLOR_REDUNDANCY,
        linewidth=2.0,
        marker="s",
        linestyle="--",
        label="Redundancy",
    )
    ax.axvline(20.0, color=COLOR_CUTOFF, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlabel("Cutoff threshold (%)", fontsize=LARGE_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("20% Regional Profile Corr.", fontsize=LARGE_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_values, labels=[str(int(value)) for value in x_values])
    ax.set_ylim(lower_limit, 1.005)
    # ax.set_title("Cutoff sensitivity of regional participation profiles", pad=8, fontsize=TITLE_FONTSIZE)
    ax.legend(frameon=False, loc="lower left", fontsize=LEGEND_FONTSIZE)
    format_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_stem, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npy_paths = sorted(args.input_dir.glob("*.npy"))
    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in {args .input_dir }")

    triplets = build_unique_triplets(args.n_regions)
    expected_triplets = math.comb(args.n_regions, 3)
    if triplets.shape[0] != expected_triplets:
        raise RuntimeError("Triplet index construction failed.")

    print(f"Loading {len (npy_paths )} HOI tensors from {args .input_dir }")
    sample_values: list[np.ndarray] = []
    for idx, path in enumerate(npy_paths, start=1):
        values = extract_unique_triplets(path, triplets, args.n_regions)
        if values.size != expected_triplets:
            raise ValueError(
                f"{path .name } produced {values .size } unique triplet values, expected {expected_triplets }"
            )
        sample_values.append(values)
        if idx % 50 == 0 or idx == len(npy_paths):
            print(f"  loaded {idx }/{len (npy_paths )} samples")

    pooled_values = np.concatenate(sample_values)
    pooled_q20, pooled_q80 = np.quantile(pooled_values, [0.2, 0.8])
    plot_abs_max = float(np.quantile(np.abs(pooled_values), 0.995))
    heatmap_main_abs_max = 2.5
    hist_edges = np.linspace(-plot_abs_max, plot_abs_max, args.hist_bins + 1)
    percentile_grid = np.linspace(0.0, 1.0, args.curve_points)
    cutoff_grid = np.array([0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float64)
    baseline_cutoff_idx = int(np.where(np.isclose(cutoff_grid, 0.20))[0][0])
    del pooled_values

    n_samples = len(sample_values)
    hist_matrix = np.empty((n_samples, args.hist_bins), dtype=np.float32)
    quantile_curves = np.empty((n_samples, args.curve_points), dtype=np.float32)
    q20_values = np.empty(n_samples, dtype=np.float32)
    q80_values = np.empty(n_samples, dtype=np.float32)
    frac_neg_fixed = np.empty(n_samples, dtype=np.float32)
    frac_pos_fixed = np.empty(n_samples, dtype=np.float32)
    frac_pooled_q20 = np.empty(n_samples, dtype=np.float32)
    frac_pooled_q80 = np.empty(n_samples, dtype=np.float32)
    frac_rank_low = np.empty(n_samples, dtype=np.float32)
    frac_rank_high = np.empty(n_samples, dtype=np.float32)
    synergy_corrs = np.empty((n_samples, cutoff_grid.size), dtype=np.float32)
    redundancy_corrs = np.empty((n_samples, cutoff_grid.size), dtype=np.float32)

    print("Computing sample-level summaries")
    for sample_idx, values in enumerate(sample_values):
        order = np.argsort(values)
        sorted_values = values[order]
        low_profiles = np.empty((cutoff_grid.size, args.n_regions), dtype=np.float32)
        high_profiles = np.empty((cutoff_grid.size, args.n_regions), dtype=np.float32)

        q20, q80 = interpolate_sorted(sorted_values, np.array([0.20, 0.80]))
        q20_values[sample_idx] = q20
        q80_values[sample_idx] = q80
        quantile_curves[sample_idx] = interpolate_sorted(sorted_values, percentile_grid)

        clipped = np.clip(values, -plot_abs_max, plot_abs_max)
        hist_matrix[sample_idx], _ = np.histogram(
            clipped, bins=hist_edges, density=True
        )

        frac_neg_fixed[sample_idx] = np.mean(values < -0.3)
        frac_pos_fixed[sample_idx] = np.mean(values > 0.3)
        frac_pooled_q20[sample_idx] = np.mean(values <= pooled_q20)
        frac_pooled_q80[sample_idx] = np.mean(values >= pooled_q80)

        n_values = values.size
        for cutoff_idx, cutoff in enumerate(cutoff_grid):
            k = max(1, int(np.floor(cutoff * n_values)))
            if np.isclose(cutoff, 0.20):
                frac_rank_low[sample_idx] = k / n_values
                frac_rank_high[sample_idx] = k / n_values

            low_triplets = triplets[order[:k]]
            high_triplets = triplets[order[-k:]]

            low_profiles[cutoff_idx] = np.bincount(
                low_triplets.ravel(), minlength=args.n_regions
            ).astype(np.float32) / (3.0 * k)
            high_profiles[cutoff_idx] = np.bincount(
                high_triplets.ravel(), minlength=args.n_regions
            ).astype(np.float32) / (3.0 * k)

        baseline_low_profile = low_profiles[baseline_cutoff_idx]
        baseline_high_profile = high_profiles[baseline_cutoff_idx]
        for cutoff_idx in range(cutoff_grid.size):
            synergy_corrs[sample_idx, cutoff_idx] = safe_corr(
                low_profiles[cutoff_idx], baseline_low_profile
            )
            redundancy_corrs[sample_idx, cutoff_idx] = safe_corr(
                high_profiles[cutoff_idx], baseline_high_profile
            )

        if (sample_idx + 1) % 50 == 0 or (sample_idx + 1) == n_samples:
            print(f"  summarised {sample_idx +1 }/{n_samples } samples")

    full_y_limits = (-plot_abs_max, plot_abs_max)
    zoom_abs_max = (
        max(
            0.60,
            float(np.quantile(np.abs(q20_values), 0.98)),
            float(np.quantile(np.abs(q80_values), 0.98)),
        )
        * 1.45
    )
    zoom_y_limits = (-zoom_abs_max, zoom_abs_max)

    print(f"Saving figures to {args .output_dir }")
    plot_sample_distribution_heatmap(
        hist_matrix=hist_matrix,
        bin_edges=hist_edges,
        main_abs_max=heatmap_main_abs_max,
        output_stem=args.output_dir / "01_sample_distribution_heatmap",
        dpi=args.dpi,
    )
    plot_rank_value_curves(
        quantile_curves=quantile_curves,
        percentile_grid=percentile_grid,
        zoom_y_limits=zoom_y_limits,
        full_y_limits=full_y_limits,
        output_stem=args.output_dir / "02_rank_value_curves",
        dpi=args.dpi,
    )
    plot_sample_quantiles(
        q20_values=q20_values,
        q80_values=q80_values,
        output_stem=args.output_dir / "03_sample_quantiles_q20_q80",
        dpi=args.dpi,
    )
    plot_selected_fraction_comparison(
        frac_neg_fixed=frac_neg_fixed,
        frac_pooled_q20=frac_pooled_q20,
        frac_rank_low=frac_rank_low,
        frac_pos_fixed=frac_pos_fixed,
        frac_pooled_q80=frac_pooled_q80,
        frac_rank_high=frac_rank_high,
        output_stem=args.output_dir / "04_selected_fraction_comparison",
        dpi=args.dpi,
    )
    plot_cutoff_sensitivity(
        cutoff_grid=cutoff_grid,
        synergy_corrs=synergy_corrs,
        redundancy_corrs=redundancy_corrs,
        output_stem=args.output_dir / "05_cutoff_sensitivity_region_profile",
        dpi=args.dpi,
    )
    print("Finished.")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    main()

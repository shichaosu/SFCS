#!/usr/bin/env python3
# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
"""
Compute feature-wise correlations from CN npy files and draw a mixed heatmap:
- Upper triangle: Pearson
- Lower triangle: Spearman
- Diagonal: 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np


def collect_feature_subject_means(root_dir: Path) -> Dict[str, Dict[str, float]]:
    """Return {feature_name: {subject_id: mean_value}} from root/feature/CN/*.npy."""
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir }")

    feature_values: Dict[str, Dict[str, float]] = {}
    for feature_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        feature_name = feature_dir.name
        cn_dir = feature_dir / "CN"
        if not cn_dir.is_dir():
            continue

        npy_files = sorted(cn_dir.glob("*.npy"))
        if not npy_files:
            continue

        subject_map: Dict[str, float] = {}
        for npy_path in npy_files:
            subject_id = npy_path.stem
            arr = np.load(npy_path, allow_pickle=False)
            subject_map[subject_id] = float(np.nanmean(arr))

        if feature_name in feature_values:
            raise ValueError(f"Duplicate feature name in one root: {feature_name }")
        feature_values[feature_name] = subject_map

    return feature_values


def infer_source_tag(root_dir: Path) -> str:
    """Infer source tag from root path."""
    tag = root_dir.name.strip()
    return tag.upper() if tag else "UNKNOWN"


def source_full_name(source_tag: str) -> str:
    """Map source tag to display name along diagonal gap."""
    mapping = {
        "SC": "Communication Models",
        "MSN": "Morphological Similarity Networks",
    }
    return mapping.get(source_tag, source_tag)


def build_ordered_features(
    roots: List[Path], feature_maps: List[Dict[str, Dict[str, float]]]
) -> Tuple[List[str], List[Dict[str, float]], List[Tuple[str, str, int, int]]]:
    """
    Keep feature order grouped by source roots (e.g., SC then MSN).
    Returns:
    - feature_names
    - feature_subject_maps
    - group_ranges: [(source_tag, source_label, start_idx, end_idx_exclusive), ...]
    """
    if len(roots) != len(feature_maps):
        raise ValueError("roots and feature_maps must have the same length.")

    feature_names: List[str] = []
    feature_subject_maps: List[Dict[str, float]] = []
    group_ranges: List[Tuple[str, str, int, int]] = []
    used_names = set()

    cursor = 0
    for root, fmap in zip(roots, feature_maps):
        source_tag = infer_source_tag(root)
        source_label = source_full_name(source_tag)
        start = cursor

        for feature_name in sorted(fmap.keys()):
            out_name = feature_name
            if out_name in used_names:
                out_name = f"{source_tag }_{feature_name }"
            if out_name in used_names:
                suffix = 2
                while f"{out_name }_{suffix }" in used_names:
                    suffix += 1
                out_name = f"{out_name }_{suffix }"

            used_names.add(out_name)
            feature_names.append(out_name)
            feature_subject_maps.append(fmap[feature_name])
            cursor += 1

        end = cursor
        if end > start:
            group_ranges.append((source_tag, source_label, start, end))

    if not feature_names:
        raise ValueError("No valid features found.")
    return feature_names, feature_subject_maps, group_ranges


def build_matrix(
    feature_names: List[str],
    feature_subject_maps: List[Dict[str, float]],
) -> Tuple[List[str], List[str], np.ndarray]:
    """Build subject x feature matrix using common subjects across all features."""
    if not feature_names or not feature_subject_maps:
        raise ValueError("No valid features found.")

    subject_sets = [set(v.keys()) for v in feature_subject_maps]
    common_subjects = sorted(set.intersection(*subject_sets))
    if len(common_subjects) < 2:
        raise ValueError(
            "Less than 2 common subjects across all features; cannot compute correlations."
        )

    data = np.zeros((len(common_subjects), len(feature_names)), dtype=float)

    for j, subject_map in enumerate(feature_subject_maps):
        data[:, j] = np.asarray(
            [subject_map[sid] for sid in common_subjects], dtype=float
        )

    return common_subjects, feature_names, data


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Compute average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def corrcoef_safe(data: np.ndarray) -> np.ndarray:
    """Compute feature correlation matrix (columns as variables), replacing NaN with 0."""
    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def compute_correlations(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return Pearson and Spearman correlation matrices."""
    pearson = corrcoef_safe(data)

    ranked = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        ranked[:, j] = rankdata_average(data[:, j])
    spearman = corrcoef_safe(ranked)
    return pearson, spearman


def build_mixed_matrix(pearson: np.ndarray, spearman: np.ndarray) -> np.ndarray:
    """Upper triangle from Pearson, lower triangle from Spearman, diagonal as 0."""
    n = pearson.shape[0]
    mixed = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i < j:
                mixed[i, j] = pearson[i, j]
            elif i > j:
                mixed[i, j] = spearman[i, j]
            else:
                mixed[i, j] = 0.0
    return mixed


def plot_heatmap(
    labels: List[str],
    mixed: np.ndarray,
    output_path: Path,
    dpi: int,
    group_ranges: List[Tuple[str, str, int, int]],
) -> None:
    """Plot mixed correlation heatmap and save to file."""
    n = len(labels)

    cmap = LinearSegmentedColormap.from_list(
        "journal_diverging", ["#3B4CC0", "#F7F7F7", "#B40426"], N=256
    )
    cmap.set_bad(color="#FFFFFF")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig_w = max(12.0, 0.55 * n + 3.0)
    fig_h = max(10.0, 0.50 * n + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    display_matrix = mixed.copy()
    im = ax.imshow(display_matrix, cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticklabels(labels, fontsize=18)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title(
        "Upper: Pearson  |  Lower: Spearman",
        fontsize=20,
        pad=10,
    )

    # Grid lines for cleaner publication-style look.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Create a wider diagonal white band to separate upper/lower triangles
    # without staircase erosion around diagonal cells.
    gap_width = 1.45
    normal_offset = gap_width / (2.0 * np.sqrt(2.0))
    x0, y0 = -0.5, -0.5
    x1, y1 = n - 0.5, n - 0.5
    band_pts = np.array(
        [
            [x0 + normal_offset, y0 - normal_offset],
            [x1 + normal_offset, y1 - normal_offset],
            [x1 - normal_offset, y1 + normal_offset],
            [x0 - normal_offset, y0 + normal_offset],
        ]
    )
    ax.add_patch(
        Polygon(
            band_pts,
            closed=True,
            facecolor="white",
            edgecolor="none",
            zorder=3,
        )
    )

    # Draw group boundaries (e.g., SC vs MSN) as black separator lines.
    for _, _, _, end in group_ranges[:-1]:
        boundary = end - 0.5
        ax.axhline(boundary, color="black", linewidth=1.8, zorder=5)
        ax.axvline(boundary, color="black", linewidth=1.8, zorder=5)

        # Write source labels along the diagonal gap.
    for source_tag, source_label, start, end in group_ranges:
        center = 0.5 * (start + end - 1)
        fontsize = 16 if source_tag == "SC" else 17
        ax.text(
            center,
            center,
            source_label,
            rotation=-45,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="semibold",
            color="black",
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.0},
        )

        # Per-cell annotation.
    for i in range(n):
        for j in range(n):
            # Skip diagonal and the first off-diagonal on both sides:
            # these cells are intersected by the diagonal gap band.
            if abs(i - j) <= 1:
                continue
            val = mixed[i, j]
            ax.text(
                j,
                i,
                f"{val :.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                zorder=4,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed heatmap: upper Pearson, lower Spearman, diagonal 0."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "data/SC/aparc/communication/SC/",
            "data/MSN/aparc/MSN/",
        ],
        help="Root directories that contain feature folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/visulization/figure_result/20/",
        help="Output directory for figure file.",
    )
    parser.add_argument(
        "--output-name",
        default="CN_feature_mixed_corr_heatmap.png",
        help="Output heatmap filename.",
    )
    parser.add_argument("--dpi", type=int, default=500, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.roots]

    feature_maps = [collect_feature_subject_means(root) for root in roots]
    feature_names, feature_subject_maps, group_ranges = build_ordered_features(
        roots, feature_maps
    )
    common_subjects, feature_names, data = build_matrix(
        feature_names, feature_subject_maps
    )

    pearson, spearman = compute_correlations(data)
    mixed = build_mixed_matrix(pearson, spearman)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / args.output_name
    plot_heatmap(feature_names, mixed, fig_path, args.dpi, group_ranges)

    print(f"Done. Heatmap saved to: {fig_path }")
    print(f"Features: {len (feature_names )}")
    print(f"Common CN subjects across all features: {len (common_subjects )}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Functional decoding or transcriptomic analysis helper.
# Update the path variables below before running this script.

import argparse
import csv
import functools
import inspect
import os
import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


POS_COLOR = "#C96B27"
NEG_COLOR = "#2C7A7B"
NEUTRAL_COLOR = "#3B3B3B"
LIGHT_GREY = "#D9D9D9"
MID_GREY = "#8A8A8A"
BG_GREY = "#F7F7F5"


def patch_pandas_compat():
    try:
        import pandas as pd
    except Exception:
        return

    try:
        params = inspect.signature(pd.DataFrame.set_axis).parameters
        if "inplace" not in params:
            original_set_axis = pd.DataFrame.set_axis

            @functools.wraps(original_set_axis)
            def set_axis_compat(self, labels, axis=0, inplace=None, **kwargs):
                copy = kwargs.pop("copy", None)
                if copy is None and inplace is False:
                    copy = True
                return original_set_axis(self, labels, axis=axis, copy=copy)

            pd.DataFrame.set_axis = set_axis_compat
    except Exception:
        pass

    try:
        if not hasattr(pd.DataFrame, "append"):
            original_concat = pd.concat

            def append_compat(
                self, other, ignore_index=False, verify_integrity=False, sort=False
            ):
                def normalize(obj):
                    if isinstance(obj, pd.Series):
                        return obj.to_frame().T
                    if isinstance(obj, dict):
                        return pd.DataFrame([obj])
                    return obj

                if isinstance(other, (list, tuple)):
                    others = [normalize(obj) for obj in other]
                    return original_concat(
                        [self, *others],
                        ignore_index=ignore_index,
                        verify_integrity=verify_integrity,
                        sort=sort,
                    )

                return original_concat(
                    [self, normalize(other)],
                    ignore_index=ignore_index,
                    verify_integrity=verify_integrity,
                    sort=sort,
                )

            pd.DataFrame.append = append_compat
    except Exception:
        pass


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create top-journal-style supplementary figures for the DK68 Pos PLS gene analysis."
    )
    parser.add_argument(
        "--pls-csv",
        default="data/data_use/brainstat/Gene/T_value/DK68_Gene_PLS.csv",
        help="Path to the Pos PLS result CSV exported by Gene_analysis_PLS.py",
    )
    parser.add_argument(
        "--t-values-path",
        default="data/data_use/brainstat/DK68/T_value.npy",
        help="Path to the DK68 Pos phenotype map used for the original PLS analysis",
    )
    parser.add_argument(
        "--output-dir",
        default="data/visulization/brainstat/Gene/",
        help="Directory where figures will be written",
    )
    parser.add_argument(
        "--n-components-explained",
        type=int,
        default=10,
        help="Maximum number of PLS components to evaluate for the scree plot",
    )
    parser.add_argument(
        "--top-tail",
        type=int,
        default=14,
        help="Number of genes to show at each tail in the stability-driver panel",
    )
    parser.add_argument(
        "--enrichment-table",
        default=None,
        help="Optional CSV/TSV table for positive-gene enrichment results",
    )
    parser.add_argument(
        "--enrichment-topn",
        type=int,
        default=15,
        help="Number of enrichment terms to display when an enrichment table is provided",
    )
    parser.add_argument(
        "--skip-model-reconstruction",
        action="store_true",
        help="Skip PLS model reconstruction panels even if abagen is available",
    )
    return parser.parse_args(argv)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#202020",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 400,
            "figure.dpi": 160,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.8,
            "legend.frameon": False,
        }
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_table(path, delimiter=","):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(value, default=np.nan):
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none"}:
        return default
    try:
        return float(text)
    except Exception:
        return default


def safe_bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_pls_results(csv_path):
    rows = read_table(csv_path, delimiter=",")
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path }")

    genes = np.array([row["Gene"] for row in rows], dtype=object)
    weights = np.array(
        [safe_float(row.get("PLS1_Weight")) for row in rows], dtype=float
    )
    zscores = np.array(
        [safe_float(row.get("Bootstrap_Z")) for row in rows], dtype=float
    )
    pvals = np.array([safe_float(row.get("P_Value")) for row in rows], dtype=float)
    qvals = np.array([safe_float(row.get("P_Value_FDR")) for row in rows], dtype=float)
    sig = np.array(
        [safe_bool(row.get("Significant_FDR_0.05")) for row in rows], dtype=bool
    )

    meta = {
        "PLS1_r": safe_float(rows[0].get("PLS1_r")),
        "PLS1_R2": safe_float(rows[0].get("PLS1_R2")),
        "N_Regions": int(safe_float(rows[0].get("N_Regions"), default=0)),
        "N_Genes": int(safe_float(rows[0].get("N_Genes"), default=len(rows))),
        "N_Boot": int(safe_float(rows[0].get("N_Boot"), default=0)),
        "Random_Seed": int(safe_float(rows[0].get("Random_Seed"), default=0)),
        "FDR_Alpha": safe_float(rows[0].get("FDR_Alpha"), default=0.05),
    }

    return {
        "genes": genes,
        "weights": weights,
        "zscores": zscores,
        "pvals": pvals,
        "qvals": qvals,
        "sig": sig,
        "meta": meta,
    }


def save_figure(fig, output_base):
    png_path = f"{output_base }.png"
    pdf_path = f"{output_base }.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {png_path }")
    print(f"[+] Saved {pdf_path }")


def select_tail_indices(values, n_each):
    order = np.argsort(values)
    neg = order[:n_each]
    pos = order[-n_each:]
    return np.unique(np.concatenate([neg, pos]))


def select_top_abs_indices(values, n_total):
    order = np.argsort(np.abs(values))
    return np.unique(order[-n_total:])


def plot_fdr_sensitivity(results, output_base):
    weights = results["weights"]
    qvals = results["qvals"]
    valid = np.isfinite(qvals)
    if not valid.any():
        return

    thresholds = np.linspace(0.001, 0.2, 200)
    total_counts = np.array(
        [np.sum(valid & (qvals <= thr)) for thr in thresholds], dtype=float
    )
    pos_counts = np.array(
        [np.sum(valid & (qvals <= thr) & (weights > 0)) for thr in thresholds],
        dtype=float,
    )
    neg_counts = np.array(
        [np.sum(valid & (qvals <= thr) & (weights < 0)) for thr in thresholds],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(
        thresholds,
        total_counts,
        color=NEUTRAL_COLOR,
        lw=2.2,
        label="All significant genes",
    )
    ax.plot(
        thresholds, pos_counts, color=POS_COLOR, lw=2.0, label="Positive-weight genes"
    )
    ax.plot(
        thresholds, neg_counts, color=NEG_COLOR, lw=2.0, label="Negative-weight genes"
    )
    ax.axvline(0.05, color=MID_GREY, lw=1.0, ls="--")
    ax.text(
        0.05,
        ax.get_ylim()[1] * 0.97,
        "q = 0.05",
        ha="left",
        va="top",
        color=MID_GREY,
        fontsize=11,
    )
    ax.set_xlabel("FDR threshold (q)", fontsize=15)
    ax.set_ylabel("Number of genes retained", fontsize=15)
    # ax.set_title("Pos PLS Gene Yield Across FDR Thresholds")
    ax.legend(loc="upper left", fontsize=12)
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.set_facecolor(BG_GREY)
    save_figure(fig, output_base)


def plot_stability_tail_panels(results, output_base_c, output_base_d, top_tail=14):
    genes = results["genes"]
    weights = results["weights"]
    zscores = results["zscores"]
    sig = results["sig"]

    # Plot C: Weight-Stability Landscape
    fig_c, ax_c = plt.subplots(figsize=(7.4, 6.6))

    pos_mask = weights >= 0
    neg_mask = weights < 0
    pos_sig_mask = sig & pos_mask
    neg_sig_mask = sig & neg_mask
    nonsig_mask = ~sig
    point_size = 24
    pos_nonsig_color = "#E3B07F"
    neg_nonsig_color = "#72B1B2"

    ax_c.scatter(
        weights[nonsig_mask & neg_mask],
        zscores[nonsig_mask & neg_mask],
        s=point_size,
        color=neg_nonsig_color,
        alpha=0.72,
        linewidth=0,
        zorder=1,
    )
    ax_c.scatter(
        weights[nonsig_mask & pos_mask],
        zscores[nonsig_mask & pos_mask],
        s=point_size,
        color=pos_nonsig_color,
        alpha=0.72,
        linewidth=0,
        zorder=1,
    )
    ax_c.scatter(
        weights[neg_sig_mask],
        zscores[neg_sig_mask],
        s=point_size,
        color=NEG_COLOR,
        alpha=0.95,
        zorder=3,
    )
    ax_c.scatter(
        weights[pos_sig_mask],
        zscores[pos_sig_mask],
        s=point_size,
        color=POS_COLOR,
        alpha=0.95,
        zorder=3,
    )
    ax_c.axhline(0, color=MID_GREY, lw=1.0, ls="--")
    ax_c.axvline(0, color=MID_GREY, lw=1.0, ls="--")
    # ax_c.set_title("C  Weight-Stability Landscape")
    ax_c.set_xlabel("PLS1 weight", fontsize=22)
    ax_c.set_ylabel("Bootstrap Z", fontsize=22)
    ax_c.text(
        0.02,
        0.98,
        (
            f"Positive-tail genes: {int (np .sum (pos_sig_mask )):,}\n"
            f"Negative-tail genes: {int (np .sum (neg_sig_mask )):,}"
        ),
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        color=NEUTRAL_COLOR,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.78),
    )
    # Set x-axis ticks for plot C with specific values
    xticks = [-0.02, -0.01, 0, 0.01, 0.02]
    ax_c.set_xticks(xticks)
    xticklabels = ["-0.02", "-0.01", "0", "0.01", "0.02"]
    ax_c.set_xticklabels(xticklabels, fontsize=16)
    ax_c.set_facecolor(BG_GREY)
    # Increase ytick label font size for plot C
    ax_c.tick_params(axis="y", labelsize=16)
    save_figure(fig_c, output_base_c)

    # Plot D: Tail Genes Driving the Pos PLS Signature
    fig_d, ax_d = plt.subplots(figsize=(7.4, 6.6))

    pos_idx = np.argsort(zscores)[-top_tail:]
    neg_idx = np.argsort(zscores)[:top_tail]
    show_idx = np.concatenate([neg_idx, pos_idx])
    show_z = zscores[show_idx]
    show_genes = genes[show_idx]
    order_show = np.argsort(show_z)
    ordered_z = show_z[order_show]
    ordered_genes = show_genes[order_show]
    ordered_colors = np.where(ordered_z >= 0, POS_COLOR, NEG_COLOR)
    y_pos = np.arange(ordered_z.size)

    ax_d.axvline(0, color=MID_GREY, lw=1.0, ls="--")
    ax_d.hlines(y_pos, 0, ordered_z, color=ordered_colors, lw=1.8, alpha=0.95)
    ax_d.scatter(
        ordered_z,
        y_pos,
        s=42,
        color=ordered_colors,
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(ordered_genes, fontsize=11)
    ax_d.set_xlabel("Bootstrap Z", fontsize=15)
    ax_d.set_ylabel("")
    # ax_d.set_title("D  Tail Genes Driving the Pos PLS Signature")
    ax_d.set_facecolor(BG_GREY)
    save_figure(fig_d, output_base_d)


def infer_delimiter(path):
    return "\t" if str(path).lower().endswith((".tsv", ".tab", ".txt")) else ","


def pick_first_existing(keys, candidates):
    lookup = {str(key).strip().lower(): key for key in keys}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def parse_ratio_or_float(value):
    text = str(value).strip()
    if not text:
        return np.nan
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            num = safe_float(parts[0])
            den = safe_float(parts[1])
            if np.isfinite(num) and np.isfinite(den) and den != 0:
                return num / den
    if "%" in text:
        number = safe_float(text.replace("%", ""))
        return number / 100.0 if np.isfinite(number) else np.nan
    return safe_float(text)


def load_enrichment_table(path):
    rows = read_table(path, delimiter=infer_delimiter(path))
    if not rows:
        raise ValueError(f"Empty enrichment table: {path }")

    keys = rows[0].keys()
    term_key = pick_first_existing(
        keys,
        [
            "Description",
            "Term",
            "Pathway",
            "name",
            "Name",
            "Term name",
            "GO_term",
            "ID",
        ],
    )
    q_key = pick_first_existing(
        keys,
        [
            "p.adjust",
            "Adjusted P-value",
            "adj_p_value",
            "FDR",
            "qvalue",
            "padj",
            "q_value",
        ],
    )
    count_key = pick_first_existing(
        keys,
        [
            "Count",
            "Gene count",
            "gene_count",
            "Intersection_size",
            "Overlap_count",
            "Hits",
        ],
    )
    ratio_key = pick_first_existing(
        keys,
        ["GeneRatio", "RichFactor", "FoldEnrichment", "EnrichmentRatio", "Ratio"],
    )

    if term_key is None or q_key is None:
        raise ValueError(
            "Could not infer enrichment columns. Need a term column and an adjusted-p/FDR column."
        )

    clean_rows = []
    for row in rows:
        qval = safe_float(row.get(q_key))
        if not np.isfinite(qval):
            continue
        term = str(row.get(term_key, "")).strip()
        if not term:
            continue
        count = parse_ratio_or_float(row.get(count_key, "")) if count_key else np.nan
        ratio = parse_ratio_or_float(row.get(ratio_key, "")) if ratio_key else np.nan
        clean_rows.append(
            {
                "term": term,
                "qval": qval,
                "count": count,
                "ratio": ratio,
            }
        )

    if not clean_rows:
        raise ValueError("No valid enrichment rows were found after parsing.")
    return clean_rows


def plot_enrichment_dotplot(enrichment_rows, output_base, topn=15):
    rows = sorted(enrichment_rows, key=lambda item: (item["qval"], item["term"]))[:topn]
    rows = list(reversed(rows))

    terms = [row["term"] for row in rows]
    qvals = np.array([row["qval"] for row in rows], dtype=float)
    counts = np.array([row["count"] for row in rows], dtype=float)
    ratios = np.array([row["ratio"] for row in rows], dtype=float)
    y = np.arange(len(rows))

    x_values = ratios.copy()
    x_label = "Gene ratio"
    if not np.isfinite(x_values).any():
        x_values = counts.copy()
        x_label = "Gene count"
    if not np.isfinite(x_values).any():
        x_values = np.arange(1, len(rows) + 1, dtype=float)
        x_label = "Rank"

    size_values = counts.copy()
    if not np.isfinite(size_values).any():
        size_values = np.ones_like(x_values)
    size_values = np.where(
        np.isfinite(size_values),
        size_values,
        (
            np.nanmin(size_values[np.isfinite(size_values)])
            if np.isfinite(size_values).any()
            else 1.0
        ),
    )
    size_scale = 60 + 240 * (size_values - np.nanmin(size_values)) / (
        np.nanmax(size_values) - np.nanmin(size_values) + 1e-12
    )

    color_values = -np.log10(np.clip(qvals, 1e-300, None))

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    scatter = ax.scatter(
        x_values,
        y,
        s=size_scale,
        c=color_values,
        cmap="YlOrBr",
        edgecolor="#333333",
        linewidth=0.45,
        alpha=0.92,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=9.5)
    ax.set_xlabel(x_label)
    ax.set_title("Positive-Weight Significant Genes: Enrichment Summary")
    ax.set_facecolor(BG_GREY)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("-log10(FDR)")

    ax.grid(axis="x", color="white", linewidth=1.1)
    ax.set_axisbelow(True)

    save_figure(fig, output_base)


def _safe_zscore_matrix(X):
    X = X.astype(float, copy=True)
    col_mean = np.nanmean(X, axis=0)
    valid_gene_mask = ~np.isnan(col_mean)
    X = X[:, valid_gene_mask]
    col_mean = col_mean[valid_gene_mask]

    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    Xz = (X - X_mean) / X_std
    return Xz, valid_gene_mask


def _standardize_vector(y):
    y = np.asarray(y, dtype=float).ravel()
    y_std = y.std(ddof=0)
    if y_std == 0:
        raise ValueError("Input phenotype variance is zero.")
    return (y - y.mean()) / y_std


def _fit_pls_sklearn(Xz, yz, n_components):
    from sklearn.cross_decomposition import PLSRegression

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xz, yz)
    y_pred = pls.predict(Xz).ravel()
    T = np.asarray(pls.x_scores_, dtype=float)
    P = np.asarray(pls.x_loadings_, dtype=float)
    W = np.asarray(pls.x_weights_, dtype=float)
    return {
        "x_scores": T,
        "x_loadings": P,
        "x_weights": W,
        "coef": np.asarray(pls.coef_).reshape(-1),
        "y_pred": y_pred,
    }


def _fit_pls_numpy(Xz, yz, n_components):
    X_res = np.asarray(Xz, dtype=float).copy()
    y_res = np.asarray(yz, dtype=float).reshape(-1, 1).copy()
    n_samples, n_features = X_res.shape
    max_comp = int(min(n_components, n_samples - 1, n_features))
    if max_comp < 1:
        raise ValueError("Not enough data to fit PLS.")

    weights = []
    scores = []
    loadings = []
    y_loadings = []

    for _ in range(max_comp):
        w = X_res.T @ y_res[:, 0]
        norm = float(np.linalg.norm(w))
        if not np.isfinite(norm) or norm == 0.0:
            break
        w = w / norm
        t = X_res @ w
        tt = float(np.dot(t, t))
        if not np.isfinite(tt) or tt == 0.0:
            break
        p = (X_res.T @ t) / tt
        q = float((y_res[:, 0] @ t) / tt)

        X_res = X_res - np.outer(t, p)
        y_res = y_res - np.outer(t, np.array([q]))

        weights.append(w)
        scores.append(t)
        loadings.append(p)
        y_loadings.append(q)

    if not weights:
        raise ValueError("Failed to derive any valid PLS components.")

    W = np.column_stack(weights)
    T = np.column_stack(scores)
    P = np.column_stack(loadings)
    Q = np.asarray(y_loadings, dtype=float).reshape(-1, 1)
    coef = W @ np.linalg.pinv(P.T @ W) @ Q
    y_pred = Xz @ coef

    return {
        "x_scores": T,
        "x_loadings": P,
        "x_weights": W,
        "coef": coef.reshape(-1),
        "y_pred": y_pred.ravel(),
    }


def fit_pls(Xz, yz, n_components):
    try:
        return _fit_pls_sklearn(Xz, yz, n_components)
    except Exception:
        return _fit_pls_numpy(Xz, yz, n_components)


def compute_fit_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return r, r2


def compute_r2x(Xz, model):
    X_hat = model["x_scores"] @ model["x_loadings"].T
    ssx = float(np.sum(Xz**2))
    if ssx == 0:
        return float("nan")
    return float(1.0 - (np.sum((Xz - X_hat) ** 2) / ssx))


def load_atlas_rows(info_path):
    rows = read_table(info_path, delimiter=",")
    atlas_rows = []
    for row in rows:
        out = dict(row)
        out["id"] = int(row["id"])
        atlas_rows.append(out)
    return atlas_rows


def reconstruct_pos_pls_context(t_values_path, n_components_explained=10):
    patch_pandas_compat()
    try:
        import abagen
    except Exception as exc:
        raise RuntimeError(
            "abagen is required for model-level reconstruction panels."
        ) from exc

    t_values_full = np.load(t_values_path).flatten()
    if t_values_full.size != 68:
        raise ValueError(f"Expected 68 DK68 values, got {t_values_full .size }")

    dk_atlas = abagen.fetch_desikan_killiany()
    expression_df = abagen.get_expression_data(dk_atlas["image"], dk_atlas["info"])
    atlas_rows = load_atlas_rows(dk_atlas["info"])
    cortex_ids = [
        row["id"]
        for row in atlas_rows
        if str(row.get("structure", "")).strip().lower() == "cortex"
    ]

    kept_ids = None
    t_values = t_values_full.copy()
    if cortex_ids:
        present_mask = [region_id in expression_df.index for region_id in cortex_ids]
        kept_ids = [region_id for region_id, ok in zip(cortex_ids, present_mask) if ok]
        expression_df = expression_df.loc[kept_ids]
        if t_values_full.size == len(cortex_ids):
            t_values = t_values_full[np.asarray(present_mask, dtype=bool)]

    if expression_df.shape[0] != t_values.size:
        raise ValueError(
            f"Expression rows ({expression_df .shape [0 ]}) do not align with phenotype values ({t_values .size })."
        )

    X = expression_df.to_numpy(dtype=float, copy=True)
    Xz, valid_gene_mask = _safe_zscore_matrix(X)
    yz = _standardize_vector(t_values)

    atlas_lookup = {row["id"]: row for row in atlas_rows}
    if kept_ids is None:
        kept_ids = [int(x) for x in expression_df.index.astype(int).tolist()]

    region_labels = []
    hemisphere = []
    for rid in kept_ids:
        row = atlas_lookup.get(int(rid), {})
        hemi = str(row.get("hemisphere", "")).strip()
        label = str(row.get("label", rid)).strip()
        region_labels.append(f"{hemi [:1 ].upper ()}-{label }" if hemi else label)
        hemisphere.append(hemi)

    max_comp = int(min(n_components_explained, Xz.shape[0] - 1, Xz.shape[1]))
    scree_rows = []
    prev_r2x = 0.0
    prev_r2y = 0.0
    for k in range(1, max_comp + 1):
        model_k = fit_pls(Xz, yz, k)
        _, r2y = compute_fit_metrics(yz, model_k["y_pred"])
        r2x = compute_r2x(Xz, model_k)
        r2x = max(prev_r2x, r2x) if np.isfinite(r2x) else prev_r2x
        r2y = max(prev_r2y, r2y) if np.isfinite(r2y) else prev_r2y
        scree_rows.append(
            {
                "component": k,
                "R2X_cum": r2x,
                "R2X_delta": r2x - prev_r2x,
                "R2Y_cum": r2y,
                "R2Y_delta": r2y - prev_r2y,
            }
        )
        prev_r2x = r2x
        prev_r2y = r2y

    pls1 = fit_pls(Xz, yz, 1)
    r, r2 = compute_fit_metrics(yz, pls1["y_pred"])

    return {
        "y_true": yz,
        "y_pred": pls1["y_pred"],
        "x_scores": pls1["x_scores"][:, 0],
        "region_labels": np.array(region_labels, dtype=object),
        "hemisphere": np.array(hemisphere, dtype=object),
        "scree_rows": scree_rows,
        "r": r,
        "r2": r2,
    }


def plot_model_summary(context, output_base_a, output_base_b):
    y_true = np.asarray(context["y_true"], dtype=float)
    y_pred = np.asarray(context["y_pred"], dtype=float)
    scree_rows = context["scree_rows"]

    scree_components = np.array([row["component"] for row in scree_rows], dtype=int)
    r2y_delta = np.array([row["R2Y_delta"] for row in scree_rows], dtype=float)
    r2y_cum = np.array([row["R2Y_cum"] for row in scree_rows], dtype=float)

    # Plot A: PLS1 Fit Scatter
    fig_a, ax_a = plt.subplots(figsize=(7.4, 5.8))
    ax_a.scatter(
        y_true,
        y_pred,
        s=48,
        color=POS_COLOR,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
    )
    coef = np.polyfit(y_true, y_pred, 1)
    xline = np.linspace(y_true.min(), y_true.max(), 200)
    ax_a.plot(xline, coef[0] * xline + coef[1], color=NEUTRAL_COLOR, lw=2.2)
    pad_x = (y_true.max() - y_true.min()) * 0.05 + 1e-6
    pad_y = (y_pred.max() - y_pred.min()) * 0.05 + 1e-6
    ax_a.set_xlim(y_true.min() - pad_x, y_true.max() + pad_x)
    ax_a.set_ylim(y_pred.min() - pad_y, y_pred.max() + pad_y)
    ax_a.set_xlabel("Observed regional phenotype (z)", fontsize=22)
    ax_a.set_ylabel("PLS1 fitted phenotype (z)", fontsize=22)
    ax_a.tick_params(axis="both", labelsize=16)
    # ax_a.set_title("A  PLS1 Fit Scatter")
    ax_a.text(
        0.03,
        0.97,
        f"r = {context ['r']:.3f}\nR² = {context ['r2']:.3f}\nn = {y_true .size }",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        color=NEUTRAL_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
    )
    ax_a.set_facecolor(BG_GREY)
    save_figure(fig_a, output_base_a)

    # Plot B: PLS Component Contribution to Phenotype Variance
    fig_b, ax_b = plt.subplots(figsize=(7.4, 5.8))
    ax_b.bar(
        scree_components,
        r2y_delta,
        color=POS_COLOR,
        alpha=0.86,
        width=0.72,
        label="_nolegend_",
    )
    ax_b.plot(
        scree_components,
        r2y_cum,
        marker="o",
        color=NEUTRAL_COLOR,
        lw=2.0,
        ms=5.5,
        label="Cumulative R²Y",
    )
    ax_b.set_xlabel("PLS component", fontsize=22)
    ax_b.set_ylabel("Explained phenotype variance", fontsize=22)
    ax_b.tick_params(axis="both", labelsize=16)
    # ax_b.set_title("B  PLS Component Contribution to Phenotype Variance")
    ax_b.set_xticks(scree_components)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(
        loc="upper left",
        fontsize=16,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    ax_b.set_facecolor(BG_GREY)
    save_figure(fig_b, output_base_b)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_plot_style()

    ensure_dir(args.output_dir)
    results = load_pls_results(args.pls_csv)

    prefix = Path(args.pls_csv).stem

    # Save four subplots individually
    stability_tail_output_base_c = os.path.join(args.output_dir, f"{prefix }_figure_C")
    stability_tail_output_base_d = os.path.join(args.output_dir, f"{prefix }_figure_D")
    plot_stability_tail_panels(
        results,
        stability_tail_output_base_c,
        stability_tail_output_base_d,
        top_tail=args.top_tail,
    )

    if args.skip_model_reconstruction:
        print("[!] Model reconstruction was skipped by flag.")
        return

    try:
        context = reconstruct_pos_pls_context(
            t_values_path=args.t_values_path,
            n_components_explained=args.n_components_explained,
        )
    except Exception as exc:
        warnings.warn(f"Skipped model-level figures: {exc }")
        return

    model_output_base_a = os.path.join(args.output_dir, f"{prefix }_figure_A")
    model_output_base_b = os.path.join(args.output_dir, f"{prefix }_figure_B")
    plot_model_summary(context, model_output_base_a, model_output_base_b)


if __name__ == "__main__":
    main()

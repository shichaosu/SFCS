# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os

# ==========================================
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, Normalize
from pathlib import Path
from scipy.interpolate import interpn
import scipy.ndimage
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

# ==========================================
# ==========================================
ENABLE_ZSCORE = False
USE_LOG_COLOR = True
PLOT_LINE_MODE = "fit"

DENSITY_BINS = 800
DENSITY_SMOOTH = 2.0
HIST_BINS = 300

LOG_RANGE_NEG = 50.0
LOG_RANGE_POS = 50.0

TARGET_SUBJECT = "022_S_6716_2021"
base_path = Path("data/data_use/aparc")
output_dir = Path("data/visulization/figure_result/20/scatter_hist/")
FONT_NO_DATA = 24
FONT_CBAR_LABEL = 22
FONT_CBAR_HL = 18
FONT_R_TEXT = 36
FONT_N_TEXT = 28
FONT_SUBTITLE = 40
FONT_AXIS_LABEL = 32
FONT_TICK = 18

recon_neg_dir = base_path / "GLM_Sparsify_Neg/20reconstruct"
recon_pos_dir = base_path / "GLM_Sparsify_Pos/20reconstruct"
real_neg_dir = base_path / "HOI_GSR_Z_Sparsification_Neg/20"
real_pos_dir = base_path / "HOI_GSR_Z_Sparsification_Pos/20"


# ==========================================
# ==========================================
def density_scatter_ultra(x, y, bins=500, smoothing=2.0):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 10:
        return np.zeros_like(x), valid

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = scipy.ndimage.gaussian_filter(data, sigma=smoothing)

    x_idx = 0.5 * (x_e[1:] + x_e[:-1])
    y_idx = 0.5 * (y_e[1:] + y_e[:-1])
    z_values = interpn(
        (x_idx, y_idx),
        z,
        np.vstack([x, y]).T,
        method="cubic",
        bounds_error=False,
        fill_value=0,
    )
    return z_values, valid

    # ==========================================
    # ==========================================


def plot_combined(ax_scatter, ax_hist, x_data, y_data, title_text, dynamic_range):
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_f, y_f = x_data[mask], y_data[mask]

    if len(x_f) < 10:
        ax_scatter.text(
            0.5, 0.5, "No Data", ha="center", va="center", fontsize=FONT_NO_DATA
        )
        ax_hist.axis("off")
        return

    r_val = np.corrcoef(x_f, y_f)[0, 1]
    x_min, x_max = x_f.min(), x_f.max()
    x_pad = (x_max - x_min) * 0.02
    x_lim_min, x_lim_max = x_min - x_pad, x_max + x_pad

    y_min, y_max = y_f.min(), y_f.max()
    y_pad = (y_max - y_min) * 0.02
    y_lim_min, y_lim_max = y_min - y_pad, y_max + y_pad

    line_min = min(x_lim_min, y_lim_min)
    line_max = max(x_lim_max, y_lim_max)

    # --- Scatter ---
    z_density, valid_idx = density_scatter_ultra(
        x_f, y_f, bins=DENSITY_BINS, smoothing=DENSITY_SMOOTH
    )
    x_plot = x_f[valid_idx]
    y_plot = y_f[valid_idx]

    sort_idx = z_density.argsort()
    x_sorted = x_plot[sort_idx]
    y_sorted = y_plot[sort_idx]
    z_sorted = z_density[sort_idx]

    vmax_val = np.percentile(z_sorted, 99.9)
    if USE_LOG_COLOR:
        vmin_val = vmax_val / dynamic_range
        if vmin_val <= 0:
            vmin_val = 1e-8
        norm_setting = LogNorm(vmin=vmin_val, vmax=vmax_val)
    else:
        norm_setting = Normalize(vmin=0, vmax=vmax_val)

    scatter = ax_scatter.scatter(
        x_sorted,
        y_sorted,
        c=z_sorted,
        s=5,
        alpha=1.0,
        cmap="jet",
        norm=norm_setting,
        edgecolor="none",
    )

    # [x, y, width, height]
    cax_scatter = ax_scatter.inset_axes([1.05, 0.25, 0.04, 0.5])
    cbar = plt.colorbar(scatter, cax=cax_scatter)
    cbar.ax.yaxis.set_major_locator(ticker.NullLocator())
    cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())
    cbar.ax.tick_params(which="both", length=0, width=0)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel("density", rotation=90, fontsize=FONT_CBAR_LABEL, labelpad=5)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.text(
        0.5,
        1.05,
        "H",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONT_CBAR_HL,
    )
    cbar.ax.text(
        0.5,
        -0.05,
        "L",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_CBAR_HL,
    )
    # =========================================

    if PLOT_LINE_MODE == "fit":
        slope, intercept = np.polyfit(x_plot, y_plot, 1)
        ax_scatter.plot(
            [x_lim_min, x_lim_max],
            [slope * x_lim_min + intercept, slope * x_lim_max + intercept],
            "k-",
            alpha=0.8,
        )
    else:
        ax_scatter.plot([line_min, line_max], [line_min, line_max], "k--", alpha=0.3)

    ax_scatter.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax_scatter.axhline(0, color="gray", lw=0.5, alpha=0.5)
    n_points = len(x_sorted)
    r_text = TextArea(
        f"r={r_val :.3f}",
        textprops=dict(fontsize=FONT_R_TEXT, fontweight="bold", color="black"),
    )
    n_text = TextArea(
        f" (n={n_points })",
        textprops=dict(fontsize=FONT_N_TEXT, fontweight="normal", color="black"),
    )
    packed_text = HPacker(children=[r_text, n_text], align="center", pad=0, sep=2)
    info_box = AnchoredOffsetbox(
        loc="upper left",
        child=packed_text,
        frameon=True,
        bbox_to_anchor=(0.05, 0.95),
        bbox_transform=ax_scatter.transAxes,
        borderpad=0.4,
        pad=0.2,
    )
    info_box.patch.set_boxstyle("round,pad=0.3")
    info_box.patch.set_facecolor("white")
    info_box.patch.set_alpha(0.8)
    info_box.patch.set_edgecolor("none")
    ax_scatter.add_artist(info_box)
    ax_scatter.set_title(title_text, fontsize=FONT_SUBTITLE, fontweight="bold", pad=15)
    ax_scatter.set_xlabel("Real Value", fontsize=FONT_AXIS_LABEL)
    ax_scatter.set_ylabel("Recon Value", fontsize=FONT_AXIS_LABEL)
    ax_scatter.tick_params(axis="both", which="major", labelsize=FONT_TICK)

    # --- Hist ---
    ax_hist.hist(
        x_f, bins=HIST_BINS, color="#1f77b4", alpha=0.7, density=True, edgecolor="none"
    )
    ax_hist.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax_hist.invert_yaxis()
    ax_hist.axis("off")
    ax_scatter.set_xlim(x_lim_min, x_lim_max)
    ax_hist.set_xlim(x_lim_min, x_lim_max)
    ax_scatter.set_ylim(y_lim_min, y_lim_max)
    ax_scatter.set_box_aspect(1)

    # ==========================================
    # ==========================================


def process_single_subject(file_path):
    filename = file_path.name
    save_name = filename.replace(".npy", ".png")
    save_path = output_dir / save_name

    path_real_neg = real_neg_dir / filename
    path_real_pos = real_pos_dir / filename
    path_recon_neg = recon_neg_dir / filename
    path_recon_pos = recon_pos_dir / filename

    if not (
        path_real_neg.exists()
        and path_real_pos.exists()
        and path_recon_neg.exists()
        and path_recon_pos.exists()
    ):
        return f"Skipped (Missing Files): {filename }"

    try:
        t_neg = np.load(path_real_neg)
        t_pos = np.load(path_real_pos)
        r_neg = np.load(path_recon_neg)
        r_pos = np.load(path_recon_pos)

        i_idx, j_idx, k_idx = np.indices(t_neg.shape)
        tri_mask = (i_idx <= j_idx) & (j_idx <= k_idx)

        real_neg = t_neg[tri_mask]
        recon_neg = r_neg[tri_mask]
        real_pos = t_pos[tri_mask]
        recon_pos = r_pos[tri_mask]

        del t_neg, t_pos, r_neg, r_pos, i_idx, j_idx, k_idx, tri_mask

        mask_n = real_neg != 0
        real_neg, recon_neg = real_neg[mask_n], recon_neg[mask_n]

        mask_p = real_pos != 0
        real_pos, recon_pos = real_pos[mask_p], recon_pos[mask_p]

        if len(real_neg) == 0 or len(real_pos) == 0:
            return f"Skipped (No Data): {filename }"

        if ENABLE_ZSCORE:
            real_neg = (real_neg - real_neg.mean()) / (real_neg.std() + 1e-8)
            recon_neg = (recon_neg - recon_neg.mean()) / (recon_neg.std() + 1e-8)
            real_pos = (real_pos - real_pos.mean()) / (real_pos.std() + 1e-8)
            recon_pos = (recon_pos - recon_pos.mean()) / (recon_pos.std() + 1e-8)

        fig = plt.figure(figsize=(20, 11))
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], wspace=0.4, hspace=0.0)

        # Negative
        ax_ns = fig.add_subplot(gs[0, 0])
        ax_nh = fig.add_subplot(gs[1, 0], sharex=ax_ns)
        plot_combined(ax_ns, ax_nh, real_neg, recon_neg, "Synergy", LOG_RANGE_NEG)

        # Positive
        ax_ps = fig.add_subplot(gs[0, 1])
        ax_ph = fig.add_subplot(gs[1, 1], sharex=ax_ps)
        plot_combined(ax_ps, ax_ph, real_pos, recon_pos, "Redundancy", LOG_RANGE_POS)

        plt.subplots_adjust(bottom=0.08, left=0.08, right=0.92)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return None

    except Exception as e:
        plt.close("all")
        return f"Error ({filename }): {str (e )}"

        # ==========================================
        # ==========================================


if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)

    target_file = real_neg_dir / f"{TARGET_SUBJECT }.npy"
    print(f"Processing message.")
    print(f"Processing message.")

    if not target_file.exists():
        print(f"Processing message.")
        exit()

    result = process_single_subject(target_file)

    print("Processing message.")
    if result is None:
        print(f"Processing message.")
    else:
        print(result)

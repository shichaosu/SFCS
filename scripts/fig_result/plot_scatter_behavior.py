# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

path_pos = "data/data_use/table/result_predict_Pos/BrainGNN_addLoss_trans/Predict_EcogSPOrgan.csv"
path_neg = (
    "data/data_use/table/result_predict_Neg/BrainGNN_addLoss/Predict_EcogSPOrgan.csv"
)
df_pos = pd.read_csv(path_pos)
df_neg = pd.read_csv(path_neg)
df_pos.columns = df_pos.columns.str.strip().str.lower()
df_neg.columns = df_neg.columns.str.strip().str.lower()
r_pos, p_pos = pearsonr(df_pos["true"], df_pos["pred"])
r_neg, p_neg = pearsonr(df_neg["true"], df_neg["pred"])


def format_p(p_val):
    return f"p < 0.001" if p_val < 0.001 else f"p = {p_val :.3f}"


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
min_true = min(df_pos["true"].min(), df_neg["true"].min())
max_true = max(df_pos["true"].max(), df_neg["true"].max())
min_pred = min(df_pos["pred"].min(), df_neg["pred"].min())
max_pred = max(df_pos["pred"].max(), df_neg["pred"].max())

margin_x = (max_true - min_true) * 0.05
margin_y = (max_pred - min_pred) * 0.05
xlims = (min_true - margin_x, max_true + margin_x)
ylims = (min_pred - margin_y, max_pred + margin_y)
sns.regplot(
    x="true",
    y="pred",
    data=df_pos,
    ax=axes[0],
    color="#4472C4",
    scatter_kws={"alpha": 0.5, "s": 60, "edgecolor": "white", "linewidths": 0.5},
    line_kws={"linewidth": 3},
)
axes[0].text(
    0.05,
    0.95,
    f"Redundancy\nr = {r_pos :.3f}\n{format_p (p_pos )}",
    transform=axes[0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.8),
)

axes[0].set_title("Redundancy Constraint", fontsize=18, fontweight="bold", pad=15)
axes[0].set_xlabel("Actual EcogSPOrgan Score", fontsize=16, fontweight="bold")
axes[0].set_ylabel("Predicted Score", fontsize=16, fontweight="bold")
sns.regplot(
    x="true",
    y="pred",
    data=df_neg,
    ax=axes[1],
    color="#ED7D31",
    scatter_kws={"alpha": 0.5, "s": 60, "edgecolor": "white", "linewidths": 0.5},
    line_kws={"linewidth": 3},
)

axes[1].text(
    0.05,
    0.95,
    f"Synergy\nr = {r_neg :.3f}\n{format_p (p_neg )}",
    transform=axes[1].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.8),
)

axes[1].set_title("Synergy Constraint", fontsize=18, fontweight="bold", pad=15)
axes[1].set_xlabel("Actual EcogSPOrgan Score", fontsize=16, fontweight="bold")
axes[1].set_ylabel("", fontsize=16)
for ax in axes:
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

plt.tight_layout()
save_path = "data/visulization/figure_result/20/Scatter_EcogSPOrgan_Comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Processing message.")

plt.close(fig)

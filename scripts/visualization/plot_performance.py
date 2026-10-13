# Visualization helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# ==========================================
# ==========================================
PLOT_TYPE = 4
FILE_PATH = r"data/CN_performance_BrainGNN.xlsx"
OUTPUT_DIR = r"data/fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)
COLOR_POS = "#E64B35"
COLOR_NEG = "#4DBBD5"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# ==========================================
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Sheet3")
    df = df.iloc[:, :6]
    df.columns = ["Metric", "Pos_r", "Pos_p", "Neg_r", "Neg_p", "average_r"]
    df = df.sort_values(by="average_r", ascending=False).reset_index(drop=True)
    return df


def get_asterisks(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""

    # ==========================================
    # ==========================================


def plot_dumbbell(df, save_path):
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.hlines(
        y=df["Metric"],
        xmin=df[["Pos_r", "Neg_r"]].min(axis=1),
        xmax=df[["Pos_r", "Neg_r"]].max(axis=1),
        color="grey",
        alpha=0.4,
        linewidth=2,
        zorder=1,
    )
    ax.scatter(
        df["Pos_r"],
        df["Metric"],
        color=COLOR_POS,
        s=120,
        label="Redundancy",
        zorder=2,
        edgecolors="white",
        linewidth=1,
    )
    ax.scatter(
        df["Neg_r"],
        df["Metric"],
        color=COLOR_NEG,
        s=120,
        label="Synergy",
        zorder=2,
        edgecolors="white",
        linewidth=1,
    )
    for i, row in df.iterrows():
        pos_star = get_asterisks(row["Pos_p"])
        neg_star = get_asterisks(row["Neg_p"])

        if pos_star:
            ax.text(
                row["Pos_r"],
                i + 0.15,
                pos_star,
                ha="center",
                va="bottom",
                color=COLOR_POS,
                fontsize=12,
                fontweight="bold",
            )
        if neg_star:
            ax.text(
                row["Neg_r"],
                i + 0.15,
                neg_star,
                ha="center",
                va="bottom",
                color=COLOR_NEG,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xlabel("Correlation (r)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    sns.despine(top=True, right=True)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Processing message.")


def plot_bubble(df, save_path):
    df_long = pd.DataFrame(
        {
            "Metric": df["Metric"].tolist() * 2,
            "Condition": ["Redundancy"] * len(df) + ["Synergy"] * len(df),
            "r_value": df["Pos_r"].tolist() + df["Neg_r"].tolist(),
            "p_value": df["Pos_p"].tolist() + df["Neg_p"].tolist(),
        }
    )

    df_long["-log10(p)"] = -np.log10(df_long["p_value"] + 1e-10)

    fig, ax = plt.subplots(figsize=(6, 10))

    scatter = ax.scatter(
        x=df_long["Condition"],
        y=df_long["Metric"],
        c=df_long["r_value"],
        cmap="Reds",
        s=df_long["-log10(p)"] * 80,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )

    sns.despine(top=True, right=True, left=True, bottom=True)
    ax.grid(axis="both", linestyle="--", alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (r)", rotation=270, labelpad=15, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Processing message.")


def plot_butterfly(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))

    y_pos = np.arange(len(df["Metric"]))
    height = 0.6

    bars_pos = ax.barh(y_pos, df["Pos_r"], height, color=COLOR_POS, label="Redundancy")
    bars_neg = ax.barh(y_pos, -df["Neg_r"], height, color=COLOR_NEG, label="Synergy")

    for i, row in df.iterrows():
        pos_star = get_asterisks(row["Pos_p"])
        neg_star = get_asterisks(row["Neg_p"])

        if pos_star:
            ax.text(
                row["Pos_r"] + 0.005,
                i,
                pos_star,
                va="center",
                color=COLOR_POS,
                fontsize=12,
                fontweight="bold",
            )
        if neg_star:
            ax.text(
                -row["Neg_r"] - 0.005,
                i,
                neg_star,
                va="center",
                ha="right",
                color=COLOR_NEG,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Metric"])
    ax.set_xlabel("Correlation (r)", fontsize=14, fontweight="bold")

    formatter = ticker.FuncFormatter(lambda x, pos: f"{abs (x ):.2f}")
    ax.xaxis.set_major_formatter(formatter)

    ax.legend(loc="upper right", frameon=False, fontsize=12)
    sns.despine(top=True, right=True, left=True)
    ax.axvline(0, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Processing message.")


def plot_grouped_bar(df, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))

    x_pos = np.arange(len(df["Metric"]))
    bar_width = 0.38

    bars_pos = ax.bar(
        x_pos - bar_width / 2,
        df["Pos_r"],
        bar_width,
        color=COLOR_POS,
        label="Redundancy",
    )
    bars_neg = ax.bar(
        x_pos + bar_width / 2, df["Neg_r"], bar_width, color=COLOR_NEG, label="Synergy"
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["Metric"], rotation=45, fontsize=18, ha="right")
    ax.set_ylabel("Correlation (r)", fontsize=20, fontweight="bold")

    ax.legend(loc="upper right", frameon=False, fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(top=True, right=True)
    y_min, y_max = ax.get_ylim()
    if y_min < 0 < y_max:
        ax.axhline(0, color="black", linewidth=1, alpha=0.8)
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.02 if (y_max - y_min) > 0 else 0.01

    for i, row in df.iterrows():
        pos_star = get_asterisks(row["Pos_p"])
        neg_star = get_asterisks(row["Neg_p"])

        if pos_star:
            bar = bars_pos[i]
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            y = height + y_offset if height >= 0 else height - y_offset
            va = "bottom" if height >= 0 else "top"
            ax.text(
                x,
                y,
                pos_star,
                ha="center",
                va=va,
                color=COLOR_POS,
                fontsize=12,
                fontweight="bold",
            )

        if neg_star:
            bar = bars_neg[i]
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            y = height + y_offset if height >= 0 else height - y_offset
            va = "bottom" if height >= 0 else "top"
            ax.text(
                x,
                y,
                neg_star,
                ha="center",
                va=va,
                color=COLOR_NEG,
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Processing message.")

    # ==========================================
    # ==========================================


if __name__ == "__main__":
    data = load_and_preprocess_data(FILE_PATH)
    if PLOT_TYPE == 1:
        print("Processing message.")
        plot_dumbbell(data, os.path.join(OUTPUT_DIR, "dumbbell_plot.png"))
    elif PLOT_TYPE == 2:
        print("Processing message.")
        plot_bubble(data, os.path.join(OUTPUT_DIR, "bubble_plot.png"))
    elif PLOT_TYPE == 3:
        print("Processing message.")
        plot_butterfly(data, os.path.join(OUTPUT_DIR, "butterfly_plot.png"))
    elif PLOT_TYPE == 4:
        print("Processing message.")
        plot_grouped_bar(data, os.path.join(OUTPUT_DIR, "grouped_bar_plot.png"))
    else:
        print("Processing message.")

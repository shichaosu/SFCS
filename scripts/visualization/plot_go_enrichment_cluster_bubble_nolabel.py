# Visualization helper for the SFCS manuscript.
# Update the path variables below before running this script.
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_js_object(path: Path, prefix: str):
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith(prefix):
        raise ValueError(f"{path } does not start with expected prefix: {prefix }")
    payload = text[len(prefix) :]
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def parse_rgb_string(value: str) -> tuple[float, float, float]:
    match = re.fullmatch(r"rgb\((\d+),(\d+),(\d+)\)", value.replace(" ", ""))
    if not match:
        raise ValueError(f"Unsupported color format: {value }")
    red, green, blue = (int(channel) / 255.0 for channel in match.groups())
    return red, green, blue


def extract_term_prefix(symbol: str, category: str) -> str:
    category = str(category).strip()
    category_map = {
        "GO Biological Processes": "GO",
        "Reactome Gene Sets": "R-HSA",
        "WikiPathways": "WP",
        "KEGG Pathway": "KEGG",
    }
    if category in category_map:
        return category_map[category]

    if symbol:
        symbol = str(symbol).strip()
        if symbol.startswith("GO:"):
            return "GO"
        if symbol.startswith("R-HSA-"):
            return "R-HSA"
        if symbol.startswith("WP"):
            return "WP"
        if symbol.startswith("hsa"):
            return "KEGG"

    return ""


def load_cluster_bubble_data(network_js: Path, style_js: Path) -> pd.DataFrame:
    networks = load_js_object(network_js, "var networks=")
    network_name = next(iter(networks))
    nodes = networks[network_name]["elements"]["nodes"]

    rows = []
    for node in nodes:
        data = node["data"]
        if float(data.get("FirstInGroupByLogP", 0)) != 1.0:
            continue
        rows.append(
            {
                "GROUP_ID": int(float(data["GROUP_ID"])),
                "Description": data["Description"],
                "Category": data["Category"],
                "Prefix": extract_term_prefix(
                    data.get("Symbol", ""), data.get("Category", "")
                ),
                "GeneCount": int(float(data["_GeneInGOAndHitList"])),
                "LogQ": float(data["Log_q_value_"]),
                "q_value": float(np.power(10.0, float(data["Log_q_value_"]))),
                "neg_log10_q": float(-float(data["Log_q_value_"])),
            }
        )

    if not rows:
        raise ValueError(
            "No representative cluster nodes were found in the network file."
        )

    styles = load_js_object(style_js, "var styles=")
    cluster_style = next(
        (style for style in styles if style.get("title") == "ColorByCluster"), None
    )
    if cluster_style is None:
        raise ValueError(
            "Could not find the 'ColorByCluster' style in the style JS file."
        )

    color_map: dict[int, tuple[float, float, float]] = {}
    for rule in cluster_style["style"]:
        selector = rule.get("selector", "")
        match = re.fullmatch(r"node\[GROUP_ID = ([0-9.]+)\]", selector)
        if not match:
            continue
        group_id = int(float(match.group(1)))
        color_map[group_id] = parse_rgb_string(rule["css"]["background-color"])

    df = pd.DataFrame(rows).sort_values("GROUP_ID").reset_index(drop=True)
    df["Color"] = df["GROUP_ID"].map(color_map)
    missing_colors = df[df["Color"].isna()]["GROUP_ID"].tolist()
    if missing_colors:
        raise ValueError(f"Missing colors for GROUP_ID values: {missing_colors }")
    return df


def gene_size_to_area(gene_count: float, scale: float = 16.0) -> float:
    return float(gene_count) * scale


def plot_cluster_bubble(
    df: pd.DataFrame, output_png: Path, title: str | None = None
) -> None:
    bubble_df = df.copy()
    y_positions = np.arange(len(bubble_df))
    bubble_sizes = bubble_df["GeneCount"].apply(gene_size_to_area)

    fig = plt.figure(figsize=(12.5, 6.2))
    ax = fig.add_axes([0.08, 0.16, 0.88, 0.76])
    ax.scatter(
        bubble_df["neg_log10_q"],
        y_positions,
        s=bubble_sizes,
        c=bubble_df["Color"].tolist(),
        alpha=0.96,
        linewidths=0.8,
        edgecolors="white",
        zorder=3,
    )

    ax.set_yticks([])
    ax.tick_params(axis="y", right=False, left=False, labelright=False, labelleft=False)
    ax.invert_yaxis()

    ax.set_xlim(0, 10)
    ax.set_xticks([5, 10])
    ax.set_xlabel(r"$-\log_{10}(p_{FDR})$", fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#333333")

    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a cluster bubble enrichment figure from Cytoscape network exports."
    )
    parser.add_argument(
        "--output",
        default="GO_Cluster_Bubble.png",
        help="Output PNG path. Default: GO_Cluster_Bubble.png",
    )
    parser.add_argument(
        "--network-js",
        required=True,
        help="Path to GONetwork.js.",
    )
    parser.add_argument(
        "--style-js",
        required=True,
        help="Path to GONetwork.style.js.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    bubble_df = load_cluster_bubble_data(Path(args.network_js), Path(args.style_js))
    plot_cluster_bubble(bubble_df, output_png=output_path, title=None)

    summary_path = output_path.with_suffix(".terms.csv")
    bubble_df[
        [
            "GROUP_ID",
            "Prefix",
            "Description",
            "Category",
            "GeneCount",
            "q_value",
            "neg_log10_q",
        ]
    ].to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved figure to: {output_path .resolve ()}")
    print(f"Saved term table to: {summary_path .resolve ()}")
    print(f"Selected clusters: {len (bubble_df )}")


if __name__ == "__main__":
    main()

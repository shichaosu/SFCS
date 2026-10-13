# Functional decoding or transcriptomic analysis helper.
# Update the path variables below before running this script.
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from brainstat.context.meta_analysis import meta_analytic_decoder, radar_plot


def run_brainstat_meta_analysis():
    # ==========================================
    # ==========================================
    data_path = "data/data_use/brainstat/vertex_fsaverage5/T_value_Neg.npy"
    fig_dir = "data/visulization/brainstat/meta"
    os.makedirs(fig_dir, exist_ok=True)

    # ==========================================
    # ==========================================
    print(f"Processing message.")
    if not os.path.exists(data_path):
        raise ValueError(f"Processing message.")

    data = np.load(data_path)
    if data.shape != (20484,):
        raise ValueError(f"Input shape does not match the expected dimensions.")

    all_data_array = data[np.newaxis, :]
    print("Processing message.")
    all_data_array[all_data_array == 0] = np.nan
    print("Processing message.")
    mean_map = np.nanmean(all_data_array, axis=0)
    nan_count = np.isnan(mean_map).sum()
    valid_count = 20484 - nan_count
    print(f"NaN handling completed.")

    if valid_count < 100:
        raise ValueError("Processing message.")

        # ==========================================
        # ==========================================

        # ==========================================
        # ==========================================
    print("Processing message.")
    meta_analysis = meta_analytic_decoder("fsaverage5", mean_map)
    csv_path = os.path.join(fig_dir, "meta_analysis_full_results_Neg.csv")
    meta_analysis.to_csv(csv_path)
    print(f"Processing message.")

    # ==========================================
    # ==========================================
    print("Processing message.")
    correlations = meta_analysis["Pearson's r"]
    positive_correlations = correlations[correlations > 0].to_dict()

    if not positive_correlations:
        print("Processing message.")
    else:
        wc = WordCloud(
            background_color="white",
            random_state=42,
            width=1000,
            height=600,
            colormap="viridis",
        )
        wc.generate_from_frequencies(frequencies=positive_correlations)

        plt.figure(figsize=(12, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Meta-analytic Decoding Word Cloud", fontsize=16)

        wordcloud_path = os.path.join(fig_dir, "meta_analytic_wordcloud_Pos.png")
        plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Processing message.")

        # ==========================================
        # ==========================================
    print("Processing message.")
    num_features = 8
    sorted_meta = meta_analysis.sort_values(by="Pearson's r", ascending=False)
    top_data = sorted_meta.head(num_features)["Pearson's r"].to_numpy()
    top_labels = sorted_meta.head(num_features).index.tolist()

    if len(top_data) == 0 or top_data[0] <= 0:
        print("Processing message.")
    else:
        min_val, max_val = np.min(top_data), np.max(top_data)
        if min_val == max_val:
            axis_range = (min_val * 0.9, max_val * 1.1)
        else:
            axis_range = (min_val * 0.95, max_val * 1.05)

        fig = plt.figure(figsize=(8, 8))
        radar_data = top_data.reshape(-1, 1)
        radar_plot(radar_data, label=top_labels, axis_range=axis_range)

        radar_path = os.path.join(fig_dir, "meta_analytic_radar_Pos.png")
        plt.savefig(radar_path, dpi=300, bbox_inches="tight", transparent=False)
        plt.close()
        print(f"Processing message.")

    print("Processing message.")


if __name__ == "__main__":
    run_brainstat_meta_analysis()

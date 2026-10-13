# Coefficient summary, group-statistics, or visualization helper.
# Update the path variables below before running this script.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    input_dir = "data/data_use/Coefficient_abs/20_Neg/CN/"
    output_dir = "data/data_use/aparc/GLM_Sparsify_Neg/average/20_absolute"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Processing message.")
    file_pattern = os.path.join(input_dir, "*.npy")
    files = glob.glob(file_pattern)

    if not files:
        print(f"Processing message.")
        return

    print(f"Processing message.")
    data_stack = []
    for file_path in files:
        try:
            data = np.load(file_path)
            if data.shape != (68, 24):
                print(f"Input shape does not match the expected dimensions.")
                continue
            data = data.astype(float)
            data[data > 30] = np.nan

            data_stack.append(data)

        except Exception as e:
            print(f"Processing message.")

    if not data_stack:
        print("Processing message.")
        return
    data_stack = np.array(data_stack)
    average_matrix = np.nanmean(data_stack, axis=0)

    print(f"Input shape does not match the expected dimensions.")
    save_data_path = os.path.join(output_dir, "CN_average_matrix.npy")
    np.save(save_data_path, average_matrix)
    np.savetxt(
        os.path.join(output_dir, "average_matrix.txt"), average_matrix, fmt="%.4f"
    )
    print(f"Processing message.")
    plt.figure(figsize=(10, 12))
    sns.heatmap(average_matrix, cmap="RdBu_r", center=0, annot=False)

    plt.title("Average Matrix Heatmap")
    plt.xlabel("Dimension 24")
    plt.ylabel("Dimension 68 (ROIs)")
    save_img_path = os.path.join(output_dir, "CN_average_heatmap.png")
    plt.savefig(save_img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Processing message.")
    print("Processing message.")


if __name__ == "__main__":
    main()

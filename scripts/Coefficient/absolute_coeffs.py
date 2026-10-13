# Coefficient summary, group-statistics, or visualization helper.
# Update the path variables below before running this script.
import os
import numpy as np


def batch_abs_npy(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Processing message.")
    else:
        print(f"Processing message.")
    file_list = os.listdir(input_dir)
    npy_files = [f for f in file_list if f.endswith(".npy")]
    total_files = len(npy_files)

    print(f"Processing message.")
    for i, filename in enumerate(npy_files):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            data = np.load(src_path)
            data_abs = np.abs(data)
            np.save(dst_path, data_abs)
            if (i + 1) % 10 == 0:
                print(f"Processing message.")

        except Exception as e:
            print(f"Processing message.")

    print("Processing message.")


if __name__ == "__main__":
    source_path = "data/data_use/aparc/GLM_Sparsify_Pos/20Coefficient/"
    target_path = "data/data_use/aparc/GLM_Sparsify_Pos/20Coefficient_absolute/"

    batch_abs_npy(source_path, target_path)

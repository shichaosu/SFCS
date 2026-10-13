# Coefficient summary, group-statistics, or visualization helper.
# Update the path variables below before running this script.
import os
import numpy as np

source_root = "data/data_use/Coefficient_abs/20_Pos/"
target_root = "data/data_use/degree/20_Pos_abs/"


def process_npy_data():
    print(f"Processing message.")

    count = 0
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith(".npy"):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_root)
                target_dir_path = os.path.join(target_root, relative_path)
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)

                target_file_path = os.path.join(target_dir_path, file)

                try:
                    data = np.load(src_file_path)
                    if data.shape != (68, 24):
                        print(f"Input shape does not match the expected dimensions.")
                        continue
                    sliced_data = data[:, 1:]
                    result = np.sum(sliced_data, axis=1)
                    np.save(target_file_path, result)

                    count += 1
                    if count % 100 == 0:
                        print(f"Processing message.")

                except Exception as e:
                    print(f"Processing message.")

    print(f"Processing message.")


if __name__ == "__main__":
    process_npy_data()

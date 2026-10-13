# Morphometric similarity network processing for the SFCS workflow.
# Update the path variables below before running this script.
import os
import numpy as np

src_root = "data/MSN/aparc/result/"
dst_root = "data/MSN/aparc/result_Z/"

for folder_name in os.listdir(src_root):
    src_folder = os.path.join(src_root, folder_name)
    if not os.path.isdir(src_folder):
        continue

    dst_folder = os.path.join(dst_root, folder_name)
    os.makedirs(dst_folder, exist_ok=True)

    for file_name in os.listdir(src_folder):
        if not file_name.endswith(".npy"):
            continue

        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)

        data = np.load(src_file)

        if data.ndim != 2:
            print(f"Processing message.")
            continue
        np.fill_diagonal(data, 0)
        mask = ~np.eye(data.shape[0], dtype=bool)
        vals = data[mask]

        mean = vals.mean()
        std = vals.std()

        if std == 0:
            print(f"Processing message.")
            vals_z = np.zeros_like(vals)
        else:
            vals_z = (vals - mean) / std
        data_z = np.zeros_like(data)
        data_z[mask] = vals_z
        np.save(dst_file, data_z)

print("Processing message.")

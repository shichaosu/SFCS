# Higher-order interaction processing and visualization helper.
# Update the path variables below before running this script.
import os
import numpy as np

input_dir = "data/fmri/aparc/HOI_GSR_Z/"
output_dir_top_val = "data/fmri/aparc/HOI_GSR_Z_Sparsification/top20/"
output_dir_top_bin = "data/fmri/aparc/Matrix_Sparsification/top20/"
output_dir_bot_val = "data/fmri/aparc/HOI_GSR_Z_Sparsification/bottom20/"
output_dir_bot_bin = "data/fmri/aparc/Matrix_Sparsification/bottom20/"
os.makedirs(output_dir_top_val, exist_ok=True)
os.makedirs(output_dir_top_bin, exist_ok=True)
os.makedirs(output_dir_bot_val, exist_ok=True)
os.makedirs(output_dir_bot_bin, exist_ok=True)

print("Processing message.")

for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)  # shape [68,68,68]
        threshold_top = np.percentile(data, 80)
        threshold_bot = np.percentile(data, 20)

        # -------------------------------------------------
        # -------------------------------------------------
        mask_top = data > threshold_top
        data_top_val = np.zeros_like(data)
        data_top_val[mask_top] = data[mask_top]
        np.save(os.path.join(output_dir_top_val, filename), data_top_val)
        data_top_bin = np.zeros_like(data, dtype=np.uint8)
        data_top_bin[mask_top] = 1
        np.save(os.path.join(output_dir_top_bin, filename), data_top_bin)

        # -------------------------------------------------
        # -------------------------------------------------
        mask_bot = data < threshold_bot
        data_bot_val = np.zeros_like(data)
        data_bot_val[mask_bot] = data[mask_bot]
        np.save(os.path.join(output_dir_bot_val, filename), data_bot_val)
        data_bot_bin = np.zeros_like(data, dtype=np.uint8)
        data_bot_bin[mask_bot] = 1
        np.save(os.path.join(output_dir_bot_bin, filename), data_bot_bin)

print("Processing message.")

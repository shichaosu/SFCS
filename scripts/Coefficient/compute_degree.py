# Coefficient summary, group-statistics, or visualization helper.
# Update the path variables below before running this script.
import numpy as np
from scipy.io import savemat
import os

input_path = (
    "data/data_use/aparc/GLM_Sparsify_Pos/average/20_absolute/average_matrix.npy"
)
output_path = "data/data_use/aparc/GLM_Sparsify_Pos/average/20_absolute/degree.mat"


def process_data():
    if not os.path.exists(input_path):
        print(f"Processing message.")
        return
    matrix = np.load(input_path)
    print(f"Input shape does not match the expected dimensions.")
    degree_data = np.sum(matrix, axis=1)

    print(f"Input shape does not match the expected dimensions.")
    print(f"Processing message.")
    savemat(output_path, {"degree": degree_data})

    print(f"Processing message.")


if __name__ == "__main__":
    process_data()

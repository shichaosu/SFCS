# Quality-control and data-conversion helper for the SFCS workflow.
# Update the path variables below before running this script.
import os
import numpy as np

directory = "data/fmri/aparc/HOI_GSR_Z/"


def nan_ratio(file_path):
    data = np.load(file_path)
    total_elements = data.size
    nan_count = np.isnan(data).sum()
    return nan_count / total_elements if total_elements > 0 else 0


def main():
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    for f in files:
        file_path = os.path.join(directory, f)
        ratio = nan_ratio(file_path)
        if ratio > 0.1:
            print(f"NaN handling completed.")
            choice = input("Processing message.").strip().lower()
            if choice == "y":
                os.remove(file_path)
                print(f"Processing message.")
            else:
                print(f"Processing message.")
        else:
            print(f"NaN handling completed.")


if __name__ == "__main__":
    main()

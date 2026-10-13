# Structural connectivity and communication-metric processing.
# Update the path variables below before running this script.
import os
import scipy.io
import numpy as np

src_dir = "data/SC/aparc/result_StdSC_addmin_cleanAndStd0_1/"
dest_base_dir = "data/data_use/aparc/SC_Std0_1/"

vars_to_extract = ["D_bin", "F", "M0", "MFPT", "Pmat", "SPL", "T", "dyn", "hops"]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


for filename in os.listdir(src_dir):
    if filename.endswith(".mat"):
        filepath = os.path.join(src_dir, filename)
        print(f"Processing {filename } ...")
        mat_data = scipy.io.loadmat(filepath)
        name_wo_ext = os.path.splitext(filename)[0]

        for var_name in vars_to_extract:
            if var_name in mat_data:
                data = mat_data[var_name]
                if data.ndim != 2:
                    print(
                        f"Warning: Variable {var_name } in {filename } is not 2D, skipping."
                    )
                    continue
                var_dir = os.path.join(dest_base_dir, var_name)
                ensure_dir(var_dir)
                out_path = os.path.join(var_dir, f"{name_wo_ext }.npy")
                np.save(out_path, data)
            else:
                print(f"Variable {var_name } not found in {filename }")

print("All files processed.")

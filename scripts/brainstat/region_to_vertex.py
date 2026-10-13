# Functional decoding or transcriptomic analysis helper.
# Update the path variables below before running this script.
import os
import numpy as np
import nibabel.freesurfer.io as fsio
from tqdm import tqdm

# ======================
# ======================
input_dir = "data/data_use/brainstat/"
output_dir = "data/data_use/brainstat/vertex_fsaverage5/"

lh_annot_path = (
    "data/atlas/Yeo_JNeurophysiol11_FreeSurfer/fsaverage5/label/lh.aparc.annot"
)
rh_annot_path = lh_annot_path.replace("lh.", "rh.")

os.makedirs(output_dir, exist_ok=True)

# ======================
# ======================
lh_labels, lh_ctab, lh_names = fsio.read_annot(lh_annot_path)
rh_labels, rh_ctab, rh_names = fsio.read_annot(rh_annot_path)

lh_names = [n.decode() for n in lh_names]
rh_names = [n.decode() for n in rh_names]
exclude = ["unknown", "corpuscallosum", "medialwall"]

lh_valid = [i for i, n in enumerate(lh_names) if n not in exclude]
rh_valid = [i for i, n in enumerate(rh_names) if n not in exclude]

assert len(lh_valid) == 34
assert len(rh_valid) == 34

# ======================
# ======================
roi_txt = os.path.join(output_dir, "DK68_region_order.txt")

with open(roi_txt, "w") as f:

    f.write("Index\tHemisphere\tRegion\n")
    for i, idx in enumerate(lh_valid):
        name = lh_names[idx]
        f.write(f"{i }\tLH\t{name }\n")
    for i, idx in enumerate(rh_valid):
        name = rh_names[idx]
        f.write(f"{i +34 }\tRH\t{name }\n")

print("ROI order file saved:", roi_txt)

# ======================
# vertex index map
# ======================
lh_vertex_map = {i: np.where(lh_labels == i)[0] for i in lh_valid}
rh_vertex_map = {i: np.where(rh_labels == i)[0] for i in rh_valid}

# ======================
# ======================
files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])

for f in tqdm(files):

    roi_values = np.load(os.path.join(input_dir, f))

    if roi_values.shape[0] != 68:
        raise ValueError(f"{f } is not 68-D")

    lh_roi = roi_values[:34]
    rh_roi = roi_values[34:]

    lh_vertex = np.zeros(len(lh_labels))
    rh_vertex = np.zeros(len(rh_labels))
    for i, idx in enumerate(lh_valid):
        lh_vertex[lh_vertex_map[idx]] = lh_roi[i]
    for i, idx in enumerate(rh_valid):
        rh_vertex[rh_vertex_map[idx]] = rh_roi[i]

    vertex_all = np.concatenate([lh_vertex, rh_vertex])

    out_path = os.path.join(output_dir, f)
    np.save(out_path, vertex_all)

print("All subjects finished!")

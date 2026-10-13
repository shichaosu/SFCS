# Visualization helper for the SFCS manuscript.
# Update the path variables below before running this script.
import pandas as pd
from scipy.io import savemat

excel_path = r"data/ROI_score.xlsx"
mat_path = r"data/ROI_score.mat"
df_red = pd.read_excel(excel_path, sheet_name="Redundancy")
df_syn = pd.read_excel(excel_path, sheet_name="Synergy")
df_red = df_red.sort_values(by="ROI_Index")
df_syn = df_syn.sort_values(by="ROI_Index")
red_values = df_red["Average"].values[:68]
syn_values = df_syn["Average"].values[:68]
savemat(mat_path, {"Redundancy": red_values, "Synergy": syn_values})

print("Processing message.")

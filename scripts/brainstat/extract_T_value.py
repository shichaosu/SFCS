# Functional decoding or transcriptomic analysis helper.
# Update the path variables below before running this script.
import pandas as pd
import numpy as np

input_file = "data/data_use/degree/20_Pos_abs/CN_AD_abs_IPW_Results.xlsx"
out_dir = "data/data_use/brainstat/"
beta_out = out_dir + "Beta_ADvsCN_Pos.npy"
t_out = out_dir + "T_value_Pos.npy"
df = pd.read_excel(input_file)
beta_array = np.zeros(68)
t_array = np.zeros(68)
for _, row in df.iterrows():
    roi = int(row["ROI_ID"])
    if 1 <= roi <= 68:
        beta_array[roi - 1] = row["Beta_ADvsCN"]
        t_array[roi - 1] = row["T_value"]
np.save(beta_out, beta_array)
np.save(t_out, t_array)
print("Beta_ADvsCN array:")
print(beta_array)

print("\nT_value array:")
print(t_array)

# Prediction or table-preparation helper for the SFCS workflow.
# Update the path variables below before running this script.
import pandas as pd
import numpy as np
import os
from scipy.stats import skew

file_path = "data/data_use/table/filtered_assessment_results.xlsx"

if not os.path.exists(file_path):
    print(f"Processing message.")
else:
    print(f"Processing message.")
    try:
        df = pd.read_excel(file_path, sheet_name="All")
        df_meta = df.iloc[:, :3].copy()
        df_scores = df.iloc[:, 3:].copy()
        df_trans_scores = df_scores.copy()

        print("Processing message.")

        for col in df_scores.columns:
            valid_data = df_scores[col].dropna()

            if len(valid_data) < 5:
                continue
            col_skewness = valid_data.skew()
            max_val = valid_data.max()
            zero_fraction = (valid_data == 0).mean()
            if zero_fraction > 0.4 and col_skewness > 0:
                df_trans_scores[col] = np.log1p(df_scores[col])
                print(f"Processing message.")
            elif col_skewness > 1.0:
                df_trans_scores[col] = np.log1p(df_scores[col])
                print(f"Processing message.")
            elif col_skewness < -1.0:
                ceiling = max_val + 1
                df_trans_scores[col] = np.log1p(ceiling - df_scores[col])
                print(f"Processing message.")

            else:
                print(f"Processing message.")
        df_final = pd.concat([df_meta, df_trans_scores], axis=1)

        with pd.ExcelWriter(
            file_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df_final.to_excel(writer, sheet_name="All_trans", index=False)
        print("Processing message.")

    except Exception as e:
        print(f"Processing message.")

# Coefficient summary, group-statistics, or visualization helper.
# Update the path variables below before running this script.
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ==============================
# PATH SETTINGS
# ==============================

DATA_ROOT = "data/data_use/degree/all/"
INFO_PATH = "data/subject_info.xlsx"

DIR_AD = os.path.join(DATA_ROOT, "AD")
DIR_CN = os.path.join(DATA_ROOT, "CN")

RESULT_PATH = os.path.join(DATA_ROOT, "CN_AD_IPW_Results.xlsx")
RESULT_PATH_LEFT = RESULT_PATH.replace(".xlsx", "_Left.xlsx")
RESULT_PATH_RIGHT = RESULT_PATH.replace(".xlsx", "_Right.xlsx")

N_ROIS = 34
N_ROIS_TOTAL = 68

# ==============================
# DATA LOADING
# ==============================


def load_and_prepare():

    print("\n[1] Loading subject information")

    df_info = pd.read_excel(INFO_PATH, sheet_name="fmri")

    df = df_info.iloc[:, [1, 7, 8]].copy()
    df.columns = ["FolderID", "Sex", "Age"]

    df["FolderID"] = df["FolderID"].astype(str).str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    sex_map = {"M": 1, "MALE": 1, "F": 0, "FEMALE": 0}

    df["Sex_Code"] = df["Sex"].astype(str).str.upper().map(sex_map)

    df = df.dropna(subset=["Age", "Sex_Code"])

    file_records = []

    for grp, path in [("AD", DIR_AD), ("CN", DIR_CN)]:

        if not os.path.exists(path):
            continue

        for f in os.listdir(path):

            if f.endswith(".npy"):

                file_records.append(
                    {
                        "FolderID": f.replace(".npy", "").strip(),
                        "Group": grp,
                        "FilePath": os.path.join(path, f),
                    }
                )

    df_files = pd.DataFrame(file_records)

    df_all = pd.merge(df_files, df, on="FolderID", how="inner")

    print("Total matched subjects:", len(df_all))

    return df_all

    # ==============================
    # PROPENSITY SCORE MODEL
    # ==============================


def compute_ipw(df):

    print("\n[2] Estimating propensity scores")

    df = df.copy()

    df["Group_Num"] = df["Group"].map({"CN": 0, "AD": 1})

    X = df[["Age", "Sex_Code"]]
    y = df["Group_Num"]

    model = LogisticRegression(solver="lbfgs", random_state=42)

    model.fit(X, y)

    ps = model.predict_proba(X)[:, 1]

    ps = np.clip(ps, 1e-3, 1 - 1e-3)

    df["PS"] = ps

    p_treated = y.mean()

    print("Treatment prevalence:", round(p_treated, 3))

    # Stabilized weights

    df["IPW"] = np.where(
        df["Group"] == "AD", p_treated / ps, (1 - p_treated) / (1 - ps)
    )

    # Weight truncation

    lower = np.percentile(df["IPW"], 1)
    upper = np.percentile(df["IPW"], 99)

    df["IPW"] = np.clip(df["IPW"], lower, upper)

    print("Weight truncation:", round(lower, 3), "-", round(upper, 3))

    return df

    # ==============================
    # EFFECTIVE SAMPLE SIZE
    # ==============================


def effective_sample_size(weights):

    return (weights.sum() ** 2) / (np.sum(weights**2))

    # ==============================
    # SMD BALANCE CHECK
    # ==============================


def compute_smd(x1, x0, w1, w0):

    ds1 = DescrStatsW(x1, weights=w1, ddof=0)
    ds0 = DescrStatsW(x0, weights=w0, ddof=0)

    m1, m0 = ds1.mean, ds0.mean
    v1, v0 = ds1.var, ds0.var

    denom = np.sqrt((v1 + v0) / 2)

    return (m1 - m0) / denom


def baseline_balance(df, weight_col=None):

    print("\n[3] Baseline balance check")

    ad = df[df.Group == "AD"]
    cn = df[df.Group == "CN"]

    if weight_col is None:

        w_ad = np.ones(len(ad))
        w_cn = np.ones(len(cn))

    else:

        w_ad = ad[weight_col].values
        w_cn = cn[weight_col].values

    smd_age = compute_smd(ad["Age"], cn["Age"], w_ad, w_cn)

    smd_sex = compute_smd(ad["Sex_Code"], cn["Sex_Code"], w_ad, w_cn)

    print("SMD Age:", round(smd_age, 3))
    print("SMD Sex:", round(smd_sex, 3))

    if weight_col:

        ess_ad = effective_sample_size(w_ad)
        ess_cn = effective_sample_size(w_cn)

        print("Effective sample size AD:", round(ess_ad, 1))
        print("Effective sample size CN:", round(ess_cn, 1))

        # ==============================
        # LOAD ROI DATA
        # ==============================


def load_brain_data(df):

    n = len(df)

    brain = np.zeros((n, N_ROIS_TOTAL))

    valid = np.ones(n, dtype=bool)

    for i, row in enumerate(df.itertuples()):

        try:

            arr = np.load(row.FilePath)

            arr = np.array(arr).flatten()

            if len(arr) < N_ROIS_TOTAL:

                valid[i] = False

            else:

                brain[i] = arr[:N_ROIS_TOTAL]

        except:

            valid[i] = False

    df = df.loc[valid].reset_index(drop=True)

    brain = brain[valid]

    print("\nValid subjects:", len(df))

    return df, brain

    # ==============================
    # ROI ANALYSIS
    # ==============================


def roi_weighted_regression(df, brain):

    print("\n[4] Weighted ROI analysis")

    group = (df["Group"] == "AD").astype(int).values

    weights = df["IPW"].values

    results = []

    n_rois = brain.shape[1]

    for r in range(n_rois):

        y = brain[:, r]

        X = sm.add_constant(group)

        model = sm.WLS(y, X, weights=weights).fit()

        beta = model.params[1]
        pval = model.pvalues[1]
        tval = model.tvalues[1]

        results.append(
            {"ROI_ID": r + 1, "Beta_ADvsCN": beta, "T_value": tval, "P_value": pval}
        )

    df_res = pd.DataFrame(results)

    reject, p_fdr, _, _ = multipletests(df_res["P_value"], method="fdr_bh")

    df_res["P_FDR"] = p_fdr
    df_res["Significant_FDR"] = reject

    return df_res

    # ==============================
    # MAIN PIPELINE
    # ==============================


def main():

    df = load_and_prepare()

    baseline_balance(df)

    df = compute_ipw(df)

    baseline_balance(df, "IPW")

    df, brain_all = load_brain_data(df)

    brain_left = brain_all[:, :N_ROIS]
    brain_right = brain_all[:, -N_ROIS:]

    print("\n[4-L] Left hemisphere analysis (first 34 ROIs)")
    results_left = roi_weighted_regression(df, brain_left)
    results_left = results_left.sort_values("P_value")
    results_left.to_excel(RESULT_PATH_LEFT, index=False)

    print("\n[5-L] Left hemisphere finished")
    print("Significant ROI (FDR):", results_left["Significant_FDR"].sum())
    print("\nTop findings (Left)")
    print(results_left.head(10))
    print("\nSaved to:", RESULT_PATH_LEFT)

    print("\n[4-R] Right hemisphere analysis (last 34 ROIs)")
    results_right = roi_weighted_regression(df, brain_right)
    results_right = results_right.sort_values("P_value")
    results_right.to_excel(RESULT_PATH_RIGHT, index=False)

    print("\n[5-R] Right hemisphere finished")
    print("Significant ROI (FDR):", results_right["Significant_FDR"].sum())
    print("\nTop findings (Right)")
    print(results_right.head(10))
    print("\nSaved to:", RESULT_PATH_RIGHT)


if __name__ == "__main__":

    main()

# Functional decoding or transcriptomic analysis helper.
# Update the path variables below before running this script.
import os
import numpy as np
import pandas as pd
import inspect
import functools
import argparse
import sys
from statsmodels.stats.multitest import fdrcorrection
import warnings

warnings.filterwarnings("ignore")


def _patch_pandas_set_axis_signature():
    try:
        params = inspect.signature(pd.DataFrame.set_axis).parameters
        if "inplace" in params:
            return
        original_set_axis = pd.DataFrame.set_axis

        @functools.wraps(original_set_axis)
        def set_axis_compat(self, labels, axis=0, inplace=None, **kwargs):
            copy = kwargs.pop("copy", None)
            if copy is None and inplace is False:
                copy = True
            return original_set_axis(self, labels, axis=axis, copy=copy)

        pd.DataFrame.set_axis = set_axis_compat
    except Exception:
        return


_patch_pandas_set_axis_signature()


def _patch_pandas_dataframe_append():
    try:
        if hasattr(pd.DataFrame, "append"):
            return
        original_concat = pd.concat

        def append_compat(
            self, other, ignore_index=False, verify_integrity=False, sort=False
        ):
            def normalize(obj):
                if isinstance(obj, pd.Series):
                    return obj.to_frame().T
                if isinstance(obj, dict):
                    return pd.DataFrame([obj])
                return obj

            if isinstance(other, (list, tuple)):
                others = [normalize(o) for o in other]
                return original_concat(
                    [self, *others],
                    ignore_index=ignore_index,
                    verify_integrity=verify_integrity,
                    sort=sort,
                )

            return original_concat(
                [self, normalize(other)],
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
                sort=sort,
            )

        pd.DataFrame.append = append_compat
    except Exception:
        return


_patch_pandas_dataframe_append()


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t-values-path",
        default="data/data_use/brainstat/T_value.npy",
    )
    parser.add_argument(
        "--t-values-path-fallback",
        default="data/data_use/brainstat/DK68/T_value.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="data/data_use/brainstat/Gene/T_value",
    )
    parser.add_argument(
        "--output-prefix",
        default="DK68_Gene_PLS",
    )
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--n-components-explained", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--fdr-method", default="indep")
    parser.add_argument(
        "--save-ranked", dest="save_ranked", action="store_true", default=True
    )
    parser.add_argument("--no-save-ranked", dest="save_ranked", action="store_false")
    return parser.parse_args(argv)


def _safe_zscore_matrix(X):
    X = X.astype(float, copy=True)
    col_mean = np.nanmean(X, axis=0)
    valid_gene_mask = ~np.isnan(col_mean)
    X = X[:, valid_gene_mask]
    col_mean = col_mean[valid_gene_mask]

    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    Xz = (X - X_mean) / X_std
    return Xz, valid_gene_mask


def _standardize_vector(y):
    y = y.astype(float, copy=True).ravel()
    y_std = y.std(ddof=0)
    if y_std == 0:
        raise ValueError("Processing message.")
    return (y - y.mean()) / y_std


def _fit_pls1_weights(Xz, yz, n_components):
    from sklearn.cross_decomposition import PLSRegression

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xz, yz)
    weights = pls.x_weights_[:, 0].astype(float, copy=False)
    y_pred = pls.predict(Xz).ravel()
    r = float(np.corrcoef(yz, y_pred)[0, 1])
    sse = float(np.sum((yz - y_pred) ** 2))
    sst = float(np.sum((yz - yz.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return weights, r, r2, pls


def _pls_explained_ratio(pls, Xz, yz):
    X = Xz
    Y = yz.reshape(-1, 1)

    T = pls.x_scores_
    P = pls.x_loadings_
    U = pls.y_scores_
    Q = pls.y_loadings_

    n_components = int(T.shape[1])
    ssx = float(np.sum(X**2))
    ssy = float(np.sum(Y**2))

    X_hat = np.zeros_like(X, dtype=float)
    Y_hat = np.zeros_like(Y, dtype=float)

    rows = []
    prev_r2x = 0.0
    prev_r2y = 0.0
    for k in range(n_components):
        X_hat = X_hat + np.outer(T[:, k], P[:, k])
        r2x = (
            float("nan") if ssx == 0 else float(1.0 - (np.sum((X - X_hat) ** 2) / ssx))
        )
        delta_x = float("nan") if np.isnan(r2x) else float(r2x - prev_r2x)
        prev_r2x = 0.0 if np.isnan(r2x) else r2x

        Y_hat = Y_hat + np.outer(U[:, k], Q[:, k])
        r2y = (
            float("nan") if ssy == 0 else float(1.0 - (np.sum((Y - Y_hat) ** 2) / ssy))
        )
        delta_y = float("nan") if np.isnan(r2y) else float(r2y - prev_r2y)
        prev_r2y = 0.0 if np.isnan(r2y) else r2y

        rows.append(
            {
                "component": k + 1,
                "R2X_cum": r2x,
                "R2X_delta": delta_x,
                "R2Y_cum": r2y,
                "R2Y_delta": delta_y,
            }
        )

    return pd.DataFrame(rows)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        import abagen
    except Exception as e:
        raise RuntimeError("Processing message.") from e

    t_values_path = args.t_values_path
    if not os.path.exists(t_values_path):
        t_values_path = args.t_values_path_fallback
    print(f"Processing message.")
    t_values_full = np.load(t_values_path).flatten()

    if t_values_full.size != 68:
        raise ValueError(f"Processing message.")

        # ---------------------------------------------------------
        # ---------------------------------------------------------
    print("Processing message.")
    print("Processing message.")
    dk_atlas = abagen.fetch_desikan_killiany()
    expression_df = abagen.get_expression_data(dk_atlas["image"], dk_atlas["info"])

    atlas_info_df = pd.read_csv(dk_atlas["info"])
    cortex_ids = (
        atlas_info_df.loc[atlas_info_df["structure"] == "cortex", "id"]
        .astype(int)
        .tolist()
    )
    kept_ids = None
    t_values = t_values_full
    if cortex_ids:
        present_mask = [i in expression_df.index for i in cortex_ids]
        kept_ids = [i for i, ok in zip(cortex_ids, present_mask) if ok]
        expression_df = expression_df.loc[kept_ids]
        if t_values_full.size == len(cortex_ids):
            t_values = t_values_full[present_mask]
    if expression_df.shape[0] != t_values.size:
        raise ValueError(f"Input shape does not match the expected dimensions.")
    print(f"Input shape does not match the expected dimensions.")

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    from scipy.stats import norm

    print("Processing message.")

    X = expression_df.to_numpy(dtype=float, copy=True)
    Xz, valid_gene_mask = _safe_zscore_matrix(X)
    gene_list = expression_df.columns.to_numpy()[valid_gene_mask]
    yz = _standardize_vector(t_values)

    w0, r, r2, pls = _fit_pls1_weights(Xz, yz, n_components=args.n_components)
    explained_pls = pls
    if int(args.n_components_explained) > int(args.n_components):
        _, _, _, explained_pls = _fit_pls1_weights(
            Xz, yz, n_components=int(args.n_components_explained)
        )
    explained_df = _pls_explained_ratio(explained_pls, Xz, yz)

    n_boot = int(args.n_boot)
    rng = np.random.RandomState(int(args.random_seed))
    boot_weights = np.full((n_boot, w0.size), np.nan, dtype=float)
    n_regions = Xz.shape[0]
    for b in range(n_boot):
        idx = rng.randint(0, n_regions, size=n_regions)
        try:
            wb, _, _, _ = _fit_pls1_weights(
                Xz[idx], yz[idx], n_components=args.n_components
            )
        except Exception:
            continue
        sign = 1.0 if float(np.dot(wb, w0)) >= 0 else -1.0
        boot_weights[b] = wb * sign

    boot_mean = np.nanmean(boot_weights, axis=0)
    boot_std = np.nanstd(boot_weights, axis=0, ddof=1)
    boot_std[boot_std == 0] = np.nan
    z = boot_mean / boot_std
    pval = 2.0 * norm.sf(np.abs(z))

    results = pd.DataFrame(
        {
            "Gene": gene_list,
            "PLS1_Weight": w0,
            "Bootstrap_Z": z,
            "P_Value": pval,
        }
    )

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    print("Processing message.")
    results["P_Value_FDR"] = np.nan
    results["Significant_FDR_0.05"] = False
    mask_p = ~np.isnan(results["P_Value"].to_numpy())
    if mask_p.any():
        rejected, pval_corrected = fdrcorrection(
            results.loc[mask_p, "P_Value"].values,
            alpha=float(args.fdr_alpha),
            method=str(args.fdr_method),
        )
        results.loc[mask_p, "P_Value_FDR"] = pval_corrected
        results.loc[mask_p, "Significant_FDR_0.05"] = rejected

    order = results["PLS1_Weight"].abs().sort_values(ascending=False).index
    result_df = results.loc[order].reset_index(drop=True)

    try:
        import sklearn

        sklearn_version = getattr(sklearn, "__version__", "unknown")
    except Exception:
        sklearn_version = "unknown"

    abagen_version = getattr(abagen, "__version__", "unknown")

    result_df["PLS1_r"] = r
    result_df["PLS1_R2"] = r2
    result_df["N_Regions"] = int(n_regions)
    result_df["N_Genes"] = int(w0.size)
    result_df["N_Boot"] = int(n_boot)
    result_df["Random_Seed"] = int(args.random_seed)
    result_df["FDR_Method"] = str(args.fdr_method)
    result_df["FDR_Alpha"] = float(args.fdr_alpha)
    result_df["abagen_version"] = abagen_version
    result_df["sklearn_version"] = sklearn_version

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"{args .output_prefix }.csv")
    result_df.to_csv(output_csv, index=False)

    explained_path = os.path.join(
        output_dir, f"{args .output_prefix }.pls_explained_ratio.tsv"
    )
    explained_df.to_csv(explained_path, sep="\t", index=False, header=True)

    cortex_order_txt = os.path.join(output_dir, "DK68_cortex_region_order.txt")
    if kept_ids is None:
        kept_ids = list(expression_df.index.astype(int))
    cortex_order_df = atlas_info_df.loc[
        atlas_info_df["id"].astype(int).isin(kept_ids)
    ].copy()
    cortex_order_df["id"] = cortex_order_df["id"].astype(int)
    cortex_order_df = cortex_order_df.set_index("id").loc[kept_ids].reset_index()
    cortex_order_df[["id", "hemisphere", "label", "structure"]].to_csv(
        cortex_order_txt, sep="\t", index=False, header=True
    )

    if bool(args.save_ranked):
        ranked_w_path = os.path.join(
            output_dir, f"{args .output_prefix }.rank_PLS1_weight.tsv"
        )
        ranked_z_path = os.path.join(
            output_dir, f"{args .output_prefix }.rank_bootstrapZ.tsv"
        )
        result_df.loc[:, ["Gene", "PLS1_Weight"]].to_csv(
            ranked_w_path, sep="\t", index=False, header=True
        )
        result_df.loc[:, ["Gene", "Bootstrap_Z"]].to_csv(
            ranked_z_path, sep="\t", index=False, header=True
        )

    print(f"Processing message.")
    print(f"Processing message.")
    print(f"PLS analysis completed.")
    print(f"Processing message.")
    print(f"PLS analysis completed.")
    print(f"PLS analysis completed.")
    print(
        result_df[["Gene", "PLS1_Weight", "Bootstrap_Z", "P_Value", "P_Value_FDR"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

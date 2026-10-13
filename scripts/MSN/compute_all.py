#!/usr/bin/env python3
# Morphometric similarity network processing for the SFCS workflow.
# Update the path variables below before running this script.
"""Script documentation for this SFCS analysis step."""
from __future__ import annotations
import os
import argparse
import traceback
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
from scipy.spatial import distance
import dcor
from tqdm import tqdm


# ----------------------
# Utilities
# ----------------------
def list_npy_files(input_dir: str) -> List[str]:
    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    files.sort()
    return files


def ensure_dirs(base_out: str, metrics: List[str]) -> Dict[str, str]:
    paths = {}
    os.makedirs(base_out, exist_ok=True)
    for m in metrics:
        p = os.path.join(base_out, m)
        os.makedirs(p, exist_ok=True)
        paths[m] = p
    return paths


def save_matrix(out_dir: str, fname: str, mat: np.ndarray) -> None:
    path = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)
    np.save(path, mat)


def pairwise_indices(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def square_from_pairs(
    n: int, pairs_values: List[Tuple[int, int, float]], diag_value: float = 1.0
) -> np.ndarray:
    M = np.zeros((n, n), dtype=float)
    for i, j, v in pairs_values:
        M[i, j] = v
        M[j, i] = v
    np.fill_diagonal(M, diag_value)
    return M

    # ----------------------
    # Metric implementations
    # ----------------------


def pairwise_spearman(
    mat: np.ndarray, sq: bool = False, use_abs: bool = False
) -> np.ndarray:
    n = mat.shape[0]
    vals = []
    for i, j in pairwise_indices(n):
        r, _ = stats.spearmanr(mat[i], mat[j])
        if np.isnan(r):
            r = 0.0
        if use_abs:
            r = abs(r)
        if sq:
            r = r * r
        vals.append((i, j, float(r)))
    M = square_from_pairs(n, vals, diag_value=1.0)
    return M


def pairwise_kendall(
    mat: np.ndarray, sq: bool = False, use_abs: bool = False
) -> np.ndarray:
    n = mat.shape[0]
    vals = []
    for i, j in pairwise_indices(n):
        tau, _ = stats.kendalltau(mat[i], mat[j])
        if np.isnan(tau):
            tau = 0.0
        if use_abs:
            tau = abs(tau)
        if sq:
            tau = tau * tau
        vals.append((i, j, float(tau)))
    M = square_from_pairs(n, vals, diag_value=1.0)
    return M


def pairwise_pdist(mat: np.ndarray, metric: str) -> np.ndarray:
    # mat shape (n_regions, F) -> pairwise between rows
    D = distance.squareform(distance.pdist(mat, metric=metric))
    np.fill_diagonal(D, 0.0)
    return D


def pairwise_dcorr(mat: np.ndarray, unbiased: bool = False) -> np.ndarray:
    n = mat.shape[0]
    vals = []
    for i, j in pairwise_indices(n):
        # ensure dcor compiled routines get float64 contiguous inputs
        x = np.ascontiguousarray(mat[i], dtype=np.float64)
        y = np.ascontiguousarray(mat[j], dtype=np.float64)
        try:
            # Use dcor.u_distance_correlation if unbiased requested and available
            if unbiased and hasattr(dcor, "u_distance_correlation"):
                v = dcor.u_distance_correlation(x, y)
            else:
                v = dcor.distance_correlation(x, y)
        except Exception:
            v = 0.0
        if np.isnan(v):
            v = 0.0
        vals.append((i, j, float(v)))
    M = square_from_pairs(n, vals, diag_value=1.0)
    return M

    # HSIC functions (RBF kernels on the feature axis)


def rbf_kernel_matrix_1d(vec: np.ndarray, sigma: float = None) -> np.ndarray:
    x = np.asarray(vec).reshape(-1, 1)
    pd = distance.squareform(distance.pdist(x, metric="euclidean"))
    if sigma is None:
        d = pd[np.triu_indices_from(pd, k=1)]
        sigma = float(np.median(d)) if d.size > 0 else 1.0
        if sigma <= 0:
            sigma = 1.0
    K = np.exp(-(pd**2) / (2.0 * (sigma**2)))
    return K


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    one_n = np.ones((n, n), dtype=float) / n
    Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return Kc


def hsic_between_vectors(x: np.ndarray, y: np.ndarray, biased: bool = True) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.shape[0]
    if n < 2:
        return 0.0
    K = rbf_kernel_matrix_1d(x, sigma=None)
    L = rbf_kernel_matrix_1d(y, sigma=None)
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    if biased:
        val = float(np.trace(Kc @ Lc) / (n * n))
    else:
        # simplified unbiased-like estimator for small F
        K_no_diag = K.copy()
        L_no_diag = L.copy()
        np.fill_diagonal(K_no_diag, 0.0)
        np.fill_diagonal(L_no_diag, 0.0)
        s = float(np.sum(K_no_diag * L_no_diag))
        denom = n * (n - 1)
        if denom <= 0:
            val = 0.0
        else:
            val = s / denom
            if n > 2:
                val = val / (n - 2)
        val = float(max(val, 0.0))
    if np.isnan(val):
        val = 0.0
    return val


def pairwise_hsic(mat: np.ndarray, biased: bool = True) -> np.ndarray:
    n = mat.shape[0]
    vals = []
    for i, j in pairwise_indices(n):
        v = hsic_between_vectors(mat[i], mat[j], biased=biased)
        vals.append((i, j, float(v)))
    M = square_from_pairs(n, vals, diag_value=1.0)
    return M

    # ----------------------
    # Per-file processing
    # ----------------------


METRICS = [
    "spearman-sq",
    "spearman",
    "kendalltau-sq",
    "kendalltau",
    "pdist_euclidean",
    "pdist_cityblock",
    "pdist_cosine",
    "pdist_chebyshev",
    "pdist_canberra",
    "pdist_braycurtis",
    "dcorr",
    "dcorr_biased",
    "hsic",
    "hsic_biased",
]


def compute_for_file(
    in_path: str, fname: str, out_dirs: Dict[str, str], cfg: Dict, n_regions: int
) -> Tuple[str, bool, str]:
    """
    Process single file: load, validate shape, compute metrics, save results.
    Returns (fname, success_bool, message)
    """
    try:
        mat = np.load(in_path)
    except Exception as e:
        return (fname, False, f"load_error: {e }")

        # Validate & possibly transpose to get shape (n_regions, F)
    if mat.ndim != 2:
        return (fname, False, f"bad_shape ndim!=2: {mat .shape }")
    if mat.shape[0] == n_regions:
        pass
    elif mat.shape[1] == n_regions:
        mat = mat.T
    else:
        return (
            fname,
            False,
            f"unexpected shape (neither ({n_regions },F) nor (F,{n_regions })): {mat .shape }",
        )

        # optional per-file zscore normalization by feature-column (columns = features)
    if cfg.get("zscore", False):
        mu = np.mean(mat, axis=0, keepdims=True)
        sd = np.std(mat, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        mat = (mat - mu) / sd

    try:
        # correlations
        M_spearman = pairwise_spearman(mat, sq=False, use_abs=False)
        M_kendall = pairwise_kendall(mat, sq=False, use_abs=False)

        # spearman-sq / kendall-sq: either square or abs depending on cfg['use_abs']
        if cfg.get("use_abs", False):
            M_spearman_sq = np.abs(M_spearman)
            M_kendall_sq = np.abs(M_kendall)
        else:
            M_spearman_sq = M_spearman**2
            M_kendall_sq = M_kendall**2

            # distances
        M_euc = pairwise_pdist(mat, metric="euclidean")
        M_city = pairwise_pdist(mat, metric="cityblock")
        M_cos = pairwise_pdist(mat, metric="cosine")
        M_cheb = pairwise_pdist(mat, metric="chebyshev")
        M_canb = pairwise_pdist(mat, metric="canberra")
        M_bray = pairwise_pdist(mat, metric="braycurtis")

        # dcorr
        M_dcorr = pairwise_dcorr(mat, unbiased=False)
        M_dcorr_b = pairwise_dcorr(mat, unbiased=True)

        # hsic
        M_hsic = pairwise_hsic(mat, biased=True)  # biased version
        M_hsic_b = pairwise_hsic(mat, biased=False)  # "unbiased-like" version
    except Exception as e:
        tb = traceback.format_exc()
        return (fname, False, f"compute_error: {e }\n{tb }")

        # save each into respective folder with same filename
    try:
        save_matrix(out_dirs["spearman-sq"], fname, M_spearman_sq)
        save_matrix(out_dirs["spearman"], fname, M_spearman)
        save_matrix(out_dirs["kendalltau-sq"], fname, M_kendall_sq)
        save_matrix(out_dirs["kendalltau"], fname, M_kendall)
        save_matrix(out_dirs["pdist_euclidean"], fname, M_euc)
        save_matrix(out_dirs["pdist_cityblock"], fname, M_city)
        save_matrix(out_dirs["pdist_cosine"], fname, M_cos)
        save_matrix(out_dirs["pdist_chebyshev"], fname, M_cheb)
        save_matrix(out_dirs["pdist_canberra"], fname, M_canb)
        save_matrix(out_dirs["pdist_braycurtis"], fname, M_bray)
        save_matrix(out_dirs["dcorr"], fname, M_dcorr)
        save_matrix(out_dirs["dcorr_biased"], fname, M_dcorr_b)
        save_matrix(out_dirs["hsic"], fname, M_hsic)
        save_matrix(out_dirs["hsic_biased"], fname, M_hsic_b)
    except Exception as e:
        return (fname, False, f"save_error: {e }")

    return (fname, True, "ok")

    # ----------------------
    # Main orchestration
    # ----------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSN metrics for per-subject .npy files and save per-metric outputs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing per-subject .npy files (shape (n_regions,F) or (F,n_regions)).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory; will create subfolders per metric.",
    )
    parser.add_argument(
        "--n_regions",
        type=int,
        default=148,
        help="Number of brain regions (e.g., 68, 148).",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="If set, standardize features (columns) per subject before computing.",
    )
    parser.add_argument(
        "--use_abs",
        action="store_true",
        help="If set, *-sq metrics store absolute values instead of squared.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    n_regions = int(args.n_regions)
    do_zscore = args.zscore
    use_abs = args.use_abs

    files = list_npy_files(input_dir)
    if len(files) == 0:
        print("No .npy files found in", input_dir)
        return

    out_dirs = ensure_dirs(output_dir, METRICS)
    cfg = {"zscore": do_zscore, "use_abs": use_abs}

    results = []
    # sequential processing with progress bar
    for fname in tqdm(files, desc="Processing files"):
        in_path = os.path.join(input_dir, fname)
        res = compute_for_file(in_path, fname, out_dirs, cfg, n_regions=n_regions)
        results.append(res)
        # optional: print immediate errors for debugging
        if not res[1]:
            print("FAILED:", res[0], res[2])

            # summary
    n_ok = sum(1 for r in results if r[1])
    n_fail = len(results) - n_ok
    print(f"Processed {len (results )} files: {n_ok } succeeded, {n_fail } failed.")
    if n_fail > 0:
        for r in results:
            if not r[1]:
                print("FAILED:", r[0], r[2])


if __name__ == "__main__":
    import sys

    if not sys.argv[1:]:
        sys.argv.extend(
            [
                "--input_dir",
                "data/MSN/aparc/all_feature_Zscore/",
                "--output_dir",
                "data/MSN/aparc/result/",
                "--n_regions",
                "68",
            ]
        )
    main()

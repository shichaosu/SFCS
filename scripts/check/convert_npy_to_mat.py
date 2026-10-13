#!/usr/bin/env python3
# Quality-control and data-conversion helper for the SFCS workflow.
# Update the path variables below before running this script.
"""
Convert all .npy files in a source directory to .mat files in a target directory.
Each .mat will contain a variable named 'sc' holding the numpy array.

Usage: run this script directly. Paths are configured by default for the user's request.
"""
import os
import sys
import numpy as np
import scipy.io

SRC_DIR = "data/SC/a2009s/SCz_npy/"
DST_DIR = "data/SC/a2009s/SCz_mat/"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def convert_file(npy_path, mat_path):
    try:
        arr = np.load(npy_path, allow_pickle=True)
        # Save with variable name 'sc'
        scipy.io.savemat(mat_path, {"sc": arr})
        return True, None
    except Exception as e:
        return False, str(e)


def main(src=SRC_DIR, dst=DST_DIR):
    if not os.path.isdir(src):
        print(f"Source directory does not exist: {src }", file=sys.stderr)
        return 2
    ensure_dir(dst)
    files = [f for f in os.listdir(src) if f.endswith(".npy")]
    if not files:
        print(f"No .npy files found in {src }")
        return 0

    total = 0
    success = 0
    failures = []

    for fname in files:
        total += 1
        npy_path = os.path.join(src, fname)
        mat_fname = os.path.splitext(fname)[0] + ".mat"
        mat_path = os.path.join(dst, mat_fname)
        ok, err = convert_file(npy_path, mat_path)
        if ok:
            success += 1
            print(f"Converted: {npy_path } -> {mat_path }")
        else:
            failures.append((npy_path, err))
            print(f"Failed: {npy_path } -> {err }", file=sys.stderr)

    print("\nSummary:")
    print(f"Total: {total }")
    print(f"Success: {success }")
    print(f"Failed: {len (failures )}")
    if failures:
        print("\nFailures detail:", file=sys.stderr)
        for p, e in failures:
            print(f"{p }: {e }", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

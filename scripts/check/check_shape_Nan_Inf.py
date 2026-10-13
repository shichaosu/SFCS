#!/usr/bin/env python3
# Quality-control and data-conversion helper for the SFCS workflow.
# Update the path variables below before running this script.
"""Script documentation for this SFCS analysis step."""

import os
import sys
import argparse
import numpy as np

TARGET_SHAPE = (148, 148)


def count_non_numeric(arr, max_chunk_elems=10_000_000):
    """Script documentation for this SFCS analysis step."""
    try:
        shape = arr.shape
        total = int(np.prod(shape)) if shape else int(arr.size)
    except Exception:
        total = int(arr.size)
        # quick checks for common dtypes
    if np.issubdtype(arr.dtype, np.floating):
        if total <= max_chunk_elems:
            non_numeric = int(np.count_nonzero(np.isnan(arr) | np.isinf(arr)))
            return non_numeric, total
        non_numeric = 0
        if shape and shape[0] > 0:
            try:
                row_len = total // shape[0]
            except Exception:
                row_len = 1
            chunk_rows = max(1, int(max_chunk_elems // max(1, row_len)))
            for start in range(0, shape[0], chunk_rows):
                stop = start + chunk_rows
                sub = arr[slice(start, stop)]
                non_numeric += int(np.count_nonzero(np.isnan(sub) | np.isinf(sub)))
            return non_numeric, total
            # fallback
        return int(np.count_nonzero(np.isnan(arr) | np.isinf(arr))), total

    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        return 0, total

        # object or other dtypes: element-wise check (may be slow)
    non_numeric = 0
    it = np.nditer(arr, flags=["refs_ok", "zerosize_ok"], op_flags=["readonly"])
    for v in it:
        val = v.item()
        try:
            # float values: check NaN/Inf
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                non_numeric += 1
            else:
                _ = float(val)
        except Exception:
            non_numeric += 1
    return non_numeric, total


def check_file(path, use_mmap=True):
    """Script documentation for this SFCS analysis step."""
    rec = {
        "path": path,
        "shape": None,
        "total": None,
        "non_numeric": None,
        "error": None,
    }
    try:
        if use_mmap:
            arr = np.load(path, mmap_mode="r")
        else:
            arr = np.load(path, allow_pickle=True)
    except Exception as e:
        rec["error"] = f"load_error: {e }"
        return rec
    try:
        rec["shape"] = tuple(arr.shape)
        rec["total"] = int(np.prod(arr.shape)) if arr.shape else int(arr.size)
        non_numeric, total = count_non_numeric(arr)
        rec["non_numeric"] = int(non_numeric)
    except Exception as e:
        rec["error"] = f"process_error: {e }"
    return rec


def confirm(prompt):
    """Script documentation for this SFCS analysis step."""
    while True:
        resp = input(prompt + " [y/n/a(s)q] ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        if resp in ("a", "always"):
            return "always"
        if resp in ("s", "skip"):
            return "skip"
        if resp in ("q", "quit"):
            raise SystemExit("User quit")
        print(
            "Please answer 'y' (yes), 'n' (no), 'a' (always delete remaining), 's' (skip remaining deletions) or 'q' (quit)."
        )


def main():
    parser = argparse.ArgumentParser(description="Processing message.")
    parser.add_argument(
        "dir",
        nargs="Processing message.",
        default="data/SC/a2009s/SCz_npy/",
        help="Processing message.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Processing message."
    )
    parser.add_argument(
        "--auto-delete",
        choices=["none", "always"],
        default="none",
        help="Processing message.",
    )
    parser.add_argument("--no-mmap", action="store_true", help="Processing message.")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"Error during processing.", file=sys.stderr)
        sys.exit(2)

        # collect files
    files = []
    if args.recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".npy"):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(".npy"):
                files.append(os.path.join(root, fn))

    if not files:
        print("Processing message.")
        return

    to_report = []
    for path in files:
        print(f"Checking: {path }")
        rec = check_file(path, use_mmap=not args.no_mmap)
        if rec["error"]:
            # treat load/process error as reportable (user may want to inspect/delete)
            to_report.append(rec)
            print(f"  ERROR: {rec ['error']}")
            continue
        bad_shape = rec["shape"] != TARGET_SHAPE
        has_non_numeric = rec["non_numeric"] and rec["non_numeric"] > 0
        if bad_shape or has_non_numeric:
            to_report.append(rec)
            reasons = []
            if bad_shape:
                reasons.append(f"shape={rec ['shape']}")
            if has_non_numeric:
                reasons.append(f"non_numeric={rec ['non_numeric']}/{rec ['total']}")
            print("  -> FLAGGED: " + "; ".join(reasons))
        else:
            pass

    if not to_report:
        print("Processing message.")
        return
    always_delete = args.auto_delete == "always"
    skip_remaining = False
    for rec in to_report:
        path = rec["path"]
        reasons = []
        if rec["error"]:
            reasons.append(f"ERROR:{rec ['error']}")
        else:
            if rec["shape"] != TARGET_SHAPE:
                reasons.append(f"shape={rec ['shape']}")
            if rec["non_numeric"] and rec["non_numeric"] > 0:
                reasons.append(f"non_numeric={rec ['non_numeric']}/{rec ['total']}")
        print("\nFile:", path)
        print("  " + "; ".join(reasons))
        if always_delete:
            try:
                os.remove(path)
                print("  Deleted (auto).")
            except Exception as e:
                print(f"  Failed to delete: {e }")
            continue
        if skip_remaining:
            print("  Skipped (skip-all mode).")
            continue

        try:
            resp = confirm("Processing message.")
        except SystemExit as e:
            print(str(e))
            break

        if resp is True:
            try:
                os.remove(path)
                print("  Deleted.")
            except Exception as e:
                print(f"  Failed to delete: {e }")
        elif resp == "always":
            # delete this and set always_delete
            try:
                os.remove(path)
                print("  Deleted.")
            except Exception as e:
                print(f"  Failed to delete: {e }")
            always_delete = True
        elif resp == "skip":
            skip_remaining = True
            print("  Skipping remaining deletions.")
        else:
            print("  Kept.")

    print("\nDone.")


if __name__ == "__main__":
    main()

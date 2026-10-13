#!/usr/bin/env python3
# Quality-control and data-conversion helper for the SFCS workflow.
# Update the path variables below before running this script.
"""Script documentation for this SFCS analysis step."""

import os
import sys
import argparse
import numpy as np

SRC_DIR_DEF = "data/fmri/aparc/HOI_GSR/"
DST_DIR_DEF = "data/fmri/aparc/HOI_GSR_Z/"
EXPECTED_SHAPE = (68, 68, 68)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def zscore_save_npy(
    src_path,
    dst_path,
    expected_shape=EXPECTED_SHAPE,
    use_mmap=True,
    chunk_elems=50_000_000,
):
    """Script documentation for this SFCS analysis step."""
    # load with mmap to avoid full memory load (read-only)
    try:
        if use_mmap:
            arr = np.load(src_path, mmap_mode="r")
        else:
            arr = np.load(src_path, allow_pickle=False)
    except Exception as e:
        return False, f"load_error: {e }"

        # check shape
    try:
        shape = tuple(arr.shape)
    except Exception as e:
        return False, f"shape_error: {e }"

    if shape != expected_shape:
        return False, f"shape_mismatch: expected {expected_shape }, got {shape }"

        # compute global mean and std in numerically stable, chunked way
        # We'll compute mean and std using two-pass or Welford-like algorithm.
        # Approach: compute sum and sumsq in chunks to get mean and variance.
    total_elems = int(np.prod(shape))
    dtype = arr.dtype

    # If dtype is integer, convert to float64 for mean/std computation
    # For float32, compute in float64 for stability.
    compute_dtype = np.float64

    # chunk strategy: iterate along first axis
    axis0 = shape[0] if len(shape) > 0 else 1
    # estimate elements per row
    elems_per_row = total_elems // axis0 if axis0 > 0 else total_elems
    rows_per_chunk = max(1, int(chunk_elems // max(1, elems_per_row)))
    # limit rows_per_chunk to axis0
    rows_per_chunk = min(axis0, rows_per_chunk)

    # First pass: compute sum and sumsq
    total_sum = np.float64(0.0)
    total_sumsq = np.float64(0.0)
    counted = 0

    for start in range(0, axis0, rows_per_chunk):
        stop = min(axis0, start + rows_per_chunk)
        try:
            block = arr[start:stop].astype(compute_dtype, copy=False)
        except Exception as e:
            return False, f"read_block_error at rows {start }:{stop } : {e }"
            # flatten block
        b = block.ravel()
        total_sum += b.sum(dtype=compute_dtype)
        total_sumsq += (b * b).sum(dtype=compute_dtype)
        counted += b.size

    if counted != total_elems:
        return False, f"count_mismatch: counted {counted } != total {total_elems }"

    mean = total_sum / total_elems
    # variance = E[x^2] - mean^2 (population variance)
    var = (total_sumsq / total_elems) - (mean * mean)
    # numerical issues: var might be slightly negative -> clip to 0
    if var < 0 and var > -1e-12:
        var = 0.0
    if var < 0:
        return False, f"negative_variance: {var }"
    std = np.sqrt(var)
    if std == 0:
        # all elements identical; result will be zeros
        std = 0.0

        # Second pass: write standardized data to dst_path
        # We'll create destination as a temporary file first then replace to avoid corruption
    tmp_dst = dst_path + ".tmp"
    try:
        # create parent dir if needed
        ensure_dir(os.path.dirname(dst_path))
        # We'll build the full output array in chunks and save to a memmapped .npy file:
        # Create an empty memmap for destination with same shape and dtype float32 (or float64)
        out_dtype = (
            np.float32
        )  # choose float32 to save space; change to float64 if you prefer
        # Use numpy.lib.format.open_memmap to write .npy file incrementally
        from numpy.lib.format import open_memmap

        out_arr = open_memmap(tmp_dst, mode="w+", dtype=out_dtype, shape=shape)
    except Exception as e:
        return False, f"create_output_error: {e }"

    try:
        if std == 0.0:
            # all values same -> standardized array is zeros
            out_arr[:] = 0.0
        else:
            # fill out_arr in same chunked manner
            for start in range(0, axis0, rows_per_chunk):
                stop = min(axis0, start + rows_per_chunk)
                try:
                    block = arr[start:stop].astype(compute_dtype, copy=False)
                except Exception as e:
                    # clean up tmp file
                    try:
                        del out_arr
                        os.remove(tmp_dst)
                    except Exception:
                        pass
                    return (
                        False,
                        f"read_block_error during second pass at {start }:{stop } : {e }",
                    )
                    # zscore: (x - mean) / std
                standardized = ((block - mean) / std).astype(out_dtype, copy=False)
                out_arr[start:stop] = standardized
                # ensure data flushed
        del out_arr
        # move tmp to final dst (atomic on same filesystem)
        os.replace(tmp_dst, dst_path)
    except Exception as e:
        # cleanup
        try:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
        except Exception:
            pass
        return False, f"write_error: {e }"

    return True, f"ok mean={mean :.6g} std={std :.6g}"


def main():
    parser = argparse.ArgumentParser(description="Processing message.")
    parser.add_argument(
        "--src", default=SRC_DIR_DEF, help="Processing message." + SRC_DIR_DEF
    )
    parser.add_argument(
        "--dst", default=DST_DIR_DEF, help="Processing message." + DST_DIR_DEF
    )
    parser.add_argument("--recursive", action="store_true", help="Processing message.")
    parser.add_argument("--no-mmap", action="store_true", help="Processing message.")
    parser.add_argument(
        "--chunk-elems", type=int, default=50_000_000, help="Processing message."
    )
    parser.add_argument("--dry-run", action="store_true", help="Processing message.")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)

    if not os.path.isdir(src_dir):
        print(f"Error during processing.", file=sys.stderr)
        sys.exit(2)
    ensure_dir(dst_dir)

    # collect files
    file_list = []
    if args.recursive:
        for dp, dns, fns in os.walk(src_dir):
            for fn in fns:
                if fn.lower().endswith(".npy"):
                    rel = os.path.relpath(dp, src_dir)
                    src_path = os.path.join(dp, fn)
                    dst_subdir = os.path.join(dst_dir, rel) if rel != "." else dst_dir
                    dst_path = os.path.join(dst_subdir, fn)
                    file_list.append((src_path, dst_path))
    else:
        for fn in os.listdir(src_dir):
            if fn.lower().endswith(".npy"):
                src_path = os.path.join(src_dir, fn)
                dst_path = os.path.join(dst_dir, fn)
                file_list.append((src_path, dst_path))

    if not file_list:
        print("No .npy files found in", src_dir)
        return

    print(
        f"Found {len (file_list )} .npy files to check (recursive={args .recursive })."
    )
    errs = []
    for i, (src_path, dst_path) in enumerate(file_list, 1):
        print(
            f"[{i }/{len (file_list )}] {src_path } -> {dst_path } ...",
            end=" ",
            flush=True,
        )
        if args.dry_run:
            print("dry-run: skipped")
            continue
        ok, msg = zscore_save_npy(
            src_path,
            dst_path,
            expected_shape=EXPECTED_SHAPE,
            use_mmap=not args.no_mmap,
            chunk_elems=args.chunk_elems,
        )
        if ok:
            print("DONE:", msg)
        else:
            print("FAILED:", msg)
            errs.append((src_path, msg))

    if errs:
        print("\nSome files failed:")
        for p, m in errs:
            print(" -", p, ":", m)
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()

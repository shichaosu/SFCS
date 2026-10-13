#!/usr/bin/env python3
# Prediction or table-preparation helper for the SFCS workflow.
# Update the path variables below before running this script.
from pathlib import Path
from openpyxl import load_workbook, Workbook

BASE_DIR = Path("data/data_use/table/result_CN")


def _read_summary(path: Path):
    wb = load_workbook(path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    header_map = {}
    for idx, h in enumerate(headers, start=1):
        if h is None:
            continue
        header_map[str(h).strip()] = idx

    required = ["Metric", "Pearson_r", "P_perm"]
    missing = [r for r in required if r not in header_map]
    if missing:
        raise ValueError(f"Missing columns {missing } in {path }")

    metrics_order = []
    data = {}
    for row in range(2, ws.max_row + 1):
        metric = ws.cell(row, header_map["Metric"]).value
        if metric is None or str(metric).strip() == "":
            continue
        metric_str = str(metric).strip()
        pearson = ws.cell(row, header_map["Pearson_r"]).value
        p_perm = ws.cell(row, header_map["P_perm"]).value
        metrics_order.append(metric_str)
        data[metric_str] = {"Pearson_r": pearson, "P_perm": p_perm}

    return metrics_order, data


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _read_roi_column(path: Path, metric: str):
    wb = load_workbook(path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]

    roi_col = None
    metric_col = None
    for idx, h in enumerate(headers, start=1):
        if h is None:
            continue
        h_str = str(h).strip()
        if h_str == "ROI_Index":
            roi_col = idx
        if h_str == metric:
            metric_col = idx

    if roi_col is None:
        roi_col = 1

    data = {}
    if metric_col is None:
        return data

    for row in range(2, ws.max_row + 1):
        roi_idx = ws.cell(row, roi_col).value
        if roi_idx is None or str(roi_idx).strip() == "":
            continue
        data[roi_idx] = ws.cell(row, metric_col).value

    return data


def _sorted_roi_indices(indices):
    def key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))

    return sorted(indices, key=key)


def process_group(group: str):
    group_dir = [
        BASE_DIR / "orign" / group / "20Coefficient",
        # BASE_DIR / "orign" / group / "20Coefficient_absolute",
        BASE_DIR / "trans" / group / "20Coefficient",
        # BASE_DIR / "trans" / group / "20Coefficient_absolute",
    ]

    summary_paths = [d / "BrainGNN_Orign_Summary.xlsx" for d in group_dir]
    roi_paths = [d / "ROI_orign_Importance_Rank.xlsx" for d in group_dir]

    for p in summary_paths + roi_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p }")

    summaries = []
    for p in summary_paths:
        metrics_order, data = _read_summary(p)
        summaries.append((metrics_order, data))

        # intersection of metrics across 4 files
    metric_sets = [set(mo) for mo, _ in summaries]
    common_metrics = set.intersection(*metric_sets)
    metrics_order = [m for m in summaries[0][0] if m in common_metrics]

    best_info = {}
    handled = set()
    for metric in metrics_order:
        if metric in handled:
            continue
            # If this is an EcogPt metric with a matching EcogSP, skip here.
        if metric.startswith("EcogPt") and f"EcogSP{metric [6 :]}" in common_metrics:
            continue
            # Handle EcogSP/EcogPt pairs: choose table by max EcogSP, and take EcogPt from same table.
        if metric.startswith("EcogSP"):
            suffix = metric[6:]
            pair_metric = f"EcogPt{suffix }"
            if pair_metric in common_metrics:
                best_idx = None
                best_val = None
                for i, (_, data) in enumerate(summaries):
                    val = _to_float(data.get(metric, {}).get("Pearson_r"))
                    if val is None:
                        continue
                    if best_val is None or val > best_val:
                        best_val = val
                        best_idx = i
                if best_idx is None:
                    continue
                p_param_sp = summaries[best_idx][1][metric].get("P_perm")
                best_info[metric] = {
                    "Pearson_r": best_val,
                    "P_perm": p_param_sp,
                    "dir_idx": best_idx,
                }
                pt_data = summaries[best_idx][1].get(pair_metric, {})
                best_info[pair_metric] = {
                    "Pearson_r": _to_float(pt_data.get("Pearson_r")),
                    "P_perm": pt_data.get("P_perm"),
                    "dir_idx": best_idx,
                }
                handled.add(metric)
                handled.add(pair_metric)
                continue

        best_idx = None
        best_val = None
        for i, (_, data) in enumerate(summaries):
            val = _to_float(data.get(metric, {}).get("Pearson_r"))
            if val is None:
                continue
            if best_val is None or val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            continue
        p_param = summaries[best_idx][1][metric].get("P_perm")
        best_info[metric] = {
            "Pearson_r": best_val,
            "P_perm": p_param,
            "dir_idx": best_idx,
        }
        handled.add(metric)

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "sheet1"
    ws1.append(["Metric", "Pearson_r", "P_perm"])
    for metric in metrics_order:
        info = best_info.get(metric)
        if not info:
            continue
        ws1.append([metric, info["Pearson_r"], info["P_perm"]])

    ws2 = wb.create_sheet("sheet2")
    metric_list = [m for m in metrics_order if m in best_info]
    ws2.append(["ROI_Index"] + metric_list)

    roi_values = {}
    roi_index_set = set()
    for metric in metric_list:
        idx = best_info[metric]["dir_idx"]
        roi_data = _read_roi_column(roi_paths[idx], metric)
        roi_values[metric] = roi_data
        roi_index_set.update(roi_data.keys())

    for roi_idx in _sorted_roi_indices(roi_index_set):
        row = [roi_idx]
        for metric in metric_list:
            row.append(roi_values.get(metric, {}).get(roi_idx))
        ws2.append(row)

    out_path = BASE_DIR / f"perm_{group .lower ()}.xlsx"
    wb.save(out_path)
    print(f"Saved {out_path }")


def main():
    for group in ["Pos", "Neg"]:
        process_group(group)


if __name__ == "__main__":
    main()

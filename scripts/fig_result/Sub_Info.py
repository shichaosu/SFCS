# Figure-generation helper for the SFCS manuscript.
# Update the path variables below before running this script.
import os
import pandas as pd
import numpy as np
from pathlib import Path

CN_DIR = Path("data/data_use/Coefficient/20_Pos/CN")
AD_DIR = Path("data/data_use/Coefficient/20_Pos/AD")
EXCEL_PATH = Path("data/subject_info.xlsx")
SHEET_NAME = "fmri"
OUT_TXT = Path("data/data_use/CN_AD_statistics.txt")


def find_column(df, candidates, required=True):
    cols = list(df.columns)
    norm = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in cols}

    for cand in candidates:
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]

    for c in cols:
        c_norm = str(c).strip().lower().replace(" ", "").replace("_", "")
        for cand in candidates:
            cand_norm = cand.strip().lower().replace(" ", "").replace("_", "")
            if cand_norm in c_norm or c_norm in cand_norm:
                return c

    if required:
        raise ValueError(f"Processing message." f"Processing message.")
    return None


def get_folder_ids(data_dir):
    files = sorted(data_dir.glob("*.npy"))
    folder_ids = [f.stem for f in files]
    subject_ids = [x[:10] for x in folder_ids]
    return pd.DataFrame({"FolderID": folder_ids, "SubjectID_from_file": subject_ids})


def summarize_group(
    group_name, file_df, info_df, folder_col, sex_col, age_col, year_col
):
    merged = file_df.merge(info_df, left_on="FolderID", right_on=folder_col, how="left")

    scan_count = len(file_df)
    subject_count = file_df["SubjectID_from_file"].nunique()
    matched_count = merged[folder_col].notna().sum()
    unmatched_count = merged[folder_col].isna().sum()

    lines = []
    lines.append("=" * 60)
    lines.append(f"Processing message.")
    lines.append("=" * 60)
    lines.append(f"Processing message.")
    lines.append(f"Processing message.")
    lines.append(f"Processing message.")
    lines.append(f"Processing message.")
    lines.append("")

    lines.append("Processing message.")
    if sex_col is not None:
        sex_series = merged[sex_col].dropna().astype(str)
        if len(sex_series) > 0:
            sex_counts = sex_series.value_counts(dropna=False)
            for sex, count in sex_counts.items():
                ratio = count / len(sex_series) * 100
                lines.append(f"  {sex }: {count } ({ratio :.2f}%)")
        else:
            lines.append("Processing message.")
    else:
        lines.append("Processing message.")
    lines.append("")

    lines.append("Processing message.")
    if age_col is not None:
        age = pd.to_numeric(merged[age_col], errors="coerce").dropna()
        if len(age) > 0:
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
            lines.append(f"Processing message.")
        else:
            lines.append("Processing message.")
    else:
        lines.append("Processing message.")
    lines.append("")

    lines.append("Processing message.")
    if year_col is not None:
        year = merged[year_col].dropna()
        year_num = pd.to_numeric(year, errors="coerce")

        if year_num.notna().sum() > 0:
            year_series = year_num.dropna().astype(int).astype(str)
        else:
            year_series = year.astype(str)

        if len(year_series) > 0:
            year_counts = year_series.value_counts().sort_index()
            for y, count in year_counts.items():
                ratio = count / len(year_series) * 100
                lines.append(f"  {y }: {count } ({ratio :.2f}%)")
        else:
            lines.append("Processing message.")
    else:
        lines.append("Processing message.")
    lines.append("")

    return lines


def main():
    cn_files = get_folder_ids(CN_DIR)
    ad_files = get_folder_ids(AD_DIR)

    info = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    info.columns = [str(c).strip() for c in info.columns]

    folder_col = find_column(
        info,
        ["FolderID", "Folder ID", "Folder", "ScanID", "Scan ID", "ImageID", "Image ID"],
    )

    sex_col = find_column(info, ["Sex", "Gender", "PTGENDER"], required=False)

    age_col = find_column(info, ["Age", "PTAGE"], required=False)

    year_col = find_column(info, ["Year", "ScanYear", "Scan Year"], required=False)
    if year_col is None:
        date_col = find_column(
            info,
            [
                "ScanDate",
                "Scan Date",
                "StudyDate",
                "Study Date",
                "EXAMDATE",
                "Acq Date",
                "AcquisitionDate",
            ],
            required=False,
        )
        if date_col is not None:
            info["__Year_from_date__"] = pd.to_datetime(
                info[date_col], errors="coerce"
            ).dt.year
            year_col = "__Year_from_date__"

    output_lines = []
    output_lines.append("Processing message.")
    output_lines.append("")
    output_lines.append(f"Processing message.")
    output_lines.append(f"Processing message.")
    output_lines.append(f"Processing message.")
    output_lines.append(f"Sheet: {SHEET_NAME }")
    output_lines.append("")
    output_lines.append(f"Processing message.")
    output_lines.append(f"Processing message.")
    output_lines.append(f"Processing message.")
    output_lines.append(f"Processing message.")
    output_lines.append("")

    output_lines.extend(
        summarize_group("CN", cn_files, info, folder_col, sex_col, age_col, year_col)
    )
    output_lines.extend(
        summarize_group("AD", ad_files, info, folder_col, sex_col, age_col, year_col)
    )

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(output_lines), encoding="utf-8")

    print(f"Processing message.")


if __name__ == "__main__":
    main()

# Coefficient summary, group-statistics, or visualization helper.
# Update the path variables below before running this script.
import os
import shutil
import pandas as pd

source_root_dir = "data/SC/aparc/communication/SC/"
excel_path = "data/subject_info.xlsx"


def normalize_id(value):
    """Script documentation for this SFCS analysis step."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "nan":
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_id_group_map():
    """Script documentation for this SFCS analysis step."""
    print(f"Processing message.")
    try:
        df = pd.read_excel(excel_path, sheet_name="fmri")
    except Exception as e:
        print(f"Processing message.")
        return None

    id_to_group = {}
    for index, row in df.iterrows():
        folder_id = normalize_id(row.iloc[1])
        group_name = normalize_id(row.iloc[6])
        if not folder_id or not group_name:
            continue
        id_to_group[folder_id] = group_name

    print(f"Processing message.")
    return id_to_group


def sort_files_by_group_in_subfolders():
    id_to_group = load_id_group_map()
    if not id_to_group:
        return

    if not os.path.isdir(source_root_dir):
        print(f"Processing message.")
        return

    subfolders = []
    for name in sorted(os.listdir(source_root_dir)):
        path = os.path.join(source_root_dir, name)
        if os.path.isdir(path):
            subfolders.append(path)

    if not subfolders:
        print(f"Processing message.")
        return

    total_success = 0
    total_unmapped = 0
    total_skipped = 0
    print(f"Processing message.")

    for subfolder in subfolders:
        count_success = 0
        count_unmapped = 0
        count_skipped = 0
        print(f"Processing message.")

        for file_name in sorted(os.listdir(subfolder)):
            src_file_path = os.path.join(subfolder, file_name)
            if not os.path.isfile(src_file_path):
                continue

            file_stem, _ = os.path.splitext(file_name)
            folder_id = normalize_id(file_stem)
            if not folder_id:
                count_skipped += 1
                continue

            group_name = id_to_group.get(folder_id)
            if not group_name:
                count_unmapped += 1
                continue

            dst_dir_path = os.path.join(subfolder, group_name)
            dst_file_path = os.path.join(dst_dir_path, file_name)

            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)
                print(f"Processing message.")

            try:
                if os.path.exists(dst_file_path):
                    os.remove(dst_file_path)
                shutil.move(src_file_path, dst_file_path)
                count_success += 1
            except Exception as e:
                print(f"Processing message.")

        total_success += count_success
        total_unmapped += count_unmapped
        total_skipped += count_skipped
        print(f"Processing message.")

    print("\n" + "=" * 30)
    print("Processing message.")
    print(f"Processing message.")
    print(f"Processing message.")
    print(f"Processing message.")
    print(f"Processing message.")


if __name__ == "__main__":
    sort_files_by_group_in_subfolders()

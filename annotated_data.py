import numpy as np
import pandas as pd
import chardet
import os
import pickle

data_path = "./data/ecg_ppg_signals"
table_path = "./data/测量记录表复核.csv"
table = pd.read_csv(table_path, encoding="utf-8")
table.dropna(axis=0, how="any", inplace=True)
table = table.sort_values(by="人员编号")
table["人员编号"] = table["人员编号"].astype("Int64").astype(str)
table["date"] = table["人员编号"].astype(str).str[:8]
df = pd.DataFrame(
    columns=[
        "id",
        "sex",
        "age",
        "ecg_b1",
        "ppg_b1",
        "ecg_b2",
        "ppg_b2",
        "ecg_g1",
        "ppg_g1",
        "ecg_g2",
        "ppg_g2",
        "sbp_b1",
        "sbp_b2",
        "sbp_g1",
        "sbp_g2",
        "dbp_b1",
        "dbp_b2",
        "dbp_g1",
        "dbp_g2",
        "file_b1",
        "file_b2",
        "file_g1",
        "file_g2",
    ]
)


def read_lines_auto_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read()
        encoding = chardet.detect(raw)["encoding"]
    with open(file_path, encoding=encoding) as f:
        lines = f.readlines()
    return lines


def safe_str_to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


for index, row in table.iterrows():
    id = str(row["人员编号"])
    date = id[:4] + "-" + id[4:6] + "-" + id[6:8]

    time_b1 = str(int(row["蓝色文件后八位(1)"])).zfill(6)
    file_b1 = (
        date
        + " "
        + time_b1[:2]
        + "_"
        + time_b1[2:4]
        + "_"
        + time_b1[4:6]
        + "_b"
        + ".csv"
    )
    path_b1 = os.path.join(data_path, id, file_b1)
    time_b2 = str(int(row["蓝色文件后八位(2)"])).zfill(6)
    file_b2 = (
        date
        + " "
        + time_b2[:2]
        + "_"
        + time_b2[2:4]
        + "_"
        + time_b2[4:6]
        + "_b"
        + ".csv"
    )
    path_b2 = os.path.join(data_path, id, file_b2)
    time_g1 = str(int(row["金色文件后八位(1)"])).zfill(6)
    file_g1 = (
        date
        + " "
        + time_g1[:2]
        + "_"
        + time_g1[2:4]
        + "_"
        + time_g1[4:6]
        + "_g"
        + ".csv"
    )
    path_g1 = os.path.join(data_path, id, file_g1)
    time_g2 = str(int(row["金色文件后八位(2)"])).zfill(6)
    file_g2 = (
        date
        + " "
        + time_g2[:2]
        + "_"
        + time_g2[2:4]
        + "_"
        + time_g2[4:6]
        + "_g"
        + ".csv"
    )
    path_g2 = os.path.join(data_path, id, file_g2)

    if not (
        os.path.exists(path_b1)
        and os.path.exists(path_b2)
        and os.path.exists(path_g1)
        and os.path.exists(path_g2)
    ):
        continue

    sbp_b1 = row["DC_血压仪高压B1"]
    dbp_b1 = row["DC_血压仪低压B1"]
    sbp_b2 = row["DC_血压仪高压B2"]
    dbp_b2 = row["DC_血压仪低压B2"]
    sbp_g1 = row["DC_血压仪高压G1"]
    dbp_g1 = row["DC_血压仪低压G1"]
    sbp_g2 = row["DC_血压仪高压G2"]
    dbp_g2 = row["DC_血压仪低压G2"]
    sex = row["性别"]
    age = row["年龄"]

    bp_pairs = [
        ("DC_血压仪高压B1", "DC_血压仪低压B1"),
        ("DC_血压仪高压B2", "DC_血压仪低压B2"),
        ("DC_血压仪高压G1", "DC_血压仪低压G1"),
        ("DC_血压仪高压G2", "DC_血压仪低压G2"),
    ]

    for high_col, low_col in bp_pairs:
        if pd.notnull(sbp_b1) and pd.notnull(dbp_b1) and dbp_b1 > sbp_b1:
            sbp_b1, dbp_b1 = dbp_b1, sbp_b1
        if pd.notnull(sbp_b2) and pd.notnull(dbp_b2) and dbp_b2 > sbp_b2:
            sbp_b2, dbp_b2 = dbp_b2, sbp_b2
        if pd.notnull(sbp_g1) and pd.notnull(dbp_g1) and dbp_g1 > sbp_g1:
            sbp_g1, dbp_g1 = dbp_g1, sbp_g1
        if pd.notnull(sbp_g2) and pd.notnull(dbp_g2) and dbp_g2 > sbp_g2:
            sbp_g2, dbp_g2 = dbp_g2, sbp_g2

    lines_b1 = read_lines_auto_encoding(path_b1)
    ecg_b1 = np.array(
        [safe_str_to_float(x) for x in lines_b1[3].strip().split(",")], dtype=float
    )
    ppg_b1 = np.array(
        [safe_str_to_float(x) for x in lines_b1[5].strip().split(",")], dtype=float
    )

    lines_b2 = read_lines_auto_encoding(path_b2)
    ecg_b2 = np.array(
        [safe_str_to_float(x) for x in lines_b2[3].strip().split(",")], dtype=float
    )
    ppg_b2 = np.array(
        [safe_str_to_float(x) for x in lines_b2[5].strip().split(",")], dtype=float
    )

    lines_g1 = read_lines_auto_encoding(path_g1)
    ecg_g1 = np.array(
        [safe_str_to_float(x) for x in lines_g1[3].strip().split(",")], dtype=float
    )
    ppg_g1 = np.array(
        [safe_str_to_float(x) for x in lines_g1[5].strip().split(",")], dtype=float
    )

    lines_g2 = read_lines_auto_encoding(path_g2)
    ecg_g2 = np.array(
        [safe_str_to_float(x) for x in lines_g2[3].strip().split(",")], dtype=float
    )
    ppg_g2 = np.array(
        [safe_str_to_float(x) for x in lines_g2[5].strip().split(",")], dtype=float
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "id": [id],
                    "sex": [sex],
                    "age": [age],
                    "ecg_b1": [ecg_b1],
                    "ppg_b1": [ppg_b1],
                    "ecg_b2": [ecg_b2],
                    "ppg_b2": [ppg_b2],
                    "ecg_g1": [ecg_g1],
                    "ppg_g1": [ppg_g1],
                    "ecg_g2": [ecg_g2],
                    "ppg_g2": [ppg_g2],
                    "sbp_b1": [sbp_b1],
                    "sbp_b2": [sbp_b2],
                    "sbp_g1": [sbp_g1],
                    "sbp_g2": [sbp_g2],
                    "dbp_b1": [dbp_b1],
                    "dbp_b2": [dbp_b2],
                    "dbp_g1": [dbp_g1],
                    "dbp_g2": [dbp_g2],
                    "file_b1": [file_b1],
                    "file_b2": [file_b2],
                    "file_g1": [file_g1],
                    "file_g2": [file_g2],
                }
            ),
        ],
        ignore_index=True,
    )

new_rows = []
for _, row in df.iterrows():
    for suffix, card in zip(['b1', 'b2', 'g1', 'g2'], ['b', 'b', 'g', 'g']):
        new_rows.append({
            "id": row["id"],
            "file": row[f"file_{suffix}"],
            "sex": row["sex"],
            "age": row["age"],
            "card": card,
            "ppg": row[f"ppg_{suffix}"],
            "ecg": row[f"ecg_{suffix}"],
            "sbp": row[f"sbp_{suffix}"],
            "dbp": row[f"dbp_{suffix}"],
        })

df_long = pd.DataFrame(new_rows)

# 打开标注文件
annotations_table = pd.read_csv("./data/信号标记.csv")
annotations_table["file_name"] = annotations_table["file_name"].astype(str)

seg_rows = []
for _, row in df_long.iterrows():
    ann = annotations_table[annotations_table["file_name"] == row["file"]]
    for _, seg in ann.iterrows():
        ecg_seg = row["ecg"][seg["start_idx"]:seg["end_idx"]]
        ppg_seg = row["ppg"][seg["start_idx"]:seg["end_idx"]]
        seg_rows.append({
            "id": row["id"],
            "file": row["file"],
            "sex": row["sex"],
            "age": row["age"],
            "card": row["card"],
            "start_idx": seg["start_idx"],
            "end_idx": seg["end_idx"],
            "ecg": ecg_seg,
            "ppg": ppg_seg,
            "sbp": row["sbp"],
            "dbp": row["dbp"],
            "ecg_label": seg["ecg"],
            "ppg_label": seg["ppg"],
        })

df_segmented = pd.DataFrame(seg_rows)
df_segmented.to_pickle("./data/超思10s标注数据.pkl")

# df_segmented_2_rows = []
# for _, row in df_long.iterrows():
#     ann = annotations_table[annotations_table["file_name"] == row["file"]].reset_index(drop=True)
#     ecg_labels = [seg["ecg"] for _, seg in ann.iterrows()]
#     ppg_labels = [seg["ppg"] for _, seg in ann.iterrows()]
#     # 生成列名
#     ecg_label_cols = {f"ecg_label_{i}": v for i, v in enumerate(ecg_labels)}
#     ppg_label_cols = {f"ppg_label_{i}": v for i, v in enumerate(ppg_labels)}
#     # 合并到一行
#     row_dict = {
#         "id": row["id"],
#         "file": row["file"],
#         "sex": row["sex"],
#         "age": row["age"],
#         "card": row["card"],
#         "ppg": row["ppg"],
#         "ecg": row["ecg"],
#         "sbp": row["sbp"],
#         "dbp": row["dbp"],
#     }
#     row_dict.update(ecg_label_cols)
#     row_dict.update(ppg_label_cols)
#     df_segmented_2_rows.append(row_dict)

# df_segmented_2 = pd.DataFrame(df_segmented_2_rows)
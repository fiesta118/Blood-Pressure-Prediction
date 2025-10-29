def extract_data_from_folder(folder_path):
    """
    读取文件夹内所有csv文件，提取ecg和ppg信号，返回DataFrame（全部转为np.array格式）。

    参数:
        folder_path (str): 包含csv文件的文件夹路径

    返回:
        pd.DataFrame: 包含file、ecg、ppg三列，ecg和ppg为np.array
    """
    import os
    import pandas as pd
    import numpy as np

    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    result = []
    for file in csv_files:
        df = pd.read_csv(file, index_col=0)
        ecg = np.array(df.loc["ecg"].values)
        ppg = np.array(df.loc["ppg"].values)
        result.append({"file": os.path.basename(file), "ecg": ecg, "ppg": ppg})

    return pd.DataFrame(result)

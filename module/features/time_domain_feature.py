import numpy as np


def ssqi(sig: np.array) -> float:
    """
    计算信号的偏度（Skewness Signal Quality Index, SSQI）。

    Args:
        sig (np.ndarray): 输入信号

    Returns:
        float: 偏度指标，反映信号分布的非对称性
    """
    num = np.mean((sig - np.mean(sig)) ** 3)
    s_sqi = num / (np.std(sig, ddof=1) ** 3)
    s_sqi_score = float(round(s_sqi, 3))
    return s_sqi_score


def ksqi(sig: np.array) -> float:
    """
    计算信号的峰度（Kurtosis Signal Quality Index, KSQI）。

    Args:
        sig (np.ndarray): 输入信号

    Returns:
        float: 峰度指标，反映信号分布的尖锐程度（已减去正态分布的3）
    """
    num = np.mean((sig - np.mean(sig)) ** 4)
    k_sqi = num / (np.std(sig, ddof=1) ** 4)
    k_sqi_fischer = k_sqi - 3.0
    k_sqi_score = float(round(k_sqi_fischer, 3))
    return k_sqi_score

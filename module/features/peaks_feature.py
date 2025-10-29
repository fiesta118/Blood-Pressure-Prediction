import numpy as np
import neurokit2 as nk
from module.filter.ecg_filter import ecg_clean


def ppg_find_peaks(ppg, sampling_rate, return_sig=False):
    """对PPG信号进行去噪后检测峰值

    Args:
        ppg (_type_): _description_
        sampling_rate (_type_): _description_
        return_sig (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
    info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate, method="bishop")
    peaks = info["PPG_Peaks"]
    if return_sig:
        return peaks, ppg_cleaned
    else:
        return peaks, None


def ecg_find_peaks(ecg, sampling_rate=500, return_sig=False, **kwargs):
    """
    对ECG信号进行去噪后检测R波峰值

    Args:
        ecg (np.ndarray): 输入ECG信号
        sampling_rate (int): 采样率，默认500Hz
        return_sig (bool): 是否返回去噪信号，默认False
        **kwargs: 传递给ecg_clean的其他参数，如cutoff, order, wavelet, threshold

    Returns:
        peaks (np.ndarray): R波峰索引
        ecg_cleaned (np.ndarray or None): 去噪信号（如果return_sig为True则返回，否则为None）
    """
    ecg_cleaned = ecg_clean(ecg, fs=sampling_rate, **kwargs)
    signals, info = nk.ecg_peaks(
        ecg_cleaned, sampling_rate=sampling_rate, method="neurokit"
    )
    peaks = info["ECG_R_Peaks"]
    if return_sig:
        return peaks, ecg_cleaned
    else:
        return peaks, None
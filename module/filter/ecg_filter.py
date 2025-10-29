import pywt
from scipy.signal import butter, filtfilt


def adb_2_ecg(adb):
    """将adb信号转换为ECG信号。

    Args:
        adb (np.ndarray): 原始adb信号

    Returns:
        np.ndarray: 转换后的ECG信号
    """
    VREF = 2.42
    G = 6
    ecg = adb * 1000 * VREF / (32767 * G)
    return ecg


def highpass_filter(sig, fs=500, cutoff=5, order=4):
    """高通滤波器，去除低频干扰。

    Args:
        sig (np.ndarray): 输入信号
        fs (int): 采样率，默认500Hz
        cutoff (float): 截止频率，默认5Hz
        order (int): 滤波器阶数，默认4

    Returns:
        np.ndarray: 滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high")
    filtered_sig = filtfilt(b, a, sig)
    return filtered_sig


def wt_filter(sig, wavelet="db4", threshold=0.1):
    """小波去噪

    Args:
        sig (np.ndarray): 输入信号
        wavelet (str, optional): 小波基，默认"db4"
        threshold (float, optional): 阈值，默认0.1

    Returns:
        np.ndarray: 去噪后的信号
    """
    coeffs = pywt.wavedec(sig, wavelet)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)


def ecg_clean(sig, fs=500, cutoff=5, order=4, wavelet="db4", threshold=0.1, **kwargs):
    """先高通滤波，再小波去噪，得到清洗后的ECG信号。

    Args:
        sig (np.ndarray): 输入信号
        fs (int): 采样率，默认500Hz
        cutoff (float): 高通滤波截止频率，默认5Hz
        order (int): 高通滤波器阶数，默认4
        wavelet (str): 小波基，默认"db4"
        threshold (float): 小波阈值，默认0.1

    Returns:
        np.ndarray: 清洗后的ECG信号
    """
    fs = kwargs.get("fs", fs)
    cutoff = kwargs.get("cutoff", cutoff)
    order = kwargs.get("order", order)
    wavelet = kwargs.get("wavelet", wavelet)
    threshold = kwargs.get("threshold", threshold)
    filtered_sig = highpass_filter(sig, fs=fs, cutoff=cutoff, order=order)
    denoised_sig = wt_filter(filtered_sig, wavelet=wavelet, threshold=threshold)
    return denoised_sig

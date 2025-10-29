import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    sig: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 8, order: int = 2
):
    """对输入信号应用带通滤波器。

    参数:
        sig (np.ndarray): 输入信号
        fs (int): 信号采样率
        lowcut (float): 带通滤波器的低截止频率
        highcut (float): 带通滤波器的高截止频率
        order (int): 滤波器阶数，默认2

    返回:
        sig_filtered (np.ndarray): Butterworth带通滤波后信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    sig_filtered = filtfilt(b, a, sig)
    return sig_filtered


def ppg_clean(sig, fs=500, lowcut=0.5, highcut=8, order=2, **kwargs):
    """
    对PPG信号进行带通滤波清洗。

    参数:
        sig (np.ndarray): 输入信号
        fs (int): 信号采样率，默认500Hz
        lowcut (float): 带通滤波器低截止频率，默认0.5Hz
        highcut (float): 带通滤波器高截止频率，默认8Hz
        order (int): 滤波器阶数，默认2
        **kwargs: 其他参数（可用于扩展）

    返回:
        np.ndarray: 清洗后的PPG信号
    """
    fs = kwargs.get("fs", fs)
    lowcut = kwargs.get("lowcut", lowcut)
    highcut = kwargs.get("highcut", highcut)
    order = kwargs.get("order", order)
    return bandpass_filter(sig, fs=fs, lowcut=lowcut, highcut=highcut, order=order)

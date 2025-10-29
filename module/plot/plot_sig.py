import os
import numpy as np
import matplotlib.pyplot as plt


def plot_signal(
    sig: np.array,
    sampling_rate,
    signal_type="ecg",
    filter_signal=True,
    savefig=False,
    save_path=None,
    savename=None,
    **kwargs
):
    """
    绘制信号质量评估结果。

    Args:
        sig (np.ndarray): 输入信号
        samplingrate (int): 采样率
        filter_signal (bool): 是否进行去噪处理，默认True
        signal_type (str): 信号类型，可选"ecg"或"ppg"，默认"ecg"
        savefig (bool): 是否保存图片，默认False
        save_path (str): 图片保存路径，默认None
        savename (str): 图片文件名，默认None
        **kwargs: 其他参数
    """
    if signal_type == "ecg":
        return _plot_ecg_signal(
            sig, sampling_rate, filter_signal, savefig, save_path, savename, **kwargs
        )
    elif signal_type == "ppg":
        return _plot_ppg_signal(
            sig, sampling_rate, filter_signal, savefig, save_path, savename, **kwargs
        )


def _plot_ecg_signal(
    sig,
    sampling_rate,
    filter_signal=True,
    savefig=False,
    save_path=None,
    savename=None,
    **kwargs
):
    import matplotlib.pyplot as plt
    from module.filter.ecg_filter import ecg_clean, adb_2_ecg

    sig = adb_2_ecg(sig)
    t = np.arange(len(sig)) / sampling_rate

    plt.figure(figsize=(12, 4))
    if filter_signal:
        sig_clean = ecg_clean(sig, fs=sampling_rate, **kwargs)
        plt.plot(t, sig_clean[: len(sig)], color="blue", label="Denoised ECG")
        plt.title("去噪后ECG信号")
    else:
        plt.plot(t, sig, color="red", label="Raw ECG")
        plt.title("原始ECG信号")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    if savefig:
        if savename is not None:
            if save_path is not None:
                full_path = os.path.join(save_path, savename)
            else:
                full_path = savename
            plt.savefig(full_path, dpi=140)
        elif save_path is not None:
            plt.savefig(save_path, dpi=140)
    plt.show() if not savefig else plt.close()


def _plot_ppg_signal(
    sig,
    sampling_rate,
    filter_signal=True,
    savefig=False,
    save_path=None,
    savename=None,
    **kwargs
):
    import matplotlib.pyplot as plt
    from module.filter.ppg_filter import ppg_clean
    from module.utils.format import normalize_data

    sig = normalize_data(sig)
    t = np.arange(len(sig)) / sampling_rate

    plt.figure(figsize=(12, 4))
    if filter_signal:
        sig_clean = ppg_clean(sig, fs=sampling_rate, **kwargs)
        plt.plot(t, sig_clean[: len(sig)], color="blue", label="Denoised PPG")
        plt.title("去噪后PPG信号")
    else:
        plt.plot(t, sig, color="red", label="Raw PPG")
        plt.title("原始PPG信号")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    if savefig:
        if savename is not None:
            if save_path is not None:
                full_path = os.path.join(save_path, savename)
            else:
                full_path = savename
            plt.savefig(full_path, dpi=140)
        elif save_path is not None:
            plt.savefig(save_path, dpi=140)
    plt.show() if not savefig else plt.close()

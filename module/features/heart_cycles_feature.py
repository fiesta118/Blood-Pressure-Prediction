import numpy as np
import pandas as pd
from scipy import signal
from module.utils.format import normalize_data
from module.features.peaks_feature import ppg_find_peaks, ecg_find_peaks
import neurokit2 as nk


def _ppg_heart_cycle_detection(ppg, sampling_rate):
    """
    Extract heart cycles from the PPG signal.

    Parameters
    ----------
    ppg : array-like
        Input PPG signal.
    sampling_rate : int or float
        Sampling rate of the PPG signal.

    Returns
    -------
    hc : list of ndarray
        List of heart cycles.
    """
    # Normalization
    ppg_normalized = normalize_data(ppg)

    # Upsampling signal by 2
    sampling_rate = sampling_rate * 2
    ppg_upsampled = signal.resample(ppg_normalized, len(ppg_normalized) * 2)

    # Systolic peak detection
    peaks, ppg_cleaned = ppg_find_peaks(
        ppg=ppg_upsampled, sampling_rate=sampling_rate, return_sig=True
    )

    # Heart cycle detection based on the peaks and fixed intervals
    hc = []
    if len(peaks) < 2:
        return hc
    # Define a fixed interval in PPG signal to detect heart cycles
    beat_bound = round((len(ppg_upsampled) / len(peaks)) / 2)
    for i in range(1, len(peaks) - 1):
        beat_start = peaks[i] - beat_bound
        beat_end = peaks[i] + beat_bound
        if beat_start >= 0 and beat_end < len(ppg_cleaned):
            beat = ppg_cleaned[beat_start:beat_end]
            if len(beat) >= beat_bound * 2:
                hc.append(beat)
    return hc


def _ecg_heart_cycle_detection(ecg, sampling_rate):
    ecg_normalized = normalize_data(ecg)
    sampling_rate = sampling_rate * 2
    ecg_upsampled = signal.resample(ecg_normalized, len(ecg_normalized) * 2)
    peaks, ecg_cleaned = ecg_find_peaks(
        ecg=ecg_upsampled, sampling_rate=sampling_rate, return_sig=True
    )
    hc = []
    if len(peaks) < 2:
        return hc
    beat_bound = round((len(ecg_upsampled) / len(peaks)) / 2)
    for i in range(1, len(peaks) - 1):
        beat_start = peaks[i] - beat_bound
        beat_end = peaks[i] + beat_bound
        if beat_start >= 0 and beat_end < len(ecg_cleaned):
            beat = ecg_cleaned[beat_start:beat_end]
            if len(beat) >= beat_bound * 2:
                hc.append(beat)
    return hc


def template_matching_features(hc):
    """
    Extract template matching features from heart cycles.

    Parameters
    ----------
    hc : list of ndarray
        List of heart cycles.

    Returns
    -------
    tm_ave_eu : float
        Average Euclidean distance to the template.
    tm_ave_corr : float
        Average correlation with the template.
    """
    hc = np.array([np.array(xi) for xi in hc if len(xi) != 0])
    template = np.mean(hc, axis=0)
    distances = []
    corrs = []
    for beat in hc:
        distances.append(np.linalg.norm(template - beat))
        corr_matrix = np.corrcoef(template, beat)
        corrs.append(corr_matrix[0, 1])
    tm_ave_eu = np.mean(distances)
    tm_ave_corr = np.mean(corrs)
    return tm_ave_eu, tm_ave_corr


def calc_signal_metrics(seg, sampling_rate, signal_type):
    """
    计算ECG或PPG信号的心率、最大RR间期、RR比值和模板相关性等特征。

    参数:
        seg (np.ndarray): 输入信号片段
        sampling_rate (int): 采样率
        signal_type (str): "ecg" 或 "ppg"

    返回:
        pd.Series: [hr, rr_max, rr_ratio, template_corr]
    """
    if signal_type == "ppg":
        peaks, _ = ppg_find_peaks(seg, sampling_rate)
        hc = _ppg_heart_cycle_detection(seg, sampling_rate)
    elif signal_type == "ecg":
        peaks, _ = ecg_find_peaks(seg, sampling_rate)
        hc = _ecg_heart_cycle_detection(seg, sampling_rate)
    else:
        raise ValueError("signal_type 只能为 'ecg' 或 'ppg'")

    if len(peaks) < 2:
        return np.nan, np.nan, np.nan, np.nan

    rr_intervals = np.diff(peaks) / sampling_rate
    hr = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else np.nan
    rr_max = np.max(rr_intervals)
    rr_min = np.min(rr_intervals)
    rr_ratio = rr_max / rr_min if rr_min > 0 else np.nan

    if len(hc) > 0:
        _, template_corr = template_matching_features(hc)
    else:
        template_corr = np.nan

    return hr, rr_max, rr_ratio, template_corr


def ecg_template_ljungbox_stat(sig, sampling_rate, max_iter=5, tol=1e-3, lags=20):
    """
    对单条ECG信号，标准化、迭代模板去周期，并计算Ljung-Box统计量。

    参数:
        sig: 1D array-like，ECG信号
        sampling_rate: int，采样率
        max_iter: int，最大迭代次数
        tol: float，收敛阈值
        lags: int，Ljung-Box统计量滞后阶数

    返回:
        lb_stat: Ljung-Box统计量
        lb_pvalue: Ljung-Box对应p值
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from module.features.peaks_feature import ecg_find_peaks
    import numpy as np

    def standardize(sig):
        return (sig - np.mean(sig)) / np.std(sig) if np.std(sig) > 0 else sig

    def remove_ecg_periodicity_template(ecg, sampling_rate, window_ms=20):
        peaks, _ = ecg_find_peaks(ecg, sampling_rate)
        ecg_cleaned = ecg.copy()
        window = int(window_ms / 1000 * sampling_rate)
        for p in peaks:
            start = max(0, p - window)
            end = min(len(ecg), p + window)
            ecg_cleaned[start:end] = np.interp(
                np.arange(start, end),
                [start - 1, end],
                [
                    ecg_cleaned[start - 1] if start > 0 else 0,
                    ecg_cleaned[end] if end < len(ecg) else 0,
                ],
            )
        return ecg_cleaned

    def remove_ecg_periodicity_template_iter(ecg, sampling_rate, max_iter=5, tol=1e-3):
        prev = ecg.copy()
        for i in range(max_iter):
            curr = remove_ecg_periodicity_template(prev, sampling_rate)
            diff = np.std(prev - curr)
            if diff < tol:
                break
            prev = curr
        return curr

    std_ecg = standardize(sig)
    template_noise_iter = remove_ecg_periodicity_template_iter(
        std_ecg, sampling_rate, max_iter=max_iter, tol=tol
    )
    lb_test = acorr_ljungbox(template_noise_iter, lags=[lags], return_df=True)
    lb_stat = lb_test["lb_stat"].values[0]
    lb_pvalue = lb_test["lb_pvalue"].values[0]
    return lb_stat, lb_pvalue

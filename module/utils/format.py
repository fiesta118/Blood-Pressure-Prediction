import numpy as np


def normalize_data(sig):
    """Normalize the input signal between zero and one

    Args:
        sig (np.ndarray): Input signal

    Returns:
        np.ndarray: Normalized signal
    """
    sig = np.array(sig)
    min_val = np.min(sig)
    max_val = np.max(sig)
    if max_val - min_val == 0:
        return np.zeros_like(sig)
    return (sig - min_val) / (max_val - min_val)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, firwin, lfilter, freqz


def pre_emphasis(signal, alpha=0.97):
    if len(signal) == 0:
        return signal
    return np.append(signal[0], signal[1:] - alpha * signal[:-1]).astype(np.float32)


def design_iir_bandpass(sr, low=50, high=8000, order=4):
    nyq = sr / 2.0
    high = min(high, nyq - 100)
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return b, a


def design_fir_bandpass(sr, low=50, high=8000, numtaps=101):
    nyq = sr / 2.0
    high = min(high, nyq - 100)
    taps = firwin(numtaps, [low / nyq, high / nyq], pass_zero=False)
    return taps


def apply_filter(signal, sr, filter_type="iir", low=20, high=10000):
    if filter_type == "iir":
        b, a = design_iir_bandpass(sr, low=low, high=high, order=4)
        filtered = filtfilt(b, a, signal)
        return filtered.astype(np.float32)

    if filter_type == "fir":
        taps = design_fir_bandpass(sr, low=low, high=high, numtaps=101)
        filtered = lfilter(taps, [1.0], signal)
        return filtered.astype(np.float32)

    raise ValueError(f"Unsupported filter_type: {filter_type}")


def get_filter_response(sr, filter_type="iir", low=50, high=8000):
    if filter_type == "iir":
        b, a = design_iir_bandpass(sr, low=low, high=high, order=4)
        w, h = freqz(b, a, worN=2048, fs=sr)
        return w, h

    if filter_type == "fir":
        taps = design_fir_bandpass(sr, low=low, high=high, numtaps=101)
        w, h = freqz(taps, [1.0], worN=2048, fs=sr)
        return w, h

    raise ValueError(f"Unsupported filter_type: {filter_type}")


def estimate_noise_power(raw_signal, processed_signal):
    """
    Ước lượng noise bị loại bỏ = raw - processed
    """
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    processed_signal = np.asarray(processed_signal, dtype=np.float32)

    min_len = min(len(raw_signal), len(processed_signal))
    if min_len == 0:
        return 0.0

    raw_signal = raw_signal[:min_len]
    processed_signal = processed_signal[:min_len]

    noise = raw_signal - processed_signal
    noise_power = np.mean(noise ** 2)
    return float(noise_power)


def estimate_snr(raw_signal, processed_signal):
    """
    SNR xấp xỉ:
    signal_power / noise_power
    với noise = raw - processed
    """
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    processed_signal = np.asarray(processed_signal, dtype=np.float32)

    min_len = min(len(raw_signal), len(processed_signal))
    if min_len == 0:
        return 0.0

    raw_signal = raw_signal[:min_len]
    processed_signal = processed_signal[:min_len]

    noise = raw_signal - processed_signal
    signal_power = np.mean(processed_signal ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12

    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


def plot_filter_response(sr, save_path, filter_type="iir", low=50, high=8000):
    """
    Vẽ magnitude response và phase response của filter
    """
    w, h = get_filter_response(sr, filter_type=filter_type, low=low, high=high)

    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.unwrap(np.angle(h))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(w, magnitude_db)
    plt.title(f"{filter_type.upper()} Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(w, phase)
    plt.title(f"{filter_type.upper()} Filter Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
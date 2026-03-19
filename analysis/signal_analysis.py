import os
from typing import Dict, Iterable, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window, welch

from preprocessing.filter import estimate_noise_power, estimate_snr, plot_filter_response
from utils.reproducibility import ensure_dir


def plot_waveform(signal: np.ndarray, sr: int, save_path: str, title: str = 'Waveform'):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_waveform_comparison(raw_signal: np.ndarray, processed_signal: np.ndarray, sr: int, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    librosa.display.waveshow(raw_signal, sr=sr, ax=axes[0])
    axes[0].set_title('Original Waveform')
    librosa.display.waveshow(processed_signal, sr=sr, ax=axes[1])
    axes[1].set_title('DSP-Processed Waveform')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_fft(signal: np.ndarray, sr: int, save_path: str, title: str = 'FFT Spectrum'):
    fft = np.fft.rfft(signal)
    magnitude = np.abs(fft)
    freq = np.fft.rfftfreq(len(signal), d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq, magnitude)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_fft_comparison(raw_signal: np.ndarray, processed_signal: np.ndarray, sr: int, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, sig, title in zip(axes, [raw_signal, processed_signal], ['Original Spectrum', 'DSP-Processed Spectrum']):
        fft = np.fft.rfft(sig)
        magnitude = np.abs(fft)
        freq = np.fft.rfftfreq(len(sig), d=1.0 / sr)
        ax.plot(freq, magnitude)
        ax.set_title(title)
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_spectrogram(signal: np.ndarray, sr: int, save_path: str, title: str = 'STFT Spectrogram', n_fft: int = 1024, hop_length: int = 256):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax, cmap='magma')
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_spectrogram_comparison(raw_signal: np.ndarray, processed_signal: np.ndarray, sr: int, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, sig, title in zip(axes, [raw_signal, processed_signal], ['Original Spectrogram', 'DSP-Processed Spectrogram']):
        stft = librosa.stft(sig, n_fft=1024, hop_length=256)
        db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(db, sr=sr, hop_length=256, x_axis='time', y_axis='log', ax=ax, cmap='magma')
        ax.set_title(title)
    fig.colorbar(img, ax=axes, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_psd(signal: np.ndarray, sr: int, save_path: str, title: str = 'Power Spectral Density'):
    f, pxx = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f, pxx)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_psd_comparison(raw_signal: np.ndarray, processed_signal: np.ndarray, sr: int, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for sig, label in [(raw_signal, 'Original'), (processed_signal, 'DSP-Processed')]:
        f, pxx = welch(sig, fs=sr, nperseg=min(1024, len(sig)))
        ax.semilogy(f, pxx, label=label)
    ax.set_title('PSD Comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def haar_wavelet_decomposition(signal: np.ndarray, levels: int = 4):
    coeffs = []
    x = np.asarray(signal, dtype=np.float32)
    for _ in range(levels):
        if len(x) % 2 == 1:
            x = x[:-1]
        approx = (x[0::2] + x[1::2]) / np.sqrt(2)
        detail = (x[0::2] - x[1::2]) / np.sqrt(2)
        coeffs.append(detail)
        x = approx
    coeffs.append(x)
    return coeffs


def plot_wavelet(signal: np.ndarray, save_path: str, title: str = 'Haar Wavelet Coefficients'):
    coeffs = haar_wavelet_decomposition(signal, levels=4)
    fig, axes = plt.subplots(len(coeffs), 1, figsize=(10, 8), sharex=False)
    for idx, (ax, coef) in enumerate(zip(axes, coeffs), start=1):
        ax.plot(coef)
        ax.set_title(f'{title} - Level {idx}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def spectral_leakage_demo(save_path: str, sr: int = 8000, freq_hz: float = 440.5, n: int = 1024):
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq_hz * t)
    rect = x * get_window('boxcar', n)
    hann = x * get_window('hann', n)

    freq = np.fft.rfftfreq(n, 1 / sr)
    rect_mag = np.abs(np.fft.rfft(rect))
    hann_mag = np.abs(np.fft.rfft(hann))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq, rect_mag, label='Rectangular window')
    ax.plot(freq, hann_mag, label='Hann window')
    ax.set_title('Spectral Leakage Demo')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def window_size_effect_demo(signal: np.ndarray, sr: int, save_path: str, window_sizes=(256, 1024, 2048)):
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(10, 10), sharex=True)
    for ax, win in zip(axes, window_sizes):
        stft = librosa.stft(signal, n_fft=win, hop_length=max(64, win // 4))
        db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(db, sr=sr, hop_length=max(64, win // 4), x_axis='time', y_axis='log', ax=ax, cmap='magma')
        ax.set_title(f'Window Size = {win}')
    fig.colorbar(img, ax=axes, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def run_full_signal_analysis(raw_signal: np.ndarray, processed_signal: np.ndarray, sr: int, out_dir: str, filter_type: str = 'iir') -> Dict[str, float]:
    ensure_dir(out_dir)
    plot_waveform(raw_signal, sr, os.path.join(out_dir, 'waveform_raw.png'), 'Raw Waveform')
    plot_waveform(processed_signal, sr, os.path.join(out_dir, 'waveform_dsp.png'), 'DSP Waveform')
    plot_waveform_comparison(raw_signal, processed_signal, sr, os.path.join(out_dir, 'waveform_comparison.png'))

    plot_fft(raw_signal, sr, os.path.join(out_dir, 'fft_raw.png'), 'Raw FFT Spectrum')
    plot_fft(processed_signal, sr, os.path.join(out_dir, 'fft_dsp.png'), 'DSP FFT Spectrum')
    plot_fft_comparison(raw_signal, processed_signal, sr, os.path.join(out_dir, 'fft_comparison.png'))

    plot_spectrogram(raw_signal, sr, os.path.join(out_dir, 'stft_raw.png'), 'Raw STFT Spectrogram')
    plot_spectrogram(processed_signal, sr, os.path.join(out_dir, 'stft_dsp.png'), 'DSP STFT Spectrogram')
    plot_spectrogram_comparison(raw_signal, processed_signal, sr, os.path.join(out_dir, 'stft_comparison.png'))

    plot_psd(raw_signal, sr, os.path.join(out_dir, 'psd_raw.png'), 'Raw PSD')
    plot_psd(processed_signal, sr, os.path.join(out_dir, 'psd_dsp.png'), 'DSP PSD')
    plot_psd_comparison(raw_signal, processed_signal, sr, os.path.join(out_dir, 'psd_comparison.png'))

    plot_wavelet(raw_signal, os.path.join(out_dir, 'wavelet_raw.png'), 'Raw Wavelet')
    plot_wavelet(processed_signal, os.path.join(out_dir, 'wavelet_dsp.png'), 'DSP Wavelet')

    spectral_leakage_demo(os.path.join(out_dir, 'spectral_leakage_demo.png'))
    window_size_effect_demo(raw_signal, sr, os.path.join(out_dir, 'window_size_effects.png'))
    plot_filter_response(sr, filter_type=filter_type, save_path=os.path.join(out_dir, f'{filter_type}_filter_response.png'))

    min_len = min(len(raw_signal), len(processed_signal))
    raw_signal = raw_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    residual = raw_signal - processed_signal
    noise_power = estimate_noise_power(raw_signal, processed_signal)
    snr_db = estimate_snr(processed_signal, residual)

    return {'snr_db': snr_db, 'noise_power': noise_power}

import numpy as np
import librosa

from preprocessing.filter import apply_filter, pre_emphasis


def load_audio(file_path, sr=22050, mono=True):
    signal, sr = librosa.load(file_path, sr=sr, mono=mono)
    return signal.astype(np.float32), sr


def trim_silence(signal, top_db=30):
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    if len(trimmed) == 0:
        return signal
    return trimmed


def normalize_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal.astype(np.float32)


def fix_length(signal, sr, duration=4.0):
    target_len = int(sr * duration)
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]
    return signal.astype(np.float32)


def preprocess_raw(signal, sr, duration=4.0):
    """Pipeline A - minimal preprocessing only."""
    signal = normalize_audio(signal)
    signal = fix_length(signal, sr, duration=duration)
    return signal.astype(np.float32)


def preprocess_dsp(signal, sr, duration=4.0, filter_type="iir"):
    """
    Pipeline B - DSP-enhanced signal.

    This branch is intentionally more informative than the raw baseline:
    - trim silence to reduce irrelevant leading/trailing segments
    - apply light pre-emphasis to highlight transient/high-frequency cues
    - apply gentle band-pass filtering to suppress very low-frequency rumble
      and extreme high-frequency noise while preserving discriminative content
    - normalize after processing for stable handcrafted features
    """
    signal = trim_silence(signal, top_db=30)
    signal = pre_emphasis(signal, alpha=0.95)
    signal = apply_filter(signal, sr, filter_type=filter_type, low=30, high=9000)
    signal = normalize_audio(signal)
    signal = fix_length(signal, sr, duration=duration)
    signal = normalize_audio(signal)
    return signal.astype(np.float32)


def waveform_to_logmelspec(
    signal,
    sr,
    n_mels=64,
    n_fft=1024,
    hop_length=512,
    fmin=20,
    fmax=None,
):
    if fmax is None:
        fmax = sr // 2

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mean = np.mean(mel_db)
    std = np.std(mel_db) + 1e-8
    mel_db = (mel_db - mean) / std
    return mel_db.astype(np.float32)

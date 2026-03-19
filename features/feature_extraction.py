import numpy as np
import librosa
from scipy.stats import entropy


def _stats_2d(feat):
    """
    For feature matrix shape (n_features, n_frames)
    """
    return np.concatenate(
        [
            np.mean(feat, axis=1),
            np.std(feat, axis=1),
            np.min(feat, axis=1),
            np.max(feat, axis=1),
        ]
    )


def _stats_1d(feat):
    feat = np.ravel(feat)
    return np.array(
        [
            np.mean(feat),
            np.std(feat),
            np.min(feat),
            np.max(feat),
        ],
        dtype=np.float32,
    )


def _band_energy(signal, sr, bands=None, n_fft=2048, hop_length=512):
    """
    Compute average band energy on predefined frequency bands.
    """
    if bands is None:
        bands = [
            (0, 250),
            (250, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 4000),
            (4000, 8000),
        ]

    stft_mag = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    energies = []
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) == 0:
            energies.append(0.0)
        else:
            band_power = np.mean(stft_mag[idx, :])
            energies.append(float(band_power))

    return np.array(energies, dtype=np.float32)


def _spectral_entropy(signal, sr, n_fft=2048, hop_length=512):
    """
    Spectral entropy averaged across frames.
    """
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_sum = np.sum(S, axis=0, keepdims=True) + 1e-12
    P = S / S_sum
    ent = [entropy(P[:, i] + 1e-12, base=2) for i in range(P.shape[1])]
    return np.array(ent, dtype=np.float32)


def extract_features(signal, sr):
    """
    Handcrafted DSP features for Pipeline B / classical models.

    Included:
    - MFCC
    - delta MFCC
    - delta-delta MFCC
    - log-mel
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff
    - zero crossing rate
    - RMS
    - chroma
    - band energy
    - spectral entropy
    """
    # Time-frequency features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=1024, hop_length=512)

    # Spectral descriptors
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)

    # Band energy + entropy
    band_energy = _band_energy(signal, sr)
    spec_entropy = _spectral_entropy(signal, sr)

    feature_vector = np.concatenate(
        [
            _stats_2d(mfcc),
            _stats_2d(delta),
            _stats_2d(delta2),
            _stats_2d(mel_db),
            _stats_2d(chroma),
            _stats_1d(centroid),
            _stats_1d(bandwidth),
            _stats_1d(rolloff),
            _stats_1d(zcr),
            _stats_1d(rms),
            band_energy,
            _stats_1d(spec_entropy),
        ]
    )

    return feature_vector.astype(np.float32)
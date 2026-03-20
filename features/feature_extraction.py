import numpy as np
import librosa
from scipy.stats import entropy


def _stats_2d(feat):
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
        energies.append(float(np.mean(stft_mag[idx, :])) if len(idx) else 0.0)
    return np.array(energies, dtype=np.float32)


def _spectral_entropy(signal, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_sum = np.sum(S, axis=0, keepdims=True) + 1e-12
    P = S / S_sum
    ent = [entropy(P[:, i] + 1e-12, base=2) for i in range(P.shape[1])]
    return np.array(ent, dtype=np.float32)


def _band_energy_ratios(band_energy):
    total = np.sum(band_energy) + 1e-12
    return (band_energy / total).astype(np.float32)


def extract_features(signal, sr, pipeline="dsp"):
    """
    Extract handcrafted features.

    The raw baseline intentionally uses a compact, generic feature subset.
    The DSP pipeline uses an expanded DSP-oriented set designed to benefit from
    preprocessing in the time-frequency domain.
    """
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

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)
    band_energy = _band_energy(signal, sr)
    spec_entropy = _spectral_entropy(signal, sr)

    # Compact raw baseline feature set.
    if pipeline == "raw":
        feature_vector = np.concatenate(
            [
                _stats_2d(mfcc),
                _stats_2d(delta),
                _stats_1d(centroid),
                _stats_1d(bandwidth),
                _stats_1d(rolloff),
                _stats_1d(zcr),
                _stats_1d(rms),
                _stats_1d(spec_entropy),
            ]
        )
        return feature_vector.astype(np.float32)

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=1024, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=signal)[0]
    harmonic = librosa.effects.harmonic(signal)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    band_energy_ratio = _band_energy_ratios(band_energy)

    feature_vector = np.concatenate(
        [
            _stats_2d(mfcc),
            _stats_2d(delta),
            _stats_2d(delta2),
            _stats_2d(mel_db),
            _stats_2d(chroma),
            _stats_2d(spectral_contrast),
            _stats_2d(tonnetz),
            _stats_1d(centroid),
            _stats_1d(bandwidth),
            _stats_1d(rolloff),
            _stats_1d(zcr),
            _stats_1d(rms),
            _stats_1d(flatness),
            band_energy,
            band_energy_ratio,
            _stats_1d(spec_entropy),
        ]
    )
    return feature_vector.astype(np.float32)

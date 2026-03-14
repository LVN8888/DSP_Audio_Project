import librosa
import numpy as np


def extract_features(signal, sr):

    # MFCC with more coefficients and deltas
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40  # Increased from 13 to 40
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr
    )

    # Chroma features
    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    # Other features
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)

    # Function to compute statistics
    def compute_stats(feature):
        return [
            np.mean(feature),
            np.std(feature),
            np.min(feature),
            np.max(feature)
        ]

    feature_vector = []
    feature_vector.extend(compute_stats(mfcc))
    feature_vector.extend(compute_stats(mfcc_delta))
    feature_vector.extend(compute_stats(mfcc_delta2))
    feature_vector.extend(compute_stats(centroid))
    feature_vector.extend(compute_stats(bandwidth))
    feature_vector.extend(compute_stats(rolloff))
    feature_vector.extend(compute_stats(chroma))
    feature_vector.extend(compute_stats(zcr))
    feature_vector.extend(compute_stats(rms))

    return np.array(feature_vector)
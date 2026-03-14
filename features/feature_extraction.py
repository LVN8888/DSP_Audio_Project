import librosa
import numpy as np


def extract_features(signal, sr):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(signal)

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        [np.mean(centroid)],
        [np.mean(bandwidth)],
        [np.mean(zcr)]
    ])

    return feature_vector
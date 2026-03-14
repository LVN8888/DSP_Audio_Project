from preprocessing.filter import bandpass_filter
from features.feature_extraction import extract_features


def pipeline_dsp(signal, sr):

    filtered = bandpass_filter(signal, sr)

    features = extract_features(filtered, sr)

    return features
from features.feature_extraction import extract_features


def pipeline_raw(signal, sr):

    features = extract_features(signal, sr)

    return features
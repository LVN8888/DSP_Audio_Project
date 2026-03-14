import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analysis.signal_analysis import load_audio
from experiments.pipelineA_raw import pipeline_raw
from experiments.pipelineB_dsp import pipeline_dsp
from models.svm_model import train_model, evaluate_model
from visualization.plots import plot_confusion_matrix


DATASET_ROOT = os.path.join("data", "UrbanSound8K")
AUDIO_ROOT = os.path.join(DATASET_ROOT, "audio")
METADATA_PATH = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
OUTPUT_ROOT = "outputs"


def collect_features_from_urbansound8k(pipeline_func, max_files=None, progress_step=25):
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)

    if max_files is not None:
        metadata = metadata.iloc[:max_files]

    total_files = len(metadata)
    print(f"Total files to process: {total_files}")

    X = []
    y = []

    for idx, (_, row) in enumerate(metadata.iterrows(), start=1):
        file_name = row["slice_file_name"]
        fold = row["fold"]
        label = row["class"]

        file_path = os.path.join(AUDIO_ROOT, f"fold{fold}", file_name)

        if not os.path.exists(file_path):
            print(f"[{idx}/{total_files}] File not found: {file_path}")
            continue

        try:
            signal, sr = load_audio(file_path)
            features = pipeline_func(signal, sr)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"[{idx}/{total_files}] Skipping {file_name} because of error: {e}")

        if idx == 1 or idx % progress_step == 0 or idx == total_files:
            print(f"Processed: {idx}/{total_files} files | Successful: {len(X)}")

    return np.array(X), np.array(y)


def filter_rare_classes(X, y, min_count=2):
    class_counts = Counter(y)
    keep_indices = [i for i, label in enumerate(y) if class_counts[label] >= min_count]

    X_filtered = X[keep_indices]
    y_filtered = y[keep_indices]

    print("\nClass distribution after filtering rare classes:")
    print(Counter(y_filtered))

    return X_filtered, y_filtered


def run_pipeline(name, pipeline_func, max_files=None, min_count=2, output_name="pipeline"):
    print(f"\n========== {name} ==========")
    print("Step 1: Loading audio and extracting features...")

    X, y = collect_features_from_urbansound8k(
        pipeline_func=pipeline_func,
        max_files=max_files,
        progress_step=25
    )

    if len(X) == 0:
        raise ValueError("No valid audio files were loaded.")

    print(f"\nStep 2: Feature extraction complete. Valid samples: {len(X)}")
    print("Original class distribution:")
    print(Counter(y))

    print("\nStep 3: Filtering rare classes...")
    X, y = filter_rare_classes(X, y, min_count=min_count)

    if len(X) == 0:
        raise ValueError("No samples remain after filtering rare classes.")

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("At least 2 valid classes are required for training.")

    print("\nStep 4: Splitting into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nStep 4.5: Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nStep 5: Training SVM model...")
    model = train_model(X_train, y_train)

    print("\nStep 6: Evaluating model...")
    y_pred = evaluate_model(model, X_test, y_test)

    print("\nStep 7: Plotting confusion matrix...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    class_names = sorted(np.unique(y))
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        save_path=os.path.join(OUTPUT_ROOT, f"{output_name}_confusion_matrix.png")
    )

    print(f"\nFinished: {name}")


if __name__ == "__main__":
    run_pipeline(
        name="Pipeline A - Raw Audio",
        pipeline_func=pipeline_raw,
        max_files=None,
        min_count=2,
        output_name="pipelineA_raw"
    )

    run_pipeline(
        name="Pipeline B - DSP Filtered",
        pipeline_func=pipeline_dsp,
        max_files=None,
        min_count=2,
        output_name="pipelineB_dsp"
    )
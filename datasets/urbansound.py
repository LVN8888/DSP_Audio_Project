import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.feature_extraction import extract_features
from preprocessing.audio import (
    load_audio,
    preprocess_dsp,
    preprocess_raw,
    waveform_to_logmelspec,
)


@dataclass
class UrbanSoundConfig:
    dataset_root: str
    sr: int = 22050
    duration: float = 4.0
    filter_type: str = "iir"

    @property
    def audio_root(self):
        return os.path.join(self.dataset_root, "audio")

    @property
    def metadata_path(self):
        return os.path.join(self.dataset_root, "metadata", "UrbanSound8K.csv")


class UrbanSound8KBuilder:
    def __init__(self, config: UrbanSoundConfig):
        self.config = config

    def load_metadata(self):
        if not os.path.exists(self.config.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.config.metadata_path}")
        return pd.read_csv(self.config.metadata_path)

    def _preprocess(self, signal, sr, pipeline):
        if pipeline == "raw":
            return preprocess_raw(signal, sr, duration=self.config.duration)
        if pipeline == "dsp":
            return preprocess_dsp(
                signal,
                sr,
                duration=self.config.duration,
                filter_type=self.config.filter_type,
            )
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    def get_example_sample(self, max_files=None):
        metadata = self.load_metadata()
        if max_files is not None and max_files > 0:
            metadata = metadata.iloc[:max_files]

        for _, row in metadata.iterrows():
            file_name = row["slice_file_name"]
            fold = int(row["fold"])
            label = row["class"]
            file_path = os.path.join(self.config.audio_root, f"fold{fold}", file_name)
            if os.path.exists(file_path):
                signal, sr = load_audio(file_path, sr=self.config.sr)
                return {
                    "file_name": file_name,
                    "fold": fold,
                    "label": label,
                    "file_path": file_path,
                    "signal": signal,
                    "sr": sr,
                }
        raise FileNotFoundError("Could not find any valid example audio file in dataset.")

    def build_feature_dataset(self, pipeline="raw", max_files=None, progress_step=50):
        metadata = self.load_metadata()

        if max_files is not None and max_files > 0:
            metadata = metadata.iloc[:max_files]

        total_files = len(metadata)
        print(f"Total files to process: {total_files}")

        X, y, folds = [], [], []

        for idx, (_, row) in enumerate(metadata.iterrows(), start=1):
            file_name = row["slice_file_name"]
            fold = int(row["fold"])
            label = row["class"]
            file_path = os.path.join(self.config.audio_root, f"fold{fold}", file_name)

            try:
                signal, sr = load_audio(file_path, sr=self.config.sr)
                processed = self._preprocess(signal, sr, pipeline)
                features = extract_features(processed, sr)

                X.append(features)
                y.append(label)
                folds.append(fold)

            except Exception as e:
                print(f"[{idx}/{total_files}] Skipping {file_name} because of error: {e}")

            if idx == 1 or idx % progress_step == 0 or idx == total_files:
                print(f"Processed: {idx}/{total_files} files | Successful: {len(X)}")

        return np.array(X, dtype=np.float32), np.array(y), np.array(folds)

    def build_spectrogram_dataset(self, pipeline="raw", max_files=None, progress_step=50):
        metadata = self.load_metadata()

        if max_files is not None and max_files > 0:
            metadata = metadata.iloc[:max_files]

        total_files = len(metadata)
        print(f"Total files to process: {total_files}")

        X, y, folds = [], [], []

        for idx, (_, row) in enumerate(metadata.iterrows(), start=1):
            file_name = row["slice_file_name"]
            fold = int(row["fold"])
            label = row["class"]
            file_path = os.path.join(self.config.audio_root, f"fold{fold}", file_name)

            try:
                signal, sr = load_audio(file_path, sr=self.config.sr)
                processed = self._preprocess(signal, sr, pipeline)
                mel = waveform_to_logmelspec(processed, sr)
                mel = np.expand_dims(mel, axis=0)

                X.append(mel.astype(np.float32))
                y.append(label)
                folds.append(fold)

            except Exception as e:
                print(f"[{idx}/{total_files}] Skipping {file_name} because of error: {e}")

            if idx == 1 or idx % progress_step == 0 or idx == total_files:
                print(f"Processed: {idx}/{total_files} files | Successful: {len(X)}")

        return np.array(X, dtype=np.float32), np.array(y), np.array(folds)

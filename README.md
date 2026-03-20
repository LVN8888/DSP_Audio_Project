# DSP Audio Classification Project (SVM-Only)

This repository contains an **SVM-only** implementation for environmental sound classification on **UrbanSound8K**. The project is intentionally organized as a **DSP-centered study**: the classifier is used to measure the effect of preprocessing and handcrafted feature design, while the main comparison is between a **raw baseline** and a **DSP-enhanced pipeline**.

## What this version focuses on

- a **Raw pipeline** with minimal preprocessing
- a **DSP pipeline** with light filtering, pre-emphasis, and richer DSP-oriented features
- grouped cross-validation with exported fold-by-fold tables
- assignment-ready plots and comparison outputs
- a final deployable `dsp_svm.joblib` artifact for prediction

## Project structure

```text
DSP_Audio/
├── analysis/                # Signal-level analysis utilities
├── datasets/                # UrbanSound8K dataset builder
├── experiments/             # Training, comparison, prediction pipeline
├── features/                # Handcrafted feature extraction
├── models/                  # Classical SVM training code
├── preprocessing/           # Audio preprocessing and filtering
├── utils/                   # Reproducibility and helpers
├── visualization/           # Tables and plots
├── main.py                  # Command-line entry point
├── requirements.txt
└── README.md
```

## Dataset layout

```text
data/
└── UrbanSound8K/
    ├── audio/
    │   ├── fold1/
    │   ├── fold2/
    │   └── ...
    └── metadata/
        └── UrbanSound8K.csv
```

Default dataset root:

```bash
data/UrbanSound8K
```

## Installation

```bash
pip install -r requirements.txt
```

## Main commands

### 1) Full training pipeline

This is the main command for assignment/report generation.

```bash
python main.py train --out-dir outputs --pipeline both
```

It will automatically:

- generate required signal-analysis plots in `outputs/analysis_required_by_pdf/`
- run grouped cross-validation
- export fold metrics, summary tables, confusion matrices, ROC curves, and reports
- compare Raw vs DSP pipelines
- fit and save the final DSP SVM model to `outputs/artifacts/dsp_svm.joblib`
- create a submission manifest in `outputs/submission/`

### 2) DSP-only training

```bash
python main.py train --out-dir outputs --pipeline dsp
```

### 3) Subset training for quick experiments

```bash
python main.py train --out-dir outputs --pipeline both --max-files 2000
```

### 4) Change number of folds

```bash
python main.py train --out-dir outputs --pipeline both --kfolds 5
```

### 5) Fit only the final model

```bash
python main.py fit-final --pipeline dsp --out-dir outputs
```

### 6) Predict a new file

```bash
python main.py predict --file path/to/test.wav --pipeline dsp --artifact outputs/artifacts/dsp_svm.joblib
```

### 7) Analyze one audio file

```bash
python main.py analyze --file path/to/sample.wav --out-dir outputs/analysis_example
```

## Pipeline design

### Raw baseline

The raw pipeline uses only minimal preprocessing:

- amplitude normalization
- fixed-duration padding/truncation

### DSP-enhanced pipeline

The DSP pipeline applies:

- silence trimming
- light pre-emphasis
- gentle band-pass filtering
- normalization after processing

It is paired with a richer DSP-oriented feature set including:

- MFCC, delta, delta-delta
- log-mel statistics
- chroma
- spectral contrast
- tonnetz
- spectral flatness
- band energy and band-energy ratios
- spectral centroid, bandwidth, rolloff
- zero-crossing rate, RMS, spectral entropy

## Important outputs

```text
outputs/
├── analysis_required_by_pdf/
├── artifacts/
│   ├── dsp_svm.joblib
│   └── dsp_svm_metadata.json
├── comparisons/
├── results/
└── submission/
    ├── submission_manifest.csv
    └── submission_manifest.json
```

## Recommended workflow

Run the full evaluation pipeline:

```bash
python main.py train --pipeline both --out-dir outputs
```

Then test a new file:

```bash
python main.py predict --file samples/example.wav --pipeline dsp --artifact outputs/artifacts/dsp_svm.joblib
```

## Notes

- default random seed: `42`
- this repository is intentionally **SVM-only**
- the primary scientific comparison is **Raw vs DSP**
- if `--max-files` is omitted, the builder uses the full available dataset

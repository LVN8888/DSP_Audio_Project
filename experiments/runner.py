import os
import joblib
import numpy as np
import pandas as pd

from analysis.signal_analysis import run_full_signal_analysis
from models.classical_models import (
    cross_validate_classical,
    compare_two_cv_results,
    fit_final_classical_model,
)
from features.feature_extraction import extract_features
from preprocessing.audio import (
    load_audio,
    preprocess_dsp,
    preprocess_raw,
)
from utils.reproducibility import ensure_dir, save_json
from visualization.plots import (
    plot_pipeline_metric_comparison,
    save_pipeline_comparison_table,
)


DISPLAY_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc_ovr"]


def print_result_tables(result, title):
    print(f"\n{title}")
    fold_df = pd.DataFrame(result["fold_metrics"])
    display_cols = [c for c in ["fold", *DISPLAY_METRICS, "best_params"] if c in fold_df.columns]
    print("\nFold-by-fold table:")
    print(fold_df[display_cols].to_string(index=False))

    summary_rows = []
    for metric, stats in result["summary"].items():
        summary_rows.append({
            "metric": metric,
            "mean": round(stats["mean"], 4),
            "ci95": round(stats["ci95"], 4),
            "lower": round(stats.get("lower", np.nan), 4),
            "upper": round(stats.get("upper", np.nan), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary table (mean ± 95% CI):")
    print(summary_df.to_string(index=False))


def generate_assignment_plots(builder, out_dir, max_files=None):
    analysis_dir = ensure_dir(os.path.join(out_dir, "analysis_required_by_pdf"))
    sample = builder.get_example_sample(max_files=max_files)

    raw_signal = preprocess_raw(
        sample["signal"],
        sample["sr"],
        duration=builder.config.duration,
    )

    dsp_signal = preprocess_dsp(
        sample["signal"],
        sample["sr"],
        duration=builder.config.duration,
    )

    stats = run_full_signal_analysis(
        raw_signal=raw_signal,
        processed_signal=dsp_signal,
        sr=sample["sr"],
        out_dir=analysis_dir,
        filter_type=builder.config.filter_type,
    )

    save_json(
        {
            "example_file": sample["file_name"],
            "example_label": sample["label"],
            "example_fold": sample["fold"],
            **stats,
        },
        os.path.join(analysis_dir, "analysis_overview.json"),
    )

    print(f"\nSaved required signal-analysis plots to: {analysis_dir}")
    print(
        f"Representative sample: {sample['file_name']} | "
        f"label={sample['label']} | fold={sample['fold']}"
    )
    return analysis_dir


def run_classical_experiment(builder, pipeline, classical_model, kfolds, out_dir, max_files=None, seed=42):
    print(f"\n========== SVM | PIPELINE={pipeline.upper()} ==========")
    print("Step 1: Loading audio and extracting features...")

    X, y, folds = builder.build_feature_dataset(
        pipeline=pipeline,
        max_files=max_files,
        progress_step=50,
    )

    if len(X) == 0:
        raise ValueError("No valid samples loaded.")

    print(f"\nStep 2: Feature extraction complete. Valid samples: {len(X)}")
    print("Step 3: Running grouped k-fold cross-validation with SVM...")

    result = cross_validate_classical(
        X=X,
        y=y,
        groups=folds,
        model_name=classical_model,
        kfolds=kfolds,
        out_dir=os.path.join(out_dir, "results"),
        experiment_name=f"{pipeline}_svm",
        seed=seed,
    )

    result_payload = {
        "pipeline": pipeline,
        "model_type": "classical",
        "classical_model": classical_model,
        "fold_metrics": result.fold_results,
        "summary": result.summary,
        "y_true": result.y_true.tolist(),
        "y_pred": result.y_pred.tolist(),
        "classes": result.classes.tolist(),
        "classification_report": result.classification_report,
    }

    print("\nStep 4: Cross-validation complete.")
    print_result_tables(result_payload, f"RESULT TABLE - {pipeline.upper()} SVM")

    return result_payload


def compare_pipeline_results(result_a, result_b, comparison_dir, metric="f1", name="comparison"):
    ensure_dir(comparison_dir)

    class DummyResult:
        def __init__(self, fold_results):
            self.fold_results = fold_results

    paired_stats = compare_two_cv_results(
        DummyResult(result_a["fold_metrics"]),
        DummyResult(result_b["fold_metrics"]),
        metric=metric,
    )

    save_json(
        paired_stats,
        os.path.join(comparison_dir, f"{name}_{metric}_paired_comparison.json"),
    )

    comparison_df = save_pipeline_comparison_table(
        result_a["summary"],
        result_b["summary"],
        paired_stats,
        os.path.join(comparison_dir, f"{name}_performance_comparison.csv"),
        paired_metric=metric,
    )

    plot_pipeline_metric_comparison(
        result_a["summary"],
        result_b["summary"],
        os.path.join(comparison_dir, f"{name}_performance_comparison.png"),
    )

    print("\nRAW vs DSP comparison table:")
    print(comparison_df.to_string(index=False))
    return paired_stats


def fit_final_model(
    builder,
    pipeline,
    model_type,
    out_dir,
    max_files=None,
    classical_model="svm",
    epochs=10,
    batch_size=32,
    lr=1e-3,
    seed=42,
):
    print(f"Pipeline   : {pipeline}")
    print("Model type : svm")

    artifacts_dir = ensure_dir(os.path.join(out_dir, "artifacts"))

    print("\nStep 1: Loading audio and extracting features...")
    X, y, folds = builder.build_feature_dataset(
        pipeline=pipeline,
        max_files=max_files,
        progress_step=50,
    )

    if len(X) == 0:
        raise ValueError("No valid samples loaded.")

    print(f"\nStep 2: Feature extraction complete. Valid samples: {len(X)}")
    artifact_path = os.path.join(artifacts_dir, f"{pipeline}_svm.joblib")

    print("Step 3: Training final SVM model with GridSearchCV...")
    payload = fit_final_classical_model(
        X=X,
        y=y,
        model_name=classical_model,
        artifact_path=artifact_path,
        seed=seed,
    )

    metadata = {
        "pipeline": pipeline,
        "model_type": "svm",
        "num_samples": int(len(X)),
        "num_classes": int(len(np.unique(y))),
        "best_params": payload["best_params"],
        "artifact_path": artifact_path,
    }
    save_json(metadata, os.path.join(artifacts_dir, f"{pipeline}_svm_metadata.json"))

    print("Step 4: Saving artifact...")
    print("Best params:", payload["best_params"])
    print("Finished training final SVM model.")
    return payload["model"], artifact_path


def export_submission_manifest(out_dir, final_artifact_path, generated_paths, cv_results=None):
    submission_dir = ensure_dir(os.path.join(out_dir, "submission"))
    rows = []
    for category, path in generated_paths.items():
        rows.append({"category": category, "path": path, "exists": os.path.exists(path)})

    if cv_results is not None:
        for pipeline_name, result in cv_results.items():
            for metric, stats in result["summary"].items():
                rows.append({
                    "category": f"summary::{pipeline_name}::{metric}",
                    "path": "in-memory",
                    "exists": True,
                    "mean": round(float(stats.get("mean", np.nan)), 6),
                    "ci95": round(float(stats.get("ci95", np.nan)), 6),
                })

    manifest_df = pd.DataFrame(rows)
    manifest_path = os.path.join(submission_dir, "submission_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    save_json(
        {
            "final_artifact_path": final_artifact_path,
            "generated_paths": generated_paths,
            "cv_pipelines": list(cv_results.keys()) if cv_results else [],
        },
        os.path.join(submission_dir, "submission_manifest.json"),
    )
    return manifest_path


def run_submission_pipeline(builder, out_dir, kfolds=5, max_files=None, seed=42, compare_pipelines=False):
    print("Step 1: Generating assignment plots...")
    analysis_dir = generate_assignment_plots(builder, out_dir, max_files=max_files)

    cv_results = {}
    comparison_stats = None
    comparison_dir = ensure_dir(os.path.join(out_dir, "comparisons"))

    if compare_pipelines:
        print("\nStep 2: Running RAW + DSP cross-validation for report tables...")
        for pipeline in ["raw", "dsp"]:
            cv_results[f"{pipeline}_svm"] = run_classical_experiment(
                builder=builder,
                pipeline=pipeline,
                classical_model="svm",
                kfolds=kfolds,
                out_dir=out_dir,
                max_files=max_files,
                seed=seed,
            )

        comparison_stats = compare_pipeline_results(
            cv_results["raw_svm"],
            cv_results["dsp_svm"],
            comparison_dir,
            metric="f1",
            name="svm",
        )
    else:
        print("\nStep 2: Running DSP cross-validation for report tables...")
        cv_results["dsp_svm"] = run_classical_experiment(
            builder=builder,
            pipeline="dsp",
            classical_model="svm",
            kfolds=kfolds,
            out_dir=out_dir,
            max_files=max_files,
            seed=seed,
        )

    print("\nStep 3: Fitting final DSP model on full selected data...")
    _, final_artifact_path = fit_final_model(
        builder=builder,
        pipeline="dsp",
        model_type="classical",
        out_dir=out_dir,
        max_files=max_files,
        classical_model="svm",
        seed=seed,
    )

    generated_paths = {
        "analysis_dir": analysis_dir,
        "results_dir": os.path.join(out_dir, "results"),
        "comparisons_dir": comparison_dir,
        "artifacts_dir": os.path.join(out_dir, "artifacts"),
        "final_artifact": final_artifact_path,
    }
    manifest_path = export_submission_manifest(
        out_dir=out_dir,
        final_artifact_path=final_artifact_path,
        generated_paths=generated_paths,
        cv_results=cv_results,
    )

    if comparison_stats is not None:
        print("\nPaired statistical comparison (f1):")
        print(comparison_stats)

    print("\nSubmission outputs ready.")
    print(f"- Plots directory      : {analysis_dir}")
    print(f"- Results directory    : {os.path.join(out_dir, 'results')}")
    print(f"- Comparisons directory: {comparison_dir}")
    print(f"- Final artifact       : {final_artifact_path}")
    print(f"- Manifest             : {manifest_path}")

    return {
        "final_artifact_path": final_artifact_path,
        "manifest_path": manifest_path,
        "comparison_stats": comparison_stats,
        "cv_results": cv_results,
    }


def _apply_pipeline_to_segment(signal, sr, pipeline, duration, filter_type):
    if pipeline == "raw":
        return preprocess_raw(signal, sr, duration=duration)
    if pipeline == "dsp":
        return preprocess_dsp(signal, sr, duration=duration)
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def _segment_signal(signal, sr, segment_duration=4.0, hop_duration=2.0):
    segment_len = int(sr * segment_duration)
    hop_len = int(sr * hop_duration)

    if len(signal) < segment_len:
        signal = np.pad(signal, (0, segment_len - len(signal)))

    segments = []
    for start in range(0, len(signal) - segment_len + 1, hop_len):
        segments.append(signal[start:start + segment_len])

    if len(segments) == 0:
        segments.append(signal[:segment_len])

    return segments


def predict_with_classical(
    file_path,
    pipeline,
    artifact_path,
    sr=22050,
    filter_type="iir",
    duration=4.0,
    segment_duration=4.0,
    hop_duration=2.0,
    threshold=0.45,
):
    payload = joblib.load(artifact_path)
    model = payload["model"]
    scaler = payload["scaler"]
    classes = np.asarray(payload["classes"])

    signal, sr = load_audio(file_path, sr=sr)
    segments = _segment_signal(signal, sr, segment_duration=segment_duration, hop_duration=hop_duration)

    probs_accum = []
    for segment in segments:
        processed = _apply_pipeline_to_segment(segment, sr, pipeline, duration=duration, filter_type=filter_type)
        feat = extract_features(processed, sr, pipeline=pipeline).reshape(1, -1)
        feat = scaler.transform(feat)
        probs_accum.append(model.predict_proba(feat)[0])

    probs = np.mean(np.vstack(probs_accum), axis=0)
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])
    prediction = classes[best_idx] if best_prob >= threshold else "unknown"

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [(str(classes[i]), float(probs[i])) for i in top_idx]

    return {
        "prediction": str(prediction),
        "confidence": best_prob,
        "top3": top3,
        "segments_used": len(segments),
    }
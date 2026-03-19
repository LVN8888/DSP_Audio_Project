import argparse
import os

from analysis.signal_analysis import run_full_signal_analysis
from datasets.urbansound import UrbanSound8KBuilder, UrbanSoundConfig
from experiments.runner import (
    fit_final_model,
    generate_assignment_plots,
    predict_with_classical,
    run_submission_pipeline,
)
from preprocessing.audio import load_audio, preprocess_dsp
from utils.reproducibility import save_json, set_global_seed


def build_parser():
    parser = argparse.ArgumentParser(description="DSP Audio Project - SVM only version")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-root", type=str, default="data/UrbanSound8K")
    common.add_argument("--sr", type=int, default=22050)
    common.add_argument("--duration", type=float, default=4.0)
    common.add_argument("--filter-type", choices=["fir", "iir"], default="iir")
    common.add_argument("--seed", type=int, default=42)

    analyze = sub.add_parser("analyze", parents=[common], help="Run signal-level DSP analysis on one file")
    analyze.add_argument("--file", type=str, required=True)
    analyze.add_argument("--out-dir", type=str, default="outputs/analysis")

    train = sub.add_parser(
        "train",
        parents=[common],
        help="One-command training pipeline: CV tables + plots + final SVM artifact",
    )
    train.add_argument("--pipeline", choices=["dsp", "both"], default="both")
    train.add_argument("--kfolds", type=int, default=5)
    train.add_argument("--max-files", type=int, default=None)
    train.add_argument("--out-dir", type=str, default="outputs")

    fit_final = sub.add_parser("fit-final", parents=[common], help="Optional: fit only the final deployable SVM model")
    fit_final.add_argument("--pipeline", choices=["raw", "dsp"], required=True)
    fit_final.add_argument("--max-files", type=int, default=None)
    fit_final.add_argument("--out-dir", type=str, default="outputs")

    predict = sub.add_parser("predict", parents=[common], help="Predict a new audio file using a trained SVM artifact")
    predict.add_argument("--file", type=str, required=True)
    predict.add_argument("--pipeline", choices=["raw", "dsp"], required=True)
    predict.add_argument("--artifact", type=str, required=True)
    predict.add_argument("--segment-duration", type=float, default=4.0)
    predict.add_argument("--hop-duration", type=float, default=2.0)
    predict.add_argument("--threshold", type=float, default=0.45)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_global_seed(args.seed)

    if args.command == "analyze":
        print("\n========== ANALYZE SIGNAL ==========")
        print("Step 1: Loading raw audio...")
        raw_signal, sr = load_audio(args.file, sr=args.sr)

        print("Step 2: Applying DSP preprocessing...")
        processed = preprocess_dsp(raw_signal, sr, duration=args.duration, filter_type=args.filter_type)

        print("Step 3: Running full signal analysis...")
        stats = run_full_signal_analysis(raw_signal, processed, sr, args.out_dir, filter_type=args.filter_type)

        print("Step 4: Saving summary...")
        save_json(stats, os.path.join(args.out_dir, "analysis_summary.json"))

        print("Saved analysis to:", args.out_dir)
        print(stats)
        return

    config = UrbanSoundConfig(
        dataset_root=args.dataset_root,
        sr=args.sr,
        duration=args.duration,
        filter_type=args.filter_type,
    )
    builder = UrbanSound8KBuilder(config)

    if args.command == "train":
        print("\n========== TRAIN (ALL-IN-ONE SVM PIPELINE) ==========")
        summary = run_submission_pipeline(
            builder=builder,
            out_dir=args.out_dir,
            kfolds=args.kfolds,
            max_files=args.max_files,
            seed=args.seed,
            compare_pipelines=(args.pipeline == "both"),
        )
        print("\nTraining pipeline completed.")
        print("Final DSP artifact:", summary["final_artifact_path"])
        print("Manifest:", summary["manifest_path"])
        return

    if args.command == "fit-final":
        print("\n========== FIT FINAL SVM MODEL ==========")
        _, artifact_path = fit_final_model(
            builder=builder,
            pipeline=args.pipeline,
            model_type="classical",
            out_dir=args.out_dir,
            max_files=args.max_files,
            classical_model="svm",
            seed=args.seed,
        )
        print("Saved artifact to:", artifact_path)
        return

    if args.command == "predict":
        print("\n========== PREDICT WITH SVM ==========")
        print("Step 1: Loading trained artifact...")
        print("Step 2: Running prediction on new audio...")

        result = predict_with_classical(
            file_path=args.file,
            pipeline=args.pipeline,
            artifact_path=args.artifact,
            sr=args.sr,
            filter_type=args.filter_type,
            duration=args.duration,
            segment_duration=args.segment_duration,
            hop_duration=args.hop_duration,
            threshold=args.threshold,
        )

        print("Prediction:", result["prediction"])
        print("Confidence:", round(result["confidence"], 4))
        print("Top-3:")
        for label, prob in result["top3"]:
            print(f"  - {label}: {prob:.4f}")


if __name__ == "__main__":
    main()

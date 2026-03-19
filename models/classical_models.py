import os
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from utils.reproducibility import confidence_interval_95, ensure_dir, paired_t_test, save_json
from visualization.plots import (
    plot_confusion_matrix,
    plot_metric_summary,
    plot_multiclass_roc,
    save_metrics_table,
)


@dataclass
class ClassicalCVResult:
    fold_results: List[dict]
    summary: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    classes: np.ndarray
    classification_report: dict


def build_classical_estimator(name: str):
    if name != "svm":
        raise ValueError("This project version only supports SVM as requested.")

    estimator = SVC(probability=True, class_weight="balanced")
    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf", "linear"],
    }
    return estimator, param_grid


def build_classical_model(model_name="svm", seed=42):
    estimator, _ = build_classical_estimator(model_name)
    return estimator


def _compute_fold_metrics(y_true, y_pred, y_proba=None, labels=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if y_proba is not None and labels is not None and len(labels) > 2:
        try:
            auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
            metrics["roc_auc_ovr"] = float(auc)
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")

    return metrics


def cross_validate_classical(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    kfolds: int,
    out_dir: str,
    experiment_name: str,
    seed: int = 42,
) -> ClassicalCVResult:
    ensure_dir(out_dir)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    outer_cv = StratifiedGroupKFold(n_splits=kfolds, shuffle=True, random_state=seed)
    fold_results = []
    y_true_all, y_pred_all, y_proba_all = [], [], []

    estimator, param_grid = build_classical_estimator(model_name)

    total_folds = kfolds
    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y_enc, groups=groups), start=1
    ):
        print(f"\n[Fold {fold_idx}/{total_folds}] Preparing train/test split...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        train_groups = groups[train_idx]

        print(f"[Fold {fold_idx}/{total_folds}] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        inner_folds = min(3, len(np.unique(train_groups)))
        inner_n_splits = max(2, inner_folds)
        inner_cv = StratifiedGroupKFold(
            n_splits=inner_n_splits,
            shuffle=True,
            random_state=seed,
        )

        print(f"[Fold {fold_idx}/{total_folds}] Running GridSearchCV for SVM...")
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
        )
        search.fit(X_train_scaled, y_train, groups=train_groups)

        model = search.best_estimator_

        print(f"[Fold {fold_idx}/{total_folds}] Evaluating...")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        fold_metrics = _compute_fold_metrics(
            y_test,
            y_pred,
            y_proba=y_proba,
            labels=np.arange(len(class_names)),
        )
        fold_metrics["fold"] = fold_idx
        fold_metrics["best_params"] = search.best_params_
        fold_results.append(fold_metrics)

        print(f"[Fold {fold_idx}/{total_folds}] Accuracy : {fold_metrics['accuracy']:.4f}")
        print(f"[Fold {fold_idx}/{total_folds}] Precision: {fold_metrics['precision']:.4f}")
        print(f"[Fold {fold_idx}/{total_folds}] Recall   : {fold_metrics['recall']:.4f}")
        print(f"[Fold {fold_idx}/{total_folds}] F1-score : {fold_metrics['f1']:.4f}")
        if "roc_auc_ovr" in fold_metrics and not np.isnan(fold_metrics["roc_auc_ovr"]):
            print(f"[Fold {fold_idx}/{total_folds}] ROC-AUC : {fold_metrics['roc_auc_ovr']:.4f}")
        print(f"[Fold {fold_idx}/{total_folds}] Best params: {search.best_params_}")

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        if y_proba is not None:
            y_proba_all.append(y_proba)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_proba_all = np.concatenate(y_proba_all) if y_proba_all else None

    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc_ovr"]:
        values = [fr[metric] for fr in fold_results if metric in fr and not np.isnan(fr[metric])]
        if values:
            summary[metric] = confidence_interval_95(values)

    class_report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    plot_confusion_matrix(
        le.inverse_transform(y_true_all),
        le.inverse_transform(y_pred_all),
        class_names,
        save_path=os.path.join(out_dir, f"{experiment_name}_confusion_matrix.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        le.inverse_transform(y_true_all),
        le.inverse_transform(y_pred_all),
        class_names,
        save_path=os.path.join(out_dir, f"{experiment_name}_confusion_matrix_norm.png"),
        normalize=True,
    )
    plot_metric_summary(
        summary,
        save_path=os.path.join(out_dir, f"{experiment_name}_metrics.png"),
    )
    if y_proba_all is not None:
        plot_multiclass_roc(
            y_true_all,
            y_proba_all,
            np.arange(len(class_names)),
            save_path=os.path.join(out_dir, f"{experiment_name}_roc.png"),
        )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(
        os.path.join(out_dir, f"{experiment_name}_fold_metrics.csv"),
        index=False,
    )
    save_metrics_table(summary, os.path.join(out_dir, f"{experiment_name}_summary_table.csv"))
    pd.DataFrame(class_report).transpose().to_csv(
        os.path.join(out_dir, f"{experiment_name}_classification_report.csv")
    )
    save_json(
        {"summary": summary, "classes": class_names.tolist(), "classification_report": class_report},
        os.path.join(out_dir, f"{experiment_name}_summary.json"),
    )

    return ClassicalCVResult(
        fold_results=fold_results,
        summary=summary,
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_proba=y_proba_all,
        classes=class_names,
        classification_report=class_report,
    )


def compare_two_cv_results(result_a: ClassicalCVResult, result_b: ClassicalCVResult, metric: str = "accuracy"):
    a_scores = [fr[metric] for fr in result_a.fold_results if metric in fr]
    b_scores = [fr[metric] for fr in result_b.fold_results if metric in fr]
    return paired_t_test(a_scores, b_scores)


def fit_final_classical_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    artifact_path: str,
    seed: int = 42,
):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator, param_grid = build_classical_estimator(model_name)

    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_scaled, y_enc)

    payload = {
        "model": search.best_estimator_,
        "scaler": scaler,
        "classes": le.classes_,
        "model_name": model_name,
        "best_params": search.best_params_,
    }

    ensure_dir(os.path.dirname(artifact_path) or ".")
    joblib.dump(payload, artifact_path)

    return payload

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str], save_path: str, normalize: bool = False):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true' if normalize else None)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45, values_format='.2f' if normalize else 'd')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_metric_summary(metric_dict: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    labels = list(metric_dict.keys())
    means = [metric_dict[k]['mean'] for k in labels]
    cis = [metric_dict[k]['ci95'] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=cis, capsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Metrics (mean ± 95% CI)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_multiclass_roc(y_true, y_proba, class_names: Sequence[str], save_path: str):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=class_names)
    if y_true_bin.shape[1] <= 1:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    macro_points = []
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.2f})', alpha=0.7)
        macro_points.append(roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title(f'Multiclass ROC (macro AUC={np.mean(macro_points):.2f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metrics_table(metric_dict: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    rows = []
    for metric, stats in metric_dict.items():
        rows.append({
            'metric': metric,
            'mean': round(float(stats.get('mean', np.nan)), 4),
            'ci95': round(float(stats.get('ci95', np.nan)), 4),
            'lower': round(float(stats.get('lower', np.nan)), 4),
            'upper': round(float(stats.get('upper', np.nan)), 4),
            'n': int(stats.get('n', 0)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


def plot_pipeline_metric_comparison(raw_summary: dict, dsp_summary: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    metrics = [m for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc_ovr'] if m in raw_summary and m in dsp_summary]
    if not metrics:
        return

    raw_means = [raw_summary[m]['mean'] for m in metrics]
    raw_cis = [raw_summary[m]['ci95'] for m in metrics]
    dsp_means = [dsp_summary[m]['mean'] for m in metrics]
    dsp_cis = [dsp_summary[m]['ci95'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, raw_means, width, yerr=raw_cis, capsize=5, label='Raw')
    ax.bar(x + width / 2, dsp_means, width, yerr=dsp_cis, capsize=5, label='DSP')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Raw vs DSP Performance Comparison (mean ± 95% CI)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_pipeline_comparison_table(raw_summary: dict, dsp_summary: dict, paired_stats: dict, save_path: str, paired_metric: str = 'f1'):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    rows = []
    all_metrics = sorted(set(raw_summary.keys()) | set(dsp_summary.keys()))
    for metric in all_metrics:
        raw = raw_summary.get(metric, {})
        dsp = dsp_summary.get(metric, {})
        rows.append({
            'metric': metric,
            'raw_mean': round(float(raw.get('mean', np.nan)), 4),
            'raw_ci95': round(float(raw.get('ci95', np.nan)), 4),
            'dsp_mean': round(float(dsp.get('mean', np.nan)), 4),
            'dsp_ci95': round(float(dsp.get('ci95', np.nan)), 4),
            'delta_dsp_minus_raw': round(float(dsp.get('mean', np.nan) - raw.get('mean', np.nan)), 4),
            'paired_metric': paired_metric if metric == paired_metric else '',
            'paired_t_stat': round(float(paired_stats.get('t_stat', np.nan)), 4) if metric == paired_metric else np.nan,
            'paired_p_value': round(float(paired_stats.get('p_value', np.nan)), 6) if metric == paired_metric else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df

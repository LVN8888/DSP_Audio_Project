import json
import math
import os
import random
from typing import Dict, Iterable, List

import numpy as np
import torch
from scipy import stats


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confidence_interval_95(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else float('nan')
    if n <= 1:
        return {'mean': mean, 'ci95': 0.0, 'lower': mean, 'upper': mean, 'n': int(n)}
    sem = stats.sem(values)
    margin = float(stats.t.ppf(0.975, df=n - 1) * sem)
    return {
        'mean': mean,
        'ci95': margin,
        'lower': mean - margin,
        'upper': mean + margin,
        'n': int(n),
    }


def paired_t_test(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {'t_stat': float('nan'), 'p_value': float('nan')}
    t_stat, p_value = stats.ttest_rel(a, b, nan_policy='omit')
    return {'t_stat': float(t_stat), 'p_value': float(p_value)}


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

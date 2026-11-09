"""Métricas de clasificación, intervalos de confianza bootstrap y tests pareados.

La métrica principal es macro-F1 (promedio del F1 de cada clase), porque no se deja engañar
por la clase mayoritaria como la accuracy.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, labels):
    """y_true / y_pred: índices enteros (posición en `labels`)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(labels)
    rng = list(range(k))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=rng, zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=rng, average="weighted",
                                                  zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f.mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "weighted_f1": float(wf),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in rng},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=rng).tolist(),
        "labels": list(labels),
    }


def _macro_f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)           # = 2tp + fp + fn
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def bootstrap_macro_f1_ci(y_true, y_pred, k, n_boot=1000, seed=0, alpha=0.05):
    y, p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = _macro_f1_fast(y[i], p[i], k)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, k, n_boot=1000, seed=0, alpha=0.05):
    """Diferencia de macro-F1 (A - B) sobre las MISMAS muestras remuestreadas.

    Devuelve (delta, ic_inf, ic_sup, p_aprox). El p-valor bilateral es aproximado:
    2 * min(P(d<=0), P(d>=0)) sobre la distribución bootstrap.
    """
    y, a, b = (np.asarray(v) for v in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        i = rng.integers(0, n, n)
        diffs[t] = _macro_f1_fast(y[i], a[i], k) - _macro_f1_fast(y[i], b[i], k)
    delta = _macro_f1_fast(y, a, k) - _macro_f1_fast(y, b, k)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(delta), float(lo), float(hi), float(p)


def holm_correction(pvalues):
    """Corrección de Holm-Bonferroni (controla el error por comparaciones múltiples)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

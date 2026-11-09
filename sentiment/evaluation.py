"""Guarda predicciones y métricas de una evaluación en un formato común.

Tanto el experimento sin entrenar como el de fine-tuning escriben exactamente lo mismo,
así 04_compare.py los trata igual.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json
from .metrics import bootstrap_macro_f1_ci, compute_metrics


def save_evaluation(out_dir, split, ids, y_true, probs, labels, meta, boot_cfg):
    """probs: (n x len(labels)) sobre las etiquetas del dataset. Devuelve el dict de métricas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = probs.argmax(axis=1)

    pred = pd.DataFrame({"id": ids,
                         "y_true": [labels[i] for i in y_true],
                         "y_pred": [labels[i] for i in y_pred]})
    for j, lab in enumerate(labels):
        pred[f"p_{lab}"] = np.round(probs[:, j], 6)
    pred.to_csv(out_dir / f"predictions_{split}.csv", index=False, encoding="utf-8")

    m = compute_metrics(y_true, y_pred, labels)
    lo, hi = bootstrap_macro_f1_ci(y_true, y_pred, len(labels),
                                   n_boot=boot_cfg["n_boot"], seed=boot_cfg["seed"])
    m.update({"macro_f1_ci95": [lo, hi], "split": split, **meta})
    write_json(out_dir / f"metrics_{split}.json", m)
    return m

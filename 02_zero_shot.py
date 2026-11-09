#!/usr/bin/env python
"""Experimento 1: modelos originales evaluados sobre tu dataset SIN entrenar con él.

Solo participan los modelos con `zero_shot: true` (los que ya traen cabeza de sentimiento).
Los modelos base (BETO, RoBERTa-BNE...) no tienen cabeza entrenada: no se pueden evaluar así.
"""
import argparse

from sentiment.common import load_config
from sentiment.experiments import run_zero_shot

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", help="nombres de modelos (por defecto, todos)")
    ap.add_argument("--splits", nargs="*", default=["test"], choices=["train", "val", "test"],
                    help="particiones a evaluar (por defecto, solo test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run_zero_shot(load_config(a.config), a.models, tuple(a.splits), a.overwrite)

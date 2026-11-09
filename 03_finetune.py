#!/usr/bin/env python
"""Experimento 2: los mismos modelos, afinados sobre train y evaluados en test.

Es reanudable: si una combinación (modelo, semilla) ya tiene resultados, se salta.
"""
import argparse

from sentiment.common import load_config
from sentiment.experiments import run_finetune

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", help="nombres de modelos (por defecto, todos)")
    ap.add_argument("--seeds", nargs="*", type=int, help="semillas (por defecto, las del config)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run_finetune(load_config(a.config), a.models, a.seeds, a.overwrite)

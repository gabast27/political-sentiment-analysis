#!/usr/bin/env python
"""Crea las particiones train/val/test (una sola vez; todos los experimentos las reutilizan)."""
import argparse

from sentiment.common import load_config
from sentiment.experiments import run_make_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    run_make_splits(load_config(ap.parse_args().config))

#!/usr/bin/env python
"""Compara todos los resultados: tablas, IC bootstrap, tests pareados y figuras."""
import argparse

from sentiment.common import load_config
from sentiment.comparison import run_compare

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    run_compare(load_config(ap.parse_args().config))

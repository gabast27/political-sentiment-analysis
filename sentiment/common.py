"""Utilidades comunes: configuración, semillas, dispositivo y trazabilidad."""
import json
import platform
import random
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
                    encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred="auto"):
    import torch
    if preferred != "auto":
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_info():
    status = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"), "tag": _git("describe", "--tags", "--always"),
            "dirty": None if status is None else bool(status)}


def library_versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["torch", "transformers", "scikit-learn", "numpy", "pandas", "PyYAML"]:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_info(cfg, device, extra=None):
    import torch
    from . import __version__
    info = {
        "code_version": __version__,
        "git": git_info(),
        "libraries": library_versions(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None,
        "config": cfg,
    }
    if extra:
        info.update(extra)
    return info

"""Carga del dataset etiquetado y partición train/val/test sin fuga de datos.

Decisiones de diseño:
- Partición ESTRATIFICADA por etiqueta (las tres particiones mantienen la misma distribución).
- Textos idénticos (tras normalizar) van siempre a la misma partición, para que el test no
  contenga comentarios que el modelo ya vio en el entrenamiento.
- Si tienes una columna de grupo (p. ej. `video_id`), la partición es por grupo: todos los
  comentarios de un vídeo caen en la misma partición. Es más exigente y más realista, porque
  los comentarios de un mismo vídeo comparten tema y vocabulario.
- Los resultados se guardan en disco: todos los experimentos usan EXACTAMENTE los mismos datos.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .common import read_json, write_json
from .labels import CANON, target_labels


def text_key(text):
    return " ".join(str(text).casefold().split())


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_labeled(dcfg):
    df = pd.read_csv(dcfg["path"], dtype=str, keep_default_na=False, encoding="utf-8")
    id_c, text_c, label_c = dcfg["id_col"], dcfg["text_col"], dcfg["label_col"]
    group_c = dcfg.get("group_col")
    needed = [id_c, text_c, label_c] + ([group_c] if group_c else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {dcfg['path']}: {missing}. Tiene: {list(df.columns)}")

    out = pd.DataFrame({
        "id": df[id_c].str.strip(),
        "text": df[text_c].str.strip(),
        "label_raw": df[label_c].str.strip(),
    })
    if group_c:
        out["group"] = df[group_c].str.strip()

    n0 = len(out)
    out = out[(out["text"] != "") & (out["label_raw"] != "")]
    if len(out) < n0:
        print(f"Aviso: {n0 - len(out)} filas con texto o etiqueta vacíos descartadas.")
    if out["id"].duplicated().any():
        raise ValueError(f"Hay {int(out['id'].duplicated().sum())} ids repetidos; deben ser únicos.")

    label_map = {str(k).strip().lower(): v for k, v in dcfg["label_map"].items()}
    mapped = out["label_raw"].str.lower().map(label_map)
    if mapped.isna().any():
        bad = sorted(out.loc[mapped.isna(), "label_raw"].unique())
        raise ValueError(f"Etiquetas sin traducir en data.label_map: {bad}. "
                         f"Añádelas en config.yaml (valores válidos: {CANON}).")
    invalid = sorted(set(mapped) - set(CANON))
    if invalid:
        raise ValueError(f"data.label_map contiene valores no válidos {invalid}; usa {CANON}.")
    out["label"] = mapped
    out["text_key"] = out["text"].map(text_key)
    return out.reset_index(drop=True)


def _group_holdout(idx, y, groups, frac, seed):
    """Separa ~frac de los datos respetando grupos y estratificando. -> (holdout_idx, rest_idx)."""
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_local, hold_local = next(sgkf.split(idx, y, groups))
    return idx[hold_local], idx[rest_local]


def make_splits(df, scfg, group_col=None):
    """Devuelve ({'train','val','test'} -> DataFrame, informe)."""
    tr, va, te = scfg["train"], scfg["val"], scfg["test"]
    if abs(tr + va + te - 1) > 1e-6:
        raise ValueError("split.train + split.val + split.test debe sumar 1")
    seed = scfg["seed"]

    counts = df["label"].value_counts()
    if counts.min() < 20:
        print(f"Aviso: la clase menos frecuente tiene solo {counts.min()} ejemplos "
              f"({counts.idxmin()}); las métricas por clase serán inestables.")

    groups_raw = df["group"] if group_col else df["text_key"]
    groups = pd.factorize(groups_raw)[0]
    y = df["label"].to_numpy()
    idx = np.arange(len(df))
    singleton = len(np.unique(groups)) == len(df)

    if singleton:   # sin grupos reales: partición estratificada exacta
        rest, test_idx = train_test_split(idx, test_size=te, stratify=y, random_state=seed)
        train_idx, val_idx = train_test_split(rest, test_size=va / (tr + va), stratify=y[rest],
                                              random_state=seed)
    else:           # con grupos: aproximada (el tamaño depende del tamaño de los grupos)
        test_idx, rest = _group_holdout(idx, y, groups, te, seed)
        val_idx, train_idx = _group_holdout(rest, y[rest], groups[rest], va / (tr + va), seed)

    parts = {"train": df.iloc[train_idx], "val": df.iloc[val_idx], "test": df.iloc[test_idx]}

    # Fuga residual: mismo texto en particiones distintas (solo posible si se parte por grupo)
    removed = {"train": 0, "val": 0}
    test_keys = set(parts["test"]["text_key"])
    for name, other in (("val", test_keys), ("train", test_keys | set(parts["val"]["text_key"]))):
        before = len(parts[name])
        parts[name] = parts[name][~parts[name]["text_key"].isin(other)]
        removed[name] = before - len(parts[name])

    dup_texts = int(df.duplicated("text_key", keep=False).sum())
    conflicts = int((df.groupby("text_key")["label"].nunique() > 1).sum())
    labels = target_labels(df["label"].unique())
    report = {
        "seed": seed,
        "strategy": ("por grupo (" + group_col + ")") if group_col else "por texto normalizado",
        "exact_stratified_split": singleton,
        "labels": labels,
        "n_total": len(df),
        "n_rows_with_duplicated_text": dup_texts,
        "n_texts_with_conflicting_labels": conflicts,
        "rows_removed_for_cross_split_overlap": removed,
        "splits": {
            name: {
                "n": len(p),
                "fraction": round(len(p) / len(df), 4),
                "label_counts": p["label"].value_counts().reindex(labels, fill_value=0).to_dict(),
                "label_fraction": p["label"].value_counts(normalize=True)
                                   .reindex(labels, fill_value=0).round(4).to_dict(),
            } for name, p in parts.items()
        },
    }
    return parts, report


def save_splits(parts, report, out_dir, source_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["id", "text", "label", "group"] if c in parts["train"].columns]
    for name, p in parts.items():
        p[cols].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    report = {**report, "source": str(source_path), "source_sha256": sha256_file(source_path)}
    write_json(out_dir / "split_report.json", report)
    return report


def load_splits(out_dir):
    out_dir = Path(out_dir)
    if not (out_dir / "split_report.json").exists():
        raise FileNotFoundError(f"No hay particiones en {out_dir}. Ejecuta antes 01_make_splits.py")
    report = read_json(out_dir / "split_report.json")
    parts = {n: pd.read_csv(out_dir / f"{n}.csv", dtype=str, keep_default_na=False, encoding="utf-8")
             for n in ("train", "val", "test")}
    return parts, report["labels"], report

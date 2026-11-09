"""Orquestación de los tres experimentos: particiones, sin entrenar y fine-tuning."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import get_device, run_info, slugify, write_json
from .data import load_labeled, load_splits, make_splits, save_splits
from .evaluation import save_evaluation
from .inference import predict_probs
from .labels import aggregate_probs, class_to_target_matrix, model_label_plan
from .preprocess import build_preprocessor
from .training import fine_tune


def _select(models, only, key=None):
    sel = [m for m in models if (not only or m["name"] in only) and (key is None or m.get(key, True))]
    unknown = set(only or []) - {m["name"] for m in models}
    if unknown:
        raise ValueError(f"Modelos desconocidos: {sorted(unknown)}. "
                         f"Disponibles: {[m['name'] for m in models]}")
    return sel


# ---------------------------------------------------------------- 1. particiones
def run_make_splits(cfg):
    df = load_labeled(cfg["data"])
    parts, report = make_splits(df, cfg["split"], cfg["data"].get("group_col"))
    report = save_splits(parts, report, cfg["split"]["out_dir"], cfg["data"]["path"])
    print(f"Total: {report['n_total']} | estrategia: {report['strategy']}")
    for name, s in report["splits"].items():
        print(f"  {name:<5} n={s['n']:<6} ({s['fraction']:.1%})  {s['label_counts']}")
    print(f"  textos duplicados: {report['n_rows_with_duplicated_text']} filas | "
          f"textos con etiquetas contradictorias: {report['n_texts_with_conflicting_labels']}")
    return report


# ---------------------------------------------------------------- 2. sin entrenar
def run_zero_shot(cfg, only=None, split_names=("test",), overwrite=False):
    parts, labels, _ = load_splits(cfg["split"]["out_dir"])
    label2id = {l: i for i, l in enumerate(labels)}
    device = get_device(cfg.get("device", "auto"))
    inf = cfg["inference"]
    results = Path(cfg["results_dir"]) / "zero_shot"
    models = _select(cfg["models"], only, key="zero_shot")
    print(f"Dispositivo: {device} | modelos: {[m['name'] for m in models]}")

    for m in models:
        out_dir = results / slugify(m["name"])
        if not overwrite and all((out_dir / f"metrics_{s}.json").exists() for s in split_names):
            print(f"[{m['name']}] ya evaluado (usa --overwrite para repetir)")
            continue
        print(f"[{m['name']}] {m['hf_id']}")
        kw = {"revision": m["revision"]} if m.get("revision") else {}
        tok = AutoTokenizer.from_pretrained(m["hf_id"], **kw)
        model = AutoModelForSequenceClassification.from_pretrained(m["hf_id"], **kw).to(device)

        plan = model_label_plan(model.config.id2label, m.get("label_map"))
        matrix = class_to_target_matrix(plan, labels, m["name"])
        prep = build_preprocessor(m.get("preprocess"))
        meta = {"model": m["name"], "hf_id": m["hf_id"], "mode": "zero_shot", "seed": None,
                "model_commit": getattr(model.config, "_commit_hash", None)}
        for split in split_names:
            df = parts[split]
            texts = [prep(t) for t in df["text"]]
            probs = predict_probs(model, tok, texts, inf["batch_size"], inf["max_length"], device,
                                  inf["precision"])
            probs = aggregate_probs(probs, matrix)
            y = df["label"].map(label2id).to_numpy()
            r = save_evaluation(out_dir, split, df["id"].tolist(), y, probs, labels, meta, cfg["bootstrap"])
            print(f"    {split}: macro-F1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}  "
                  f"IC95=[{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}]")
        write_json(out_dir / "run_info.json",
                   run_info(cfg, device, {"model": m, "label_mapping": plan, "target_labels": labels}))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- 3. fine-tuning
def run_finetune(cfg, only=None, seeds=None, overwrite=False):
    parts, labels, _ = load_splits(cfg["split"]["out_dir"])
    device = get_device(cfg.get("device", "auto"))
    tcfg = cfg["train"]
    seeds = seeds or tcfg["seeds"]
    results = Path(cfg["results_dir"]) / "finetune"
    models = _select(cfg["models"], only)
    print(f"Dispositivo: {device} | modelos: {[m['name'] for m in models]} | semillas: {seeds}")

    for m in models:
        for seed in seeds:
            out_dir = results / slugify(m["name"]) / f"seed_{seed}"
            if not overwrite and (out_dir / "metrics_test.json").exists():
                print(f"[{m['name']} seed {seed}] ya hecho (usa --overwrite para repetir)")
                continue
            print(f"[{m['name']} seed {seed}] {m['hf_id']}")
            test_m, history, meta = fine_tune(m, parts, labels, tcfg, cfg["bootstrap"], seed, device, out_dir)
            write_json(out_dir / "history.json", history)
            write_json(out_dir / "run_info.json", run_info(cfg, device, {"model": m, **meta}))
            print(f"    TEST macro-F1={test_m['macro_f1']:.4f}  acc={test_m['accuracy']:.4f}  "
                  f"(mejor epoch {meta['best_epoch']}, {meta['train_seconds']} s)")

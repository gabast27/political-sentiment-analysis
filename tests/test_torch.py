"""Pruebas de extremo a extremo con modelos diminutos locales (sin internet)."""
import copy

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import synth_sentences  # noqa: E402
from sentiment.comparison import run_compare  # noqa: E402
from sentiment.experiments import run_finetune, run_make_splits, run_zero_shot  # noqa: E402
from sentiment.training import load_for_finetuning  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

LABELS = ["NEG", "NEU", "POS"]


def test_reordenar_cabeza_conserva_las_predicciones(tiny_models):
    """Si el modelo usa el orden POS,NEG,NEU, tras reordenar a NEG,NEU,POS cada clase
    debe dar exactamente el mismo logit que antes."""
    path = tiny_models["sent_perm"]
    tok = AutoTokenizer.from_pretrained(path)
    original = AutoModelForSequenceClassification.from_pretrained(path).eval()
    nuevo, action = load_for_finetuning(path, LABELS)
    nuevo.eval()
    assert action == "reuse"
    enc = tok(["excelente apoyo gracias", "corrupto fraude"], padding=True, return_tensors="pt")
    with torch.no_grad():
        a, b = original(**enc).logits, nuevo(**enc).logits
    # original: 0=POS 1=NEG 2=NEU  ->  nuevo: 0=NEG 1=NEU 2=POS
    assert torch.allclose(a[:, 1], b[:, 0], atol=1e-5)
    assert torch.allclose(a[:, 2], b[:, 1], atol=1e-5)
    assert torch.allclose(a[:, 0], b[:, 2], atol=1e-5)
    assert nuevo.config.id2label == {0: "NEG", 1: "NEU", 2: "POS"}


def test_modelo_base_recibe_cabeza_nueva(tiny_models):
    model, action = load_for_finetuning(tiny_models["base"], LABELS)
    assert action == "fresh" and model.config.num_labels == 3


@pytest.fixture
def cfg(tmp_path, tiny_models):
    df = synth_sentences(250)
    df.to_csv(tmp_path / "data.csv", index=False)
    plain = {"method": "none", "replacements": {}}
    return {
        "device": "cpu", "results_dir": str(tmp_path / "results"),
        "data": {"path": str(tmp_path / "data.csv"), "id_col": "id", "text_col": "text",
                 "label_col": "label", "group_col": None,
                 "label_map": {"negativo": "NEG", "neutral": "NEU", "positivo": "POS"}},
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 1, "out_dir": str(tmp_path / "splits")},
        "inference": {"batch_size": 32, "max_length": 32, "precision": "fp32"},
        "train": {"seeds": [1, 2], "epochs": 15, "batch_size": 32, "eval_batch_size": 64,
                  "grad_accum_steps": 1, "lr": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1,
                  "max_length": 32, "patience": 15, "class_weights": "balanced", "precision": "fp32",
                  "save_model": False},
        "bootstrap": {"n_boot": 100, "seed": 0},
        "models": [
            {"name": "tiny_sent", "hf_id": tiny_models["sent"], "zero_shot": True, "preprocess": plain},
            {"name": "tiny_perm", "hf_id": tiny_models["sent_perm"], "zero_shot": True, "preprocess": plain},
            {"name": "tiny_base", "hf_id": tiny_models["base"], "zero_shot": False, "preprocess": plain},
        ],
    }


def test_flujo_completo(cfg):
    run_make_splits(cfg)
    run_zero_shot(cfg, split_names=("test", "val"))
    run_finetune(cfg)
    summary = run_compare(copy.deepcopy(cfg))

    res = cfg["results_dir"]
    from pathlib import Path
    for f in ["comparison/summary.csv", "comparison/summary.md", "comparison/significance.csv",
              "comparison/finetuning_gain.csv", "comparison/macro_f1.png",
              "comparison/confusion_matrices.png", "zero_shot/tiny_sent/metrics_test.json",
              "finetune/tiny_base/seed_2/history.json", "finetune/tiny_perm/seed_1/run_info.json"]:
        assert (Path(res) / f).exists(), f

    ft = summary[summary["mode"] == "fine-tuning"].set_index("model")
    zs = summary[summary["mode"] == "sin entrenar"].set_index("model")
    # Sin entrenar, una cabeza aleatoria rinde cerca del azar; tras afinar, debe aprender de sobra
    assert (zs["macro_f1"] < 0.6).all()
    assert (ft["macro_f1"] > 0.85).all(), ft["macro_f1"]
    # El modelo base no participa en el experimento sin entrenar
    assert "tiny_base" not in zs.index and "tiny_base" in ft.index
    # Reanudable: una segunda ejecución no repite nada
    run_finetune(cfg)

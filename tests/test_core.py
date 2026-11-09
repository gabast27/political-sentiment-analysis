"""Tests de la lógica sin torch: etiquetas, métricas, particiones y preprocesado."""
import numpy as np
import pandas as pd
import pytest

from sentiment.data import load_labeled, make_splits
from sentiment.labels import (aggregate_probs, canon_of, class_to_target_matrix, model_label_plan,
                              plan_head)
from sentiment.metrics import (bootstrap_macro_f1_ci, compute_metrics, holm_correction,
                               paired_bootstrap_diff)
from sentiment.preprocess import build_preprocessor


# ------------------------------------------------------------------ etiquetas
def test_canon_of():
    assert canon_of("NEG") == "NEG" and canon_of("Positive") == "POS" and canon_of("neutral") == "NEU"
    assert canon_of("Very Negative") == "NEG"
    assert canon_of("LABEL_0") is None
    assert canon_of("1 star", {"1 star": "NEG"}) == "NEG"


def test_plan_head_reutiliza_o_reinicia():
    assert plan_head({0: "NEG", 1: "NEU", 2: "POS"}, ["NEG", "NEU", "POS"]) == ("reuse", [0, 1, 2])
    assert plan_head({0: "POS", 1: "NEG", 2: "NEU"}, ["NEG", "NEU", "POS"]) == ("reuse", [1, 2, 0])
    assert plan_head({0: "LABEL_0", 1: "LABEL_1"}, ["NEG", "NEU", "POS"]) == ("fresh", None)
    assert plan_head({0: "NEG", 1: "NEU", 2: "POS"}, ["NEG", "POS"]) == ("fresh", None)


def test_agregar_estrellas_a_tres_clases():
    ov = {"1 star": "NEG", "2 stars": "NEG", "3 stars": "NEU", "4 stars": "POS", "5 stars": "POS"}
    plan = model_label_plan({0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}, ov)
    m = class_to_target_matrix(plan, ["NEG", "NEU", "POS"])
    out = aggregate_probs(np.array([[0.3, 0.3, 0.1, 0.2, 0.1]], dtype=np.float32), m)
    assert np.allclose(out, [[0.6, 0.1, 0.3]])


def test_dataset_binario_ignora_neutral():
    plan = model_label_plan({0: "NEG", 1: "NEU", 2: "POS"})
    m = class_to_target_matrix(plan, ["NEG", "POS"])
    out = aggregate_probs(np.array([[0.2, 0.6, 0.2]], dtype=np.float32), m)
    assert np.allclose(out, [[0.5, 0.5]]) and np.allclose(out.sum(), 1)


def test_clase_sin_traducir_da_error_claro():
    with pytest.raises(ValueError, match="label_map"):
        class_to_target_matrix(model_label_plan({0: "LABEL_0", 1: "LABEL_1"}), ["NEG", "POS"], "m")


# ------------------------------------------------------------------ métricas
def test_metricas_perfectas_y_bootstrap():
    y = np.array([0, 1, 2] * 50)
    m = compute_metrics(y, y, ["NEG", "NEU", "POS"])
    assert m["macro_f1"] == 1.0 and m["accuracy"] == 1.0
    lo, hi = bootstrap_macro_f1_ci(y, y, 3, n_boot=200)
    assert lo == hi == 1.0


def test_macro_f1_coincide_con_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    assert compute_metrics(y, p, ["a", "b", "c"])["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))


def test_bootstrap_pareado():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 600)
    good = np.where(rng.random(600) < 0.85, y, rng.integers(0, 3, 600))
    bad = rng.integers(0, 3, 600)
    delta, lo, hi, p = paired_bootstrap_diff(y, good, bad, 3, n_boot=300)
    assert delta > 0.3 and lo > 0 and p < 0.05
    d0, _, _, p0 = paired_bootstrap_diff(y, good, good, 3, n_boot=100)
    assert d0 == 0 and p0 == 1.0


def test_holm():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


# ------------------------------------------------------------------ particiones
def _write_csv(tmp_path, df):
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return {"path": str(p), "id_col": "id", "text_col": "text", "label_col": "label",
            "group_col": None, "label_map": {"negativo": "NEG", "neutral": "NEU", "positivo": "POS"}}


def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["negativo", "neutral", "positivo"], size=n, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({"id": [f"i{k}" for k in range(n)],
                         "text": [f"comentario único número {k} sobre política" for k in range(n)],
                         "label": labels})


SCFG = {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42}


def test_particion_estratificada_exacta(tmp_path):
    df = load_labeled(_write_csv(tmp_path, _synthetic()))
    parts, rep = make_splits(df, SCFG)
    assert rep["exact_stratified_split"]
    sizes = {k: len(v) for k, v in parts.items()}
    assert sizes == {"train": 1400, "val": 300, "test": 300}
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for p in parts.values():
        assert p["label"].value_counts(normalize=True)["NEG"] == pytest.approx(0.5, abs=0.02)


def test_textos_duplicados_van_a_la_misma_particion(tmp_path):
    base = _synthetic(900)
    dup = base.iloc[:300].copy()
    dup["id"] = [f"dup{k}" for k in range(300)]
    df = load_labeled(_write_csv(tmp_path, pd.concat([base, dup], ignore_index=True)))
    parts, rep = make_splits(df, SCFG)
    keys = [set(p["text_key"]) for p in parts.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert rep["n_rows_with_duplicated_text"] == 600


def test_particion_por_grupo_no_mezcla_videos(tmp_path):
    df0 = _synthetic(2400)
    df0["video"] = [f"v{k % 60}" for k in range(len(df0))]
    cfg = _write_csv(tmp_path, df0)
    cfg["group_col"] = "video"
    parts, rep = make_splits(load_labeled(cfg), SCFG, "video")
    groups = [set(p["group"]) for p in parts.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert 0.10 < rep["splits"]["test"]["fraction"] < 0.20


def test_etiqueta_sin_traducir(tmp_path):
    df = _synthetic(50)
    df.loc[0, "label"] = "raro"
    with pytest.raises(ValueError, match="raro"):
        load_labeled(_write_csv(tmp_path, df))


# ------------------------------------------------------------------ preprocesado
def test_preprocesado_reemplazos_y_xlmt():
    f = build_preprocessor({"method": "xlmt", "replacements": {"<USUARIO>": "@user", "<URL>": "http"}})
    assert f("<USUARIO> mira <URL> y @pepe https://x.com") == "@user mira http y @user http"
    assert build_preprocessor(None)("tal cual") == "tal cual"


def test_preprocesado_pysentimiento():
    import importlib.util
    if importlib.util.find_spec("pysentimiento") is None:
        pytest.skip("pysentimiento no instalado")
    f = build_preprocessor({"method": "pysentimiento",
                            "replacements": {"<USUARIO>": "@usuario", "<URL>": "url"}})
    out = f("<USUARIO> gracias <URL> jajajajaja 😂 noooooo")
    assert "@usuario" in out and "url" in out and "emoji" in out and "nooo" in out

"""Espacio de etiquetas común.

Los modelos usan nombres distintos (NEG/NEU/POS, negative/neutral/positive, 1-5 estrellas...).
Todo se traduce a tres etiquetas canónicas y el orden de clases del experimento es SIEMPRE
el canónico filtrado por las clases presentes en tu dataset.
"""
import numpy as np

CANON = ["NEG", "NEU", "POS"]


def canon_of(name, override=None):
    """Etiqueta canónica de una clase del modelo, o None si no se puede deducir."""
    if override and name in override:
        value = override[name]
        if value not in CANON:
            raise ValueError(f"label_map: '{name}' -> '{value}' no es una de {CANON}")
        return value
    n = str(name).strip().lower()
    if "neg" in n:
        return "NEG"
    if "neu" in n:
        return "NEU"
    if "pos" in n:
        return "POS"
    return None


def target_labels(present):
    """Orden canónico de las etiquetas presentes en el dataset."""
    return [c for c in CANON if c in set(present)]


def model_label_plan(id2label, override=None):
    """{índice_de_clase: etiqueta_canónica | None} para las clases de un modelo."""
    return {int(i): canon_of(name, override) for i, name in id2label.items()}


def class_to_target_matrix(plan, targets, model_name="modelo"):
    """Matriz (n_clases_modelo x n_targets) con 1 si la clase pertenece a esa etiqueta.

    Varias clases del modelo pueden caer en la misma etiqueta (5 estrellas -> 3 clases): sus
    probabilidades se suman. Las clases que corresponden a etiquetas ausentes de tu dataset
    (p. ej. NEU en un dataset binario) se descartan y se renormaliza.
    """
    unknown = {i: n for i, n in plan.items() if n is None}
    if unknown:
        raise ValueError(
            f"{model_name}: no sé a qué etiqueta {CANON} corresponden las clases {sorted(unknown)}. "
            f"Añade `label_map` a ese modelo en config.yaml (p. ej. {{LABEL_0: NEG, LABEL_1: POS}})."
        )
    n_cls = max(plan) + 1
    m = np.zeros((n_cls, len(targets)), dtype=np.float32)
    for i, canon in plan.items():
        if canon in targets:
            m[i, targets.index(canon)] = 1.0
    for j, t in enumerate(targets):
        if m[:, j].sum() == 0:
            print(f"AVISO {model_name}: el modelo no tiene ninguna clase para '{t}'; nunca la predecirá.")
    return m


def aggregate_probs(probs, matrix):
    """Probabilidades del modelo -> probabilidades sobre las etiquetas del dataset."""
    out = probs @ matrix
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def plan_head(id2label, targets, override=None):
    """¿Se puede reutilizar la cabeza de clasificación del modelo al afinar?

    Devuelve ("reuse", perm) si las clases del modelo son una permutación exacta de `targets`
    (perm[t] = índice de clase del modelo que corresponde al target t); si no, ("fresh", None)
    y la cabeza se reinicia.
    """
    if not id2label:
        return "fresh", None
    plan = model_label_plan(id2label, override)
    values = list(plan.values())
    if None in values or sorted(values) != sorted(targets):
        return "fresh", None
    by_canon = {c: i for i, c in plan.items()}
    return "reuse", [by_canon[t] for t in targets]

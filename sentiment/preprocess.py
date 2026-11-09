"""Preprocesado específico por modelo.

Cada modelo se entrenó con un formato de texto concreto (p. ej. RoBERTuito/BETO-sentiment
esperan `@usuario` y `url`; XLM-T espera `@user` y `http`). Usar el formato correcto importa
mucho en la evaluación SIN entrenamiento: con el formato equivocado el modelo "original" rinde
peor de lo que realmente puede.

Se aplica igual en inferencia directa y en fine-tuning.
"""
import importlib.util
import pathlib


def _load_pysentimiento_module():
    """Carga solo pysentimiento/preprocessing.py (necesita únicamente `emoji`).

    Importar `pysentimiento` completo arrastra `datasets`, `torch`, etc.; por eso se carga el
    archivo directamente. Instalación:  pip install pysentimiento --no-deps
    """
    spec = importlib.util.find_spec("pysentimiento")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("method: pysentimiento requiere `pip install pysentimiento --no-deps` "
                          "(solo se usa su módulo de preprocesado).")
    path = pathlib.Path(list(spec.submodule_search_locations)[0]) / "preprocessing.py"
    s = importlib.util.spec_from_file_location("_pysentimiento_preprocessing", path)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    return module


def _xlmt(text):
    """Preprocesado de la tarjeta del modelo cardiffnlp/twitter-xlm-roberta-base-sentiment."""
    out = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)


def build_preprocessor(spec):
    """spec = {"replacements": {"<URL>": "url", ...}, "method": none | pysentimiento | xlmt}"""
    spec = spec or {}
    replacements = spec.get("replacements") or {}
    method = spec.get("method", "none")
    if method == "none":
        fn = lambda t: t  # noqa: E731
    elif method == "xlmt":
        fn = _xlmt
    elif method == "pysentimiento":
        module = _load_pysentimiento_module()
        fn = lambda t: module.preprocess_tweet(t, lang="es")  # noqa: E731
    else:
        raise ValueError(f"preprocess.method desconocido: {method}")

    def run(text):
        for old, new in replacements.items():
            text = text.replace(old, new)
        return fn(text)

    return run

"""Compara los resultados de todos los modelos (sin entrenar y con fine-tuning).

Genera en results/comparison/: summary.csv/.md, per_class.csv, significance.csv,
finetuning_gain.csv y figuras. Todo se calcula sobre el MISMO conjunto de test.
"""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import read_json  # noqa: E402
from .metrics import holm_correction, paired_bootstrap_diff  # noqa: E402

MODE_NAME = {"zero_shot": "sin entrenar", "finetune": "fine-tuning"}
METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
           "weighted_f1", "cohen_kappa", "mcc"]


def _collect(results_dir):
    rows, per_class, systems = [], [], {}
    for path in sorted(Path(results_dir).glob("*/*/metrics_test.json")) + \
            sorted(Path(results_dir).glob("*/*/*/metrics_test.json")):
        m = read_json(path)
        rows.append({"model": m["model"], "mode": m["mode"], "seed": m.get("seed"),
                     **{k: m[k] for k in METRICS}, "ci_lo": m["macro_f1_ci95"][0],
                     "ci_hi": m["macro_f1_ci95"][1], "n": m["n"], "run_dir": str(path.parent)})
        for lab, d in m["per_class"].items():
            per_class.append({"model": m["model"], "mode": m["mode"], "seed": m.get("seed"),
                              "label": lab, **d})
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def _summary(runs):
    out = []
    for (model, mode), g in runs.groupby(["model", "mode"]):
        first = g.sort_values("seed", na_position="first").iloc[0]
        row = {"model": model, "mode": MODE_NAME[mode], "runs": len(g)}
        for k in METRICS:
            row[k] = g[k].mean()
            row[k + "_std"] = g[k].std(ddof=1) if len(g) > 1 else np.nan
        row["macro_f1_ci_lo"], row["macro_f1_ci_hi"] = first["ci_lo"], first["ci_hi"]
        out.append(row)
    return pd.DataFrame(out).sort_values(["mode", "macro_f1"], ascending=[True, False])


def _markdown(summary):
    def fmt(r, k):
        s = r[f"{k}_std"]
        return f"{r[k]:.3f}" + ("" if pd.isna(s) else f" ± {s:.3f}")
    head = "| Modelo | Modo | Runs | Macro-F1 | IC95% macro-F1* | Accuracy | Bal. acc. | Kappa | MCC |\n"
    head += "|---|---|---|---|---|---|---|---|---|\n"
    body = ""
    for _, r in summary.iterrows():
        body += (f"| {r['model']} | {r['mode']} | {r['runs']} | {fmt(r, 'macro_f1')} | "
                 f"[{r['macro_f1_ci_lo']:.3f}, {r['macro_f1_ci_hi']:.3f}] | {fmt(r, 'accuracy')} | "
                 f"{fmt(r, 'balanced_accuracy')} | {fmt(r, 'cohen_kappa')} | {fmt(r, 'mcc')} |\n")
    note = ("\n*IC95% bootstrap sobre el conjunto de test (1ª semilla en fine-tuning). "
            "± = desviación típica entre semillas.\n")
    return head + body + note


def _gain(summary):
    z = summary[summary["mode"] == MODE_NAME["zero_shot"]].set_index("model")
    f = summary[summary["mode"] == MODE_NAME["finetune"]].set_index("model")
    common = z.index.intersection(f.index)
    return pd.DataFrame({
        "macro_f1_sin_entrenar": z.loc[common, "macro_f1"],
        "macro_f1_fine_tuning": f.loc[common, "macro_f1"],
        "mejora_absoluta": f.loc[common, "macro_f1"] - z.loc[common, "macro_f1"],
    }).reset_index()


def _load_preds(run_dir):
    p = pd.read_csv(Path(run_dir) / "predictions_test.csv", dtype={"id": str})
    return p.set_index("id")


def _significance(runs, labels, boot):
    """Bootstrap pareado entre sistemas (1 run representativo por modelo y modo)."""
    reps = (runs.sort_values("seed", na_position="first")
                .groupby(["model", "mode"], as_index=False).first())
    names = [f"{r.model} [{MODE_NAME[r['mode']]}]" for _, r in reps.iterrows()]
    preds = [_load_preds(r.run_dir) for _, r in reps.iterrows()]
    label2id = {l: i for i, l in enumerate(labels)}
    base = preds[0].index
    rows = []
    for (i, a), (j, b) in combinations(list(enumerate(preds)), 2):
        if not (a.index.equals(base) and b.index.equals(base)):
            b, a = b.reindex(base), a.reindex(base)
        y = a["y_true"].map(label2id).to_numpy()
        delta, lo, hi, p = paired_bootstrap_diff(
            y, a["y_pred"].map(label2id).to_numpy(), b["y_pred"].map(label2id).to_numpy(),
            len(labels), n_boot=boot["n_boot"], seed=boot["seed"])
        rows.append({"A": names[i], "B": names[j], "delta_macro_f1_A_menos_B": delta,
                     "ic95_lo": lo, "ic95_hi": hi, "p_aprox": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_correction(out["p_aprox"].to_numpy())
        out["significativo_5pct"] = out["p_holm"] < 0.05
    return out


def _plot_bars(summary, path):
    models = list(dict.fromkeys(summary["model"]))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(models)), 4.5))
    width = 0.38
    for k, mode in enumerate(MODE_NAME.values()):
        sub = summary[summary["mode"] == mode].set_index("model")
        xs = [i + (k - 0.5) * width for i, m in enumerate(models) if m in sub.index]
        vals = [sub.loc[m, "macro_f1"] for m in models if m in sub.index]
        err = [0 if pd.isna(sub.loc[m, "macro_f1_std"]) else sub.loc[m, "macro_f1_std"]
               for m in models if m in sub.index]
        ax.bar(xs, vals, width, yerr=err, capsize=3, label=mode)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Macro-F1 en test: sin entrenar vs fine-tuning")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_confusions(runs, labels, path):
    reps = (runs.sort_values("seed", na_position="first")
                .groupby(["model", "mode"], as_index=False).first())
    n = len(reps)
    cols = min(4, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols, 3.4 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), reps.iterrows()):
        cm = np.array(read_json(Path(r.run_dir) / "metrics_test.json")["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.axis("on")
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{r['model']}\n{MODE_NAME[r['mode']]}", fontsize=9)
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_compare(cfg):
    results = Path(cfg["results_dir"])
    runs, per_class = _collect(results)
    if runs.empty:
        raise SystemExit("No hay resultados en results/. Ejecuta antes 02_zero_shot.py y/o 03_finetune.py")
    labels = read_json(Path(cfg["split"]["out_dir"]) / "split_report.json")["labels"]
    out = results / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    summary = _summary(runs)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    runs.drop(columns="run_dir").to_csv(out / "all_runs.csv", index=False)
    (per_class.groupby(["model", "mode", "label"], as_index=False)[["precision", "recall", "f1", "support"]]
              .mean().to_csv(out / "per_class.csv", index=False))
    gain = _gain(summary)
    if len(gain):
        gain.to_csv(out / "finetuning_gain.csv", index=False)
    sig = _significance(runs, labels, cfg["bootstrap"])
    sig.to_csv(out / "significance.csv", index=False)
    _plot_bars(summary, out / "macro_f1.png")
    _plot_confusions(runs, labels, out / "confusion_matrices.png")

    print(_markdown(summary))
    if len(gain):
        print("Mejora por fine-tuning (macro-F1):\n", gain.round(3).to_string(index=False))
    print(f"\nArchivos en {out}")
    return summary

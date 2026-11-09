"""Fine-tuning con un bucle de entrenamiento propio (PyTorch).

Se usa un bucle explícito en lugar de `Trainer` porque la API de `Trainer` cambia entre
versiones de transformers y aquí cada paso es visible y reproducible: AdamW con decaimiento
de pesos, calentamiento + decaimiento lineal, recorte de gradiente, precisión mixta,
selección del mejor epoch por macro-F1 de validación y parada temprana.
El conjunto de test NO se toca hasta el final.
"""
import copy
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .common import set_seed
from .evaluation import save_evaluation
from .inference import amp_dtype, predict_probs
from .labels import plan_head
from .metrics import compute_metrics
from .preprocess import build_preprocessor


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def _final_linear(model, n_out):
    last = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == n_out:
            last = m
    return last


@torch.no_grad()
def _permute_head(model, perm):
    lin = _final_linear(model, len(perm))
    if lin is None:
        raise RuntimeError("No encuentro la capa final de clasificación para reordenar las clases.")
    idx = torch.tensor(perm, device=lin.weight.device)
    lin.weight.copy_(lin.weight[idx].clone())
    lin.bias.copy_(lin.bias[idx].clone())


def load_for_finetuning(hf_id, labels, label_override=None, revision=None):
    """Carga el modelo con la cabeza adecuada a las etiquetas del dataset.

    - Modelo ya entrenado en sentimiento con las mismas clases: se REUTILIZA su cabeza,
      reordenada al orden de `labels` (así conserva lo aprendido).
    - Modelo base (sin cabeza) o con otro conjunto de clases: cabeza nueva.
    """
    kw = {"revision": revision} if revision else {}
    cfg = AutoConfig.from_pretrained(hf_id, **kw)
    action, perm = plan_head(getattr(cfg, "id2label", None), labels, label_override)
    if action == "reuse":
        model = AutoModelForSequenceClassification.from_pretrained(hf_id, **kw)
        _permute_head(model, perm)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=len(labels), ignore_mismatched_sizes=True, **kw)
    model.config.id2label = {i: l for i, l in enumerate(labels)}
    model.config.label2id = {l: i for i, l in enumerate(labels)}
    return model, action


def _collate(tokenizer, max_length):
    def fn(batch):
        texts, ys = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True,
                        return_tensors="pt")
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc
    return fn


def _optimizer(model, lr, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def fine_tune(model_cfg, parts, labels, tcfg, boot_cfg, seed, device, out_dir):
    """Afina un modelo con una semilla y evalúa en val y test. Devuelve las métricas de test."""
    t0 = time.time()
    set_seed(seed)
    hf_id, rev = model_cfg["hf_id"], model_cfg.get("revision")
    kw = {"revision": rev} if rev else {}
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **kw)
    model, head_action = load_for_finetuning(hf_id, labels, model_cfg.get("label_map"), rev)
    model.to(device)

    prep = build_preprocessor(model_cfg.get("preprocess"))
    label2id = {l: i for i, l in enumerate(labels)}
    texts = {s: [prep(t) for t in parts[s]["text"]] for s in ("train", "val", "test")}
    y = {s: parts[s]["label"].map(label2id).to_numpy() for s in ("train", "val", "test")}

    lr = model_cfg.get("lr", tcfg["lr"])
    epochs = model_cfg.get("epochs", tcfg["epochs"])
    bs, accum, max_len = tcfg["batch_size"], tcfg["grad_accum_steps"], tcfg["max_length"]

    loader = DataLoader(TextDataset(texts["train"], y["train"].tolist()), batch_size=bs, shuffle=True,
                        collate_fn=_collate(tokenizer, max_len),
                        generator=torch.Generator().manual_seed(seed))
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * epochs
    opt = _optimizer(model, lr, tcfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(tcfg["warmup_ratio"] * total_steps), total_steps)

    weight = None
    if tcfg["class_weights"] == "balanced":
        counts = np.bincount(y["train"], minlength=len(labels))
        weight = torch.tensor(len(y["train"]) / (len(labels) * np.maximum(counts, 1)),
                              dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    dtype = amp_dtype(tcfg["precision"], device)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None

    best = {"score": -1.0, "epoch": 0, "state": None}
    history, wait = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            yb = batch.pop("labels").to(device)
            batch = batch.to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), yb)
            running += loss.item() * len(yb)
            seen += len(yb)
            loss = loss / accum
            scaler.scale(loss).backward() if scaler else loss.backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        val_probs = predict_probs(model, tokenizer, texts["val"], tcfg["eval_batch_size"], max_len,
                                  device, tcfg["precision"])
        vm = compute_metrics(y["val"], val_probs.argmax(1), labels)
        val_loss = float(-np.log(np.clip(val_probs[np.arange(len(y["val"])), y["val"]], 1e-9, 1)).mean())
        history.append({"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss,
                        "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"]})
        print(f"  [{model_cfg['name']} s{seed}] epoch {epoch}/{epochs}  "
              f"train_loss={running / seen:.4f}  val_loss={val_loss:.4f}  val_macro_f1={vm['macro_f1']:.4f}")

        if vm["macro_f1"] > best["score"] + 1e-6:
            best.update(score=vm["macro_f1"], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                print(f"  parada temprana (sin mejora en {tcfg['patience']} epochs)")
                break

    model.load_state_dict(best["state"])
    meta = {"model": model_cfg["name"], "hf_id": hf_id, "mode": "finetune", "seed": seed,
            "best_epoch": best["epoch"], "epochs_run": len(history), "head": head_action,
            "lr": lr, "model_commit": getattr(model.config, "_commit_hash", None)}
    out = {}
    for split in ("val", "test"):
        probs = predict_probs(model, tokenizer, texts[split], tcfg["eval_batch_size"], max_len,
                              device, tcfg["precision"])
        out[split] = save_evaluation(out_dir, split, parts[split]["id"].tolist(), y[split], probs,
                                     labels, meta, boot_cfg)
    meta["train_seconds"] = round(time.time() - t0, 1)

    if tcfg["save_model"]:
        model.save_pretrained(out_dir / "model")
        tokenizer.save_pretrained(out_dir / "model")
    del model, opt, best
    if device == "cuda":
        torch.cuda.empty_cache()
    return out["test"], history, meta

"""Inferencia por lotes (sin gradientes) y selección de precisión."""
import numpy as np
import torch


def amp_dtype(precision, device):
    """None = fp32. En CPU siempre fp32."""
    if device != "cuda" or precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # auto


@torch.inference_mode()
def predict_probs(model, tokenizer, texts, batch_size, max_length, device, precision="auto"):
    """Probabilidades softmax (n_textos x n_clases_del_modelo), en el orden original."""
    model.eval()
    order = np.argsort([len(t) for t in texts], kind="stable")   # menos padding = más rápido
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    dtype = amp_dtype(precision, device)
    for s in range(0, len(texts), batch_size):
        ids = order[s:s + batch_size]
        enc = tokenizer([texts[i] for i in ids], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
            logits = model(**enc).logits
        out[ids] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out

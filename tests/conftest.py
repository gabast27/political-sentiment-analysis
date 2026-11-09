"""Fixtures: dataset sintético y modelos transformer diminutos (sin descargar nada)."""
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POS = "excelente bueno apoyo gracias felicidades genial admirable".split()
NEG = "corrupto malo renuncia fraude vergüenza asco desastre".split()
NEU = "presidente anuncio reunión entrevista informe fecha lugar".split()
FILLER = "el la de que hoy sobre con para".split()


def synth_sentences(n_per_class=250, seed=0):
    rnd = random.Random(seed)
    rows, i = [], 0
    for lab, words in (("negativo", NEG), ("neutral", NEU), ("positivo", POS)):
        for _ in range(n_per_class):
            toks = rnd.sample(words, 3) + rnd.sample(FILLER, 2)
            rnd.shuffle(toks)
            rows.append((f"id{i}", " ".join(toks) + f" {rnd.randint(0, 10**6)}", lab))
            i += 1
    rnd.shuffle(rows)
    return pd.DataFrame(rows, columns=["id", "text", "label"])


def build_tiny_model(path, id2label=None, n_labels=3):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    corpus = [" ".join(POS + NEG + NEU + FILLER)] * 5 + [str(k) for k in range(0, 1000, 7)]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tk.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=400, special_tokens=specials))
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tk.token_to_id("[CLS]")),
                                                 ("[SEP]", tk.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    kw = {}
    if id2label:
        kw = {"id2label": id2label, "label2id": {v: k for k, v in id2label.items()}}
    cfg = BertConfig(vocab_size=tk.get_vocab_size(), hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=4, intermediate_size=128, max_position_embeddings=64,
                     initializer_range=0.1,
                     num_labels=n_labels, **kw)
    import torch
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(path)
    fast.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    return {
        "sent": build_tiny_model(root / "sent", {0: "NEG", 1: "NEU", 2: "POS"}),
        "sent_perm": build_tiny_model(root / "sent_perm", {0: "POS", 1: "NEG", 2: "NEU"}),
        "base": build_tiny_model(root / "base", None, n_labels=2),   # LABEL_0 / LABEL_1
    }

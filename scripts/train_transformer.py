"""
Fine-tune DistilBERT for product classification on the same temporal split as the baseline.

Runs on CPU (GitHub Actions or a laptop) by training on a stratified subsample with 192-token inputs.
Writes reports/transformer_metrics.json so both models are compared on identical test complaints.

    python scripts/train_transformer.py --train-size 20000 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from complaints import classify, clean  # noqa: E402
from complaints.config import REPORTS, SEED  # noqa: E402

MODEL = "distilbert-base-uncased"


def encode(tok, texts, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, ids, mask, bs=64):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        out.append(model(input_ids=ids[i:i + bs], attention_mask=mask[i:i + bs]).logits.softmax(-1).numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    t0 = time.time()

    df, _ = clean.load()
    split = clean.temporal_split(df)
    y = classify.labels(df, split == "train")
    classes = sorted(y.unique())
    cid = {c: i for i, c in enumerate(classes)}
    tr = df[split == "train"].assign(y=y[split == "train"])
    # stratified subsample with a floor so small products are still learned
    per = max(a.train_size // len(classes), 300)
    tr = tr.sample(frac=1, random_state=SEED).groupby("y").head(per)
    te = df[split == "test"]
    y_te = y[split == "test"].to_numpy()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(classes))
    ids, mask = encode(tok, tr.text, a.max_len)
    labels = torch.tensor([cid[c] for c in tr.y])
    dl = DataLoader(TensorDataset(ids, mask, labels), batch_size=16, shuffle=True)
    counts = np.bincount(labels.numpy(), minlength=len(classes))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    for ep in range(a.epochs):
        model.train()
        for i, (bi, bm, bl) in enumerate(dl):
            loss = loss_fn(model(input_ids=bi, attention_mask=bm).logits, bl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if i % 100 == 0:
                print(f"epoch {ep + 1} step {i}/{len(dl)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    te_ids, te_mask = encode(tok, te.text, a.max_len)
    probs = predict(model, te_ids, te_mask)
    pred = np.array(classes)[probs.argmax(1)]
    metrics = classify.evaluate(y_te, pred, classes)
    metrics.update({"model": MODEL, "train_examples": len(tr), "epochs": a.epochs, "max_len": a.max_len,
                    "test_examples": len(te), "classes": classes, "minutes": round((time.time() - t0) / 60, 1)})
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(REPORTS / "transformer_test_probs.npy", probs.astype(np.float16))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1", "minutes")}, indent=2))


if __name__ == "__main__":
    main()

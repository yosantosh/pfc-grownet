"""Tiny Shakespeare char-level loader.
Contiguous splits by position -> true temporal generalization test
(test = later plays/acts the model never saw).
Plus a noise-robustness variant (10% random char swaps).
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "tinyshakespeare.txt"


def load(corpus_chars=260_000, context=8, val_chars=30_000, test_chars=30_000, seed=7):
    text = SRC.read_text(encoding="utf-8")
    # use a contiguous block from the start for reproducibility
    text = text[:corpus_chars + val_chars + test_chars]
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)

    def windows(a, b):
        # windows fully inside [a, b)
        X, Y = [], []
        for i in range(a, b - context):
            X.append(ids[i:i + context])
            Y.append(ids[i + context])
        return np.array(X), np.array(Y)

    n_tr = corpus_chars
    n_va = n_tr + val_chars
    Xtr, Ytr = windows(0, n_tr)
    Xva, Yva = windows(n_tr, n_va)
    Xte, Yte = windows(n_va, n_va + test_chars)
    rng = np.random.RandomState(seed)
    # noise variant: 10% positions in context replaced by random vocab ids
    Xno = Xte.copy()
    mask = rng.rand(*Xno.shape) < 0.10
    Xno[mask] = rng.randint(0, len(vocab), size=int(mask.sum()))
    info = {"vocab": vocab, "stoi": stoi, "itos": itos, "V": len(vocab),
            "context": context,
            "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte)}
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), (Xno, Yte.copy()), info


if __name__ == "__main__":
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), (Xn, Yn), info = load()
    print("V =", info["V"], "vocab sample:", info["vocab"][:20])
    print("train/val/test/noise:", Xtr.shape, Xva.shape, Xte.shape, Xn.shape)

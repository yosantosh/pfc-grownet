"""Tiny compositional language dataset for growth experiments.
Design: train on a subset of (subject, verb, place) combos,
hold out unseen combos to measure compositional generalization.
Next-token prediction with context window C.
"""
import random
import numpy as np

SUBJECTS = ["cat", "dog", "bird", "fish"]
VERBS = ["sat", "ran", "flew", "swam"]
PLACES = ["mat", "rug", "nest", "tank"]
FUNCS = ["the", "a", "on", "to", "<s>", "</s>"]

def all_sentences():
    sents = []
    for s in SUBJECTS:
        for v in VERBS:
            for p in PLACES:
                det1 = "the"
                det2 = "the"
                prep = "on" if v in ("sat",) else ("to" if v == "ran" else "to")
                # keep grammar simple but varied
                if (s, v) in [("bird", "flew"), ("fish", "swam"), ("cat", "sat"), ("dog", "ran")]:
                    prep = "on" if v == "sat" else "to"
                sents.append([det1, s, v, prep, det2, p])
    return sents

def build_vocab():
    vocab = sorted(set(FUNCS) | set(SUBJECTS) | set(VERBS) | set(PLACES))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos

def make_splits(seed=0, holdout_frac=0.25, context=4):
    rng = random.Random(seed)
    sents = all_sentences()  # 4*4*4 = 64 sentences
    rng.shuffle(sents)
    n_test = int(len(sents) * holdout_frac)
    test_sents = sents[:n_test]
    train_sents = sents[n_test:]
    vocab, stoi, itos = build_vocab()

    def windows(sent_list):
        X, Y = [], []
        for s in sent_list:
            toks = ["<s>"] * context + s + ["</s>"]
            ids = [stoi[t] for t in toks]
            for i in range(len(s) + 1):
                ctx = ids[i:i + context]
                nxt = ids[i + context]
                X.append(ctx)
                Y.append(nxt)
        return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)

    Xtr, Ytr = windows(train_sents)
    Xte, Yte = windows(test_sents)
    # further split train into train/val 85/15 by shuffling windows
    idx = np.arange(len(Xtr))
    rng2 = np.random.RandomState(seed)
    rng2.shuffle(idx)
    n_val = int(0.15 * len(idx))
    Xva, Yva = Xtr[idx[:n_val]], Ytr[idx[:n_val]]
    Xtr, Ytr = Xtr[idx[n_val:]], Ytr[idx[n_val:]]
    info = {
        "vocab": vocab, "stoi": stoi, "itos": itos,
        "train_sents": train_sents, "test_sents": test_sents,
        "context": context,
    }
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), info

if __name__ == "__main__":
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), info = make_splits()
    print("vocab:", info["vocab"], "V=", len(info["vocab"]))
    print("train windows:", Xtr.shape, "val:", Xva.shape, "test(unseen combos):", Xte.shape)
    print("example train sent:", info["train_sents"][0])
    print("example test sent:", info["test_sents"][0])

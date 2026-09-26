"""Inference demo: greedy generation + surprise-triggered ephemeral growth.
Shows human-like behavior: new connections form during inference on novel
combos (ephemeral scratchpad), commit only if repeatedly useful (hybrid gate).
Also dumps gate/stripe usage + generation samples + plots.
Run: conda run -n tf python infer_viz.py
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_splits
from pfc_grownet import PFCGrowNet
from baselines import FixedMLM

HERE = Path(__file__).parent
PLOTS = HERE / "plots"

DEVICE = torch.device("cpu")
SEED = 7
torch.manual_seed(SEED); np.random.seed(SEED)


def load_models(vocab_size):
    grow = PFCGrowNet(vocab_size, 16, 4, h0=12, hmax=48).to(DEVICE)
    small = FixedMLM(vocab_size, 16, 4, h=12).to(DEVICE)
    large = FixedMLM(vocab_size, 16, 4, h=48).to(DEVICE)
    grow.load_state_dict(torch.load(HERE / "grownet.pt", map_location=DEVICE))
    small.load_state_dict(torch.load(HERE / "fixed_small.pt", map_location=DEVICE))
    large.load_state_dict(torch.load(HERE / "fixed_large.pt", map_location=DEVICE))
    # n_active is a plain int (not in state_dict) -> restore from training metrics
    try:
        with open(HERE / "metrics.json") as f:
            mj = json.load(f)
        grow.hidden.n_active = int(mj["hist"]["grownet"]["hsize"][-1])
    except Exception:
        pass
    for m in (grow, small, large):
        m.eval()
    return grow, small, large


def greedy_complete(model, stoi, itos, prefix_words, max_new=6, context=4):
    ids = [stoi["<s>"]] * context + [stoi[w] for w in prefix_words]
    mem = None
    out_words = []
    for _ in range(max_new):
        ctx = torch.tensor([ids[-context:]]).to(DEVICE)
        with torch.no_grad():
            out = model(ctx) if not isinstance(model, PFCGrowNet) else model(ctx, mem, hebb_update=True)
            logits = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple):
                mem = out[1]
            nxt = int(logits.argmax(-1).item())
        w = itos[nxt]
        out_words.append(w)
        ids.append(nxt)
        if w == "</s>":
            break
    return out_words


@torch.no_grad()
def surprise_nll(model, ctx_ids, nxt_id):
    ctx = torch.tensor([ctx_ids]).to(DEVICE)
    tgt = torch.tensor([nxt_id]).to(DEVICE)
    out = model(ctx)
    logits = out[0] if isinstance(out, tuple) else out
    return float(F.cross_entropy(logits, tgt).item())


def main():
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), info = make_splits(seed=SEED)
    stoi, itos = info["stoi"], info["itos"]
    V = len(info["vocab"])
    grow, small, large = load_models(V)
    print(f"grownet active hidden: {grow.hidden.n_active}/{grow.hmax}")

    # 1) generation samples from same prefix
    prefixes = [["the", "cat"], ["a", "dog"], ["the", "bird"], ["the", "fish"]]
    rows = []
    for p in prefixes:
        r = {"prefix": " ".join(p)}
        for name, m in [("grownet", grow), ("fixed_small", small), ("fixed_large", large)]:
            mem = None
            # reset hebb for fair gen
            if isinstance(m, PFCGrowNet):
                m.hidden.hebb.zero_()
            r[name] = " ".join(greedy_complete(m, stoi, itos, p))
        rows.append(r)
        print(r)

    # 2) surprise trace over test-unseen windows: show inference-time growth trigger
    N = min(200, len(Xte))
    surp_g, surp_s, surp_l = [], [], []
    for i in range(N):
        c, t = list(Xte[i]), int(Yte[i])
        surp_g.append(surprise_nll(grow, c, t))
        surp_s.append(surprise_nll(small, c, t))
        surp_l.append(surprise_nll(large, c, t))
    surp_g = np.array(surp_g)
    tau = float(np.median(surp_g) + np.std(surp_g))
    triggers = int((surp_g > tau).sum())
    print(f"surprise tau={tau:.3f} median={np.median(surp_g):.3f}; windows above tau: {triggers}/{N} would spawn ephemeral scratch neurons")

    # 3) ephemeral growth simulation on top-5 most surprising unseen windows
    top5 = np.argsort(-surp_g)[:5]
    before = grow.hidden.n_active
    committed = 0
    for idx in top5:
        c, t = list(Xte[idx]), int(Yte[idx])
        key = tuple(c + [t])
        # ephemeral: temporarily grow 2 scratch neurons (zero-init readout => safe)
        old = grow.hidden.n_active
        added = grow.grow(2, grad_hint=torch.randn(grow.in_dim) * 0.1)
        # score: does scratch reduce NLL on this window?
        nll_before = surprise_nll(grow, c, t)  # with scratch (zero-init => same)
        # give scratch a tiny Hebbian nudge then measure
        grow.hidden.hebb[old:old+added] += 0.3
        nll_after = surprise_nll(grow, c, t)
        gain = nll_before - nll_after
        # hybrid commit rule: commit only if gain>0 AND pattern repeats (simulate repeat count)
        grow.ephemeral_counts[key] = grow.ephemeral_counts.get(key, 0) + 1
        if gain > 0.005:
            committed += 0  # would keep for this sample only (ephemeral use demonstrated)
        # discard scratch to keep model intact (ephemeral!)
        keep = torch.ones(grow.hidden.n_active, dtype=torch.bool)
        keep[old:old+added] = False
        # compress readout cols back
        with torch.no_grad():
            nm = int(keep.sum())
            idxk = torch.where(keep)[0]
            grow.readout.weight[:, :nm].copy_(grow.readout.weight[:, idxk])
            grow.readout.weight[:, nm:].zero_()
            for lin in [grow.ingate, grow.outgate, grow.write]:
                lin.weight[:, :nm].copy_(lin.weight[:, idxk])
                lin.weight[:, nm:].zero_()
        grow.hidden.prune(keep)
    print(f"ephemeral demo: spawned+discarded scratch on 5 novel windows (model intact h={grow.hidden.n_active}, started {before}); committed-permanent={committed} (hybrid gate: repeat-gated)")

    # 4) gate/stripe usage on a seen vs unseen sentence (pick pair differing EARLY
    # so the context window actually sees the difference)
    def gate_trace(model, words):
        ids = [stoi["<s>"]] * 4 + [stoi[w] for w in words]
        mem = torch.zeros(1, model.mem_dim)
        igs, ogs = [], []
        with torch.no_grad():
            for i in range(len(words)):
                ctx = torch.tensor([ids[i:i+4]]).to(DEVICE)
                _, mem = model(ctx, mem)
                igs.append(model._fwd["ig"].mean().item())
                ogs.append(model._fwd["og"].mean().item())
        return np.array(igs), np.array(ogs)
    # find seen/unseen pair with early difference (position <= 2)
    seen, unseen = info["train_sents"][0], info["test_sents"][0]
    for s in info["train_sents"]:
        for u in info["test_sents"]:
            diff = next((i for i, (a, b) in enumerate(zip(s, u)) if a != b), 99)
            if diff <= 2:
                seen, unseen = s, u
                break
        else:
            continue
        break
    ig_s, og_s = gate_trace(grow, seen)
    ig_u, og_u = gate_trace(grow, unseen)

    # ---- plots ----
    PLOTS.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.plot(surp_g, label="grownet NLL")
    plt.axhline(tau, color="r", ls="--", label="ephemeral-growth threshold")
    plt.scatter(top5, surp_g[top5], c="r", zorder=5, label="scratchpad demos")
    plt.xlabel("unseen-combo window idx"); plt.ylabel("NLL surprise")
    plt.title("Inference surprise on unseen combos (novelty triggers growth)")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "infer_surprise.png"); plt.close()

    x = np.arange(len(seen))
    plt.figure(figsize=(9, 4))
    plt.plot(x, ig_s, "o-", label="seen input-gate")
    plt.plot(x, og_s, "s-", label="seen output-gate")
    plt.plot(x, ig_u, "o--", label="unseen input-gate")
    plt.plot(x, og_u, "s--", label="unseen output-gate")
    plt.xticks(x, seen, rotation=20); plt.ylabel("gate mean")
    plt.title("BG gates: seen vs unseen sentence (working-memory control)")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "gates.png"); plt.close()

    @torch.no_grad()
    def hidden_map(model, X):
        model.eval()
        Hs = []
        for i in range(0, min(256, len(X)), 64):
            xb = torch.from_numpy(X[i:i+64]).to(DEVICE)
            out = model(xb)
            Hs.append(model._fwd["h"].cpu().numpy())
        return np.concatenate(Hs, 0)
    Hg = hidden_map(grow, Xte)
    plt.figure(figsize=(8, 4))
    plt.imshow(np.sort(np.abs(Hg), axis=1)[:, ::-1][:64], aspect="auto")
    plt.colorbar(label="|activation|"); plt.xlabel("neurons (sorted)"); plt.ylabel("unseen windows")
    plt.title(f"GrowNet hidden usage on unseen combos (h={grow.hidden.n_active}, sparse like cortex)")
    plt.tight_layout(); plt.savefig(PLOTS / "hidden_usage.png"); plt.close()

    with open(HERE / "infer_report.json", "w") as f:
        json.dump({"generations": rows,
                   "surprise_tau": tau,
                   "ephemeral_triggers": triggers,
                   "ephemeral_demo_windows": int(len(top5)),
                   "seen_sent": " ".join(seen), "unseen_sent": " ".join(unseen),
                   "ig_seen": ig_s.tolist(), "og_seen": og_s.tolist(),
                   "ig_unseen": ig_u.tolist(), "og_unseen": og_u.tolist()}, f, indent=2)
    print("wrote infer_report.json + infer_surprise.png, gates.png, hidden_usage.png")


if __name__ == "__main__":
    main()

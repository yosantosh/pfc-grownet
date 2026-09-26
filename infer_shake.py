"""Shakespeare inference: temperature generations, surprise trace, gates, robustness.
Run: conda run -n tf python -u infer_shake.py
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shake_data import load as load_shake
from pfc_grownet import PFCGrowNet
from baselines import FixedMLM

HERE = Path(__file__).parent
PLOTS = HERE / "plots_shake"
DEVICE = torch.device("cpu")
torch.manual_seed(7); np.random.seed(7)


def load_models(V):
    grow = PFCGrowNet(V, 24, 8, h0=3, hmax=96, mem_dim=24).to(DEVICE)
    small = FixedMLM(V, 24, 8, h=3).to(DEVICE)
    large = FixedMLM(V, 24, 8, h=96).to(DEVICE)
    grow.load_state_dict(torch.load(HERE / "shake_grownet.pt", map_location=DEVICE))
    small.load_state_dict(torch.load(HERE / "shake_fixed_small.pt", map_location=DEVICE))
    large.load_state_dict(torch.load(HERE / "shake_fixed_large.pt", map_location=DEVICE))
    try:
        mj = json.load(open(HERE / "shake_metrics.json"))
        grow.hidden.n_active = int(mj["hist"]["grownet"]["hsize"][-1])
        grow.deep.n_active = int(mj["hist"]["grownet"].get("dsize", [0])[-1])
    except Exception:
        pass
    for m in (grow, small, large):
        m.eval()
    return grow, small, large


@torch.no_grad()
def sample(model, stoi, itos, prompt, n_chars=200, temp=0.8, context=8):
    ids = [stoi["<s>"] if "<s>" in stoi else 0] * context + [stoi.get(c, 0) for c in prompt]
    mem = None
    out = list(prompt)
    for _ in range(n_chars):
        ctx = torch.tensor([ids[-context:]]).to(DEVICE)
        r = model(ctx, mem) if isinstance(model, PFCGrowNet) else model(ctx)
        lg = r[0] if isinstance(r, tuple) else r
        if isinstance(r, tuple):
            mem = r[1]
        nxt = int(torch.multinomial(F.softmax(lg[0] / temp, dim=-1), 1).item())
        out.append(itos[nxt]); ids.append(nxt)
    return "".join(out)


@torch.no_grad()
def nll_of(model, c, t):
    o = model(torch.tensor([c]).to(DEVICE))
    lg = o[0] if isinstance(o, tuple) else o
    return float(F.cross_entropy(lg, torch.tensor([t]).to(DEVICE)).item())


def main():
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), (Xno, Yno), info = load_shake(
        corpus_chars=100_000, val_chars=12_000, test_chars=12_000)
    stoi, itos, V = info["stoi"], info["itos"], info["V"]
    grow, small, large = load_models(V)
    print(f"grownet h={grow.hidden.n_active}")

    prompts = ["To be, or not", "KING:\n", "All the world's"]
    gens = {}
    for p in prompts:
        gens[p] = {}
        for name, m in [("grownet", grow), ("fixed_small", small), ("fixed_large", large)]:
            if isinstance(m, PFCGrowNet):
                m.hidden.hebb.zero_()
            gens[p][name] = sample(m, stoi, itos, p)
            print(f"--- {name} | {p!r} ---\n" + gens[p][name][:300] + "\n")

    # surprise trace on future test (first 1500 windows)
    N = 1500
    sg = np.array([nll_of(grow, list(Xte[i]), int(Yte[i])) for i in range(N)])
    sl = np.array([nll_of(large, list(Xte[i]), int(Yte[i])) for i in range(N)])
    tau = float(np.median(sg) + np.std(sg))
    print(f"surprise: grow median {np.median(sg):.3f} mean {sg.mean():.3f} | tau {tau:.3f} | "
          f"windows>tau {(sg > tau).sum()}/{N} would spawn ephemeral scratch")

    # per-position accuracy over context? + noise degradation curve
    corrupts = [0.0, 0.05, 0.10, 0.20, 0.30]
    rng = np.random.RandomState(0)
    deg = {"grownet": [], "fixed_small": [], "fixed_large": []}
    sub = 3000
    sel = np.random.choice(len(Xte), sub, replace=False)
    for p in corrupts:
        Xc = Xte[sel].copy()
        if p > 0:
            mk = rng.rand(*Xc.shape) < p
            Xc[mk] = rng.randint(0, V, size=int(mk.sum()))
        for name, m in [("grownet", grow), ("fixed_small", small), ("fixed_large", large)]:
            m.eval(); tot, cor = 0.0, 0
            with torch.no_grad():
                for i in range(0, sub, 512):
                    xb = torch.from_numpy(Xc[i:i+512]).to(DEVICE)
                    yb = torch.from_numpy(Yte[sel][i:i+512]).to(DEVICE)
                    o = m(xb); lg = o[0] if isinstance(o, tuple) else o
                    tot += F.cross_entropy(lg, yb, reduction="sum").item()
                    cor += (lg.argmax(-1) == yb).sum().item()
            deg[name].append((tot / sub, cor / sub))

    # gates on seen-ish vs future snippet
    @torch.no_grad()
    def gate_trace(model, s):
        ids = [0] * 8 + [stoi.get(c, 0) for c in s]
        mem = torch.zeros(1, model.mem_dim)
        igs, ogs = [], []
        for i in range(len(s)):
            _, mem = model(torch.tensor([ids[i:i+8]]).to(DEVICE), mem)
            igs.append(model._fwd["ig"].mean().item())
            ogs.append(model._fwd["og"].mean().item())
        return np.array(igs), np.array(ogs)
    a = "To be, or not"
    b = "".join(itos[int(x)] for x in Xte[0])
    ig_a, og_a = gate_trace(grow, a)
    ig_b, og_b = gate_trace(grow, b)

    plt.figure(figsize=(10, 4))
    plt.plot(sg[:600], label="grownet NLL", alpha=0.8)
    plt.plot(sl[:600], label="fixed_large NLL", alpha=0.8)
    plt.axhline(tau, color="r", ls="--", label="ephemeral threshold")
    plt.xlabel("future-test window"); plt.ylabel("NLL surprise")
    plt.title("Inference surprise on unseen Shakespeare segment")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_surprise.png"); plt.close()

    plt.figure(figsize=(8, 4))
    for name in deg:
        plt.plot(corrupts, [d[0] for d in deg[name]], "o-", label=name)
    plt.xlabel("input corruption rate"); plt.ylabel("CE loss (future test)")
    plt.title("Robustness: loss under char corruption (lower = better)")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_robust.png"); plt.close()

    plt.figure(figsize=(8, 4))
    for name in deg:
        plt.plot(corrupts, [d[1] for d in deg[name]], "o-", label=name)
    plt.xlabel("input corruption rate"); plt.ylabel("accuracy")
    plt.title("Robustness: accuracy under corruption")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_robust_acc.png"); plt.close()

    xa = np.arange(len(a))
    plt.figure(figsize=(10, 4))
    plt.plot(xa, ig_a, "o-", label="prompt input-gate")
    plt.plot(xa, og_a, "s-", label="prompt output-gate")
    plt.xticks(xa, list(a)); plt.ylabel("gate mean")
    plt.title("BG working-memory gates over prompt 'To be, or not'")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_gates.png"); plt.close()

    with open(HERE / "shake_infer.json", "w") as f:
        json.dump({"generations": gens, "tau": tau,
                   "median_nll_grow": float(np.median(sg)), "median_nll_large": float(np.median(sl)),
                   "robustness": {k: [{"corrupt": c, "loss": float(l), "acc": float(a_)}
                                      for (l, a_), c in zip(v, corrupts)] for k, v in deg.items()}}, f, indent=2)
    print("saved shake_surprise/robust/gates + shake_infer.json")


if __name__ == "__main__":
    main()

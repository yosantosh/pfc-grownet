"""Train + compare on Tiny Shakespeare (char-level, 260k windows).
Same protocol as train_compare.py, scaled: C=8, d=24, H0=24/HMAX=96.
Latest techniques: AdamW, cosine, clip, label-smooth, dropout, residual skip,
DA-LR, GradMax-PC growth, prune, EWC-lite, sleep replay.
Run: conda run -n tf python train_shake.py   (~6-10 min CPU)
"""
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shake_data import load as load_shake
from pfc_grownet import PFCGrowNet, SEED_N
from baselines import FixedMLM
from growth_trial import select_growth, hard_inputs, pc_hint

HERE = Path(__file__).parent
PLOTS = HERE / "plots_shake"
PLOTS.mkdir(exist_ok=True)

SEED = 7
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

CTX, D_EMB = 8, 24
H0, HMAX, MEM = SEED_N, 96, 24  # born with 3 neurons; deep layer born empty
EPOCHS, BATCH = 6, 1024
BASE_LR, WD, LS, CLIP = 3e-3, 1e-2, 0.05, 1.0
EVAL_CAP = 6000  # eval subsample for speed on 2-CPU box


def label_smooth_ce(logits, target, eps=0.05):
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return ((1 - eps) * nll + eps * (-logp.mean(dim=-1))).mean()


def train_one(model, opt, Xtr, Ytr, base_lr, ewc_lam=0.0, da_avg=2.0):
    # single-forward per batch: DA factor carried from previous batch loss (no 2x cost)
    model.train()
    tot, n = 0.0, 0
    da, prev_nll = 1.0, da_avg
    perm = np.random.permutation(len(Xtr))
    for s in range(0, len(Xtr), BATCH):
        j = perm[s:s + BATCH]
        xb = torch.from_numpy(Xtr[j]).to(DEVICE)
        yb = torch.from_numpy(Ytr[j]).to(DEVICE)
        for g in opt.param_groups:
            g["lr"] = base_lr * da
        opt.zero_grad()
        o = model(xb)
        lg = o[0] if isinstance(o, tuple) else o
        loss = label_smooth_ce(lg, yb, LS)
        if ewc_lam > 0 and hasattr(model, "ewc_penalty"):
            loss = loss + ewc_lam * model.ewc_penalty()
        loss.backward()
        if hasattr(model, "update_utility"):
            try:
                model.update_utility()
            except Exception:
                pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        lv = loss.item()
        tot += lv * len(xb); n += len(xb)
        da = 1.0 + max(-0.3, min(1.0, (lv - da_avg) / (da_avg + 1e-6)))
    return tot / n, da


@torch.no_grad()
def evaluate(model, X, Y, bs=2048):
    if len(X) > EVAL_CAP:  # stratified-ish: uniform stride subsample
        sel = np.linspace(0, len(X) - 1, EVAL_CAP).astype(int)
        X, Y = X[sel], Y[sel]
    model.eval()
    tl, tc, n = 0.0, 0, 0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEVICE)
        yb = torch.from_numpy(Y[i:i+bs]).to(DEVICE)
        o = model(xb)
        lg = o[0] if isinstance(o, tuple) else o
        tl += F.cross_entropy(lg, yb, reduction="sum").item()
        tc += (lg.argmax(-1) == yb).sum().item(); n += len(xb)
    return tl / n, tc / n


@torch.no_grad()
def grow_stats(model, X, Y):
    model.eval()
    xb = torch.from_numpy(X[:2048]).to(DEVICE)
    yb = torch.from_numpy(Y[:2048]).to(DEVICE)
    o = model(xb)
    lg = o[0] if isinstance(o, tuple) else o
    nll = F.cross_entropy(lg, yb, reduction="none")
    surp = float(nll.mean()); med = float(nll.median())
    h = model._fwd["h"]
    act = h.abs().mean(0)
    dorm = float((act < 0.12).float().mean())
    n = h.size(1)
    model.hidden.utility[:n] = 0.9 * model.hidden.utility[:n] + 0.1 * act.cpu()
    E = model._fwd["e"].detach().cpu().numpy()
    hard = np.argsort(-nll.cpu().numpy())[:512]
    Eh = E[hard] - E[hard].mean(0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(Eh, full_matrices=False)
        hint = torch.from_numpy(Vt[0].astype(np.float32))
    except Exception:
        hint = torch.randn(model.in_dim) * 0.1
    return surp, dorm, hint, med


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), (Xno, Yno), info = load_shake(
        corpus_chars=100_000, val_chars=12_000, test_chars=12_000)
    V = info["V"]
    print(f"V={V} train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    grow = PFCGrowNet(V, D_EMB, CTX, h0=H0, hmax=HMAX, mem_dim=MEM).to(DEVICE)
    small = FixedMLM(V, D_EMB, CTX, h=H0).to(DEVICE)
    large = FixedMLM(V, D_EMB, CTX, h=HMAX).to(DEVICE)
    models = {"grownet": grow, "fixed_small": small, "fixed_large": large}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=BASE_LR, weight_decay=WD) for k, m in models.items()}
    hist = {k: {"train_loss": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [],
                "noise_loss": [], "noise_acc": [], "hsize": [], "dsize": [], "edges": []} for k in models}
    sig = {"surprise": [], "dormant": []}
    replay_x, replay_y = [], []
    ewc_on, da_avg = False, 2.5

    for epoch in range(EPOCHS):
        base_lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * epoch / EPOCHS))
        for k, m in models.items():
            lam = 1e-5 if (ewc_on and k == "grownet") else 0.0
            tl, _ = train_one(m, opts[k], Xtr, Ytr, base_lr, lam, da_avg)
            hist[k]["train_loss"].append(tl)
        da_avg = 0.9 * da_avg + 0.1 * float(np.mean([hist[k]["train_loss"][-1] for k in models]))
        for k, m in models.items():
            vl, va = evaluate(m, Xva, Yva)
            te, tea = evaluate(m, Xte, Yte)
            nl, nla = evaluate(m, Xno, Yno)
            hist[k]["val_loss"].append(vl); hist[k]["val_acc"].append(va)
            hist[k]["test_loss"].append(te); hist[k]["test_acc"].append(tea)
            hist[k]["noise_loss"].append(nl); hist[k]["noise_acc"].append(nla)
        surp, dorm, hint, med = grow_stats(grow, Xva, Yva)
        sig["surprise"].append(surp); sig["dormant"].append(dorm)
        hist["grownet"]["hsize"].append(int(grow.hidden.n_active))
        hist["grownet"]["dsize"].append(int(grow.deep.n_active))
        hist["grownet"]["edges"].append(int(grow.hidden.active_edges() + grow.deep.active_edges()))
        hist["fixed_small"]["hsize"].append(H0); hist["fixed_large"]["hsize"].append(HMAX)
        hist["fixed_small"]["dsize"].append(0); hist["fixed_large"]["dsize"].append(0)
        hist["fixed_small"]["edges"].append(0); hist["fixed_large"]["edges"].append(0)
        # replay harvest: hardest 128 train windows (small pool for 2-CPU speed)
        with torch.no_grad():
            si = np.random.choice(len(Xtr), 2048, replace=False)
            xb = torch.from_numpy(Xtr[si]).to(DEVICE); yb = torch.from_numpy(Ytr[si]).to(DEVICE)
            grow.eval(); o = grow(xb); lg = o[0] if isinstance(o, tuple) else o
            nll = F.cross_entropy(lg, yb, reduction="none").cpu().numpy()
            for i in si[np.argsort(-nll)[:128]]:
                replay_x.append(Xtr[i]); replay_y.append(Ytr[i])
            replay_x, replay_y = replay_x[-512:], replay_y[-512:]
        # LOSS-DRIVEN growth in all dimensions: trial widen vs deepen vs sprout.
        cur = hist["grownet"]["val_loss"][-1]
        grow.hidden.prune_edges(0.05)
        grow.deep.prune_edges(0.05)
        rngp = np.random.RandomState(SEED + epoch)
        pti = rngp.choice(len(Xtr), 512, replace=False)
        pvi = rngp.choice(len(Xva), min(512, len(Xva)), replace=False)
        choice, res = select_growth(grow, Xtr[pti], Ytr[pti], Xva[pvi], Yva[pvi],
                                    k_widen=8, k_deep=4, k_sprout=64,
                                    steps=2, seed=epoch)
        print(f"[ep {epoch+1}] growth trials: " + ", ".join(
            f"{d}: drop {res[d]['drop']:+.4f}/{res[d]['dparams']}p" for d in ("widen", "deepen", "sprout"))
            + f" -> {choice} (h={grow.hidden.n_active}, d={grow.deep.n_active})", flush=True)
        import torch as _t
        _t.manual_seed(epoch)
        if choice == "widen":
            E, _ = hard_inputs(grow, Xtr[pti], Ytr[pti])
            hh = pc_hint(E)
            grow.grow_hidden(8, grad_hint=None if hh is None else hh.to(grow.hidden.W.device))
        elif choice == "deepen":
            _, H = hard_inputs(grow, Xtr[pti], Ytr[pti])
            hh = pc_hint(H)
            grow.grow_deep(4, grad_hint=None if hh is None else hh.to(grow.deep.W.device))
        elif choice == "sprout":
            grow.hidden.sprout_edges(64, seed=epoch)
            grow.deep.sprout_edges(32, seed=epoch + 1)
        if (epoch + 1) % 2 == 0 and len(replay_x) > 128:
            RX = np.stack(replay_x[-256:]); RY = np.array(replay_y[-256:])
            tl, _ = train_one(grow, opts["grownet"], RX, RY, base_lr * 0.5, 0.0, da_avg)
            grow.snapshot_ewc(None); ewc_on = True
            print(f"[ep {epoch+1}] SLEEP replay 256 hards (loss {tl:.3f}); EWC on", flush=True)
        if (epoch + 1) % 4 == 0:
            b = (grow.hidden.n_active, grow.deep.n_active)
            grow.prune_hidden_dormant(min_keep=SEED_N, thresh=0.05)
            grow.prune_deep_dormant(thresh=0.05)
            a = (grow.hidden.n_active, grow.deep.n_active)
            if a != b:
                print(f"[ep {epoch+1}] PRUNE h {b[0]}->{a[0]}, d {b[1]}->{a[1]}", flush=True)
        print(f"ep {epoch+1}/{EPOCHS} | " + " | ".join(
            f"{k}: tr {hist[k]['train_loss'][-1]:.3f} va {hist[k]['val_loss'][-1]:.3f}/{hist[k]['val_acc'][-1]:.3f} "
            f"te {hist[k]['test_loss'][-1]:.3f}/{hist[k]['test_acc'][-1]:.3f} nz {hist[k]['noise_loss'][-1]:.3f}/{hist[k]['noise_acc'][-1]:.3f}"
            for k in models) + f" | h={grow.hidden.n_active} d={grow.deep.n_active}", flush=True)

    for k, m in models.items():
        torch.save(m.state_dict(), HERE / f"shake_{k}.pt")
    summary = {}
    for k in models:
        tr, va, te, nz = hist[k]["train_loss"][-1], hist[k]["val_loss"][-1], hist[k]["test_loss"][-1], hist[k]["noise_loss"][-1]
        summary[k] = {"train_loss": tr, "val_loss": va, "test_loss": te, "noise_loss": nz,
                      "train_ppl": float(np.exp(min(tr, 12))), "test_ppl": float(np.exp(min(te, 12))),
                      "noise_ppl": float(np.exp(min(nz, 12))),
                      "val_acc": hist[k]["val_acc"][-1], "test_acc": hist[k]["test_acc"][-1],
                      "noise_acc": hist[k]["noise_acc"][-1],
                      "gen_gap_future": te - va, "gen_gap_noise": nz - te,
                      "params_total": count_params(models[k]), "hsize": hist[k]["hsize"][-1],
                      "dsize": hist[k]["dsize"][-1], "edges": hist[k]["edges"][-1]}
    with open(HERE / "shake_metrics.json", "w") as f:
        json.dump({"summary": summary, "hist": hist, "signals": sig}, f, indent=2)
    print(json.dumps(summary, indent=2))

    ep = np.arange(1, EPOCHS + 1)
    plt.figure(figsize=(9, 5))
    for k in models:
        plt.plot(ep, hist[k]["val_loss"], label=f"{k} val")
        plt.plot(ep, hist[k]["test_loss"], "--", label=f"{k} test-future")
        plt.plot(ep, hist[k]["noise_loss"], ":", label=f"{k} test-noisy")
    plt.xlabel("epoch"); plt.ylabel("CE loss"); plt.title("Shakespeare: val vs future-test vs noisy-test")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(PLOTS / "shake_loss.png"); plt.close()

    plt.figure(figsize=(9, 5))
    for k in models:
        plt.plot(ep, hist[k]["test_acc"], label=k)
    plt.xlabel("epoch"); plt.ylabel("next-char acc (future segment)"); plt.title("Future-segment accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_acc.png"); plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(ep, hist["grownet"]["hsize"], marker="o", label="L1 neurons")
    plt.plot(ep, hist["grownet"]["dsize"], marker="s", label="L2 neurons")
    plt.xlabel("epoch"); plt.ylabel("active neurons"); plt.title("GrowNet growth on Shakespeare (3-neuron seed)")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_growth.png"); plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(ep, sig["surprise"], "r-", label="surprise"); ax1.set_xlabel("epoch"); ax1.set_ylabel("surprise", color="r")
    ax2 = ax1.twinx(); ax2.plot(ep, sig["dormant"], "b-", label="dormant"); ax2.set_ylabel("dormant frac", color="b")
    plt.title("Growth signals"); fig.tight_layout(); plt.savefig(PLOTS / "shake_signals.png"); plt.close()

    labels = list(summary.keys())
    x = np.arange(len(labels)); w = 0.35
    gf = [summary[k]["gen_gap_future"] for k in labels]
    gn = [summary[k]["gen_gap_noise"] for k in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(x - w/2, gf, w, label="future - val"); plt.bar(x + w/2, gn, w, label="noisy - future")
    plt.xticks(x, labels); plt.ylabel("loss gap"); plt.title("Generalization gaps (lower = better)")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "shake_gaps.png"); plt.close()
    print("saved", PLOTS)


if __name__ == "__main__":
    main()

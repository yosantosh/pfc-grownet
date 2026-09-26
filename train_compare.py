"""Train + compare: PFC-GrowNet vs Fixed-Small vs Fixed-Large on tiny text.
Latest techniques: AdamW, cosine decay, grad-clip, label-smooth CE, dropout,
LayerNorm, DA-modulated LR, function-preserving growth, gradient-vote direction,
dormancy prune, EWC-lite, sleep replay, hybrid ephemeral inference hook.
Saves metrics.json + plots/*.png. Run with: conda run -n tf python train_compare.py
"""
import json, math, os, random
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
PLOTS.mkdir(exist_ok=True)

SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
CTX, D_EMB = 4, 16
H0, HMAX = 12, 48
EPOCHS = 30
BATCH = 64
BASE_LR = 3e-3
WD = 1e-2
LS = 0.05  # label smoothing
CLIP = 1.0


def label_smooth_ce(logits, target, eps=0.05):
    V = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    smooth = -logp.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


def batch_iter(X, Y, bs, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, len(X), bs):
        j = idx[s:s + bs]
        yield torch.from_numpy(X[j]).to(DEVICE), torch.from_numpy(Y[j]).to(DEVICE)


@torch.no_grad()
def evaluate(model, X, Y, bs=256):
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    all_nll = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEVICE)
        yb = torch.from_numpy(Y[i:i+bs]).to(DEVICE)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits, yb, reduction="none")
        tot_loss += loss.sum().item()
        tot_correct += (logits.argmax(-1) == yb).sum().item()
        n += len(xb)
        all_nll.extend(loss.cpu().tolist())
    return tot_loss / n, tot_correct / n, np.array(all_nll)


@torch.no_grad()
def model_stats_grownet(model, X, Y):
    """dormancy, surprise, utility snapshot, grad-hint direction (top-PC of hard inputs)."""
    model.eval()
    xb = torch.from_numpy(X[:512]).to(DEVICE)
    yb = torch.from_numpy(Y[:512]).to(DEVICE)
    out = model(xb)
    logits = out[0] if isinstance(out, tuple) else out
    nll = F.cross_entropy(logits, yb, reduction="none")
    surprise = float(nll.mean())
    h = model._fwd["h"]  # (B,n)
    act = h.abs().mean(dim=0)
    dormant_frac = float((act < 0.12).float().mean())
    # utility update (act-only; grad part added in train step)
    n = h.size(1)
    model.hidden.utility[:n] = 0.9 * model.hidden.utility[:n] + 0.1 * act.cpu()
    # grad-hint: top PC of hardest 25% input embeddings
    E = model._fwd["e"].detach().cpu().numpy()  # (B,in)
    hard = np.argsort(-nll.detach().cpu().numpy())[:max(32, len(E)//4)]
    Eh = E[hard] - E[hard].mean(0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(Eh, full_matrices=False)
        hint = torch.from_numpy(Vt[0].astype(np.float32))
    except Exception:
        hint = torch.randn(model.in_dim) * 0.1
    return surprise, dormant_frac, hint, float(nll.median())


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def cosine_lr(epoch, base=BASE_LR, epochs=EPOCHS):
    return base * 0.5 * (1 + math.cos(math.pi * epoch / epochs))


def train_one(model, opt, Xtr, Ytr, base_lr, epoch, ewc_lam=0.0, da_avg=None):
    model.train()
    tot, n, da_sum, da_n = 0.0, 0, 0.0, 0
    for xb, yb in batch_iter(Xtr, Ytr, BATCH, True):
        # DA-modulated LR: surprise vs running avg
        with torch.no_grad():
            out0 = model(xb)
            lg0 = out0[0] if isinstance(out0, tuple) else out0
            b_nll = float(F.cross_entropy(lg0, yb).item())
            if da_avg is None or da_avg <= 0:
                da = 1.0
            else:
                da = 1.0 + max(-0.3, min(1.0, (b_nll - da_avg) / (da_avg + 1e-6)))
            for g in opt.param_groups:
                g["lr"] = base_lr * da
            da_sum += da; da_n += 1
        opt.zero_grad()
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        loss = label_smooth_ce(logits, yb, LS)
        if ewc_lam > 0 and hasattr(model, "ewc_penalty"):
            loss = loss + ewc_lam * model.ewc_penalty()
        loss.backward()
        # utility with grad signal (grownet only)
        if hasattr(model, "update_utility"):
            try:
                # grad of pre-readout hidden: approximate via readout grad
                model.update_utility(grad_h=None)
            except Exception:
                pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        tot += loss.item() * len(xb); n += len(xb)
    return tot / n, (da_sum / max(1, da_n))


def main():
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), info = make_splits(seed=SEED)
    V = len(info["vocab"])
    print(f"vocab({V}): {info['vocab']}")
    print(f"train {Xtr.shape} val {Xva.shape} test-unseen-combos {Xte.shape}")

    grow = PFCGrowNet(V, D_EMB, CTX, h0=H0, hmax=HMAX).to(DEVICE)
    small = FixedMLM(V, D_EMB, CTX, h=H0).to(DEVICE)
    large = FixedMLM(V, D_EMB, CTX, h=HMAX).to(DEVICE)
    print("params: grow(seed)=%d small=%d large=%d" % (count_params(grow), count_params(small), count_params(large)))

    models = {"grownet": grow, "fixed_small": small, "fixed_large": large}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=BASE_LR, weight_decay=WD) for k, m in models.items()}
    hist = {k: {"train_loss": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [],
                "hsize": [], "dormant": [], "surprise": [], "da": []} for k in models}
    # replay buffer: list of (x, y)
    replay_x, replay_y = [], []
    best_val = {k: 1e9 for k in models}
    no_improve = {"grownet": 0}
    ewc_on = False
    da_avg = 2.0
    prev_val = None

    for epoch in range(EPOCHS):
        base_lr = cosine_lr(epoch)
        for k, m in models.items():
            lam = 1e-5 if (ewc_on and k == "grownet") else 0.0
            tl, da = train_one(m, opts[k], Xtr, Ytr, base_lr, epoch, lam, da_avg)
            hist[k]["train_loss"].append(tl)
            hist[k]["da"].append(da)
        # eval
        for k, m in models.items():
            vl, va, _ = evaluate(m, Xva, Yva)
            te_l, te_a, _ = evaluate(m, Xte, Yte)
            hist[k]["val_loss"].append(vl); hist[k]["val_acc"].append(va)
            hist[k]["test_loss"].append(te_l); hist[k]["test_acc"].append(te_a)
            if vl < best_val[k]:
                best_val[k] = vl
        da_avg = 0.9 * da_avg + 0.1 * float(np.mean([hist[k]["train_loss"][-1] for k in models]))
        # grownet instrum
        surp, dorm, hint, med = model_stats_grownet(grow, Xva, Yva)
        hist["grownet"]["hsize"].append(int(grow.hidden.n_active))
        hist["grownet"]["dormant"].append(dorm)
        hist["grownet"]["surprise"].append(surp)
        for k in ("fixed_small", "fixed_large"):
            hist[k]["hsize"].append(H0 if k == "fixed_small" else HMAX)
            hist[k]["dormant"].append(0.0); hist[k]["surprise"].append(surp)
        # collect replay: hardest train samples this epoch
        with torch.no_grad():
            grow.eval()
            si = np.random.choice(len(Xtr), min(512, len(Xtr)), replace=False)
            xb = torch.from_numpy(Xtr[si]).to(DEVICE); yb = torch.from_numpy(Ytr[si]).to(DEVICE)
            out = grow(xb); lg = out[0] if isinstance(out, tuple) else out
            nll = F.cross_entropy(lg, yb, reduction="none").cpu().numpy()
            hard = si[np.argsort(-nll)[:64]]
            for i in hard:
                replay_x.append(Xtr[i]); replay_y.append(Ytr[i])
            replay_x, replay_y = replay_x[-256:], replay_y[-256:]

        # growth decision every 2 epochs (directional: hint from hard-PC)
        if (epoch + 1) % 2 == 0:
            cur = hist["grownet"]["val_loss"][-1]
            improved = (prev_val is None) or (prev_val - cur > 1e-3)
            prev_val = cur
            no_improve["grownet"] = 0 if improved else no_improve["grownet"] + 1
            if (not improved) and no_improve["grownet"] >= 1 and grow.hidden.n_active < HMAX and (dorm > 0.10 or surp > med):
                added = grow.grow(6, grad_hint=hint.to(grow.hidden.W.device))
                # baby boost: bump LR next epoch (maturity gate already soft-starts)
                for g in opts["grownet"].param_groups:
                    g["lr"] = base_lr * 1.5
                print(f"[epoch {epoch+1}] GROW +{added} -> h={grow.hidden.n_active} (surp={surp:.3f} dorm={dorm:.2f})")
                no_improve["grownet"] = 0
        # sleep replay + EWC snapshot every 4 epochs
        if (epoch + 1) % 4 == 0 and len(replay_x) > 64:
            RX = np.stack(replay_x[-256:]); RY = np.array(replay_y[-256:])
            tl, _ = train_one(grow, opts["grownet"], RX, RY, base_lr * 0.5, epoch, 0.0, da_avg)
            grow.snapshot_ewc(None)
            ewc_on = True
            print(f"[epoch {epoch+1}] SLEEP replay on {len(RX)} hard samples (loss {tl:.3f}); EWC on")
        # prune every 6 epochs
        if (epoch + 1) % 6 == 0:
            before = grow.hidden.n_active
            grow.prune_dormant(min_keep=10, thresh=0.05)
            if grow.hidden.n_active != before:
                print(f"[epoch {epoch+1}] PRUNE {before} -> {grow.hidden.n_active}")
        print(f"ep {epoch+1:02d} | " + " | ".join(
            f"{k}: tr {hist[k]['train_loss'][-1]:.3f} va {hist[k]['val_loss'][-1]:.3f}/{hist[k]['val_acc'][-1]:.2f} te {hist[k]['test_loss'][-1]:.3f}/{hist[k]['test_acc'][-1]:.2f}"
            for k in models) + f" | h={grow.hidden.n_active}")

    # final save
    torch.save(grow.state_dict(), HERE / "grownet.pt")
    torch.save(small.state_dict(), HERE / "fixed_small.pt")
    torch.save(large.state_dict(), HERE / "fixed_large.pt")
    # perplexity + generalization gap
    summary = {}
    for k in models:
        tr, va, te = hist[k]["train_loss"][-1], hist[k]["val_loss"][-1], hist[k]["test_loss"][-1]
        summary[k] = {
            "train_loss": tr, "val_loss": va, "test_loss": te,
            "train_ppl": float(np.exp(min(tr, 10))), "test_ppl": float(np.exp(min(te, 10))),
            "val_acc": hist[k]["val_acc"][-1], "test_acc": hist[k]["test_acc"][-1],
            "gen_gap_loss": te - tr, "gen_gap_acc": hist[k]["val_acc"][-1] - hist[k]["test_acc"][-1],
            "params": count_params(models[k]), "hsize": hist[k]["hsize"][-1],
        }
    with open(HERE / "metrics.json", "w") as f:
        json.dump({"summary": summary, "hist": hist, "vocab": info["vocab"],
                   "test_sents": [" ".join(s) for s in info["test_sents"]],
                   "train_sents": [" ".join(s) for s in info["train_sents"][:8]]}, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- plots ----
    ep = np.arange(1, EPOCHS + 1)
    plt.figure(figsize=(9, 5))
    for k in models:
        plt.plot(ep, hist[k]["val_loss"], label=f"{k} val")
        plt.plot(ep, hist[k]["test_loss"], "--", label=f"{k} test-unseen")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Generalization: val vs unseen-combo test loss")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "loss_curves.png"); plt.close()

    plt.figure(figsize=(9, 5))
    for k in models:
        plt.plot(ep, hist[k]["test_acc"], label=k)
    plt.xlabel("epoch"); plt.ylabel("next-token acc (unseen combos)"); plt.title("Unseen-combination accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "test_acc.png"); plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(ep, hist["grownet"]["hsize"], marker="o")
    plt.xlabel("epoch"); plt.ylabel("active hidden neurons"); plt.title("GrowNet growth trajectory (seed 12 -> max 48)")
    plt.tight_layout(); plt.savefig(PLOTS / "growth.png"); plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(ep, hist["grownet"]["surprise"], "r-", label="surprise (val NLL)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("surprise", color="r")
    ax2 = ax1.twinx()
    ax2.plot(ep, hist["grownet"]["dormant"], "b-", label="dormant frac")
    ax2.set_ylabel("dormant frac", color="b")
    plt.title("Growth signals: surprise vs dormancy")
    fig.tight_layout(); plt.savefig(PLOTS / "signals.png"); plt.close()

    # generalization bar
    labels = list(summary.keys())
    gaps = [summary[k]["gen_gap_loss"] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, gaps)
    plt.ylabel("test_loss - train_loss"); plt.title("Generalization gap (lower = better)")
    plt.tight_layout(); plt.savefig(PLOTS / "gen_gap.png"); plt.close()
    print("saved plots to", PLOTS)


if __name__ == "__main__":
    main()

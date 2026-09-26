"""Loss-driven growth-direction selection (trial-gated).

For each candidate direction (widen / deepen / sprout):
  snapshot structure -> install candidate (function-preserving, loss unchanged)
  -> S quick tune steps on probe-train -> measure drop on probe-val
  -> restore structure.
Commit the winner iff its held-out drop beats `min_drop`.
Score = drop / (dparams + 1): pure loss reduction per added parameter (BIC-lite).
"""
import torch
import torch.nn.functional as F


def ce_loss(model, X, Y):
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.long)
        yb = torch.as_tensor(Y, dtype=torch.long)
        o = model(xb)
        lg = o[0] if isinstance(o, tuple) else o
        return float(F.cross_entropy(lg, yb).item())


def pc_hint(mat, n_hard=64):
    """Top principal direction of a matrix (for GradMax-style init)."""
    import numpy as np
    M = np.asarray(mat, dtype=np.float64)
    M = M - M.mean(0, keepdims=True)
    try:
        _, _, Vt = __import__("numpy").linalg.svd(M, full_matrices=False)
        return torch.from_numpy(Vt[0].astype("float32"))
    except Exception:
        return None


@torch.no_grad()
def hard_inputs(model, X, Y, n=256):
    """Return (E_hard, hpad_hard): inputs / hidden states of worst-NLL probe rows."""
    import numpy as np
    model.eval()
    xb = torch.as_tensor(X[:1024], dtype=torch.long)
    yb = torch.as_tensor(Y[:1024], dtype=torch.long)
    o = model(xb)
    lg = o[0] if isinstance(o, tuple) else o
    nll = F.cross_entropy(lg, yb, reduction="none").cpu().numpy()
    hard = np.argsort(-nll)[: min(n, len(nll))]
    E = model._fwd["e"].detach().cpu().numpy()[hard]
    H = model._fwd["hpad"].detach().cpu().numpy()[hard]
    return E, H


def trial_direction(model, direction, Xpt, Ypt, Xpv, Ypv, k_widen=4, k_deep=4,
                    k_sprout=32, steps=3, lr=1e-3, seed=0):
    """Returns (drop, dparams). Restores structure before returning."""
    import numpy as np
    base = ce_loss(model, Xpv, Ypv)
    snap = model.structural_snapshot()
    dparams, tag = 0, direction
    try:
        torch.manual_seed(seed)
        if direction == "widen":
            if model.hidden.n_active >= model.hidden.max_neurons:
                return 0.0, 0
            E, _ = hard_inputs(model, Xpt, Ypt)
            hint = pc_hint(E)
            if hint is None:
                return 0.0, 0
            added = model.grow_hidden(k_widen, grad_hint=hint.to(model.hidden.W.device))
            if not added:
                return 0.0, 0
            dparams = added * (model.hidden.K * model.hidden.in_dim + model.hidden.K
                               + model.vocab + 3 * model.mem_dim)
        elif direction == "deepen":
            if model.deep.n_active >= model.deep.max_neurons:
                return 0.0, 0
            _, H = hard_inputs(model, Xpt, Ypt)
            hint = pc_hint(H)
            added = model.grow_deep(k_deep,
                                    grad_hint=None if hint is None else hint.to(model.deep.W.device))
            if not added:
                return 0.0, 0
            dparams = added * (model.deep.K * model.deep.in_dim + model.deep.K + model.vocab)
        elif direction == "sprout":
            target = model.hidden if model.hidden.n_active else None
            if target is None or int((target.emask[:target.n_active] == 0).sum()) < 8:
                # nothing masked yet -> create pool by pruning weakest edges first
                model.hidden.prune_edges(0.05)
                model.deep.prune_edges(0.05)
            got = model.hidden.sprout_edges(k_sprout, seed=seed)
            got += model.deep.sprout_edges(k_sprout // 2, seed=seed + 1)
            if not got:
                return 0.0, 0
            dparams = 0  # synapses reuse existing capacity: no new parameters
        else:
            return 0.0, 0
        # S quick-tune steps on probe-train (full-batch; probe is tiny)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        xb = torch.as_tensor(Xpt, dtype=torch.long)
        yb = torch.as_tensor(Ypt, dtype=torch.long)
        for _ in range(steps):
            opt.zero_grad()
            o = model(xb)
            lg = o[0] if isinstance(o, tuple) else o
            F.cross_entropy(lg, yb).backward()
            opt.step()
        after = ce_loss(model, Xpv, Ypv)
        return base - after, dparams
    finally:
        model.structural_restore(snap)


def select_growth(model, Xpt, Ypt, Xpv, Ypv, k_widen=4, k_deep=4, k_sprout=32,
                  steps=3, lr=1e-3, min_drop=1e-4, seed=0):
    """Try all directions; return (best_dir or 'none', info dict)."""
    results = {}
    for d in ("widen", "deepen", "sprout"):
        drop, dp = trial_direction(model, d, Xpt, Ypt, Xpv, Ypv,
                                   k_widen, k_deep, k_sprout, steps, lr, seed)
        results[d] = {"drop": drop, "dparams": dp, "score": drop / (dp + 1)}
    best = max(results, key=lambda d: results[d]["score"])
    if results[best]["drop"] < min_drop:
        return "none", results
    return best, results

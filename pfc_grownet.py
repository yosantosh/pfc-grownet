"""PFC-GrowNet v2: seed = 3 neurons, grows in ALL dimensions by loss reduction.

Fixed skeleton: embed(C*d) -> growable posterior (L1) -> growable deep (L2, born empty)
  -> PFC memory stripes (BG-gated) -> output. Residual skip guarantees a floor.

Growth dimensions (trial-gated by measured held-out loss drop):
  omega  widen L1/L2 ........... add dendritic neurons (zero-output birth)
  delta  deepen ................ create/widen L2 on top of L1 (zero-output birth)
  sigma  synaptogenesis ........ sprout masked input edges at zero init, prune weak ones
Selection: snapshot -> install candidate (function-preserving) -> S quick tune
  steps on probe-train -> measure drop on probe-val -> restore -> commit the
  winner iff drop > 0. Score = drop / (dparams + 1)  (BIC-lite efficiency).

Neuron model: K dendrites, soma = sum + gamma*max, tanh*0.35, maturity gate.
No cross-neuron norms anywhere (they would couple old/new units and break
exact function preservation at birth).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED_N = 3  # every net is born with exactly 3 posterior neurons


class DendriticGrowLayer(nn.Module):
    """Growable layer of dendritic neurons + edge-level (synapse) mask.

    W: (max_n, K, in_dim) weights, emask: same-shape 0/1 synapse mask.
    Effective weight = W * emask. Sprouted edges are zero-init -> birth is
    function-preserving. n_active may be 0 (used for the not-yet-born L2).
    """
    def __init__(self, in_dim, n_neurons=SEED_N, K=3, gamma=0.5, dropout=0.1, max_neurons=64):
        super().__init__()
        self.in_dim = in_dim
        self.K = K
        self.gamma = gamma
        self.max_neurons = max_neurons
        self.n_active = n_neurons
        self.W = nn.Parameter(torch.randn(max_neurons, K, in_dim) * math.sqrt(2.0 / (in_dim * K)))
        self.b = nn.Parameter(torch.zeros(max_neurons, K))
        self.ln_w = nn.Parameter(torch.ones(max_neurons))   # legacy, unused in fwd
        self.ln_b = nn.Parameter(torch.zeros(max_neurons))  # legacy, unused in fwd
        self.drop = nn.Dropout(dropout)
        self.register_buffer("emask", torch.ones(max_neurons, K, in_dim))
        self.register_buffer("edge_util", torch.zeros(max_neurons, K, in_dim))
        self.register_buffer("maturity", torch.ones(max_neurons))
        self.register_buffer("age", torch.ones(max_neurons) * 100.0)
        self.register_buffer("hebb", torch.zeros(max_neurons))
        self.register_buffer("utility", torch.zeros(max_neurons))
        self.m0 = 50.0

    def forward(self, x, baby_boost=False):
        n = self.n_active
        if n == 0:
            return torch.zeros(x.size(0), 0, device=x.device, dtype=x.dtype)
        Weff = self.W[:n] * self.emask[:n]  # (n,K,in)
        b = self.b[:n]
        d = torch.einsum("bi,nki->bnk", x, Weff) + b.unsqueeze(0)
        d = F.relu(d)
        soma = d.sum(dim=-1) + self.gamma * d.max(dim=-1).values
        soma = torch.tanh(soma * 0.35)
        if baby_boost:
            soma = 0.5 * soma + 0.5 * (F.softplus(soma) - 0.6931) * 2.0
        g = (self.age[:n] / self.m0).clamp(0, 1).to(soma.device)
        out = soma * g.unsqueeze(0)
        out = out * (1.0 + 0.2 * self.hebb[:n].unsqueeze(0))
        out = self.drop(out)
        self._cache = {"soma": soma.detach(), "d": d.detach(), "x": x.detach()}
        return out

    # ---------- neuron-level growth ----------
    def grow_neurons(self, n_new, init="gradmax", grad_hint=None):
        n_new = min(n_new, self.max_neurons - self.n_active)
        if n_new <= 0:
            return 0
        with torch.no_grad():
            s, e = self.n_active, self.n_active + n_new
            if init == "gradmax" and grad_hint is not None and torch.isfinite(grad_hint).all():
                gh = grad_hint / (grad_hint.norm() + 1e-8)
                for i in range(s, e):
                    for k in range(self.K):
                        self.W[i, k].copy_(gh * 0.5 + torch.randn_like(gh) * 0.05)
                    self.b[i].zero_()
            elif s > 0:
                mu = self.W[:s].mean(dim=0)  # MixtureGrowth-lite
                for i in range(s, e):
                    self.W[i].copy_(mu + torch.randn_like(mu) * 0.05)
                    self.b[i].zero_()
            else:  # growing from empty (L2 birth): small random fan-in
                for i in range(s, e):
                    self.W[i].normal_(0, 0.05)
                    self.b[i].zero_()
            self.emask[s:e].fill_(1.0)
            self.edge_util[s:e].zero_()
            self.age[s:e].zero_()
            self.maturity[s:e].zero_()
            self.hebb[s:e].zero_()
            self.utility[s:e].zero_()
            self.n_active = e
        return n_new

    def prune(self, keep_mask):
        with torch.no_grad():
            idx = torch.where(keep_mask)[0]
            n_new = len(idx)
            if n_new == 0:
                self.n_active = 0
                return 0
            self.W[:n_new].copy_(self.W[idx])
            self.b[:n_new].copy_(self.b[idx])
            self.ln_w[:n_new].copy_(self.ln_w[idx])
            self.ln_b[:n_new].copy_(self.ln_b[idx])
            self.emask[:n_new].copy_(self.emask[idx])
            self.edge_util[:n_new].copy_(self.edge_util[idx])
            self.age[:n_new].copy_(self.age[idx])
            self.utility[:n_new].copy_(self.utility[idx])
            self.hebb[:n_new].zero_()
            self.n_active = n_new
        return self.n_active

    # ---------- synapse-level growth ----------
    def update_edge_utility(self):
        """Call after loss.backward(), before optimizer step. Uses |W * grad|."""
        if self.W.grad is None or self.n_active == 0:
            return
        with torch.no_grad():
            u = (self.W[:self.n_active].detach() * self.W.grad[:self.n_active].detach()).abs()
            self.edge_util[:self.n_active] = 0.9 * self.edge_util[:self.n_active] + 0.1 * u

    def active_edges(self):
        if self.n_active == 0:
            return 0
        return int(self.emask[:self.n_active].sum().item())

    def prune_edges(self, frac=0.05):
        """Zero the weakest `frac` of active edges, keeping >=1 edge per dendrite."""
        n = self.n_active
        if n == 0:
            return 0
        with torch.no_grad():
            m = self.emask[:n].clone()
            before = int(m.sum().item())
            for i in range(n):
                for k in range(self.K):
                    row_u = self.edge_util[i, k]
                    row_m = m[i, k]
                    live = torch.where(row_m > 0)[0]
                    if len(live) <= 8:
                        continue
                    kdrop = max(1, int(len(live) * frac))
                    worst = live[torch.argsort(row_u[live])[:kdrop]]
                    m[i, k, worst] = 0.0
            self.emask[:n].copy_(m)
            return before - int(m.sum().item())

    def sprout_edges(self, n_edge, neuron_weights=None, seed=None):
        """Unmask `n_edge` currently-masked edges at ZERO weight (loss-safe birth),
        biased toward high-utility neurons. Returns # sprouted."""
        n = self.n_active
        if n == 0:
            return 0
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        with torch.no_grad():
            masked = torch.where(self.emask[:n] == 0)
            if len(masked[0]) == 0:
                return 0
            if neuron_weights is None:
                neuron_weights = (self.utility[:n].clamp_min(1e-6)).cpu()
            probs = neuron_weights / neuron_weights.sum()
            idx = torch.multinomial(probs, min(n_edge, len(masked[0])), replacement=True, generator=g)
            # map sampled neurons -> one random masked edge each
            done = 0
            for ni in idx.tolist():
                cand = torch.where(self.emask[ni] == 0)
                if len(cand[0]) == 0:
                    continue
                j = torch.randint(0, len(cand[0]), (1,), generator=g).item()
                k, d = int(cand[0][j]), int(cand[1][j])
                self.emask[ni, k, d] = 1.0
                self.W[ni, k, d].zero_()  # zero-init: soma unchanged at birth
                self.edge_util[ni, k, d] = 0.0
                done += 1
        return done


class BGGates(nn.Module):
    """Legacy unused helper (kept for compat); gating lives inline in PFCGrowNet."""
    def __init__(self, h_dim, mem_dim=16):
        super().__init__()
        self.to_ingate = nn.Linear(h_dim, mem_dim)
        self.to_outgate = nn.Linear(h_dim, mem_dim)
        self.to_write = nn.Linear(h_dim, mem_dim)

    def forward(self, h, mem):
        ig = torch.sigmoid(self.to_ingate(h))
        og = torch.sigmoid(self.to_outgate(h))
        cand = torch.tanh(self.to_write(h))
        return (1 - ig) * mem + ig * cand, og * ((1 - ig) * mem + ig * cand)


class PFCGrowNet(nn.Module):
    def __init__(self, vocab, d_emb=16, context=4, h0=SEED_N, hmax=48, K=3,
                 mem_dim=16, dropout=0.1, deepmax=32):
        super().__init__()
        self.vocab = vocab
        self.context = context
        self.d_emb = d_emb
        self.emb = nn.Embedding(vocab, d_emb)
        in_dim = context * d_emb
        self.in_dim = in_dim
        self.hmax = hmax
        self.deepmax = deepmax
        self.mem_dim = mem_dim
        self.hidden = DendriticGrowLayer(in_dim, n_neurons=h0, K=K, dropout=dropout, max_neurons=hmax)
        self.deep = DendriticGrowLayer(hmax, n_neurons=0, K=K, dropout=dropout, max_neurons=deepmax)
        self.ingate = nn.Linear(hmax, mem_dim)
        self.outgate = nn.Linear(hmax, mem_dim)
        self.write = nn.Linear(hmax, mem_dim)
        # readout over [h_padded(hmax) + deep_padded(deepmax) + read(mem)]
        self.readout = nn.Linear(hmax + deepmax + mem_dim, vocab)
        self.skip = nn.Linear(in_dim, vocab, bias=False)
        with torch.no_grad():
            self.readout.weight.zero_()
            nn.init.xavier_uniform_(self.readout.weight[:, :h0])
            self.readout.bias.zero_()
            nn.init.xavier_uniform_(self.skip.weight, gain=0.5)
            for lin in [self.ingate, self.outgate, self.write]:
                nn.init.xavier_uniform_(lin.weight[:, :h0])
                lin.bias.zero_()
        self.register_buffer("fisher", torch.zeros(hmax + deepmax + mem_dim))
        self._snap = None
        self.replay = []
        self.ephemeral_counts = {}

    # column offsets into readout input: [0:hmax | hmax:hmax+deepmax | rest]
    @property
    def _deep_off(self):
        return self.hmax

    def forward(self, ctx, mem=None, hebb_update=False):
        B = ctx.size(0)
        dev = ctx.device
        e = self.emb(ctx).view(B, -1)
        h = self.hidden(e)
        n = h.size(1)
        hpad = torch.zeros(B, self.hmax, device=dev)
        if n:
            hpad[:, :n] = h
        d = self.deep(hpad)
        dn = d.size(1)
        dpad = torch.zeros(B, self.deepmax, device=dev)
        if dn:
            dpad[:, :dn] = d
        if mem is None:
            mem = torch.zeros(B, self.mem_dim, device=dev)
        ig = torch.sigmoid(self.ingate(hpad))
        og = torch.sigmoid(self.outgate(hpad))
        cand = torch.tanh(self.write(hpad))
        mem_new = (1 - ig) * mem + ig * cand
        read = og * mem_new
        logits = self.readout(torch.cat([hpad, dpad, read], dim=-1)) + self.skip(e)
        if hebb_update:
            with torch.no_grad():
                pre = e.mean(dim=1, keepdim=True)
                if n:
                    self.hidden.hebb[:n] += 0.05 * ((pre * h).mean(0) - self.hidden.hebb[:n])
                if dn:
                    self.deep.hebb[:dn] += 0.05 * ((hpad.mean(1, keepdim=True) * d).mean(0) - self.deep.hebb[:dn])
        self._fwd = {"h": h.detach(), "d": d.detach(), "hpad": hpad.detach(),
                     "e": e.detach(), "mem": mem_new.detach(),
                     "ig": ig.detach(), "og": og.detach()}
        return logits, mem_new

    # ---- growth plumbing (all births are zero-output -> loss cannot jump) ----
    def zero_new_hidden(self, old_n):
        with torch.no_grad():
            self.readout.weight[:, old_n:self.hidden.n_active].zero_()
            self.ingate.weight[:, old_n:self.hidden.n_active].zero_()
            self.outgate.weight[:, old_n:self.hidden.n_active].zero_()
            self.write.weight[:, old_n:self.hidden.n_active].zero_()
            self.deep.W[:, :, old_n:self.hidden.n_active].zero_()  # new L1 outputs hit L2 at 0

    def zero_new_deep(self, old_dn):
        o = self._deep_off
        with torch.no_grad():
            self.readout.weight[:, o + old_dn:o + self.deep.n_active].zero_()

    def grow_hidden(self, n_new, grad_hint=None):
        old = self.hidden.n_active
        added = self.hidden.grow_neurons(n_new, init="gradmax", grad_hint=grad_hint)
        if added:
            self.zero_new_hidden(old)
        return added

    def grow_deep(self, n_new, grad_hint=None):
        old = self.deep.n_active
        added = self.deep.grow_neurons(n_new, init="gradmax", grad_hint=grad_hint)
        if added:
            self.zero_new_deep(old)
        return added

    # ---- structural snapshot / restore (for trial-gated growth) ----
    def structural_snapshot(self):
        with torch.no_grad():
            return {
                "hn": self.hidden.n_active, "dn": self.deep.n_active,
                "t": {k: v.detach().clone() for k, v in self.named_parameters()}
                    | {k: v.detach().clone() for k, v in self.named_buffers()
                       if k.startswith("hidden.") or k.startswith("deep.")},
            }

    def structural_restore(self, snap):
        with torch.no_grad():
            for k, v in snap["t"].items():
                try:
                    ref = dict(self.named_parameters()).get(k, dict(self.named_buffers()).get(k))
                    if ref is not None:
                        ref.copy_(v)
                except Exception:
                    pass
            self.hidden.n_active = snap["hn"]
            self.deep.n_active = snap["dn"]

    # ---- pruning ----
    def prune_hidden_dormant(self, min_keep=SEED_N, thresh=0.05):
        n = self.hidden.n_active
        if n <= min_keep:
            return 0
        u = self.hidden.utility[:n]
        order = torch.argsort(u, descending=True)
        keep = torch.zeros(n, dtype=torch.bool)
        keep[order[:min_keep]] = True
        keep[u > thresh] = True
        if keep.all():
            return 0
        with torch.no_grad():
            idx = torch.where(keep)[0]
            self.readout.weight[:, :len(idx)].copy_(self.readout.weight[:, idx])
            self.readout.weight[:, len(idx):self.hmax].zero_()
            for lin in [self.ingate, self.outgate, self.write]:
                lin.weight[:, :len(idx)].copy_(lin.weight[:, idx])
                lin.weight[:, len(idx):self.hmax].zero_()
            self.deep.W[:, :, :len(idx)].copy_(self.deep.W[:, :, idx])
            self.deep.W[:, :, len(idx):self.hmax].zero_()
            self.deep.emask[:, :, :len(idx)].copy_(self.deep.emask[:, :, idx])
            self.deep.emask[:, :, len(idx):self.hmax].zero_()
        self.hidden.prune(keep)
        return n - int(keep.sum())

    def prune_deep_dormant(self, thresh=0.05):
        n = self.deep.n_active
        if n == 0:
            return 0
        u = self.deep.utility[:n]
        keep = u > thresh
        if keep.all():
            return 0
        o = self._deep_off
        with torch.no_grad():
            idx = torch.where(keep)[0]
            if len(idx):
                self.readout.weight[:, o:o + len(idx)].copy_(self.readout.weight[:, o + idx])
            self.readout.weight[:, o + len(idx):o + self.deepmax].zero_()
        self.deep.prune(keep)
        return n - int(keep.sum())

    # ---- stats ----
    def update_utility(self, grad_h=None):
        with torch.no_grad():
            for layer in (self.hidden, self.deep):
                hh = self._fwd["h"] if layer is self.hidden else self._fwd["d"]
                if hh.size(1) == 0:
                    continue
                act = hh.abs().mean(dim=0)
                nn_ = hh.size(1)
                layer.utility[:nn_] = 0.9 * layer.utility[:nn_] + 0.1 * act.cpu()
                layer.age[:nn_] += 1.0
            self.hidden.update_edge_utility()
            self.deep.update_edge_utility()

    def snapshot_ewc(self, fisher):
        self._snap = {k: v.detach().clone() for k, v in self.named_parameters()}
        if fisher is not None:
            self.fisher.zero_()

    def ewc_penalty(self):
        if self._snap is None:
            return 0.0
        pen = 0.0
        for (n, p) in self.named_parameters():
            if n in self._snap:
                pen = pen + ((p - self._snap[n]) ** 2).sum()
        return pen

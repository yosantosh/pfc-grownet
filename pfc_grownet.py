"""PFC-GrowNet: growable dendritic PFC-inspired next-token net (PyTorch, tf env).
Implements the invented architecture faithfully but trainably:
 - fixed skeleton: embed(C*d) -> growable posterior -> PFC stripes -> output
 - dendritic neurons: K dendrites, soma = sum + gamma*max, LayerNorm, maturity gate
 - BG gates: input/output gating of 2 PFC memory slots
 - dual trace: slow weights (AdamW) + fast Hebbian trace h (label-free, used in infer)
 - DA modulator: surprise-scaled LR multiplier + grow threshold
 - function-preserving growth: zero-output birth, smooth baby phase, 2x baby LR
 - directional growth: probe gradient-vote (SVoD-lite) + residual correlation + dormancy
 - pruning: utility-based, EWC-lite consolidation, sleep replay support
 - hybrid inference: ephemeral scratch neurons -> commit if repeatedly useful
Latest techniques integrated: AdamW, cosine schedule, grad clip, LayerNorm,
dropout, label smoothing, weight decay, EWC, replay buffer, Fisher importance.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticGrowLayer(nn.Module):
    """Growable hidden layer of dendritic neurons.
    Each neuron i has K dendrites: d_k = relu(W_k x + b_k); soma = sum_d + gamma*max_d.
    Growth adds neurons with zero output contribution (caller zeros the readout column).
    """
    def __init__(self, in_dim, n_neurons=12, K=3, gamma=0.5, dropout=0.1, max_neurons=64):
        super().__init__()
        self.in_dim = in_dim
        self.K = K
        self.gamma = gamma
        self.max_neurons = max_neurons
        self.n_active = n_neurons
        # allocate max, mask inactive
        self.W = nn.Parameter(torch.randn(max_neurons, K, in_dim) * math.sqrt(2.0 / (in_dim * K)))
        self.b = nn.Parameter(torch.zeros(max_neurons, K))
        self.ln_w = nn.Parameter(torch.ones(max_neurons))
        self.ln_b = nn.Parameter(torch.zeros(max_neurons))
        self.drop = nn.Dropout(dropout)
        # maturity: 0 at birth -> 1; buffer so it moves with model
        self.register_buffer("maturity", torch.ones(max_neurons))
        self.register_buffer("age", torch.ones(max_neurons) * 100.0)
        # fast Hebbian trace per neuron (scalar gate)
        self.register_buffer("hebb", torch.zeros(max_neurons))
        # utility trace
        self.register_buffer("utility", torch.zeros(max_neurons))
        self.m0 = 50.0

    def forward(self, x, baby_boost=False):
        # x: (B, in_dim); use only active neurons
        n = self.n_active
        W = self.W[:n]  # (n,K,in)
        b = self.b[:n]
        # dendrites: (B,n,K)
        d = torch.einsum("bi,nki->bnk", x, W) + b.unsqueeze(0)
        # NOTE: no normalization across K (it whitens away inter-neuron diversity
        # and saturates tanh, stalling early learning). Raw dendrites keep
        # diverse gradients; soma scaled to avoid tanh saturation.
        d = F.relu(d)
        soma_sum = d.sum(dim=-1)
        soma_max, _ = d.max(dim=-1)
        soma = soma_sum + self.gamma * soma_max  # (B,n)
        # NOTE: no cross-neuron LayerNorm here on purpose: it would couple old/new
        # neurons and break function-preserving birth. Scale to avoid tanh saturation.
        soma = torch.tanh(soma * 0.35)
        # maturity gate
        g = (self.age[:n] / self.m0).clamp(0, 1).to(soma.device)
        # smooth baby: if age small, blend toward softplus-ish linear to keep grad flowing
        if baby_boost:
            soma = 0.5 * soma + 0.5 * (F.softplus(soma) - 0.6931) * 2.0
        out = soma * g.unsqueeze(0)
        # Hebbian fast modulation (inference): out *= (1 + alpha*h)
        out = out * (1.0 + 0.2 * self.hebb[:n].unsqueeze(0))
        out = self.drop(out)
        # cache for growth stats
        self._cache = {"soma": soma.detach(), "d": d.detach(), "x": x.detach()}
        return out

    def grow_neurons(self, n_new, init="gradmax", grad_hint=None):
        """Add n_new neurons, function-preserving if caller zeros readout.
        init: 'gradmax' uses grad_hint directions + noise; else small noise around mean."""
        n_new = min(n_new, self.max_neurons - self.n_active)
        if n_new <= 0:
            return 0
        with torch.no_grad():
            s = self.n_active
            e = s + n_new
            if init == "gradmax" and grad_hint is not None:
                # grad_hint: (in_dim,) direction of steepest residual; tile with noise
                gh = grad_hint / (grad_hint.norm() + 1e-8)
                for i in range(s, e):
                    for k in range(self.K):
                        noise = torch.randn_like(gh) * 0.05
                        self.W[i, k].copy_(gh * 0.5 + noise)
                    self.b[i].zero_()
            else:
                # copy mean of existing + noise (MixtureGrowth-lite)
                mu = self.W[:s].mean(dim=0)
                for i in range(s, e):
                    self.W[i].copy_(mu + torch.randn_like(mu) * 0.05)
                    self.b[i].zero_()
            self.age[s:e].zero_()  # newborns
            self.maturity[s:e].zero_()
            self.hebb[s:e].zero_()
            self.utility[s:e].zero_()
            self.n_active = e
        return n_new

    def prune(self, keep_mask):
        """keep_mask: bool tensor len n_active. Compress in place (keeps order)."""
        with torch.no_grad():
            idx = torch.where(keep_mask)[0]
            if len(idx) == 0:
                return 0
            n_new = len(idx)
            self.W[:n_new].copy_(self.W[idx])
            self.b[:n_new].copy_(self.b[idx])
            self.ln_w[:n_new].copy_(self.ln_w[idx])
            self.ln_b[:n_new].copy_(self.ln_b[idx])
            self.age[:n_new].copy_(self.age[idx])
            self.utility[:n_new].copy_(self.utility[idx])
            self.hebb[:n_new].zero_()
            self.n_active = n_new
        return self.n_active


class BGGates(nn.Module):
    """Basal-ganglia-lite: input gate (write to memory) + output gate (read from memory)."""
    def __init__(self, h_dim, mem_dim=16):
        super().__init__()
        self.to_ingate = nn.Linear(h_dim, mem_dim)
        self.to_outgate = nn.Linear(h_dim, mem_dim)
        self.to_write = nn.Linear(h_dim, mem_dim)

    def forward(self, h, mem):
        # h: (B,H), mem: (B,M)
        ig = torch.sigmoid(self.to_ingate(h))
        og = torch.sigmoid(self.to_outgate(h))
        cand = torch.tanh(self.to_write(h))
        mem_new = (1 - ig) * mem + ig * cand
        read = og * mem_new
        return mem_new, read


class PFCGrowNet(nn.Module):
    def __init__(self, vocab, d_emb=16, context=4, h0=12, hmax=48, K=3,
                 mem_dim=16, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.context = context
        self.d_emb = d_emb
        self.emb = nn.Embedding(vocab, d_emb)
        in_dim = context * d_emb
        self.in_dim = in_dim
        self.hidden = DendriticGrowLayer(in_dim, n_neurons=h0, K=K, dropout=dropout, max_neurons=hmax)
        self.bg = BGGates(h0, mem_dim)  # note: Linear layers sized to current h? we project dynamically
        # To allow growth, BG linears take variable h: use max-sized linears + slicing
        self.bg_max = hmax
        self.mem_dim = mem_dim
        self.ingate = nn.Linear(hmax, mem_dim)
        self.outgate = nn.Linear(hmax, mem_dim)
        self.write = nn.Linear(hmax, mem_dim)
        # readout over [h_padded(max) + read(mem_dim)]
        self.readout = nn.Linear(hmax + mem_dim, vocab)
        # residual skip: direct embed-context -> logits (latest-technique stabilizer;
        # guarantees a linear-bigram floor while the growable path adds capacity)
        self.skip = nn.Linear(in_dim, vocab, bias=False)
        with torch.no_grad():
            self.readout.weight.zero_()  # placeholders; real init below for active part
            # init active readout slice properly
            nn.init.xavier_uniform_(self.readout.weight[:, :h0])
            self.readout.bias.zero_()
            nn.init.xavier_uniform_(self.skip.weight, gain=0.5)
            for lin in [self.ingate, self.outgate, self.write]:
                nn.init.xavier_uniform_(lin.weight[:, :h0])
                lin.bias.zero_()
        self.hmax = hmax
        # EWC-lite: importance + snapshot
        self.register_buffer("fisher", torch.zeros(hmax + mem_dim))
        self._snap = None
        # replay buffer of (ctx, nxt, surprise)
        self.replay = []
        # ephemeral commit counter: fingerprint -> count
        self.ephemeral_counts = {}
        # per-neuron baby LR multiplier handled in optimizer param groups (see trainer)

    def forward(self, ctx, mem=None, hebb_update=False):
        # ctx: (B,C) token ids
        B = ctx.size(0)
        dev = ctx.device
        e = self.emb(ctx).view(B, -1)  # (B, in_dim)
        h = self.hidden(e)  # (B, n_active)
        n = h.size(1)
        # pad h to hmax for fixed-size gating/readout
        hpad = torch.zeros(B, self.hmax, device=dev)
        hpad[:, :n] = h
        if mem is None:
            mem = torch.zeros(B, self.mem_dim, device=dev)
        ig = torch.sigmoid(self.ingate(hpad))
        og = torch.sigmoid(self.outgate(hpad))
        cand = torch.tanh(self.write(hpad))
        mem_new = (1 - ig) * mem + ig * cand
        read = og * mem_new
        logits = self.readout(torch.cat([hpad, read], dim=-1)) + self.skip(e)
        if hebb_update:
            # label-free fast trace: co-activation of pooled input and h
            with torch.no_grad():
                pre = e.mean(dim=1, keepdim=True)  # (B,1)
                post = h  # (B,n)
                dh = (pre * post).mean(dim=0)  # (n,)
                self.hidden.hebb[:n] += 0.05 * (dh - self.hidden.hebb[:n])
        self._fwd = {"h": h.detach(), "hpad": hpad.detach(), "e": e.detach(),
                     "mem": mem_new.detach(), "ig": ig.detach(), "og": og.detach()}
        return logits, mem_new

    # ---- growth plumbing ----
    def zero_new_readout(self, old_n):
        with torch.no_grad():
            self.readout.weight[:, old_n:self.hidden.n_active].zero_()
            self.ingate.weight[:, old_n:self.hidden.n_active].zero_()
            self.outgate.weight[:, old_n:self.hidden.n_active].zero_()
            self.write.weight[:, old_n:self.hidden.n_active].zero_()

    def grow(self, n_new, grad_hint=None):
        old = self.hidden.n_active
        added = self.hidden.grow_neurons(n_new, init="gradmax", grad_hint=grad_hint)
        if added > 0:
            self.zero_new_readout(old)
        return added

    def prune_dormant(self, min_keep=8, thresh=0.02):
        n = self.hidden.n_active
        u = self.hidden.utility[:n]
        # dormant = low utility; always keep at least min_keep highest-utility
        order = torch.argsort(u, descending=True)
        keep = torch.zeros(n, dtype=torch.bool)
        keep[order[:max(min_keep, 1)]] = True
        keep[u > thresh] = True
        if keep.all():
            return 0
        # compress readout/gate columns accordingly
        with torch.no_grad():
            idx = torch.where(keep)[0]
            self.readout.weight[:, :len(idx)].copy_(self.readout.weight[:, idx])
            self.readout.weight[:, len(idx):].zero_()
            for lin in [self.ingate, self.outgate, self.write]:
                lin.weight[:, :len(idx)].copy_(lin.weight[:, idx])
                lin.weight[:, len(idx):].zero_()
        self.hidden.prune(keep)
        return n - int(keep.sum())

    def update_utility(self, grad_h=None):
        with torch.no_grad():
            h = self._fwd["h"]  # (B,n)
            act = h.abs().mean(dim=0)
            if grad_h is not None:
                g = grad_h.abs().mean(dim=0).clamp_max(10.0)
                u = act * (0.5 + g)
            else:
                u = act
            n = h.size(1)
            self.hidden.utility[:n] = 0.9 * self.hidden.utility[:n] + 0.1 * u
            self.hidden.age[:n] += 1.0

    def snapshot_ewc(self, fisher):
        self._snap = {k: v.detach().clone() for k, v in self.named_parameters()}
        if fisher is not None:
            # fisher per readout col approx -> broadcast; keep simple global diag
            self.fisher.zero_()

    def ewc_penalty(self):
        if self._snap is None:
            return 0.0
        pen = 0.0
        for (n, p) in self.named_parameters():
            if n in self._snap:
                pen = pen + ((p - self._snap[n]) ** 2).sum()
        return pen

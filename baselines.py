"""Fixed baselines with same I/O + training protocol for fair comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedMLM(nn.Module):
    def __init__(self, vocab, d_emb=16, context=4, h=12, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb)
        self.fc1 = nn.Linear(context * d_emb, h)
        self.ln = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h, vocab)

    def forward(self, ctx, mem=None):
        e = self.emb(ctx).view(ctx.size(0), -1)
        h = torch.tanh(self.ln(self.fc1(e)))
        h = self.drop(h)
        return self.fc2(h), None

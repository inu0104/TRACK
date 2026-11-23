"""Instance encoders. Every variant maps one transaction to one vector."""
from __future__ import annotations

import torch
import torch.nn as nn


def make_encoder(d_in: int, d_hid: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(d_hid, d_hid), nn.ReLU())


class SetEncoder(nn.Module):
    """Per-instance encoder followed by self-attention across the bag.

    The MLP encoder above sees one transaction at a time, which sits badly with
    what this work claims. Laundering typologies are temporal by definition --
    structuring is "many transfers just under a threshold within a short
    window", not a property any single transfer has -- and a per-instance
    encoder cannot represent that. We had been recovering it through
    hand-built features (`log_acct_gap`, `log_pair_seq`, `new_counterparty`),
    which is the same summarise-then-model move the paper criticises aggregate
    features for.

    Self-attention over the bag lets a transaction be represented in terms of
    the others around it, so "one of twelve near-identical transfers in two
    days" becomes representable rather than pre-computed. TransMIL (NeurIPS
    2021) does the same thing for pathology slides.

    The cost is that instance representations stop being independent, so an
    instance score becomes "suspicious in this context" rather than
    "suspicious". Since instance attribution is one of the claims here, both
    bag and instance metrics are reported for this encoder rather than only
    the bag ones.
    """

    def __init__(self, d_in: int, d_hid: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.proj = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.qkv = nn.Linear(d_hid, 3 * d_hid)
        self.out = nn.Linear(d_hid, d_hid)
        self.norm1, self.norm2 = nn.LayerNorm(d_hid), nn.LayerNorm(d_hid)
        self.ff = nn.Sequential(nn.Linear(d_hid, d_hid), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(d_hid, d_hid))
        self._mask = None                 # set by MILNet before each call

    def forward(self, x):
        # Attention is written out rather than delegated to
        # nn.MultiheadAttention, whose fast path faults with an illegal memory
        # access on this input. Bags are heavily padded here -- the cap is the
        # 99th percentile of bag size while the median account has a handful of
        # transactions -- so most rows are mostly mask, and a row that is
        # entirely mask would send a softmax over all -inf to NaN. Slot 0 is
        # therefore always left attendable; its output is meaningless for an
        # empty bag but the pooling masks it out anyway, and a finite
        # meaningless number is survivable where a NaN is not.
        h = self.proj(x)
        b, k, d = h.shape
        dh = d // self.heads
        q, kk, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, k, self.heads, dh)
        q = q.view(shape).transpose(1, 2)
        kk = kk.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        attn_mask = None
        if self._mask is not None:
            keep = self._mask > 0.5
            keep = keep.clone()
            keep[:, 0] = True
            attn_mask = keep[:, None, None, :].expand(b, self.heads, k, k)
        a = torch.nn.functional.scaled_dot_product_attention(q, kk, v,
                                                             attn_mask=attn_mask)
        a = self.out(a.transpose(1, 2).reshape(b, k, d))
        h = self.norm1(h + a)
        return self.norm2(h + self.ff(h))


def build_encoder(kind: str, d_in: int, d_hid: int) -> nn.Module:
    if kind == "mlp":
        return make_encoder(d_in, d_hid)
    if kind == "set":
        return SetEncoder(d_in, d_hid)
    raise ValueError(kind)

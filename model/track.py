"""TRACK: score every transaction, pool selectively into an account.

The label rule in anti-money laundering is a maximum -- an account is reported
when any one of its transactions is suspicious -- while the input to an
account-level model is a mean over the same transactions. This module closes
that gap: MILNet scores each transaction with a shared encoder and pools the
scores under the account label, so the loss constrains only the pooled value
and the model is left to decide which transaction carries the bag.

The pooled score is the account prediction; the instance scores are the
by-product that names the transaction, and they are never supervised."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from model.encoders import build_encoder


class MILNet(nn.Module):
    def __init__(self, d_in, d_hid=64, pooling="max", encoder=None,
                 encoder_kind="mlp"):
        super().__init__()
        self.pooling = pooling
        self.enc = (encoder if encoder is not None
                    else build_encoder(encoder_kind, d_in, d_hid))
        self.att_v, self.att_u = nn.Linear(d_hid, d_hid), nn.Linear(d_hid, d_hid)
        self.att_w = nn.Linear(d_hid, 1)
        self.head = nn.Linear(d_hid, 1)

    def forward(self, x, mask):
        if hasattr(self.enc, "_mask"):        # a set encoder needs to know which
            self.enc._mask = mask             # slots are padding
        h = self.enc(x)
        m1 = mask.unsqueeze(-1)
        inst = self.head(h).squeeze(-1) + (1.0 - mask) * -1e9
        if self.pooling == "maxscore":
            # The operation the label rule actually names. y_u = max_k y_k says
            # to score each transaction and let the highest one decide the
            # account, and that is this line.
            #
            # Every other pooling here combines embeddings first and classifies
            # the combination -- the standard embedding-level formulation, and
            # the one Ilse et al. (ICML 2018) use. It is a legitimate and
            # usually stronger architecture, but it is not what our argument
            # describes: an elementwise max over the hidden dimensions takes
            # dimension 1 from one transaction and dimension 2 from another, so
            # the pooled vector corresponds to no transaction that happened.
            # Motivating max pooling by the label rule and then implementing
            # that is a gap a reviewer can open, so both are measured.
            return inst.max(1).values, torch.softmax(inst, dim=1), inst
        if self.pooling == "mean":
            z = (h * m1).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            a = mask / mask.sum(1, keepdim=True).clamp(min=1)
        elif self.pooling == "max":
            hm = (h + (1.0 - m1) * -1e9)
            z, win = hm.max(1)
            # Credit through the max, which is the attribution this architecture
            # actually licenses. The pooled vector takes each hidden dimension
            # from whichever transaction won it, and the head reads that vector
            # with one weight per dimension, so a transaction's share of the
            # account score is the sum of w_d * z_d over the dimensions it won.
            # Unlike the per-instance head score in `inst`, which asks a
            # counterfactual -- what would this transaction score on its own --
            # this is an exact decomposition: the contributions sum to the
            # account logit less its bias. The two need not agree, so both are
            # returned and both are scored.
            a = torch.zeros_like(mask).scatter_add_(
                1, win, self.head.weight.view(1, -1) * z)
        else:
            e = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))
            a = torch.softmax(e + (1.0 - m1) * -1e9, dim=1).squeeze(-1)
            z = (h * a.unsqueeze(-1)).sum(1)
        return self.head(z).squeeze(-1), a, inst


def infer(model, packed, mask, idx, device, extras=False, chunk=4096):
    # The chunk is a number of bags, but the set encoder's activation is
    # quadratic in the bag width: one attention matrix is chunk x heads x cap x
    # cap. At AMLSim's cap of 1024 the default chunk asks for 69 GB in a single
    # allocation and the kernel takes the process, which is what happened twice
    # before this bound existed. Per-instance encoders have no such term and
    # keep the default.
    if hasattr(model.enc, "_mask"):
        k = packed.shape[1]
        chunk = max(8, min(chunk, 1_500_000_000 // (16 * max(1, k * k))))
    out, atts, isc = [], [], []
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        logit, a, s = model(torch.from_numpy(packed[b]).to(device),
                            torch.from_numpy(mask[b]).to(device))
        out.append(torch.sigmoid(logit).cpu().numpy())
        if extras:
            atts.append(a.cpu().numpy())
            isc.append(s.cpu().numpy())
    s = np.concatenate(out)
    return (s, np.concatenate(atts), np.concatenate(isc)) if extras else s


def train_eval(packed, mask, y, split, pooling, seed, device, epochs, lr, batch,
               d_hid=64, patience=30, pretrained=None, train_drop=0.0,
               encoder_kind="mlp", py=None, drop_flagged=False):
    torch.manual_seed(seed)
    tr, va, te = split["train"], split["valid"], split["test"]
    enc = build_encoder(encoder_kind, packed.shape[2], d_hid)
    if pretrained is not None:
        enc.load_state_dict(pretrained)     # fresh copy per run, runs stay independent
    model = MILNet(packed.shape[2], d_hid=d_hid, pooling=pooling,
                   encoder=enc).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    pos = max(1, int(y[tr].sum()))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        (len(tr) - pos) / pos, dtype=torch.float32, device=device))
    xt = torch.from_numpy(packed[tr]).to(device)
    mt = torch.from_numpy(mask[tr]).to(device)
    yt = torch.from_numpy(y[tr].astype(np.float32)).to(device)
    # Removing exactly the flagged transactions, rather than a random share of
    # all of them, is the pointed version of the same test. Uniform dropout at
    # 0.1 and 0.3 changed nothing on any development dataset, but with one or two
    # flagged transactions in a bag of roughly two hundred it only turns a tenth
    # or a third of the positive bags into label noise, and the account's other
    # transactions still carry the pattern. Dropping the flagged ones leaves
    # every positive bag positive with nothing positive left in it. If the bag
    # label really is max over instances and the model reads it that way, this
    # has to hurt; if it does not, the model is scoring the account rather than
    # the transaction, whatever the pooling operator says.
    pf = (torch.from_numpy(py[tr].astype(np.float32)).to(device)
          if (drop_flagged and py is not None) else None)

    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            m_b = mt[b]
            if train_drop > 0:
                # Instance dropout as a supervised training augmentation. This
                # is where the burst structure bites: the bag keeps its positive
                # label while the transaction that earned it is removed, so a
                # share of positive bags is presented to the loss as positives
                # containing nothing positive, which is exactly the pathology
                # this ablation is meant to induce.
                keep = (torch.rand_like(m_b) >= train_drop).float()
                m_b = m_b * keep
                empty = m_b.sum(1) == 0          # never hand the model an empty bag
                if empty.any():
                    m_b[empty] = mt[b][empty]
            if pf is not None:
                m_b = m_b * (1.0 - pf[b])
                empty = m_b.sum(1) == 0
                if empty.any():
                    m_b[empty] = mt[b][empty]
            logit, _, _ = model(xt[b], m_b)
            loss = lossf(logit, yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(y[va], infer(model, packed, mask, va, device))
        if ap > best + 1e-5:
            best, state, bad = ap, {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        vs = infer(model, packed, mask, va, device)
        ts, att, isc = infer(model, packed, mask, te, device, extras=True)
    return vs, ts, att, isc, best

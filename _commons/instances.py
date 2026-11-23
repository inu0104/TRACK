"""Transactions as instances, accounts as bags.

The account-level pipelines this work compares against fold an account's
transactions into one vector before learning starts. These functions keep them:
build_instances turns each transaction into a feature row expressed relative to
its own account's history, attach_counterparty appends the other endpoint's
context, and pack_bags stacks the rows of one account into a padded block the
model can pool over.

All features are causal -- computed from what was visible at the transaction's
own timestamp -- so a bag can be cut by time without leaking forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Windows as a fraction of each dataset's own observed horizon. AMLworld spans
# 10 days after the tail cut and AMLNet 186, so a literal "last 7 days" means
# almost everything on one and little on the other, and the two datasets would
# not be running the same feature.
WINDOW_FRACTIONS = (1 / 6, 0.4, 0.8)


# --------------------------------------------------------------------------

# Columns that describe the transfer itself rather than either account, so the
# two endpoint rows carry identical values and copying them across adds width
# without information. `direction` is excluded for the same reason: the
# counterparty's is exactly the negation of this row's.
TX_INTRINSIC = ("log_amount", "direction", "hour_sin", "hour_cos", "ch1", "ch2")


def attach_counterparty(x: np.ndarray, names: list[str], eid: np.ndarray
                        ) -> tuple[np.ndarray, list[str]]:
    """Append, to each row, the account-context features of the *other* endpoint.

    One hop of message passing, precomputed. It tests the premise underneath the
    coupled-bag idea -- that a transaction is better judged with both accounts'
    histories than with one -- without restructuring the model, so if the
    premise fails the larger rewrite is not worth doing.

    Note what this is not: averaging the two views' scores, which we measured as
    harmful (mean -7 to -8%, min -15 to -48%). That combines two verdicts and
    keeps their intersection. This gives one verdict a wider input, so a
    typology visible from only one side stays visible.
    """
    order = np.argsort(eid, kind="stable")
    e = eid[order]
    starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    counts = np.diff(np.r_[starts, len(e)])
    partner = np.arange(len(eid))                  # unpaired rows point at self
    two = starts[counts == 2]
    a, b = order[two], order[two + 1]
    partner[a], partner[b] = b, a
    keep = [i for i, n in enumerate(names) if n not in TX_INTRINSIC]
    unpaired = float((partner == np.arange(len(eid))).mean())
    if unpaired:
        print(f"  {unpaired:.2%} of rows have no counterparty row and reuse their "
              f"own context", flush=True)
    return (np.concatenate([x, x[partner][:, keep]], axis=1),
            names + [f"cp_{names[i]}" for i in keep])


def build_instances(ev: pd.DataFrame, accounts: pd.Index, with_eid: bool = False,
                    counterparty_context: bool = False):
    """One row per (account, incident transaction), described from that side.

    Every feature except the raw amount, hour and channel is a function of
    (transaction, that account's history up to and including it). That is the
    design rule, and it is what makes the two rows of one transfer differ: the
    same $9,500 is "the twelfth near-identical transfer this account has sent
    this week" from the sender's side and "one of forty arrivals today" from the
    receiver's. An earlier feature set described the transaction almost alone --
    seven of its twelve columns were bit-identical across the two rows, so the
    encoder saw the transfer and almost nothing about either account.

    Everything is causal: expanding and windowed statistics use only what had
    happened by that transaction. A deployed system cannot see an account's
    later transfers, so a feature that reads them would report a number no
    deployment could reproduce.

    `with_eid` additionally returns, per row, the index of the transaction it
    came from. A transaction has two endpoints and therefore produces two rows,
    one in each endpoint's bag, so the eid is what identifies the two rows as
    the same real-world event. Off by default: adding it to the return tuple
    unconditionally would break callers that unpack four values.
    """
    idx = {a: i for i, a in enumerate(accounts)}
    has_hour = "hour" in ev.columns
    chans = [c for c in ("ch1", "ch2") if c in ev.columns]

    frames = []
    for actor, other, sign in (("source", "target", 1.0), ("target", "source", -1.0)):
        cols = [actor, other, "t", "amount", "flag"] + (["hour"] if has_hour else []) + chans
        d = ev[cols].copy()
        d.columns = ["acct", "cp", "t", "amount", "flag"] + \
                    (["hour"] if has_hour else []) + chans
        d["eid"] = np.arange(len(ev), dtype=np.int64)
        d = d[d["acct"].isin(idx)]
        d = d.sort_values(["acct", "t"], kind="mergesort")
        pair = d.groupby(["acct", "cp"], sort=False)
        d["pair_seq"] = pair.cumcount()
        d["pair_gap"] = pair["t"].diff()
        d["acct_gap"] = d.groupby("acct", sort=False)["t"].diff()
        d["direction"] = sign
        frames.append(d)

    inst = pd.concat(frames, ignore_index=True)
    inst = inst.sort_values(["acct", "t"], kind="mergesort").reset_index(drop=True)

    n = len(inst)
    acct_code = inst["acct"].map(idx).to_numpy(np.int64)
    t = inst["t"].to_numpy(np.float64)
    amount = inst["amount"].to_numpy(np.float64)
    is_in = (inst["direction"].to_numpy() < 0)
    grp = inst.groupby("acct", sort=False)
    horizon = float(np.nanmax(t) - np.nanmin(t)) or 1.0

    # Account-relative windows. One searchsorted per window finds, for every
    # row, the first row of the *same* account still inside it.
    #
    # The key is built from the row's *position group*, not its account code.
    # Rows are sorted by (account, time) so an account's rows are contiguous,
    # but the account code comes from `accounts`, whose order is the aggregate
    # runner's and is not sorted on every dataset. Keying on
    # the account code therefore produced a non-monotone array on exactly those
    # datasets, and searchsorted on unsorted input returns nonsense silently,
    # or here an out-of-range index. Numbering the groups in row order makes the
    # key monotone by construction whatever the account order happens to be.
    seq = np.arange(n)
    new_acct = np.r_[True, acct_code[1:] != acct_code[:-1]]
    grp_ord = np.cumsum(new_acct) - 1
    t0 = t - np.nanmin(t)
    big = float(np.nanmax(t0)) + 1.0
    key = grp_ord * big + t0
    assert np.all(np.diff(key) >= 0), "instance rows are not grouped by account"
    win_counts = {}
    for f in WINDOW_FRACTIONS:
        lo = np.searchsorted(key, grp_ord * big + (t0 - horizon * f), side="left")
        win_counts[f] = (seq - lo + 1).astype(np.float64), lo

    ex = grp["amount"].expanding()
    mu = ex.mean().reset_index(level=0, drop=True).reindex(inst.index).to_numpy()
    sd = ex.std().reset_index(level=0, drop=True).reindex(inst.index).to_numpy()
    rank = ex.rank(pct=True).reset_index(level=0, drop=True).reindex(inst.index).to_numpy()
    cum_amt = grp["amount"].cumsum().to_numpy()
    n_so_far = (grp.cumcount().to_numpy() + 1).astype(np.float64)

    gap = inst["acct_gap"].to_numpy()
    gmu = grp["acct_gap"].expanding().mean().reset_index(level=0, drop=True) \
        .reindex(inst.index).to_numpy()
    gsd = grp["acct_gap"].expanding().std().reset_index(level=0, drop=True) \
        .reindex(inst.index).to_numpy()

    # Repeated near-identical amounts are the signature of structuring, so count
    # how many of the account's earlier transactions fell in this one's amount
    # bucket. Bucketing on log amount keeps it to a cumcount instead of the
    # quadratic pairwise comparison the literal definition would need.
    bucket = np.floor(np.log1p(np.clip(amount, 0, None)) * 8.0).astype(np.int64)
    inst["_b"] = bucket
    rep = inst.groupby(["acct", "_b"], sort=False).cumcount().to_numpy()
    inst.drop(columns="_b", inplace=True)

    new_cp = (inst["pair_seq"].to_numpy() == 0).astype(np.float64)
    cum_new = np.r_[0.0, np.cumsum(new_cp)]
    cum_out = np.r_[0.0, np.cumsum(~is_in)]
    cum_in_amt = np.r_[0.0, np.cumsum(np.where(is_in, amount, 0.0))]
    cum_out_amt = np.r_[0.0, np.cumsum(np.where(is_in, 0.0, amount))]
    grp_start = np.flatnonzero(new_acct)
    first_of_acct = np.repeat(grp_start, np.diff(np.r_[grp_start, n]))

    def since_start(cum):
        return cum[seq + 1] - cum[first_of_acct]

    n_out = since_start(cum_out)
    amt_in = since_start(cum_in_amt)
    amt_out = since_start(cum_out_amt)

    # Layering leaves money resting briefly: an account receives, then sends a
    # similar amount soon after. Both halves are recorded relative to the
    # account, so they differ between the two endpoint views of one transfer.
    last_in_t = np.where(is_in, t, np.nan)
    last_in_a = np.where(is_in, amount, np.nan)
    s = pd.Series(last_in_t, index=inst.index).groupby(acct_code).ffill().shift(1)
    a_prev = pd.Series(last_in_a, index=inst.index).groupby(acct_code).ffill().shift(1)
    same_acct = np.r_[False, acct_code[1:] == acct_code[:-1]]
    pass_gap = np.where(same_acct, t - s.to_numpy(), np.nan)
    pass_ratio = np.where(same_acct, amount / np.maximum(a_prev.to_numpy(), 1e-9), np.nan)

    z = np.clip((amount - mu) / np.where((sd > 0) & np.isfinite(sd), sd, np.nan), -10, 10)
    cols = {
        # transaction attributes -- identical in both views, kept for scale
        "log_amount": np.log1p(np.clip(amount, 0, None)),
        "direction": inst["direction"].to_numpy(),
        # amount, relative to this account's own past
        "amt_z": z,
        "amt_rank": rank,
        "amt_share": amount / np.maximum(cum_amt, 1e-9),
        "amt_repeat": rep / np.maximum(n_so_far - 1, 1.0),
        # rhythm, relative to this account's own past
        "log_acct_gap": np.log1p(np.clip(np.nan_to_num(gap, nan=-1.0), 0, None)),
        "gap_z": np.clip((gap - gmu) / np.where((gsd > 0) & np.isfinite(gsd), gsd,
                                                np.nan), -10, 10),
        "first_event": np.isnan(gap).astype(np.float64),
        "log_seq": np.log1p(n_so_far - 1),
        "log_tenure": np.log1p(np.clip(t - t[first_of_acct], 0, None)),
        # counterparty structure, relative to this account's own past
        "new_counterparty": new_cp,
        "log_pair_seq": np.log1p(inst["pair_seq"].to_numpy()),
        "log_pair_gap": np.log1p(np.clip(np.nan_to_num(
            inst["pair_gap"].to_numpy(), nan=-1.0), 0, None)),
        # direction balance so far
        "dir_count_balance": (2.0 * n_out - n_so_far) / n_so_far,
        "dir_amount_balance": ((amt_out - amt_in)
                               / np.maximum(amt_out + amt_in, 1e-9)),
        # passthrough
        "log_passthrough_gap": np.log1p(np.clip(np.nan_to_num(pass_gap, nan=-1.0),
                                                0, None)),
        "passthrough_ratio": np.clip(np.nan_to_num(pass_ratio, nan=0.0), 0, 10),
    }
    for f in WINDOW_FRACTIONS:
        c, lo = win_counts[f]
        tag = f"w{f:.2f}".replace("0.", "")
        cols[f"log_count_{tag}"] = np.log1p(c)
        cols[f"new_cp_rate_{tag}"] = (cum_new[seq + 1] - cum_new[lo]) / c
    if has_hour:
        cols["hour_sin"] = np.sin(2 * np.pi * inst["hour"].to_numpy() / 24.0)
        cols["hour_cos"] = np.cos(2 * np.pi * inst["hour"].to_numpy() / 24.0)
    for c in chans:
        cols[c] = inst[c].to_numpy(np.float64)

    feats = pd.DataFrame(cols)
    x = np.nan_to_num(feats.to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bag = inst["acct"].map(idx).to_numpy(np.int64)
    names = list(feats.columns)
    eids = inst["eid"].to_numpy(np.int64)
    if counterparty_context:
        x, names = attach_counterparty(x, names, eids)
    if with_eid:
        return x, bag, inst["flag"].to_numpy(np.int64), names, eids
    return x, bag, inst["flag"].to_numpy(np.int64), names


def label_survival(sizes: np.ndarray, n_flag: np.ndarray, y: np.ndarray,
                   cap: int) -> float:
    """Share of positive bags that still hold a flagged instance after capping.

    Reported, never used to choose anything -- it reads instance labels, so
    letting it pick the cap would leak. It is here because the cap is the one
    preprocessing step that can silently invent the exact pathology this work
    studies: a bag that keeps its positive label after its only flagged
    transaction was sampled away is a positive bag with no positive instance,
    and max pooling cannot find what is no longer there. The percentile rule
    below is label-free, and this number is how we check it was generous enough.
    """
    pos = np.flatnonzero(y == 1)
    if not len(pos):
        return float("nan")
    s = sizes[pos].astype(float)
    f = n_flag[pos].astype(float)
    miss = np.where(s <= cap, 0.0,
                    hypergeom.pmf(0, np.maximum(s, 1), np.maximum(f, 1),
                                  np.minimum(cap, s)))
    return float(1.0 - miss.mean())


def pack_bags(x, bag, inst_y, n_acct, cap, rng, eid=None):
    order = np.argsort(bag, kind="stable")
    x, bag, inst_y = x[order], bag[order], inst_y[order]
    eid = None if eid is None else eid[order]
    starts = np.searchsorted(bag, np.arange(n_acct), side="left")
    ends = np.searchsorted(bag, np.arange(n_acct), side="right")
    packed = np.zeros((n_acct, cap, x.shape[1]), np.float32)
    mask = np.zeros((n_acct, cap), np.float32)
    py = np.zeros((n_acct, cap), np.int8)
    peid = None if eid is None else np.full((n_acct, cap), -1, np.int64)
    for i in range(n_acct):
        s, e = starts[i], ends[i]
        n = e - s
        if n == 0:
            continue
        sel = np.arange(s, e) if n <= cap else s + rng.choice(n, cap, replace=False)
        k = len(sel)
        packed[i, :k], mask[i, :k], py[i, :k] = x[sel], 1.0, inst_y[sel]
        if peid is not None:
            peid[i, :k] = eid[sel]
    return (packed, mask, py) if peid is None else (packed, mask, py, peid)

#!/usr/bin/env python3
"""Time-ordered split, as an alternative to the random one both runners use.

Why
---
The random partition (`fixed_test_partition`) is what the fraud-GNN baselines we
compare against use. PC-GNN states "the train-test split is based on the
stratified sampling provided by Scikit-learn"; CARE-GNN, BWGNN and AMLNet do the
same. The AML transaction datasets we run on say the opposite. AMLworld "split[s]
the transaction indices after ordering them by their timestamps" 60-20-20, and
Elliptic++ uses "a 70/30 train-test split with respect to time steps". Reviewer 2
asked for temporal or case-level splits for exactly this reason: a random split
can put two accounts of one laundering scheme on both sides.

What was tried first, and why it was thrown away
------------------------------------------------
The cheap version assigns each whole account to a side by its *last*
transaction, which keeps the Bag definition untouched. Measured, it is
unusable. Bag size correlates with the label, and an account's last transaction
is late exactly when the account is long-lived, so the positives pile up on the
test side: the development pool holds 46 of AMLworld's 5,862 positives against
1,172 under the random split, and 595 of Elliptic++'s 14,266 against 2,853. That
is a selection effect of the assignment rule, not a property of laundering, and
it would have produced a table that says nothing about time.

What this does instead
----------------------
Transactions are cut at two time quantiles, following AMLworld: the first 60%
train, the next 20% validation, the last 20% test. A bag is then an
(account, window) pair, holding that account's transactions inside that window
and labelled by the max over them. An account active throughout appears in all
three windows, holding different transactions each time, which is what removes
the size-driven selection.

Instance features are built once over the whole history and are causal by
construction, so a test-window row still knows the account's past. Only bag
membership is cut. That is the deployment reading: an analyst scoring today's
activity knows what the account did last year.

Cost of the choice
------------------
Proportions follow AMLworld rather than the random protocol's 20/80, so numbers
here are not comparable cell-by-cell with the random-split table. The claim this
run can support is the within-table one: whether the ordering of methods holds
when the split respects time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# AMLworld: "a 60-20-20 temporal train-validation-test split, i.e., we split the
# transaction indices after ordering them by their timestamps".
DEFAULT_FRACS = (0.6, 0.8)


def window_bounds(ev: pd.DataFrame,
                  fracs: tuple[float, float] = DEFAULT_FRACS) -> tuple[float, float]:
    """The two timestamps that cut the transaction sequence into three windows."""
    t = np.sort(ev["t"].to_numpy(np.float64))
    return tuple(float(t[min(len(t) - 1, int(round(f * len(t))))]) for f in fracs)


def window_of(t: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """0 train, 1 validation, 2 test, by transaction time."""
    t1, t2 = bounds
    return np.where(t < t1, 0, np.where(t < t2, 1, 2)).astype(np.int8)


class RowView:
    """`packed`-shaped access for (account, window) rows without copying it.

    Every reader of the packed tensor slices it once with a row index list and
    moves the slice to the device, so the only two things needed are
    ``.shape[2]`` and ``__getitem__``. Materialising the rows instead would cost
    a copy per window, and AMLworld's packed tensor alone is tens of gigabytes.
    """

    def __init__(self, packed: np.ndarray, acct_of: np.ndarray):
        self._packed, self._acct = packed, acct_of
        self.shape = (len(acct_of), packed.shape[1], packed.shape[2])
        self.dtype = packed.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        return self._packed[self._acct[idx]]


def build_rows(slot_t: np.ndarray, mask: np.ndarray, py: np.ndarray,
               bounds: tuple[float, float]):
    """Expand accounts into (account, window) rows.

    Returns
    -------
    acct_of : int64 [R]      account index behind each row
    win_of  : int8  [R]      0 train, 1 validation, 2 test
    rmask   : f32   [R, cap] the account's mask restricted to that window
    ry      : int64 [R]      max over the window's instance labels
    """
    t1, t2 = bounds
    edges = ((-np.inf, t1), (t1, t2), (t2, np.inf))
    acct, win, masks, ys = [], [], [], []
    for w, (lo, hi) in enumerate(edges):
        mw = mask * ((slot_t >= lo) & (slot_t < hi))
        keep = np.flatnonzero(mw.sum(1) > 0)
        acct.append(keep)
        win.append(np.full(len(keep), w, np.int8))
        masks.append(mw[keep])
        ys.append((py[keep] * (mw[keep] > 0)).max(1))
        del mw
    return (np.concatenate(acct).astype(np.int64), np.concatenate(win),
            np.concatenate(masks).astype(np.float32),
            np.concatenate(ys).astype(np.int64))


def temporal_budget_split(win_of: np.ndarray, ry: np.ndarray, seed: int,
                          budget: float) -> dict[str, np.ndarray]:
    """Labelled subset of the two past windows; the future window is the test set.

    The budget is applied to the past as a whole rather than to the training
    window alone, because the past is what a deployed system has labels for. The
    train/validation division inside it is the time boundary, not a ratio, so
    validation is always later than training.
    """
    rng = np.random.default_rng(seed)
    past = np.flatnonzero(win_of < 2)
    labelled = []
    for cls in np.unique(ry):
        pool = past[ry[past] == cls].copy()
        rng.shuffle(pool)
        want = min(len(pool), max(2, int(round(len(pool) * budget))))
        labelled.append(pool[:want])
    labelled = np.concatenate(labelled)
    tr = labelled[win_of[labelled] == 0]
    va = labelled[win_of[labelled] == 1]
    # A budget small enough to empty either side would make the run meaningless
    # rather than merely small, so it is reported as a failure, not padded.
    if len(tr) < 2 or len(va) < 2:
        raise ValueError(f"budget {budget:g} leaves train={len(tr)} valid={len(va)}")
    return {"train": np.sort(tr), "valid": np.sort(va),
            "test": np.sort(np.flatnonzero(win_of == 2)),
            "unused": np.sort(np.setdiff1d(past, labelled))}


def describe(win_of: np.ndarray, ry: np.ndarray, bounds: tuple[float, float]) -> str:
    parts = []
    for w, tag in enumerate(("train", "valid", "test")):
        sel = win_of == w
        n, p = int(sel.sum()), int(ry[sel].sum())
        parts.append(f"{tag} {n:,}/{p:,} ({p / max(1, n):.3%})")
    return f"windows t1={bounds[0]:.4g} t2={bounds[1]:.4g} · " + " · ".join(parts)

"""Did the pooled model point at the right transaction?

Instance labels exist in these benchmarks but are held out of training, which
is what makes these numbers a measurement rather than a circular check."""
from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score


def packed_top(py, te, pp, score, valid):
    s = np.where(valid, score[pp], -np.inf)
    return py[te[pp], s.argmax(axis=1)].mean()


def instance_ranking_metrics(py, te, pp, score, valid, ks=(1, 3, 5, 10),
                             fracs=(0.01, 0.05, 0.10)) -> dict:
    """How well does the model point an analyst at the right transaction?

    P@1 alone answers "is the single top-ranked transaction the flagged one",
    which matters when a reviewer opens one item. Recall@k is the operational
    question: an analyst opening the top k transactions of a flagged account
    wants to know whether the evidence is in there. Both are only computable
    because AML ships transaction-level labels -- the pathology MIL work this
    architecture comes from cannot check its own attention at all.

    Ranking is by the per-instance head score, which exists for every pooling,
    rather than by attention weights, which exist only for one.
    """
    s = np.where(valid, score[pp], -np.inf)
    truth = py[te[pp]]                                   # n_pos_bags x cap
    n_flag = (truth * valid).sum(axis=1)
    sizes = valid.sum(axis=1)
    order = np.argsort(-s, axis=1)
    out = {}
    for k in ks:
        topk = np.take_along_axis(truth, order[:, :k], axis=1)
        hit = topk.sum(axis=1)
        # Three different questions, and they diverge for k > 1.
        #   recall@k  what share of this account's flagged transactions surfaced
        #   hit@k     did the analyst find anything at all in the top k
        #   prec@k    of the k items opened, how many were flagged
        # recall is the one for the main table: hit saturates as k grows, and
        # precision is capped by the bag's positive count -- a median of 2
        # flagged among 268 transactions bounds precision@10 at 0.2 no matter
        # how good the ranking is, which reads as a weak model when it is a
        # property of the data.
        with np.errstate(invalid="ignore", divide="ignore"):
            rec = np.where(n_flag > 0, hit / np.maximum(n_flag, 1), np.nan)
        out[f"instance_recall_at_{k}"] = float(np.nanmean(rec))
        out[f"instance_hit_at_{k}"] = float((hit > 0).mean())
        out[f"instance_prec_at_{k}"] = float(np.mean(hit / min(k, 1e9)))
    # chance for "any hit in top-k" under a random ranking of the bag
    sizes = valid.sum(axis=1)
    # Rank-based summaries that do not depend on a k at all, plus the bag
    # statistics needed to read any of the @k numbers. Recorded whether or not
    # they are reported: which metric a question needs is rarely known when the
    # run is launched, and re-running to add a column is expensive.
    flat_y, flat_s, flat_grp = [], [], []
    for i in range(len(pp)):
        m = valid[i]
        flat_y.append(truth[i][m])
        flat_s.append(s[i][m])
        flat_grp.append(np.full(int(m.sum()), i))
    fy = np.concatenate(flat_y) if flat_y else np.zeros(0)
    fs = np.concatenate(flat_s) if flat_s else np.zeros(0)
    if fy.size and 0 < fy.sum() < fy.size:
        out["instance_auprc_pooled"] = float(average_precision_score(fy, fs))
        out["instance_auroc_pooled"] = float(roc_auc_score(fy, fs))
    # per-bag average precision and reciprocal rank: the analyst opens one
    # account at a time, so the within-bag ranking is what they experience
    aps, rrs = [], []
    for i in range(len(pp)):
        m = valid[i]
        yi, si = truth[i][m], s[i][m]
        if yi.sum() == 0 or yi.sum() == yi.size:
            continue
        aps.append(average_precision_score(yi, si))
        rank = np.argsort(-si)
        rrs.append(1.0 / (1 + int(np.flatnonzero(yi[rank])[0])))
    if aps:
        out["instance_map_per_bag"] = float(np.mean(aps))
        out["instance_mrr_per_bag"] = float(np.mean(rrs))
    # Fixed k is only half the story. Bag sizes span two orders of magnitude
    # here -- Elliptic++ has a median of 2 transactions per account, AMLSim 268
    # -- so one constant k means "the whole bag" on one dataset and "the top
    # 4%" on another, and recall@k saturates to 1.0 wherever k exceeds the bag.
    # Two data-derived cuts are added alongside, both standard elsewhere:
    #   k = the bag's own positive count, which is what the anomaly-detection
    #       and pathology-MIL literature use ("sample as many as there are
    #       annotated patches"); at that k, precision and recall coincide
    #   k = a fraction of the bag, the instance-level analogue of the 0.1/0.5/1%
    #       alert budgets already used at account level
    for i_k, kk in ((f"pos", n_flag.astype(int)),):
        hits = np.zeros(len(pp))
        for i in range(len(pp)):
            k_i = max(1, int(kk[i]))
            hits[i] = truth[i][order[i, :k_i]].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(n_flag > 0, hits / np.maximum(n_flag, 1), np.nan)
        out[f"instance_recall_at_{i_k}"] = float(np.nanmean(r))
        out[f"instance_prec_at_{i_k}"] = float(np.nanmean(
            np.where(n_flag > 0, hits / np.maximum(kk, 1), np.nan)))
        out[f"instance_hit_at_{i_k}"] = float((hits > 0).mean())

    for f in fracs:
        hits = np.zeros(len(pp))
        kk = np.maximum(1, np.floor(sizes * f)).astype(int)
        for i in range(len(pp)):
            hits[i] = truth[i][order[i, :kk[i]]].sum()
        tag = f"{f:.0%}".replace("%", "pct")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(n_flag > 0, hits / np.maximum(n_flag, 1), np.nan)
        out[f"instance_recall_at_{tag}"] = float(np.nanmean(r))
        out[f"instance_prec_at_{tag}"] = float(np.mean(hits / kk))
        out[f"instance_hit_at_{tag}"] = float((hits > 0).mean())
        out[f"instance_k_at_{tag}_median"] = float(np.median(kk))

    out["instance_bag_size_median"] = float(np.median(sizes))
    out["instance_bag_size_mean"] = float(np.mean(sizes))
    out["instance_flagged_median"] = float(np.median(n_flag))
    out["instance_flagged_mean"] = float(np.mean(n_flag))
    out["instance_n_pos_bags"] = int(len(pp))
    return out


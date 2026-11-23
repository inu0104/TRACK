"""Split rules and metrics, shared by every runner.

Every choice that could touch the test set is made on validation: the decision
threshold, the epoch, the width. Test is read once, at the end.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from sklearn.metrics import (f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedShuffleSplit

# Alert budgets an investigator could actually work through, as a share of the
# scored population. Precision at a fixed budget is what an AML team asks for;
# a threshold-free average is not something anyone operates at.
ALERT_FRACTIONS = (0.001, 0.005, 0.01)


def budget_tag(budget: float) -> str:
    return f"{budget:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def split_hash(parts: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in parts:
        normalized = np.asarray(array, dtype=np.int64)
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def fixed_test_partition(y: np.ndarray, seed: int, test_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified development pool and fixed test partition."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    dev_idx, test_idx = next(splitter.split(np.zeros(len(y), dtype=np.uint8), y))
    return np.sort(dev_idx.astype(np.int64)), np.sort(test_idx.astype(np.int64))


def strict_budget_split(
    y: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    budget: float,
    train_share: float,
) -> dict[str, np.ndarray]:
    """Create a class-stratified, nested labelled subset inside ``dev_idx``.

    Each class is shuffled once per seed.  Taking prefixes therefore makes the
    1% labelled set a subset of 5%, which is a subset of 10% (subject only to
    classwise rounding).  The fixed test set is identical across budgets.
    """
    rng = np.random.default_rng(seed)
    labelled_by_class: list[np.ndarray] = []
    unused_by_class: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_dev = dev_idx[y[dev_idx] == cls].copy()
        rng.shuffle(cls_dev)
        requested = int(round(float(np.sum(y == cls)) * budget))
        requested = min(len(cls_dev), max(2, requested))
        labelled_by_class.append(cls_dev[:requested])
        unused_by_class.append(cls_dev[requested:])

    labelled = np.concatenate(labelled_by_class)
    unused = np.concatenate(unused_by_class)
    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_labelled = labelled[y[labelled] == cls]
        n_train = int(np.floor(len(cls_labelled) * train_share))
        n_train = min(len(cls_labelled) - 1, max(1, n_train))
        train_parts.append(cls_labelled[:n_train])
        valid_parts.append(cls_labelled[n_train:])

    split = {
        "train": np.sort(np.concatenate(train_parts).astype(np.int64)),
        "valid": np.sort(np.concatenate(valid_parts).astype(np.int64)),
        "test": np.sort(test_idx.astype(np.int64)),
        "unused": np.sort(unused.astype(np.int64)),
    }
    all_parts = np.concatenate(list(split.values()))
    if len(np.unique(all_parts)) != len(y) or len(all_parts) != len(y):
        raise RuntimeError("split partitions overlap or do not cover every node")
    for key in ("train", "valid", "test"):
        if len(np.unique(y[split[key]])) != 2:
            raise RuntimeError(f"{key} partition does not contain both classes")
    return split


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5, float(f1_score(y_true, score >= 0.5, zero_division=0))
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def alert_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score, kind="mergesort")
    prevalence = float(np.mean(y_true == 1))
    total_pos = max(1, int(np.sum(y_true == 1)))
    result: dict[str, float] = {"test_prevalence": prevalence}
    for fraction in ALERT_FRACTIONS:
        k = max(1, int(np.ceil(len(y_true) * fraction)))
        positives = int(np.sum(y_true[order[:k]] == 1))
        suffix = f"{fraction * 100:g}pct".replace(".", "p")
        precision = positives / k
        result[f"precision_at_{suffix}"] = precision
        result[f"recall_at_{suffix}"] = positives / total_pos
        result[f"lift_at_{suffix}"] = precision / prevalence if prevalence > 0 else float("nan")
        result[f"alerts_per_tp_at_{suffix}"] = k / positives if positives else float("inf")
    return result


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(np.uint8)
    return {
        "threshold": float(threshold),
        "f1_susp": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_susp": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_susp": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "predicted_positive_rate": float(np.mean(pred == 1)),
    }

#!/usr/bin/env python3
"""Run TRACK on one dataset.

    python run_track.py --dataset amlworld --output-dir results/main_amlworld \
        --split temporal --encoder mlp --poolings max --counterparty-context \
        --seeds 2024 2025 2026 2027 2028 --budgets 1.0 0.10 --device cuda:0

Writes metrics/seed_metrics.csv with one row per (pooling, seed, budget) and
fifty-one columns, preds/*.npy with the raw scores behind them, and
manifest.json with the configuration and the code revision that produced them.
Every threshold and every hyper-parameter is chosen on validation; test is read
once.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from _commons.instances import (build_instances, label_survival,
                                pack_bags)
from _commons.metrics import (alert_metrics, best_f1_threshold,
                              classification_metrics, fixed_test_partition,
                              strict_budget_split)
from _commons.provenance import git_sha, utc_now
from _commons.ranking import instance_ranking_metrics, packed_top
from _datasets.loaders import LOADERS
from _datasets.temporal_split import (RowView, build_rows,
                                      describe as describe_windows,
                                      temporal_budget_split, window_bounds)
from model.track import infer, train_eval



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("random", "temporal"), default="random",
                    help="random: stratified account partition, what the "
                         "fraud-GNN baselines use. temporal: AMLworld's "
                         "60-20-20 cut on transaction time, one bag per "
                         "(account, window).")
    ap.add_argument("--cap", type=int, default=None,
                    help="max instances kept per bag. Left unset it is the "
                         "--cap-quantile of the bag-size distribution, because a "
                         "fixed constant silently destroys the label on dense "
                         "datasets: AMLNet averages 198 transactions per account "
                         "with ~1.6 flagged, so a cap of 64 keeps a random third "
                         "and drops the only positive instance from most positive "
                         "bags. Truncation cannot be label-aware without leaking, "
                         "so the cap simply has to be large enough.")
    ap.add_argument("--cap-quantile", type=float, default=0.99)
    ap.add_argument("--cap-max", type=int, default=1024,
                    help="hard ceiling; padded tensors are bags x cap x features")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10])
    ap.add_argument("--poolings", nargs="+", default=["max", "mean", "attention"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3],
                    help="searched identically for every pooling. Tuning only "
                         "the proposal and leaving the comparison fixed would "
                         "decide the result by budget rather than by method.")
    ap.add_argument("--hidden", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--save-instance-scores", action="store_true",
                    help="also write per-slot test scores, slot labels and the "
                         "padding mask. Needed to draw one account's "
                         "transactions in rank order; 13 MB per cell on AMLNet")
    ap.add_argument("--drop-flagged", action="store_true",
                    help="remove the flagged transactions from every training "
                         "bag, so each positive bag keeps its label with nothing "
                         "positive left in it. The pointed form of --train-drop, "
                         "which changed nothing at 0.1 or 0.3. Training only; "
                         "validation and test bags are untouched")
    ap.add_argument("--train-drop", type=float, default=0.0,
                    help="instance dropout during supervised MIL training. This "
                         "is the one that damages the label: 82.6%% of positive "
                         "AMLworld bags hold exactly one flagged transaction, so "
                         "a drop rate p hands the loss a positive bag with no "
                         "positive instance in it a share p of the time. Raise it "
                         "only to demonstrate that damage.")
    ap.add_argument("--counterparty-context", action="store_true",
                    help="append the other endpoint's account-context features "
                         "to each instance -- one precomputed hop of message "
                         "passing on the bag-instance graph")
    ap.add_argument("--encoder", default="mlp", choices=["mlp", "set"],
                    help="mlp scores each transaction alone; set adds "
                         "self-attention across the bag so a transaction can be "
                         "represented relative to the others around it")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--test-ratio", type=float, default=0.80)
    ap.add_argument("--train-share", type=float, default=0.80)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = args.output_dir.resolve()
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    # test-set scores, so the comparison against the baselines can bootstrap
    # over accounts paired within a seed rather than over ten seed-level numbers
    (out / "preds").mkdir(parents=True, exist_ok=True)
    run_tag = ""
    if args.counterparty_context:
        run_tag += "_cpctx"
    if args.encoder != "mlp":
        run_tag += f"_{args.encoder}"
    if args.train_drop:
        run_tag += f"_traindrop{args.train_drop:g}"
    if args.drop_flagged:
        run_tag += "_dropflagged"
    if args.split != "random":
        run_tag += f"_{args.split}"
    device = torch.device(args.device)

    print(f"[{utc_now()}] loading {args.dataset}", flush=True)
    ev, accounts, y = LOADERS[args.dataset]()
    print(f"  {len(ev):,} events, {len(accounts):,} bags, "
          f"positives {int(y.sum()):,} ({y.mean():.4%})", flush=True)
    # A packed slot carries no timestamp. The transaction id does, and pack_bags
    # already threads one extra per-instance array through its sampling, so the
    # temporal split rides on `eid` rather than needing its own plumbing.
    if args.split == "temporal":
        x, bag, inst_y, names, eid = build_instances(
            ev, accounts, with_eid=True,
            counterparty_context=args.counterparty_context)
    else:
        x, bag, inst_y, names = build_instances(
                ev, accounts, counterparty_context=args.counterparty_context)
        eid = None
    print(f"  {len(x):,} instances x {len(names)} features: {names}", flush=True)

    raw_sizes = np.bincount(bag, minlength=len(accounts))
    cap = args.cap or int(min(args.cap_max,
                              max(8, np.quantile(raw_sizes, args.cap_quantile))))
    over = float((raw_sizes > cap).mean())
    kept = float(np.minimum(raw_sizes, cap).sum() / max(1, raw_sizes.sum()))
    mem = len(accounts) * cap * x.shape[1] * 4 / 1e9
    print(f"  bag size: median {np.median(raw_sizes):.0f} mean {raw_sizes.mean():.1f} "
          f"max {raw_sizes.max()}", flush=True)
    survival = label_survival(raw_sizes, np.bincount(bag, weights=inst_y,
                                                     minlength=len(accounts)), y, cap)
    print(f"  cap={cap} (q{args.cap_quantile:g}); bags truncated {over:.2%}, "
          f"instances kept {kept:.2%}, padded tensor {mem:.2f} GB", flush=True)
    print(f"  positive-bag label survival after capping: {survival:.2%}", flush=True)
    packed_out = pack_bags(x, bag, inst_y, len(accounts), cap,
                           np.random.default_rng(0), eid=eid)
    if eid is not None:
        packed, mask, py, peid = packed_out
    else:
        packed, mask, py = packed_out
    # Under the temporal split a bag is an (account, window) pair, so the packed
    # tensor stays as it is and the rows around it change. `acct_of` is what maps
    # a row back to the account whose slots it reads.
    bounds = acct_of = None
    if args.split == "temporal":
        bounds = window_bounds(ev)
        ev_t = ev["t"].to_numpy(np.float64)
        slot_t = np.where(peid >= 0, ev_t[np.maximum(peid, 0)], np.inf)
        acct_of, win_of, mask, y = build_rows(slot_t, mask, py, bounds)
        py = py[acct_of]
        del slot_t
        packed = RowView(packed, acct_of)
        print(f"  {describe_windows(win_of, y, bounds)}", flush=True)
        print(f"  rows {len(acct_of):,} over {len(accounts):,} accounts "
              f"({len(acct_of) / len(accounts):.2f} windows per account)", flush=True)


    rows = []
    chosen: dict[tuple[str, float], dict] = {}   # (pooling, budget) -> winning config
    np.save(out / "preds" / "y.npy", y)
    for seed in args.seeds:
        if args.split == "temporal":
            test_idx = np.flatnonzero(win_of == 2)
        else:
            dev_idx, test_idx = fixed_test_partition(y, seed, args.test_ratio)
        np.save(out / "preds" / f"test_idx_s{seed}.npy", test_idx)
        for budget in args.budgets:
            if args.split == "temporal":
                try:
                    split = temporal_budget_split(win_of, y, seed, budget)
                except ValueError as e:
                    print(f"  skip budget {budget:g}: {e}", flush=True)
                    continue
            else:
                split = strict_budget_split(y, dev_idx, test_idx, seed, budget,
                                            args.train_share)
            for pooling in args.poolings:
                # Configuration is chosen on validation AUPRC only; test is read
                # once, for the winner. Early stopping already uses validation,
                # so this adds no new label use beyond a wider search.
                #
                # The search runs at the first seed of each (pooling, budget) and
                # the winner is reused for the rest, which is what the baseline
                # runner does. Repeating the full grid per seed multiplies the
                # cost fourfold and changes nothing about the protocol -- the
                # choice is still made on validation, test is still read once.
                # What it gives up is per-seed adaptation, so the reused
                # configuration is written into every row and can be checked.
                grid = ([chosen[(pooling, budget)]] if (pooling, budget) in chosen
                        else [{"lr": a, "hidden": b}
                              for a in args.lrs for b in args.hidden])
                best, fit_s, fit_mb = None, 0.0, 0.0
                for c in grid:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    t_fit = time.time()
                    r = train_eval(packed, mask, y, split, pooling, seed, device,
                                   args.epochs, c["lr"], args.batch, c["hidden"],
                                   args.patience, None,
                                   train_drop=args.train_drop,
                                   encoder_kind=args.encoder,
                                   py=py,
                                   drop_flagged=args.drop_flagged)
                    # Time the winning configuration only, so the number is one
                    # fit rather than a search, and is comparable to the
                    # baseline runner's fit_seconds.
                    if best is None or r[4] > best[0][4]:
                        best, fit_s = (r, c), time.time() - t_fit
                        fit_mb = (torch.cuda.max_memory_allocated() / 2 ** 20
                                  if device.type == "cuda" else 0.0)
                (vs, ts, att, isc, val_ap), cfg = best
                chosen.setdefault((pooling, budget), cfg)
                te = split["test"]
                # kept so the comparison against the baselines can bootstrap
                # over accounts paired within a seed, rather than over the ten
                # seed-level numbers
                np.save(out / "preds" / f"{pooling}{run_tag}_s{seed}_b{budget:g}.npy",
                        np.asarray(ts, np.float32))
                if args.save_instance_scores:
                    # Per-slot scores for the test bags, with the slot labels and
                    # the padding mask beside them. Enough to draw one account's
                    # transactions in rank order and mark which of them was the
                    # laundering transfer, which no aggregate model can produce.
                    # Off by default: on AMLNet this is 13 MB per cell.
                    tag = f"{pooling}{run_tag}_s{seed}_b{budget:g}"
                    np.save(out / "preds" / f"inst_{tag}.npy", np.asarray(isc, np.float32))
                    np.save(out / "preds" / f"instlab_{tag}.npy", py[te].astype(np.int8))
                    np.save(out / "preds" / f"instmask_{tag}.npy", mask[te].astype(np.int8))
                thr, _ = best_f1_threshold(y[split["valid"]], vs)
                m = classification_metrics(y[te], ts, thr)
                m["auprc"] = float(average_precision_score(y[te], ts))
                m["auroc"] = float(roc_auc_score(y[te], ts))
                m.update(alert_metrics(y[te], ts))

                pp = np.flatnonzero(y[te] == 1)
                if len(pp):
                    valid = mask[te[pp]] > 0
                    m["instance_p_at_1"] = float(packed_top(py, te, pp, isc, valid))
                    m.update(instance_ranking_metrics(py, te, pp, isc, valid))
                    m["attention_p_at_1"] = (float(packed_top(py, te, pp, att, valid))
                                             if pooling == "attention" else float("nan"))
                    # Same ranking metrics over the max-credit attribution, so
                    # the two ways of asking "which transaction carried this
                    # bag" can be read off one row.
                    if pooling == "max":
                        m.update({k.replace("instance_", "maxcredit_"): v for k, v
                                  in instance_ranking_metrics(py, te, pp, att,
                                                              valid).items()})
                    m["instance_chance"] = float((py[te[pp]] * valid).sum() / valid.sum())
                else:
                    m["instance_p_at_1"] = m["attention_p_at_1"] = float("nan")
                    m["instance_chance"] = float("nan")

                rows.append({"dataset": args.dataset, "pooling": pooling,
                             "tag": run_tag,
                             "seed": seed, "budget": budget,
                             "sel_lr": cfg["lr"], "sel_hidden": cfg["hidden"],
                             "fit_seconds": round(fit_s, 1),
                             "peak_gpu_mb": round(fit_mb, 1),
                             "valid_auprc": val_ap, **m})
                print(f"  {pooling:9} seed={seed} b={budget}: "
                      f"AUPRC={m['auprc']:.4f} F1={m['f1_susp']:.4f} "
                      f"instR@5={m.get('instance_recall_at_5', float('nan')):.4f} "
                      f"chance={m['instance_chance']:.4f} "
                      f"[lr={cfg['lr']:g} h={cfg['hidden']}]", flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics" / "seed_metrics.csv", index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    print("\n=== summary ===")
    print(frame.groupby("pooling")[["auprc", "f1_susp", "instance_p_at_1",
                                    "instance_chance"]].agg(["mean", "std"])
          .round(4).to_string(), flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "experiment": f"mil_gate_{args.dataset}",
        "finished_at_utc": utc_now(), "command": [sys.executable, *sys.argv],
        "commit_sha": git_sha(), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__,
        "instance_features": names,
        "bags": int(len(accounts)), "positives": int(y.sum()),
        "instances": int(len(x)), "capped_share": over,
        "instances_kept_share": kept,
        "label_survival_after_cap": survival,
        "protocol": {"cap": cap, "cap_quantile": args.cap_quantile,
                     "search_lrs": args.lrs, "search_hidden": args.hidden,
                     "patience": args.patience,
                     "selection": "validation AUPRC, identical budget per pooling",
                     "train_drop": args.train_drop, "encoder": args.encoder,
                     "counterparty_context": bool(args.counterparty_context),
                     "seeds": args.seeds, "budgets": args.budgets,
                     "epochs": args.epochs, "test_ratio": args.test_ratio,
                     "train_share": args.train_share, "split": args.split,
                     "window_bounds": list(bounds) if bounds is not None else None,
                     "rows": int(len(acct_of)) if acct_of is not None else None,
                     "threshold_selection": "validation_only_max_f1"},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

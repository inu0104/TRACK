"""Raw transaction tables, one loader per dataset.

Every loader returns the same triple -- an event frame with (source, target,
amount, t, flag), the account index those endpoints refer to, and the account
label -- so the rest of the pipeline never learns which dataset it is running
on. `flag` is the transaction-level label; it is carried for evaluation only
and never reaches training.

See _datasets/README.md for how to obtain each source."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _account_index(df: pd.DataFrame) -> pd.Index:
    """Reproduce the aggregate runners' index construction exactly.

    They take it from pd.concat of two groupby frames; sorted(set|set) looks
    equivalent but orders differently, which silently gives the same seed a
    different train/test split and makes the AUPRC numbers incomparable.
    """
    o = df.groupby("source")["amount"].mean().rename("o")
    i = df.groupby("target")["amount"].mean().rename("i")
    return pd.concat([o, i], axis=1).index


def load_amlworld(max_day: float = 9.999):
    df = pd.read_csv(REPO / "datasets/amlworld/HI-Small_Trans.csv")
    df.columns = ["Timestamp", "From_Bank", "From_Account", "To_Bank", "To_Account",
                  "Amount_Received", "Receiving_Currency", "Amount_Paid",
                  "Payment_Currency", "Payment_Format", "Is_Laundering"]
    ts = pd.to_datetime(df["Timestamp"])
    day = (ts - ts.min()).dt.total_seconds() / 86400.0
    keep = day <= max_day                      # drop the degenerate tail
    df, ts, day = df[keep], ts[keep], day[keep]
    ev = pd.DataFrame({
        "source": df["From_Bank"].astype(str) + ":" + df["From_Account"].astype(str),
        "target": df["To_Bank"].astype(str) + ":" + df["To_Account"].astype(str),
        "t": day.to_numpy() * 24.0,
        "amount": df["Amount_Paid"].fillna(0.0).to_numpy(),
        "flag": df["Is_Laundering"].to_numpy(),
        "hour": ts.dt.hour.to_numpy(),
        "ch1": df["Payment_Format"].astype("category").cat.codes.to_numpy(),
        "ch2": df["Payment_Currency"].astype("category").cat.codes.to_numpy(),
    })
    acc = _account_index(ev)
    fl = ev[ev.flag == 1]
    y = acc.isin(set(fl.source) | set(fl.target)).astype(np.int64)
    return ev, acc, y


def load_amlnet(max_day: float = 186.0):
    df = pd.read_csv(REPO / "datasets/amlnet/AMLNet_August 2025.csv",
                     usecols=["type", "amount", "nameOrig", "nameDest",
                              "isMoneyLaundering", "hour", "day_of_month", "month"])
    t = ((df["month"] - df["month"].min()) * 31 + df["day_of_month"]) * 24 + df["hour"]
    day = (t - t.min()) / 24.0
    keep = day <= max_day
    df, day = df[keep], day[keep]
    ev = pd.DataFrame({
        "source": df["nameOrig"].to_numpy(), "target": df["nameDest"].to_numpy(),
        "t": day.to_numpy() * 24.0,
        "amount": df["amount"].fillna(0.0).to_numpy(),
        "flag": df["isMoneyLaundering"].to_numpy(),
        "hour": df["hour"].to_numpy(),
        "ch1": df["type"].astype("category").cat.codes.to_numpy(),
    })
    acc = _account_index(ev)
    fl = ev[ev.flag == 1]
    y = acc.isin(set(fl.source) | set(fl.target)).astype(np.int64)
    return ev, acc, y


def load_ellipticpp_wallet(max_day: float = np.inf):
    """Wallet layer. Each AddrAddr entry is one transaction's input-output pair.

    Amount and time come from the transaction table, the same join the
    20260812 directional run used. Instance labels are the transaction classes,
    so 'flag' is 1 only for transactions the original Elliptic marked illicit;
    the 77% marked unknown count as 0, which understates instance recall and is
    recorded as such.

    This is the layer whose account labels are defined by incidence -- illicit
    if it touches one illicit transaction -- so the bag rule holds by
    construction here. That makes it the sharpest test of the MIL framing and
    simultaneously the layer where a win proves least about laundering.
    """
    A = REPO / "datasets/ellipticpp/actors"
    T = REPO / "datasets/ellipticpp/transactions"
    tx = pd.read_csv(T / "txs_features.csv",
                     usecols=["txId", "Time step", "total_BTC",
                              "num_input_addresses", "num_output_addresses"])
    tx["pair_btc"] = tx["total_BTC"] / (tx["num_input_addresses"]
                                        * tx["num_output_addresses"]).clip(lower=1)
    cls = pd.read_csv(T / "txs_classes.csv")
    tx = tx.merge(cls, on="txId", how="left")

    a2t = pd.read_csv(A / "AddrTx_edgelist.csv")
    t2a = pd.read_csv(A / "TxAddr_edgelist.csv")
    j = (a2t.merge(t2a, on="txId", how="inner")
            .merge(tx[["txId", "Time step", "pair_btc", "class"]], on="txId", how="left"))
    ev = pd.DataFrame({
        "source": j["input_address"].to_numpy(), "target": j["output_address"].to_numpy(),
        "t": j["Time step"].fillna(0).to_numpy().astype(float),
        "amount": j["pair_btc"].fillna(0.0).to_numpy(),
        "flag": (j["class"] == 1).astype(int).to_numpy(),
    })

    # Bag order has to be the aggregate run's order, not wallets_classes order.
    # 20260812_directional_agg_ellipticpp indexes by wallets_features after
    # keeping each address's last time step, then restricts to labelled rows;
    # taking wallets_classes order instead gives the same accounts in a
    # different sequence, so the same seed splits a different set and the two
    # AUPRC numbers cannot be compared. This bit me on AMLworld already.
    wf = pd.read_csv(A / "wallets_features.csv", usecols=["address", "Time step"])
    wf = wf.sort_values("Time step").groupby("address", as_index=False).last()
    wc = pd.read_csv(A / "wallets_classes.csv")
    wf = wf.merge(wc, on="address", how="left")
    labelled = wf[wf["class"].isin([1, 2])]
    acc = pd.Index(labelled["address"])
    y = (labelled["class"] == 1).astype(np.int64).to_numpy()
    ev = ev[ev.source.isin(acc) | ev.target.isin(acc)]
    return ev, acc, y


def load_amlsim(max_day: float = np.inf):
    """AMLSim, the IBM simulator behind much of the earlier AML GNN work.

    Taken from the Kaggle snapshot (anshankul/ibm-amlsim-example-dataset) rather
    than regenerated, so the instance is fixed and citable. 1,323,234 transactions
    over 200 steps across 9,999 accounts, 1,719 flagged transactions (0.13%),
    which derives to 1,639 flagged accounts (16.4%).

    Two things make it useful beyond being a fifth dataset. It ships its own
    account-level IS_FRAUD column -- 1,685 accounts against the 1,639 our
    incidence rule derives, so the rule can be checked against the simulator's
    own intent. And alerts.csv ties each flagged transaction to a typology
    (cycle 936, fan_in 783), which lets instance-level precision be broken down
    by pattern; AMLworld's pattern file does not join back to its transactions,
    so this is the only place that check is available.

    It also has no degenerate tail: the last twelve steps hold 6.1k-7.2k
    transactions at 0.04-0.24% flagged. AMLworld and AMLNet both collapse. That
    rules out "pattern-injection simulators do this" as an explanation and
    confines the defect to those two datasets.
    """
    D = REPO / "datasets/amlsim"
    t = pd.read_csv(D / "transactions.csv")
    ev = pd.DataFrame({
        "source": t["SENDER_ACCOUNT_ID"].to_numpy(),
        "target": t["RECEIVER_ACCOUNT_ID"].to_numpy(),
        "t": t["TIMESTAMP"].to_numpy().astype(float) * 24.0,   # steps -> hours
        "amount": t["TX_AMOUNT"].fillna(0.0).to_numpy(),
        "flag": (t["IS_FRAUD"].astype(str).str.lower() == "true").astype(int).to_numpy(),
        "ch1": t["TX_TYPE"].astype("category").cat.codes.to_numpy(),
    })
    keep = (ev["t"] / 24.0) <= max_day
    ev = ev[keep]
    acc = _account_index(ev)
    fl = ev[ev.flag == 1]
    y = acc.isin(set(fl.source) | set(fl.target)).astype(np.int64)
    return ev, acc, y


# The original Elliptic dataset has no loader on purpose. Its nodes are
# transactions and its labels are per-transaction, so there is no bag: the
# supervision is already at instance level and MIL has nothing to weaken.
# It was also verified to be a strict subset of Elliptic++'s transaction layer
# -- identical txIds, identical 234,355 edges, labels matching 100% after the
# unknown/3 encoding is mapped, and the 166 shared features agreeing to 4e-16.

LOADERS = {"amlworld": load_amlworld, "amlnet": load_amlnet,
           "amlsim": load_amlsim, "ellipticpp": load_ellipticpp_wallet}

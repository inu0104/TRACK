# Datasets

No dataset is redistributed here. Each one comes from its own source under its
own terms. `fetch.py` downloads the two whose hosts allow scripted downloads.

| Dataset | Accounts | Transactions | Suspicious accounts | Source |
|---|---:|---:|---:|---|
| AMLworld HI-Small | 515,078 | 5,077,097 | 1.14% | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` |
| AMLNet | 11,000 | 1,088,410 | 13.35% | Zenodo record `16736515` |
| AMLSim | 9,999 | 1,323,234 | 16.39% | Kaggle `anshankul/ibm-amlsim-example-dataset` |
| Elliptic++ | 265,354 | 1,661,916 | 5.38% | https://github.com/git-disl/EllipticPlusPlus |
| HOFINET | 452,816 | 4,732,130 | 2.13% | AI Hub dataset 71925 |

The counts are what the loaders return after each loader's tail cut. They are
not the raw upstream file sizes. "Suspicious accounts" is the share of accounts
that are ever flagged. The share of `(account, window)` samples flagged in the
test window is lower: 0.46%, 2.25%, 4.23%, 5.24% and 1.31% in the order above.

**AMLSim** is a published snapshot of IBM's AMLSim generator, not output we
produced. Its publisher does not document the generator configuration behind
the snapshot, so the snapshot itself is the reproducible artifact.

**HOFINET** is a synthetic interbank settlement network published on AI Hub
("이상 판별을 위한 금융거래 정보 및 사용자 패턴 합성데이터",
https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71925). It is
generated to reflect patterns in real electronic fund transfers. AI Hub
requires an account to download and applies its own terms of use. The counts
above are from the release used in the paper; AI Hub updated the dataset in
June 2026, so a newer download may give slightly different numbers.

## Retrieval

```bash
python _datasets/fetch.py            # AMLworld HI-Small and AMLNet
python _datasets/fetch.py amlworld   # or one at a time
python _datasets/fetch.py amlnet
```

AMLworld needs a Kaggle API token at `~/.kaggle/kaggle.json`
(see https://www.kaggle.com/docs/api). AMLNet is public on Zenodo.

Download AMLSim, Elliptic++ and HOFINET by hand from the links above. For
HOFINET, save the electronic-transfer table (header starting with `거래일자`)
as `datasets/hofinet/HOFINET.txt`. Every dataset is
read from `datasets/<name>/`, which this repository ignores:

```
datasets/
├── amlworld/HI-Small_Trans.csv
├── amlnet/AMLNet_August 2025.csv
├── amlsim/transactions.csv
├── ellipticpp/
│   ├── actors/AddrTx_edgelist.csv
│   ├── actors/TxAddr_edgelist.csv
│   ├── actors/wallets_features.csv
│   ├── actors/wallets_classes.csv
│   ├── transactions/txs_features.csv
│   └── transactions/txs_classes.csv
└── hofinet/HOFINET.txt
```

## What the loaders return

Every loader returns the same triple, so nothing downstream knows which dataset
it is running on:

- `ev`: one row per transaction with `source`, `target`, `amount`, `t`, `flag`
- `accounts`: the index those endpoints refer to
- `y`: the account label, which is the maximum of `flag` over the account

`flag` is the transaction-level label. It is used for evaluation only and never
reaches training.

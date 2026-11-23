#!/usr/bin/env python3
"""Download the AMLworld (HI-Small) and AMLNet raw datasets.

No dataset is redistributed here. AMLworld and AMLNet live behind Kaggle and
Zenodo, so they must be fetched before either can be run. Elliptic++ and AMLSim
are downloaded manually as described in _datasets/README.md.

AMLworld's Kaggle dataset is 8.18 GB across HI-/LI- Small/Medium/Large; we pull
only the three HI-Small files (~0.51 GB) since that is the split the paper uses.
Kaggle sometimes wraps a single-file download in a zip, so responses are sniffed
for a PK header rather than trusted.

Destinations match the paths _datasets/loaders.py reads:
    datasets/amlworld/HI-Small_Trans.csv
    datasets/amlnet/AMLNet_August 2025.csv
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
KAGGLE_FILES = ["HI-Small_Trans.csv", "HI-Small_accounts.csv", "HI-Small_Patterns.txt"]
ZENODO_RECORD = "16736515"
UA = "Mozilla/5.0 (X11; Linux aarch64) dataset-fetcher/1.0"


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def stream(req: urllib.request.Request, dest: Path, label: str) -> None:
    """Download to a .part file, then sniff for a zip wrapper before committing."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with tmp.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (64 << 20) < (1 << 20):
                    print(f"    {label}: {human(done)} / {human(total)}", flush=True)

    with tmp.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":
        print(f"    {label}: unwrapped from a zip", flush=True)
        with zipfile.ZipFile(tmp) as z:
            inner = z.namelist()[0]
            with z.open(inner) as src, dest.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
        tmp.unlink()
    else:
        tmp.replace(dest)
    print(f"  OK  {dest.name}  ({human(dest.stat().st_size)})", flush=True)


def kaggle() -> None:
    out = ROOT / "datasets/amlworld"
    out.mkdir(parents=True, exist_ok=True)
    cred = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    tok = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()

    print(f"[AMLworld] Kaggle {KAGGLE_SLUG} -> {out}")
    for name in KAGGLE_FILES:
        dest = out / name
        if dest.exists():
            print(f"  HAVE {name} ({human(dest.stat().st_size)})")
            continue
        url = (f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}/"
               + urllib.parse.quote(name))
        req = urllib.request.Request(
            url, headers={"Authorization": f"Basic {tok}", "User-Agent": UA})
        stream(req, dest, name)


def zenodo() -> None:
    out = ROOT / "datasets/amlnet"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[AMLNet] Zenodo {ZENODO_RECORD} -> {out}")
    req = urllib.request.Request(
        f"https://zenodo.org/api/records/{ZENODO_RECORD}", headers={"User-Agent": UA})
    rec = json.loads(urllib.request.urlopen(req, timeout=60).read())
    for f in rec["files"]:
        dest = out / f["key"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"  HAVE {f['key']} ({human(f['size'])})")
            continue
        stream(urllib.request.Request(f["links"]["self"], headers={"User-Agent": UA}),
               dest, f["key"])


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "amlworld"):
        kaggle()
    if which in ("all", "amlnet"):
        zenodo()
    print("\n=== summary ===")
    for d in (ROOT / "datasets/amlworld", ROOT / "datasets/amlnet"):
        if d.is_dir():
            for p in sorted(d.iterdir()):
                print(f"  {human(p.stat().st_size):>10}  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

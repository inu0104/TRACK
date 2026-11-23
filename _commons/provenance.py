"""Run provenance recorded beside every result."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                       capture_output=True, text=True, check=False)
    return r.stdout.strip()


# --------------------------------------------------------------------------
# loaders -> (events, accounts, y)
#   events: source, target, t (hours or steps), amount, flag [, hour, ch1, ch2]
#   accounts: pd.Index fixing bag order
#   y: account labels aligned to accounts
# --------------------------------------------------------------------------

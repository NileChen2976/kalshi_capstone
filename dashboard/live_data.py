# -*- coding: utf-8 -*-
"""
live_data.py -- read what live_collector.py writes (data/live/latest.json and the hourly JSONL files)
and express order books in D terms so the -D and -R books of one event share one price axis.

D terms for a book:
    D contract:  bid side = yes levels (price p, qty)            ask side = no levels -> price 1 - p
    R contract:  bid side = no levels  (price p: buying R-no = buying D at p)
                 ask side = yes levels -> price 1 - p (buying R-yes = selling D at 1 - p)
"""
from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import pandas as pd

from kalshi_data import DATA_DIR
from pair_data import is_r

LIVE_DIR = DATA_DIR / "live"
LATEST = LIVE_DIR / "latest.json"


def latest() -> dict | None:
    """Most recent round written by the collector, or None when it is not running."""
    if not LATEST.exists():
        return None
    for _ in range(3):                      # the file is replaced atomically; retry a torn read anyway
        try:
            return json.loads(LATEST.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def ladder(rec: dict, ticker: str) -> pd.DataFrame:
    """Book of one record as a ladder in D terms: columns side ('bid'/'ask'), price, qty. Sorted by price."""
    ob = rec["orderbook_fp"]
    yes = pd.DataFrame(ob.get("yes_dollars") or [], columns=["p", "q"]).astype(float)
    no = pd.DataFrame(ob.get("no_dollars") or [], columns=["p", "q"]).astype(float)
    if is_r(ticker):
        bid = pd.DataFrame({"side": "bid", "price": no["p"], "qty": no["q"]})
        ask = pd.DataFrame({"side": "ask", "price": (1 - yes["p"]).round(4), "qty": yes["q"]})
    else:
        bid = pd.DataFrame({"side": "bid", "price": yes["p"], "qty": yes["q"]})
        ask = pd.DataFrame({"side": "ask", "price": (1 - no["p"]).round(4), "qty": no["q"]})
    return pd.concat([bid, ask]).sort_values("price").reset_index(drop=True)


def top(lad: pd.DataFrame) -> dict:
    b, a = lad[lad["side"] == "bid"], lad[lad["side"] == "ask"]
    if b.empty or a.empty:
        return {"best_bid": None, "best_ask": None, "spread": None, "bid_top": None, "ask_top": None, "mid": None}
    bb, ba = b["price"].max(), a["price"].min()
    return {"best_bid": bb, "best_ask": ba, "spread": round(ba - bb, 4), "mid": (bb + ba) / 2,
            "bid_top": float(b.loc[b["price"].idxmax(), "qty"]), "ask_top": float(a.loc[a["price"].idxmin(), "qty"])}


def top_levels(lad: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """n best levels per side."""
    b = lad[lad["side"] == "bid"].nlargest(n, "price")
    a = lad[lad["side"] == "ask"].nsmallest(n, "price")
    return pd.concat([b, a]).sort_values("price").reset_index(drop=True)


def age_seconds(rec: dict) -> float:
    ts = dt.datetime.fromisoformat(rec["ts"])
    return (dt.datetime.now(dt.timezone.utc) - ts).total_seconds()


def history(ticker: str, minutes: int = 60) -> pd.DataFrame:
    """Top-of-book (D terms) per poll for the last `minutes`, from the hourly JSONL files."""
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(minutes=minutes)
    files = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= now:
        files.append(LIVE_DIR / t.strftime("%Y-%m-%d") / (t.strftime("%H") + ".jsonl"))
        t += dt.timedelta(hours=1)
    rows = []
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if f'"ticker": "{ticker}"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = dt.datetime.fromisoformat(rec["ts"])
                if ts < start:
                    continue
                tp = top(ladder(rec, ticker))
                rows.append({"ts": ts, **tp})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    j = latest()
    print("collector round:", j and j["round_ts"])
    if j:
        for tk, rec in j["books"].items():
            lad = ladder(rec, tk)
            print(f"{tk:18s} age {age_seconds(rec):5.1f}s  levels {len(lad):3d}  top {top(lad)}")
        h = history("CONTROLS-2026-D", 30)
        print("history rows (30 min):", len(h))

# -*- coding: utf-8 -*-
"""
replay_data.py -- turn the raw live captures (data/live/<UTC day>/<HH>.jsonl, 100 levels per side,
every 10 s) into compact replay files and serve them to the Replay tab.

    data/replay/<UTC day>.parquet   one row per (round, ticker, side, level):
                                    round (epoch s), ts (UTC), ticker, side ('bid'/'ask' in D terms), price, qty
                                    -- the LEVELS best levels per side, both contracts of every event in D terms

    python replay_data.py                       # export every captured day that has no replay file yet
    python replay_data.py 2026-09-08 --every 6  # one day, keep every 6th round (1-minute resolution, for git)

A day is loaded once (lru_cache) and indexed by round, so a slider move is a numpy slice.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_data import DATA_DIR
from live_data import LIVE_DIR, ladder, top_levels
from pair_data import events

REPLAY_DIR = DATA_DIR / "replay"
LEVELS = 10
ET = "America/New_York"


# --------------------------------------------------------------------------- export
def raw_days() -> list[str]:
    return sorted(p.name for p in LIVE_DIR.glob("20??-??-??") if p.is_dir())


def exported_days() -> list[str]:
    return sorted(p.stem for p in REPLAY_DIR.glob("20??-??-??.parquet"))


def export_day(day: str, levels: int = LEVELS, every: int = 1) -> Path:
    files = sorted(glob.glob(str(LIVE_DIR / day / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no live captures for {day}")
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r = int(rec["round"])
                if every > 1 and (r // 10) % every:
                    continue
                lv = top_levels(ladder(rec, rec["ticker"]), levels)
                for side, price, qty in lv[["side", "price", "qty"]].itertuples(index=False):
                    rows.append((r, rec["ts"], rec["ticker"], side, float(price), float(qty)))
    df = pd.DataFrame(rows, columns=["round", "ts", "ticker", "side", "price", "qty"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["round", "ticker", "side", "price"]).reset_index(drop=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = REPLAY_DIR / f"{day}.parquet"
    df.to_parquet(out, index=False)
    return out


# --------------------------------------------------------------------------- load
@dataclass
class ReplayDay:
    day: str
    df: pd.DataFrame            # sorted by round
    rounds: np.ndarray          # unique round epochs, ascending
    ts_et: pd.DatetimeIndex     # one naive-ET timestamp per round
    starts: np.ndarray          # df row where each round starts
    ends: np.ndarray            # df row where each round ends (exclusive)
    tob: pd.DataFrame           # index round; columns (ticker, 'bid'|'ask'|'bid_sz'|'ask_sz')

    def index_at(self, i: int) -> int:
        return int(min(max(i, 0), len(self.rounds) - 1))

    def books_at(self, i: int) -> dict[str, pd.DataFrame]:
        i = self.index_at(i)
        chunk = self.df.iloc[self.starts[i]:self.ends[i]]
        return {tk: g[["side", "price", "qty"]].reset_index(drop=True) for tk, g in chunk.groupby("ticker")}

    def label(self, i: int) -> str:
        i = self.index_at(i)
        return f"{self.ts_et[i]:%Y-%m-%d %H:%M:%S} ET  (round {i + 1} / {len(self.rounds)})"


@lru_cache(maxsize=3)
def load_day(day: str) -> ReplayDay:
    path = REPLAY_DIR / f"{day}.parquet"
    if not path.exists():
        export_day(day)
    df = pd.read_parquet(path).sort_values(["round", "ticker", "side", "price"]).reset_index(drop=True)
    rounds = np.sort(df["round"].unique())
    rv = df["round"].to_numpy()
    starts = np.searchsorted(rv, rounds, side="left")
    ends = np.searchsorted(rv, rounds, side="right")
    first_ts = df.groupby("round")["ts"].min().reindex(rounds)
    ts_et = pd.DatetimeIndex(first_ts).tz_convert(ET).tz_localize(None)
    bid = df[df["side"] == "bid"].sort_values("price").groupby(["round", "ticker"]).last()[["price", "qty"]]
    ask = df[df["side"] == "ask"].sort_values("price").groupby(["round", "ticker"]).first()[["price", "qty"]]
    tob = pd.concat({"bid": bid["price"], "bid_sz": bid["qty"], "ask": ask["price"], "ask_sz": ask["qty"]}, axis=1)
    tob = tob.unstack("ticker").swaplevel(axis=1).sort_index(axis=1)
    return ReplayDay(day, df, rounds, ts_et, starts, ends, tob)


def available_days() -> list[str]:
    return sorted(set(raw_days()) | set(exported_days()))


def event_series(rd: ReplayDay, event: str) -> pd.DataFrame:
    """Per round: D mid, R-as-D mid, ask_D + ask_R - 1, bid_D + bid_R - 1 (all in dollars), index = naive ET time."""
    ev = events()[event]
    out = pd.DataFrame(index=rd.ts_et)
    t = rd.tob.reindex(rd.rounds)
    for side in ("D", "R"):
        tk = ev[side]
        if tk in t.columns.get_level_values(0):
            out[f"bid_{side}"] = t[(tk, "bid")].to_numpy()
            out[f"ask_{side}"] = t[(tk, "ask")].to_numpy()
            out[f"mid_{side}"] = (out[f"bid_{side}"] + out[f"ask_{side}"]) / 2
    if {"ask_D", "bid_R", "bid_D", "ask_R"} <= set(out.columns):
        out["ask_sum_m1"] = out["ask_D"] - out["bid_R"]       # ask_D + (1 - bid_Rd) - 1
        out["bid_sum_m1"] = out["bid_D"] - out["ask_R"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("day", nargs="?", default=None, help="UTC capture day, e.g. 2026-09-08; default = all days without a replay file")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th 10-second round (6 = one minute)")
    ap.add_argument("--levels", type=int, default=LEVELS)
    a = ap.parse_args()
    days = [a.day] if a.day else [d for d in raw_days() if d not in exported_days()]
    for d in days:
        p = export_day(d, a.levels, a.every)
        rd = load_day.__wrapped__(d)
        print(f"{d}: {p.name} {p.stat().st_size / 1e6:.1f} MB, {len(rd.rounds)} rounds, {rd.ts_et[0]:%H:%M} - {rd.ts_et[-1]:%H:%M} ET")

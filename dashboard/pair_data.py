# -*- coding: utf-8 -*-
"""
pair_data.py -- the two binary contracts of one Kalshi event (-D / -R) expressed in one frame.

Conventions ("D terms"):
    R yes price p  ==  D yes price 1 - p            (buying R-yes is selling D-yes)
    R yes-bid  -> D ask at 1 - p ;  R yes-ask -> D bid at 1 - p
    trade direction: pro-D = buys D-yes or buys R-no ; pro-R = buys R-yes or buys D-no
    event probability P(D) = mean(mid_D, 1 - mid_R)
"""
from __future__ import annotations

import glob
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from kalshi_data import DATA_DIR, kalshi_1min, kalshi_daily, kalshi_tickers, kalshi_trades

_SUFFIX = re.compile(r"-(D|R)$")


def event_of(ticker: str) -> str:
    return _SUFFIX.sub("", ticker)


def side_of(ticker: str) -> str | None:
    m = _SUFFIX.search(ticker)
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def events() -> dict[str, dict[str, str]]:
    """{'CONTROLS-2026': {'D': 'CONTROLS-2026-D', 'R': 'CONTROLS-2026-R'}, ...} for complete pairs only."""
    ev: dict[str, dict[str, str]] = {}
    for tk in kalshi_tickers():
        s = side_of(tk)
        if s:
            ev.setdefault(event_of(tk), {})[s] = tk
    return {k: v for k, v in ev.items() if "D" in v and "R" in v}


def is_r(ticker: str) -> bool:
    return side_of(ticker) == "R"


# --------------------------------------------------------------------------- daily
def _n_trades_by_day(ticker: str) -> pd.Series:
    m = pd.read_parquet(DATA_DIR / "trades_manifest.parquet")
    m = m[m["ticker"] == ticker]
    s = pd.Series(m["n_trades"].to_numpy(), index=pd.to_datetime(m["date"]))
    return s.groupby(level=0).sum()


def pair_daily(event: str) -> pd.DataFrame:
    """Daily frame of both contracts in D terms plus cross-book checks and activity stats."""
    tD, tR = events()[event]["D"], events()[event]["R"]
    D = kalshi_daily(tD).set_index("date")
    R = kalshi_daily(tR).set_index("date")
    idx = D.index.union(R.index)
    D, R = D.reindex(idx), R.reindex(idx)
    df = pd.DataFrame(index=idx)
    df["mid_D"], df["bid_D"], df["ask_D"] = D["mid_close"], D["yes_bid_close"], D["yes_ask_close"]
    df["mid_R_raw"], df["bid_R_raw"], df["ask_R_raw"] = R["mid_close"], R["yes_bid_close"], R["yes_ask_close"]
    df["mid_Rd"], df["bid_Rd"], df["ask_Rd"] = 1 - R["mid_close"], 1 - R["yes_ask_close"], 1 - R["yes_bid_close"]
    df["residual"] = df["mid_D"] - df["mid_Rd"]                          # D mid minus R-implied D mid
    df["ask_sum_m1"] = D["yes_ask_close"] + R["yes_ask_close"] - 1        # > 0 normal, < 0 = buy-both arbitrage
    df["bid_sum_m1"] = D["yes_bid_close"] + R["yes_bid_close"] - 1        # < 0 normal, > 0 = sell-both arbitrage
    df["p_d"] = df[["mid_D", "mid_Rd"]].mean(axis=1)
    df["vol_D"], df["vol_R"] = D["volume"], R["volume"]
    df["oi_D"], df["oi_R"] = D["open_interest"], R["open_interest"]
    df["trades_D"] = _n_trades_by_day(tD).reindex(idx).fillna(0)
    df["trades_R"] = _n_trades_by_day(tR).reindex(idx).fillna(0)
    return df.reset_index().rename(columns={"index": "date"})


# --------------------------------------------------------------------------- intraday
def pair_1min(event: str, day: str) -> dict[str, pd.DataFrame]:
    """{'D': ..., 'R': ...} 1-min frames with columns t, mid, bid, ask in D terms."""
    out = {}
    for side, tk in events()[event].items():
        m = kalshi_1min(tk, day)
        if m.empty:
            out[side] = m
            continue
        f = pd.DataFrame({"t": m["t"]})
        if side == "R":
            f["mid"], f["bid"], f["ask"] = 1 - m["mid_close"], 1 - m["yes_ask_close"], 1 - m["yes_bid_close"]
        else:
            f["mid"], f["bid"], f["ask"] = m["mid_close"], m["yes_bid_close"], m["yes_ask_close"]
        f["volume"] = m["volume"]
        out[side] = f
    return out


def merged_trades(event: str, day: str) -> pd.DataFrame:
    """Trades of both contracts on one timeline, price in D terms, direction pro-D / pro-R."""
    parts = []
    for side, tk in events()[event].items():
        t = kalshi_trades(tk, day)
        if t.empty:
            continue
        t = t.copy()
        t["contract"] = side
        t["price_d"] = (1 - t["yes_price"]) if side == "R" else t["yes_price"]
        pro_d = (t["taker_side"] == "no") if side == "R" else (t["taker_side"] == "yes")
        t["direction"] = np.where(pro_d, "pro-D", "pro-R")
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_values("t").reset_index(drop=True)


def day_flow(tr: pd.DataFrame) -> pd.DataFrame:
    """Contracts and trade counts by contract x direction for one day's merged trades."""
    if tr.empty:
        return pd.DataFrame()
    g = tr.groupby(["contract", "direction"]).agg(trades=("count", "size"), contracts=("count", "sum")).reset_index()
    return g


@lru_cache(maxsize=8)
def daily_net_flow(event: str) -> pd.DataFrame:
    """Per ET day, both contracts merged: net = pro-D − pro-R contracts, plus pro_d, pro_r, n_trades, gross.
    Days without any trade are absent (callers fill with 0)."""
    tD, tR = events()[event]["D"], events()[event]["R"]
    parts = []
    for f in sorted(glob.glob(str(DATA_DIR / "trades" / "*.parquet"))):
        t = pd.read_parquet(f, columns=["ticker", "count", "taker_side", "trade_date_et"])
        t = t[t["ticker"].isin((tD, tR))]
        if not t.empty:
            parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["date", "net", "pro_d", "pro_r", "n_trades", "gross"])
    t = pd.concat(parts, ignore_index=True)
    pro_d = np.where(t["ticker"] == tD, t["taker_side"] == "yes", t["taker_side"] == "no")
    t["pro_d"] = np.where(pro_d, t["count"], 0.0)
    t["pro_r"] = np.where(pro_d, 0.0, t["count"])
    t["date"] = pd.to_datetime(t["trade_date_et"])
    g = t.groupby("date").agg(pro_d=("pro_d", "sum"), pro_r=("pro_r", "sum"), n_trades=("count", "size"), gross=("count", "sum"))
    g["net"] = g["pro_d"] - g["pro_r"]
    return g.reset_index()[["date", "net", "pro_d", "pro_r", "n_trades", "gross"]]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(events())
    for ev in events():
        pdaily = pair_daily(ev)
        print(ev, len(pdaily), "days | residual mean %.4f | ask_sum-1 mean %.4f min %.3f | bid_sum-1 mean %.4f max %.3f" % (
            pdaily.residual.mean(), pdaily.ask_sum_m1.mean(), pdaily.ask_sum_m1.min(), pdaily.bid_sum_m1.mean(), pdaily.bid_sum_m1.max()))
        last = pdaily.dropna(subset=["mid_D", "mid_Rd"]).date.max().strftime("%Y-%m-%d")
        tr = merged_trades(ev, last)
        print("  ", last, "merged trades:", len(tr), "|", day_flow(tr).to_dict("records") if len(tr) else "")

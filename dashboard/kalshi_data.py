# -*- coding: utf-8 -*-
"""
kalshi_data.py -- thin readers for the local Kalshi parquet data (daily / 1-min / trades / manifests).

The day-partitioned files under data/1min and data/trades can hold several Kalshi tickers
(the batch downloader writes every contract into the same day file), so every reader
filters by ticker.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_KALSHI_TICKER = "CONTROLS-2026-D"

DAILY_COLS = ["date", "price_open", "price_high", "price_low", "price_close",
              "yes_bid_close", "yes_ask_close", "mid_close", "spread_close",
              "volume", "open_interest"]


@lru_cache(maxsize=1)
def _daily_all() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "daily" / "kalshi_daily.parquet")
    df["date"] = pd.to_datetime(df["trade_date_et"])
    return df


def kalshi_tickers() -> list[str]:
    """All Kalshi tickers present in the daily file, default contract first."""
    t = sorted(_daily_all()["ticker"].unique().tolist())
    if DEFAULT_KALSHI_TICKER in t:
        t.remove(DEFAULT_KALSHI_TICKER)
        t.insert(0, DEFAULT_KALSHI_TICKER)
    return t


def kalshi_daily(ticker: str = DEFAULT_KALSHI_TICKER) -> pd.DataFrame:
    df = _daily_all()
    df = df[df["ticker"] == ticker]
    return df.sort_values("date")[DAILY_COLS].reset_index(drop=True)


def _read_day(kind: str, ticker: str, day: str, time_col: str) -> pd.DataFrame:
    p = DATA_DIR / kind / f"{day}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df = df[df["ticker"] == ticker].sort_values(time_col).reset_index(drop=True)
    if df.empty:
        return df
    df["t"] = df[time_col].dt.tz_localize(None)   # naive ET, for plotting
    return df


def kalshi_1min(ticker: str, day: str) -> pd.DataFrame:
    """1-minute candles for one ET day. Only minutes with activity have a row."""
    return _read_day("1min", ticker, day, "period_start_et")


def kalshi_trades(ticker: str, day: str) -> pd.DataFrame:
    return _read_day("trades", ticker, day, "timestamp_et")


@lru_cache(maxsize=None)
def _manifest_days(name: str, ticker: str, mask_col: str | None = None) -> tuple[str, ...]:
    m = pd.read_parquet(DATA_DIR / name)
    m = m[m["ticker"] == ticker]
    if mask_col:
        m = m[m[mask_col] > 0]
    return tuple(sorted(pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d").unique().tolist()))


def trade_days(ticker: str) -> list[str]:
    """ET days with at least one trade (from trades_manifest)."""
    return list(_manifest_days("trades_manifest.parquet", ticker, "n_trades"))


def all_days(ticker: str) -> list[str]:
    """Every ET day covered by the 1-min download (from 1min_manifest)."""
    return list(_manifest_days("1min_manifest.parquet", ticker))


if __name__ == "__main__":
    print("tickers:", kalshi_tickers())
    for tk in kalshi_tickers():
        d = kalshi_daily(tk)
        td = trade_days(tk)
        last = td[-1] if td else None
        print(f"{tk:18s} daily {len(d):3d} rows {d.date.min():%Y-%m-%d}..{d.date.max():%Y-%m-%d} | "
              f"trade days {len(td):3d} | last day 1min {len(kalshi_1min(tk, last)) if last else 0:4d} rows, "
              f"trades {len(kalshi_trades(tk, last)) if last else 0:4d}")

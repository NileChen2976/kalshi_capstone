# -*- coding: utf-8 -*-
"""
stock_data.py -- minimal daily stock data layer for the dashboard.

* Source:   yfinance (auto_adjust=False + actions), from 2024-06-01
* Cache:    data/stocks/daily/{TICKER}.parquet  (one file per ticker, refreshed if older than 3 days)
* Universe: data/stocks/universe.csv            (current S&P 500 constituents from Wikipedia:
                                                 ticker, company name, GICS sector)
* Adjustment: "hfq" = backward-adjusted. The first day's price is kept as traded and later
  prices are scaled UP by the cumulative dividend/split factor:
      adj_factor_t = AdjClose_t / Close_t ;  close_hfq_t = Close_t * adj_factor_t / adj_factor_0
  (Yahoo's own "Adj Close" is forward-adjusted: today's price fixed, history scaled down.)

Offline mode: set STOCK_DATA_OFFLINE=1 (used on the hosted dashboard) and the layer never
calls Yahoo; it only serves the parquet cache. Run prefetch_stocks.py before deploying.

To swap in your own daily database later, keep get_stock_daily() returning the same COLUMNS.
"""
from __future__ import annotations

import io
import os
import threading
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STOCK_DIR = DATA_DIR / "stocks" / "daily"
UNIVERSE_CSV = DATA_DIR / "stocks" / "universe.csv"
START = "2024-06-01"
OFFLINE = os.environ.get("STOCK_DATA_OFFLINE", "").strip() not in ("", "0", "false", "False")

COLUMNS = [
    "ticker", "date",
    "open", "high", "low", "close", "adj_close", "volume",
    "dividends", "splits", "adj_factor",
    "open_hfq", "high_hfq", "low_hfq", "close_hfq",
]


def normalize_ticker(t: str) -> str:
    """BRK.B -> BRK-B (Yahoo notation)."""
    return (t or "").strip().upper().replace(".", "-")


def load_universe(refresh: bool = False) -> pd.DataFrame:
    """S&P 500 constituents: columns ticker, name, sector. Cached in universe.csv."""
    if UNIVERSE_CSV.exists() and not refresh:
        return pd.read_csv(UNIVERSE_CSV)
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
        )
        r.raise_for_status()
        t = pd.read_html(io.StringIO(r.text))[0]
        uni = t[["Symbol", "Security", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
        )
        uni["ticker"] = uni["ticker"].map(normalize_ticker)
    except Exception:  # noqa: BLE001
        uni = pd.DataFrame({"ticker": ["AAPL"], "name": ["Apple Inc."], "sector": ["Information Technology"]})
    uni = uni.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(UNIVERSE_CSV, index=False)
    return uni


def universe_info(ticker: str) -> dict:
    """{'name': ..., 'sector': ...} for a ticker, or empty strings if it is not in the universe."""
    uni = load_universe()
    row = uni[uni["ticker"] == normalize_ticker(ticker)]
    if row.empty:
        return {"name": "", "sector": ""}
    return {"name": str(row.iloc[0]["name"]), "sector": str(row.iloc[0]["sector"])}


# One lock per ticker: several dashboard callbacks call get_stock_daily() for the same ticker
# at the same time; without the lock two threads download and write the same file concurrently
# and corrupt the parquet.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(ticker: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if ticker not in _LOCKS:
            _LOCKS[ticker] = threading.Lock()
        return _LOCKS[ticker]


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Read the cache; delete a corrupt file and return None so the caller re-downloads."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001  (thrift / magic bytes / truncated file ...)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _download(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf  # lazy import: offline deployments never need it

    h = yf.Ticker(ticker).history(start=start, auto_adjust=False, actions=True)
    if h.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    df = h.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"stock_splits": "splits"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["adj_factor"] = df["adj_close"] / df["close"]
    f0 = df["adj_factor"].iloc[0]
    for c in ("open", "high", "low", "close"):
        df[f"{c}_hfq"] = df[c] * df["adj_factor"] / f0
    df["ticker"] = ticker
    return df[COLUMNS].sort_values("date").reset_index(drop=True)


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file then os.replace(), so the real file is always complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def get_stock_daily(ticker: str, start: str = START, max_age_days: int = 3) -> pd.DataFrame:
    """Daily bars (with hfq columns) for one ticker. Serves the cache when fresh enough;
    otherwise downloads (unless OFFLINE). A corrupt cache file is deleted and re-downloaded."""
    ticker = normalize_ticker(ticker)
    path = STOCK_DIR / f"{ticker}.parquet"
    with _lock_for(ticker):
        df = _read_cache(path)
        if df is not None and not df.empty:
            if OFFLINE or (date.today() - df["date"].max().date()).days <= max_age_days:
                return df
        if OFFLINE:
            raise ValueError(f"{ticker} is not in the local cache (offline mode)")
        df = _download(ticker, start)
        _atomic_write(df, path)
        return df


if __name__ == "__main__":
    print(load_universe().head())
    print(universe_info("AAPL"))
    print(get_stock_daily("AAPL").tail(3).to_string())

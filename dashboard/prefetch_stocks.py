# -*- coding: utf-8 -*-
"""
prefetch_stocks.py -- download and cache daily bars for every ticker in the S&P 500 universe.

Run this locally before deploying: the hosted dashboard runs with STOCK_DATA_OFFLINE=1 and
only reads the parquet cache under data/stocks/daily/, because cloud IPs are frequently
rate-limited by Yahoo Finance.

    python prefetch_stocks.py            # download what is missing or stale
    python prefetch_stocks.py --sleep 1  # be more polite
"""
from __future__ import annotations

import argparse
import time

from stock_data import STOCK_DIR, get_stock_daily, load_universe, normalize_ticker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds to wait between downloads")
    ap.add_argument("--max-age-days", type=int, default=3)
    a = ap.parse_args()

    tickers = [normalize_ticker(t) for t in load_universe()["ticker"].tolist()]
    ok, failed = 0, []
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        cached_before = (STOCK_DIR / f"{tk}.parquet").exists()
        try:
            df = get_stock_daily(tk, max_age_days=a.max_age_days)
            ok += 1
            print(f"[{i}/{len(tickers)}] {tk:6s} {len(df):4d} rows  {'cached' if cached_before else 'downloaded'}", flush=True)
            if not cached_before:
                time.sleep(a.sleep)
        except Exception as e:  # noqa: BLE001
            failed.append((tk, str(e)[:80]))
            print(f"[{i}/{len(tickers)}] {tk:6s} FAILED: {e}", flush=True)
            time.sleep(a.sleep * 2)
    print(f"\ndone in {time.time() - t0:.0f}s: {ok} ok, {len(failed)} failed")
    for tk, msg in failed:
        print("  ", tk, msg)


if __name__ == "__main__":
    main()

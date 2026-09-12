# -*- coding: utf-8 -*-
"""
live_collector.py -- poll the Kalshi REST order book for every dashboard contract and save it.

    python live_collector.py                 # every 10 s, depth 100, all tickers in the daily file
    python live_collector.py --interval 5 --tickers CONTROLS-2026-D CONTROLS-2026-R

Output (all under data/live/, which is git-ignored):
    YYYY-MM-DD/HH.jsonl   one line per (poll round, ticker):
                          {"ts": "<UTC ISO, local receive time>", "round": <epoch s>, "ticker": ..., "latency_ms": ...,
                           "orderbook_fp": {"yes_dollars": [[price, qty], ...], "no_dollars": [[price, qty], ...]}}
    latest.json           the most recent book of every ticker, rewritten atomically after each round;
                          the dashboard's Live tab reads this file.

The REST endpoint returns the resting book (verified against the WebSocket feed: identical top-of-book
95.8% of the time with zero lag), capped at 100 levels per side.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalshi_data import DATA_DIR, kalshi_tickers  # noqa: E402

BASE = "https://api.elections.kalshi.com/trade-api/v2"
LIVE_DIR = DATA_DIR / "live"
LATEST = LIVE_DIR / "latest.json"


def fetch_book(session: requests.Session, ticker: str, depth: int) -> tuple[dict, float]:
    t0 = time.time()
    for attempt in range(4):
        r = session.get(f"{BASE}/markets/{ticker}/orderbook", params={"depth": depth}, timeout=20)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        r.raise_for_status()
        return r.json()["orderbook_fp"], (time.time() - t0) * 1000
    raise RuntimeError(f"429 x4 for {ticker}")


def atomic_write_json(path: Path, obj: dict) -> None:
    """Write via a temp file and os.replace(). On Windows the replace fails with PermissionError while
    another process/thread has the target open for reading (the dashboard reads latest.json every 10 s),
    so retry briefly and finally fall back to an in-place write rather than let the collector die."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    try:
        path.write_text(json.dumps(obj), encoding="utf-8")     # non-atomic fallback; readers retry torn reads
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def run(interval: float = 10.0, depth: int = 100, tickers: list[str] | None = None, stop_event=None) -> None:
    """Poll forever (or until stop_event is set). Safe to call from a background thread of app.py."""
    tickers = tickers or kalshi_tickers()
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "User-Agent": "kalshi-capstone-live/1.0"})
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"live collector: {len(tickers)} tickers, every {interval}s, depth {depth} -> {LIVE_DIR}", flush=True)

    for stale in LIVE_DIR.glob("latest.*.tmp"):            # leftovers from an interrupted replace
        try:
            stale.unlink()
        except OSError:
            pass
    n_round = 0
    while stop_event is None or not stop_event.is_set():
        t_round = time.time()
        now = dt.datetime.now(dt.timezone.utc)
        try:
            hour_file = LIVE_DIR / now.strftime("%Y-%m-%d") / (now.strftime("%H") + ".jsonl")
            hour_file.parent.mkdir(parents=True, exist_ok=True)
            latest = {"round": int(t_round), "round_ts": now.isoformat(timespec="milliseconds"), "books": {}}
            with open(hour_file, "a", encoding="utf-8") as f:
                for tk in tickers:
                    try:
                        ob, lat = fetch_book(s, tk, depth)
                        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
                               "round": int(t_round), "ticker": tk, "latency_ms": round(lat, 1), "orderbook_fp": ob}
                        f.write(json.dumps(rec) + "\n")
                        latest["books"][tk] = rec
                    except Exception as e:  # noqa: BLE001
                        print(f"{now:%H:%M:%S} {tk} ERROR {e}", flush=True)
            if latest["books"]:
                atomic_write_json(LATEST, latest)
            n_round += 1
            if n_round % 30 == 1:
                print(f"{now:%Y-%m-%d %H:%M:%S}Z round {n_round}: {len(latest['books'])}/{len(tickers)} books, "
                      f"{(time.time() - t_round) * 1000:.0f} ms", flush=True)
        except Exception as e:  # noqa: BLE001  -- never let one bad round kill the collector
            print(f"{now:%Y-%m-%d %H:%M:%S}Z round ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(max(0.0, interval - (time.time() - t_round)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between rounds")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--tickers", nargs="*", default=None)
    a = ap.parse_args()
    run(interval=a.interval, depth=a.depth, tickers=a.tickers)


if __name__ == "__main__":
    main()

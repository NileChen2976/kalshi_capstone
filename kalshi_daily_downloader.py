"""
kalshi_daily_downloader.py

Daily Kalshi candlesticks with correct bar-date attribution.

Kalshi `end_period_ts` is treated as the END of the candle.
A daily bar ending 2026-01-16 00:00 ET belongs to trade_date_et=2026-01-15.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kalshi_utils import (
    BASE_URL,
    ET_TZ,
    build_session,
    request_with_retry,
    resolve_download_dates,
)


DEFAULT_OUTPUT = Path("data/daily/kalshi_daily.parquet")

NUMERIC_COLS = [
    "price_open", "price_high", "price_low", "price_close",
    "price_mean", "price_previous",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "volume", "open_interest",
]


def flatten_daily(candles: list[dict], ticker: str) -> pd.DataFrame:
    rows = []

    for x in candles:
        p = x.get("price") or {}
        b = x.get("yes_bid") or {}
        a = x.get("yes_ask") or {}

        rows.append(
            {
                "ticker": ticker,
                "end_period_ts": x.get("end_period_ts"),
                "price_open": p.get("open_dollars"),
                "price_high": p.get("high_dollars"),
                "price_low": p.get("low_dollars"),
                "price_close": p.get("close_dollars"),
                "price_mean": p.get("mean_dollars"),
                "price_previous": p.get("previous_dollars"),
                "yes_bid_open": b.get("open_dollars"),
                "yes_bid_high": b.get("high_dollars"),
                "yes_bid_low": b.get("low_dollars"),
                "yes_bid_close": b.get("close_dollars"),
                "yes_ask_open": a.get("open_dollars"),
                "yes_ask_high": a.get("high_dollars"),
                "yes_ask_low": a.get("low_dollars"),
                "yes_ask_close": a.get("close_dollars"),
                "volume": x.get("volume_fp"),
                "open_interest": x.get("open_interest_fp"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df["period_end_utc"] = pd.to_datetime(
        df["end_period_ts"],
        unit="s",
        utc=True,
    )
    df["period_end_et"] = (
        df["period_end_utc"]
        .dt.tz_convert(ET_TZ)
    )

    # Calendar-day attribution.
    df["period_start_et"] = (
        df["period_end_et"] - pd.DateOffset(days=1)
    )
    df["period_start_utc"] = (
        df["period_start_et"]
        .dt.tz_convert("UTC")
    )
    df["trade_date_et"] = (
        df["period_start_et"].dt.date
    )

    df["mid_close"] = (
        df["yes_bid_close"] + df["yes_ask_close"]
    ) / 2
    df["spread_close"] = (
        df["yes_ask_close"] - df["yes_bid_close"]
    )

    return (
        df.drop_duplicates(
            subset=["ticker", "end_period_ts"],
            keep="last",
        )
        .sort_values(["period_start_utc", "ticker"])
        .reset_index(drop=True)
    )


def download_daily(
    ticker: str,
    series_ticker: str,
    start_date: str | None,
    end_date: str | None,
    output_path: Path,
) -> pd.DataFrame:
    session = build_session()

    start_et, end_et = resolve_download_dates(
        ticker,
        start_date,
        end_date,
        session,
    )

    # Request through next midnight because end_period_ts is bar END.
    request_end_et = (
        end_et + pd.DateOffset(days=1)
    )

    url = (
        f"{BASE_URL}/series/{series_ticker}"
        f"/markets/{ticker}/candlesticks"
    )
    params = {
        "start_ts": int(
            start_et.tz_convert("UTC").timestamp()
        ),
        "end_ts": int(
            request_end_et.tz_convert("UTC").timestamp()
        ),
        "period_interval": 1440,
    }

    r = request_with_retry(
        session,
        url,
        params=params,
    )

    df = flatten_daily(
        r.json().get("candlesticks", []),
        ticker,
    )

    if df.empty:
        print("No daily candles returned.")
        return df

    df = df[
        (df["trade_date_et"] >= start_et.date())
        & (df["trade_date_et"] <= end_et.date())
    ].copy()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        old = pd.read_parquet(output_path)
        if "ticker" in old.columns:
            old = old[
                old["ticker"] != ticker
            ].copy()
        combined = pd.concat(
            [old, df],
            ignore_index=True,
        )
    else:
        combined = df.copy()

    combined = (
        combined.drop_duplicates(
            subset=["ticker", "end_period_ts"],
            keep="last",
        )
        .sort_values(["period_start_utc", "ticker"])
        .reset_index(drop=True)
    )

    combined.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    print(
        f"{ticker}: saved {len(df)} daily rows "
        f"({df['trade_date_et'].min()} -> "
        f"{df['trade_date_et'].max()})"
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    a = p.parse_args()

    download_daily(
        ticker=a.ticker,
        series_ticker=a.series,
        start_date=a.start_date,
        end_date=a.end_date,
        output_path=Path(a.output),
    )


if __name__ == "__main__":
    main()

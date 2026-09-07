"""
kalshi_1min_downloader.py

1-minute Kalshi candlesticks partitioned by ET trade date.

Important:
`end_period_ts` is the END of the candle.
A candle ending 2026-01-16 00:00 ET represents
2026-01-15 23:59 -> 2026-01-16 00:00 and is saved in 2026-01-15.parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from kalshi_utils import (
    BASE_URL,
    ET_TZ,
    build_session,
    et_day_bounds,
    load_manifest,
    manifest_is_complete,
    request_with_retry,
    resolve_download_dates,
    save_date_partition,
    upsert_manifest_row,
)


DEFAULT_DATA_DIR = Path("data/1min")
DEFAULT_MANIFEST = Path("data/1min_manifest.parquet")

MANIFEST_COLS = [
    "ticker", "date", "status",
    "n_candles", "message",
    "updated_at_utc",
]

NUMERIC_COLS = [
    "price_open", "price_high", "price_low", "price_close",
    "price_mean", "price_previous",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "volume", "open_interest",
]


def flatten_1min(
    candles: list[dict],
    ticker: str,
) -> pd.DataFrame:
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
    df["period_start_et"] = (
        df["period_end_et"]
        - pd.Timedelta(minutes=1)
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


def download_one_day(
    ticker: str,
    series_ticker: str,
    date_str: str,
    session,
) -> pd.DataFrame:
    start_et, end_et = et_day_bounds(
        date_str
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
            end_et.tz_convert("UTC").timestamp()
        ),
        "period_interval": 1,
    }

    r = request_with_retry(
        session,
        url,
        params=params,
    )

    df = flatten_1min(
        r.json().get("candlesticks", []),
        ticker,
    )

    if df.empty:
        return df

    # Correct date assignment is by candle START.
    return (
        df[df["trade_date_et"] == start_et.date()]
        .copy()
        .reset_index(drop=True)
    )


def download_range(
    ticker: str,
    series_ticker: str,
    start_date: str | None,
    end_date: str | None,
    data_dir: Path,
    manifest_path: Path,
    sleep_seconds: float,
    force: bool,
) -> None:
    session = build_session()

    start_et, end_et = resolve_download_dates(
        ticker,
        start_date,
        end_date,
        session,
    )

    dates = pd.date_range(
        start_et,
        end_et,
        freq="D",
    )

    manifest = load_manifest(
        manifest_path,
        MANIFEST_COLS,
    )

    for i, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")

        if not force and manifest_is_complete(
            manifest,
            ticker,
            date_str,
            ("success", "zero_candles"),
        ):
            print(
                f"[{i}/{len(dates)}] "
                f"{date_str}: complete, skip"
            )
            continue

        try:
            df = download_one_day(
                ticker,
                series_ticker,
                date_str,
                session,
            )

            if df.empty:
                status = "zero_candles"
                n = 0
                print(
                    f"[{i}/{len(dates)}] "
                    f"{date_str}: 0 candles"
                )
            else:
                save_date_partition(
                    df=df,
                    date_str=date_str,
                    ticker=ticker,
                    data_dir=data_dir,
                    dedupe_cols=[
                        "ticker",
                        "end_period_ts",
                    ],
                    sort_cols=[
                        "period_start_utc",
                        "ticker",
                    ],
                )
                status = "success"
                n = len(df)
                print(
                    f"[{i}/{len(dates)}] "
                    f"{date_str}: {n} candles | "
                    f"{df['period_start_et'].min()} -> "
                    f"{df['period_end_et'].max()}"
                )

            upsert_manifest_row(
                manifest_path,
                {
                    "ticker": ticker,
                    "date": date_str,
                    "status": status,
                    "n_candles": n,
                    "message": "",
                    "updated_at_utc": pd.Timestamp.now(
                        tz="UTC"
                    ),
                },
            )
            manifest = load_manifest(
                manifest_path,
                MANIFEST_COLS,
            )
            time.sleep(sleep_seconds)

        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"[{i}/{len(dates)}] "
                f"{date_str}: ERROR -> {message}"
            )

            upsert_manifest_row(
                manifest_path,
                {
                    "ticker": ticker,
                    "date": date_str,
                    "status": "error",
                    "n_candles": 0,
                    "message": message,
                    "updated_at_utc": pd.Timestamp.now(
                        tz="UTC"
                    ),
                },
            )
            manifest = load_manifest(
                manifest_path,
                MANIFEST_COLS,
            )
            time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
    )
    p.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--force",
        action="store_true",
    )
    a = p.parse_args()

    download_range(
        ticker=a.ticker,
        series_ticker=a.series,
        start_date=a.start_date,
        end_date=a.end_date,
        data_dir=Path(a.data_dir),
        manifest_path=Path(a.manifest),
        sleep_seconds=a.sleep,
        force=a.force,
    )


if __name__ == "__main__":
    main()

"""
kalshi_trade_downloader.py

Public Kalshi trade downloader.

Features:
- Historical + recent trade endpoints
- Uses /historical/cutoff -> trades_created_ts
- Cursor pagination
- ET calendar-day partitioning
- One parquet per ET date
- Ticker stored inside parquet
- 429 retry / exponential backoff
- Manifest with n_trades and n_pages
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
    get_historical_cutoff,
    load_manifest,
    manifest_is_complete,
    request_with_retry,
    resolve_download_dates,
    save_date_partition,
    upsert_manifest_row,
)


DEFAULT_DATA_DIR = Path("data/trades")
DEFAULT_MANIFEST = Path(
    "data/trades_manifest.parquet"
)

MANIFEST_COLS = [
    "ticker", "date", "status",
    "n_trades", "n_pages",
    "source", "message",
    "updated_at_utc",
]


def fetch_trade_pages(
    session,
    url: str,
    ticker: str,
    min_ts: int,
    max_ts: int,
    page_sleep: float = 0.25,
) -> tuple[list[dict], int]:
    trades = []
    cursor = None
    n_pages = 0

    while True:
        params = {
            "ticker": ticker,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "limit": 1000,
        }

        if cursor:
            params["cursor"] = cursor

        r = request_with_retry(
            session,
            url,
            params=params,
        )
        payload = r.json()

        trades.extend(
            payload.get("trades", [])
        )
        n_pages += 1

        cursor = payload.get("cursor")
        if not cursor:
            break

        time.sleep(page_sleep)

    return trades, n_pages


def fetch_one_day_raw(
    ticker: str,
    date_str: str,
    session,
    cutoff_utc: pd.Timestamp,
) -> tuple[list[dict], int, str]:
    start_et, end_et = et_day_bounds(
        date_str
    )

    start_utc = start_et.tz_convert("UTC")
    end_utc = end_et.tz_convert("UTC")

    historical_url = (
        f"{BASE_URL}/historical/trades"
    )
    recent_url = (
        f"{BASE_URL}/markets/trades"
    )

    all_trades = []
    total_pages = 0
    sources = []

    # Entire ET day is before cutoff.
    if end_utc <= cutoff_utc:
        x, pages = fetch_trade_pages(
            session,
            historical_url,
            ticker,
            int(start_utc.timestamp()),
            int(end_utc.timestamp()),
        )
        all_trades.extend(x)
        total_pages += pages
        sources.append("historical")

    # Entire ET day is at/after cutoff.
    elif start_utc >= cutoff_utc:
        x, pages = fetch_trade_pages(
            session,
            recent_url,
            ticker,
            int(start_utc.timestamp()),
            int(end_utc.timestamp()),
        )
        all_trades.extend(x)
        total_pages += pages
        sources.append("recent")

    # Cutoff lies inside this ET day.
    else:
        x, pages = fetch_trade_pages(
            session,
            historical_url,
            ticker,
            int(start_utc.timestamp()),
            int(cutoff_utc.timestamp()),
        )
        all_trades.extend(x)
        total_pages += pages
        sources.append("historical")

        x, pages = fetch_trade_pages(
            session,
            recent_url,
            ticker,
            int(cutoff_utc.timestamp()),
            int(end_utc.timestamp()),
        )
        all_trades.extend(x)
        total_pages += pages
        sources.append("recent")

    return (
        all_trades,
        total_pages,
        "+".join(sources),
    )


def flatten_trades(
    trades: list[dict],
    ticker: str,
    date_str: str,
) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    rename = {
        "count_fp": "count",
        "yes_price_dollars": "yes_price",
        "no_price_dollars": "no_price",
    }
    df = df.rename(columns=rename)

    # Ensure expected columns exist even if schema evolves.
    expected = [
        "ticker",
        "trade_id",
        "created_time",
        "count",
        "yes_price",
        "no_price",
        "taker_side",
        "taker_outcome_side",
        "taker_book_side",
        "is_block_trade",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    df["ticker"] = df["ticker"].fillna(
        ticker
    )

    for col in [
        "count",
        "yes_price",
        "no_price",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["created_time"],
        utc=True,
    )
    df["timestamp_et"] = (
        df["timestamp_utc"]
        .dt.tz_convert(ET_TZ)
    )
    df["trade_date_et"] = (
        df["timestamp_et"].dt.date
    )

    target_date = pd.Timestamp(
        date_str,
        tz=ET_TZ,
    ).date()

    df = df[
        df["trade_date_et"] == target_date
    ].copy()

    if df.empty:
        return df

    # Useful research fields.
    df["yes_notional"] = (
        df["count"] * df["yes_price"]
    )
    df["no_notional"] = (
        df["count"] * df["no_price"]
    )

    return (
        df.drop_duplicates(
            subset=["trade_id"],
            keep="last",
        )
        .sort_values(
            ["timestamp_utc", "trade_id"]
        )
        .reset_index(drop=True)
    )


def download_range(
    ticker: str,
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

    cutoff_payload = get_historical_cutoff(
        session
    )
    cutoff_utc = pd.Timestamp(
        cutoff_payload["trades_created_ts"]
    )

    print(
        f"Trade cutoff UTC: {cutoff_utc}"
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
            ("success", "zero_trades"),
        ):
            print(
                f"[{i}/{len(dates)}] "
                f"{date_str}: complete, skip"
            )
            continue

        try:
            raw, n_pages, source = (
                fetch_one_day_raw(
                    ticker=ticker,
                    date_str=date_str,
                    session=session,
                    cutoff_utc=cutoff_utc,
                )
            )

            df = flatten_trades(
                raw,
                ticker,
                date_str,
            )

            if df.empty:
                status = "zero_trades"
                n_trades = 0
                print(
                    f"[{i}/{len(dates)}] "
                    f"{date_str}: 0 trades | "
                    f"pages={n_pages} | "
                    f"source={source}"
                )
            else:
                save_date_partition(
                    df=df,
                    date_str=date_str,
                    ticker=ticker,
                    data_dir=data_dir,
                    dedupe_cols=["trade_id"],
                    sort_cols=[
                        "timestamp_utc",
                        "ticker",
                    ],
                )
                status = "success"
                n_trades = len(df)

                print(
                    f"[{i}/{len(dates)}] "
                    f"{date_str}: {n_trades} trades | "
                    f"pages={n_pages} | "
                    f"source={source} | "
                    f"{df['timestamp_et'].min()} -> "
                    f"{df['timestamp_et'].max()}"
                )

            upsert_manifest_row(
                manifest_path,
                {
                    "ticker": ticker,
                    "date": date_str,
                    "status": status,
                    "n_trades": n_trades,
                    "n_pages": n_pages,
                    "source": source,
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
                    "n_trades": 0,
                    "n_pages": 0,
                    "source": "",
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
        start_date=a.start_date,
        end_date=a.end_date,
        data_dir=Path(a.data_dir),
        manifest_path=Path(a.manifest),
        sleep_seconds=a.sleep,
        force=a.force,
    )


if __name__ == "__main__":
    main()

"""
kalshi_utils.py
Shared utilities for Kalshi downloaders.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
ET_TZ = "America/New_York"


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "kalshi-research-pipeline/1.0",
        }
    )
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    max_retries: int = 8,
    timeout: int = 60,
) -> requests.Response:
    """
    GET with exponential backoff for HTTP 429.
    Honors Retry-After when present.
    """
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=timeout)

        if r.status_code == 200:
            return r

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                wait = float(retry_after)
            else:
                wait = min(
                    60.0,
                    (2 ** attempt) + random.uniform(0, 1),
                )

            print(
                f"429 rate limit -> retry "
                f"{attempt + 1}/{max_retries} in {wait:.1f}s"
            )
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError(f"Max retries exceeded: {url}")


def get_market_metadata(
    ticker: str,
    session: Optional[requests.Session] = None,
) -> dict:
    session = session or build_session()
    r = request_with_retry(
        session,
        f"{BASE_URL}/markets/{ticker}",
    )
    return r.json()["market"]


def get_historical_cutoff(
    session: Optional[requests.Session] = None,
) -> dict:
    session = session or build_session()
    r = request_with_retry(
        session,
        f"{BASE_URL}/historical/cutoff",
    )
    return r.json()


def et_day_bounds(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return [start_et, end_et] for an ET calendar day.
    DST-safe because America/New_York is timezone-aware.
    """
    start_et = pd.Timestamp(date_str, tz=ET_TZ)
    end_et = start_et + pd.DateOffset(days=1)
    return start_et, end_et


def resolve_download_dates(
    ticker: str,
    start_date: Optional[str],
    end_date: Optional[str],
    session: requests.Session,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    market = get_market_metadata(ticker, session)

    market_open_et = (
        pd.Timestamp(market["open_time"])
        .tz_convert(ET_TZ)
        .normalize()
    )

    if start_date:
        requested_start = pd.Timestamp(
            start_date,
            tz=ET_TZ,
        ).normalize()
        start_et = max(requested_start, market_open_et)
    else:
        start_et = market_open_et

    today_et = pd.Timestamp.now(
        tz=ET_TZ
    ).normalize()

    if end_date:
        requested_end = pd.Timestamp(
            end_date,
            tz=ET_TZ,
        ).normalize()
        end_et = min(requested_end, today_et)
    else:
        end_et = today_et

    if end_et < start_et:
        raise ValueError(
            f"end_date {end_et.date()} is before "
            f"start_date {start_et.date()}"
        )

    return start_et, end_et


def load_manifest(
    manifest_path: Path,
    columns: list[str],
) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_parquet(manifest_path)

    return pd.DataFrame(columns=columns)


def upsert_manifest_row(
    manifest_path: Path,
    row: dict,
    key_cols: tuple[str, ...] = ("ticker", "date"),
) -> None:
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = list(row.keys())
    manifest = load_manifest(
        manifest_path,
        columns=columns,
    )

    new_row = pd.DataFrame([row])

    manifest = pd.concat(
        [manifest, new_row],
        ignore_index=True,
    )

    manifest = (
        manifest.drop_duplicates(
            subset=list(key_cols),
            keep="last",
        )
        .sort_values(list(key_cols))
        .reset_index(drop=True)
    )

    manifest.to_parquet(
        manifest_path,
        index=False,
        engine="pyarrow",
    )


def manifest_is_complete(
    manifest: pd.DataFrame,
    ticker: str,
    date_str: str,
    complete_statuses: tuple[str, ...],
) -> bool:
    if manifest.empty:
        return False

    mask = (
        (manifest["ticker"] == ticker)
        & (manifest["date"] == date_str)
        & manifest["status"].isin(complete_statuses)
    )
    return bool(mask.any())


def save_date_partition(
    df: pd.DataFrame,
    date_str: str,
    ticker: str,
    data_dir: Path,
    dedupe_cols: list[str],
    sort_cols: list[str],
) -> int:
    """
    Save one ET-date parquet while preserving other tickers already
    stored in the same file. Existing rows for `ticker` are replaced.
    """
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = data_dir / f"{date_str}.parquet"

    if path.exists():
        existing = pd.read_parquet(path)
        if "ticker" in existing.columns:
            existing = existing[
                existing["ticker"] != ticker
            ].copy()
        combined = pd.concat(
            [existing, df],
            ignore_index=True,
        )
    else:
        combined = df.copy()

    combined = (
        combined.drop_duplicates(
            subset=dedupe_cols,
            keep="last",
        )
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    combined.to_parquet(
        path,
        index=False,
        engine="pyarrow",
    )

    return len(combined)


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

UNIVERSE = [
    {"event": "Senate Control", "ticker": "CONTROLS-2026-D", "series": "CONTROLS"},
    {"event": "Senate Control", "ticker": "CONTROLS-2026-R", "series": "CONTROLS"},
    {"event": "Texas Senate", "ticker": "SENATETX-26-D", "series": "SENATETX"},
    {"event": "Texas Senate", "ticker": "SENATETX-26-R", "series": "SENATETX"},
    {"event": "Ohio Senate", "ticker": "SENATEOHS-26-D", "series": "SENATEOHS"},
    {"event": "Ohio Senate", "ticker": "SENATEOHS-26-R", "series": "SENATEOHS"},
]

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "kalshi_daily_downloader.py"
MIN1 = ROOT / "kalshi_1min_downloader.py"
TRADES = ROOT / "kalshi_trade_downloader.py"


def run(cmd):
    print("\nRUN:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def date_args(start_date, end_date):
    out = []
    if start_date:
        out += ["--start-date", start_date]
    if end_date:
        out += ["--end-date", end_date]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["daily", "1min", "trades"],
        default=["daily", "1min", "trades"],
    )
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    for f in [DAILY, MIN1, TRADES]:
        if not f.exists():
            raise FileNotFoundError(
                f"{f.name} not found. Keep batch entry in the same folder "
                "as the three downloader scripts."
            )

    exclude = set(args.exclude)
    only = set(args.only)

    universe = [
        x for x in UNIVERSE
        if x["ticker"] not in exclude
        and (not only or x["ticker"] in only)
    ]

    print("Contracts:")
    for x in universe:
        print(f"  {x['ticker']:<20} {x['event']}")

    failures = []

    for i, x in enumerate(universe, 1):
        ticker = x["ticker"]
        series = x["series"]
        print(f"\n=== {i}/{len(universe)} {ticker} ===")

        for dataset in args.datasets:
            try:
                common_dates = date_args(args.start_date, args.end_date)

                if dataset == "daily":
                    cmd = [
                        sys.executable, str(DAILY),
                        "--ticker", ticker,
                        "--series", series,
                    ] + common_dates

                elif dataset == "1min":
                    cmd = [
                        sys.executable, str(MIN1),
                        "--ticker", ticker,
                        "--series", series,
                        "--sleep", str(args.sleep),
                    ] + common_dates
                    if args.force:
                        cmd.append("--force")

                else:
                    cmd = [
                        sys.executable, str(TRADES),
                        "--ticker", ticker,
                        "--sleep", str(args.sleep),
                    ] + common_dates
                    if args.force:
                        cmd.append("--force")

                run(cmd)

            except Exception as e:
                failures.append((ticker, dataset, str(e)))
                print(f"FAILED {ticker} / {dataset}: {e}")
                print("Continuing...")

    print("\nBatch finished.")

    if failures:
        print("Failures:")
        for ticker, dataset, err in failures:
            print(f"  {ticker} / {dataset}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

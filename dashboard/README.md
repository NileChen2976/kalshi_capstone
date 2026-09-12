# Kalshi x S&P 500 dashboard

Plotly Dash app with three tabs. Kalshi political contracts (2026 Senate control, Texas and Ohio
Senate races), each event as a -D / -R pair, next to S&P 500 stocks.

## Run locally

```
pip install -r ../requirements.txt          # Python 3.10+
cd dashboard
python app.py                               # -> http://127.0.0.1:8050
```

Order-book capture (every 10 s, depth 100, into `data/live/`) should run as its own process:
`run_collector.cmd` (logs to `data/live/collector.log`), or the Windows scheduled task
**KalshiLiveCollector** (runs `run_collector_hidden.vbs` at logon, restarts on failure). `python app.py`
checks whether `data/live/latest.json` is being refreshed; only when nothing else is capturing does it
start a collector thread of its own (`LIVE_COLLECTOR=1` forces it, `0` disables it). Never run two
collectors at once: they would append to the same files.

## Tabs

**Event pair (D vs R)** — `tab_pair.py`
The two contracts of one event on one P(D) axis: the R contract is shown as 1 − price (its bid
becomes an ask and vice versa). Daily panel: mids and the event P(D) = mean(D mid, 1 − R mid);
cross-book checks (D mid − R-implied mid, ask_D + ask_R − 1, bid_D + bid_R − 1: a negative ask sum
or a positive bid sum would be a risk-free arbitrage); daily volume and trade counts per contract.
Intraday panel for one ET day: 1-min quotes of both contracts, merged trades with direction
(pro-D = buys D-yes or R-no, pro-R = buys R-yes or D-no) and contract symbol, cumulative net flow,
and a trade table that follows the visible range.

**Stocks × Kalshi** — `tab_stock.py`
One stock vs several Kalshi series (contracts and/or event P(D)); stock by ticker or paged through
the S&P 500, optionally inside one GICS sector. Closeness table: correlations of daily changes
aligned at 16:00 ET (same day, Kalshi leads, stock leads, best lag in ±3 days) and the event-day hit
rate / lift. Event-window statistics for one selected series (thresholds and window adjustable).
Intraday view of one contract with its trades.

**Live books** — `tab_live.py`
Every 10 s reads `data/live/latest.json` written by `live_collector.py` and draws, per event, the
mirrored ladder of both contracts in D terms (10 best levels per side; 100 per side are saved),
a metrics table (best bid/ask, spread, top sizes, cross-book sums, age, latency) and the cross-book
sums over the last 60 minutes. The collector polls the public REST order book (`depth=100`) for all
contracts in the daily file and appends hourly JSONL files under `data/live/YYYY-MM-DD/`
(git-ignored). The REST book was checked against the WebSocket feed: identical top-of-book 95.8%
of the time with zero lag, so 10-s polling is a faithful sample of the resting book.

**Replay** — `tab_replay.py`
Scrub through a captured day: pick the capture day (UTC folder of the collector) and an event,
drag the slider, step one round at a time, or press play (1 round, 1 min, 5 min or 30 min per tick).
The ladder and metrics show the book at that round; the day overview (mids and cross-book sums)
marks the current time and jumps on click. Replay reads `data/replay/<day>.parquet`, produced from
the raw captures by `python replay_data.py` (all days without a replay file) or
`python replay_data.py 2026-09-08 --every 6` (one-minute resolution, small enough to commit for the
hosted copy). Ten levels per side are kept; one day at 10-second resolution is under 1 MB.

## Modules

| File | Purpose |
|---|---|
| `app.py` | Dash app shell: tabs, callback registration, `server` for gunicorn |
| `kalshi_data.py` | Readers for `data/daily`, `data/1min`, `data/trades` and manifests, filtered by ticker |
| `pair_data.py` | Event pairs, D-terms mapping, merged trades, daily pair frame, event P(D) |
| `stock_data.py` | Stock daily bars: yfinance download, parquet cache, backward adjustment, S&P 500 universe with sector; offline mode |
| `event_study.py` | Event-window statistics for any series key (`TICKER` or `EVENT:<event>`); `python event_study.py AAPL EVENT:CONTROLS-2026` |
| `closeness.py` | Correlation / lead-lag / event-day table for several series vs one stock |
| `live_collector.py`, `live_data.py` | REST order-book poller and readers (ladders in D terms, top-of-book history) |
| `replay_data.py` | Exports raw captures to `data/replay/<day>.parquet` (10 levels per side) and serves them to the Replay tab |
| `prefetch_stocks.py` | Cache daily bars for all 503 constituents (run before deploying) |

## Data

- `data/daily/kalshi_daily.parquet`, `data/1min/*.parquet`, `data/trades/*.parquet`: from the
  downloaders in the repo root (`kalshi_batch_download.py`); day files hold several Kalshi tickers.
- `data/stocks/universe.csv`: current S&P 500 constituents (ticker, name, GICS sector).
- `data/stocks/daily/{TICKER}.parquet`: cached stock bars. Backward adjustment keeps the first day's
  price as traded and scales later prices by `AdjClose/Close` relative to the first day.
- `data/live/`: live order-book captures, local only.

## Hosting

`render.yaml` in the repo root deploys the app on Render (free plan) with `STOCK_DATA_OFFLINE=1`
(the hosted app only reads the committed parquet cache). The Live tab shows "collector not running"
there; live capture is local for now, replay of saved captures is planned.

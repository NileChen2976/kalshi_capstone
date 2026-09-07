# Kalshi x S&P 500 dashboard

Interactive Plotly Dash app that puts Kalshi political contracts (2026 Senate control and
state races) next to S&P 500 stocks.

## Run locally

```
cd dashboard
python app.py            # -> http://127.0.0.1:8050
```

Requirements: `pip install -r ../requirements.txt` (Python 3.10+).

## Files

| File | Purpose |
|---|---|
| `app.py` | Dash app: panel A (daily comparison), event-window statistics, panel B (Kalshi intraday), trade table |
| `kalshi_data.py` | Readers for the local Kalshi parquet data (`data/daily`, `data/1min`, `data/trades`, manifests), filtered by ticker |
| `stock_data.py` | Stock daily data layer: yfinance download, parquet cache, backward adjustment, S&P 500 universe with GICS sector |
| `event_study.py` | Event-window statistics (also runnable standalone: `python event_study.py AAPL CONTROLS-2026-R`) |
| `prefetch_stocks.py` | Downloads and caches daily bars for every S&P 500 ticker (run before deploying) |

## Data

- `data/daily/kalshi_daily.parquet`, `data/1min/*.parquet`, `data/trades/*.parquet`: produced by the
  downloaders in the repo root (`kalshi_batch_download.py`). Day files hold several Kalshi tickers.
- `data/stocks/universe.csv`: current S&P 500 constituents (ticker, name, GICS sector) from Wikipedia;
  paging order is this list sorted by ticker.
- `data/stocks/daily/{TICKER}.parquet`: one cached file per stock.
  Backward adjustment: the first day's price is kept as traded and later prices are scaled by
  `AdjClose/Close` relative to the first day (Yahoo's own Adj Close is forward-adjusted).

## Interactions

- Panel A: choose the Kalshi contract; type a ticker + Enter, or page Prev/Next through the S&P 500
  (optionally inside one GICS sector); the company name and sector are shown under the controls.
  Drag the range slider to pick a period; click a day to send panel B there.
- Event windows: thresholds (Kalshi ±k% as relative return or probability points, stock |r| ≥ s%,
  window ±K trading days) recompute for the current pair. The Kalshi return is the change of the
  mid between consecutive trading days at 16:00 ET (as-of from 1-min + daily data), aligned with
  the stock's close-to-close return. Hit rate = share of events whose stock |r| exceeds the threshold
  at that offset, compared with the unconditional baseline (lift). Click an event row to zoom
  panel A to ±45 days around it and move panel B to that day.
- Panel B: wheel to zoom, drag to pan; prev/next day, prev/next day with trades. The trade table
  follows the visible range; clicking a row centres the chart on that trade (±10 minutes).

## Hosting

`render.yaml` in the repo root deploys the app on Render (free plan) with
`STOCK_DATA_OFFLINE=1`, so the hosted app only reads the committed parquet cache and never
calls Yahoo. To publish new data: run the downloaders / `prefetch_stocks.py` locally, commit the
parquet files, push; Render redeploys automatically.

## Known limitations

- Kalshi 1-minute candles exist only for minutes with activity, so intraday quotes are drawn as
  step lines, not candlesticks.
- Order-book panel (depth ladder / liquidity heat map) is not built yet; it needs continuous
  WebSocket collection first.

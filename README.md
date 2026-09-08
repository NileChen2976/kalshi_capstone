# kalshi_capstone

Kalshi political contracts (2026 Senate control, Texas and Ohio Senate races) next to S&P 500 stocks.

| Path | What it is |
|---|---|
| `dashboard/` | Plotly Dash app: event pair (synthetic book), stocks × Kalshi, live order books, replay. See `dashboard/README.md`. |
| `kalshi_daily_downloader.py`, `kalshi_1min_downloader.py`, `kalshi_trade_downloader.py` | Kalshi public-API downloaders (daily candles, 1-minute candles, trades), partitioned by ET day with manifests. |
| `kalshi_batch_download.py` | Runs the three downloaders for every contract in its `UNIVERSE` list. |
| `kalshi_utils.py` | Shared HTTP session, retry on 429, ET date helpers, manifest helpers. |
| `data/` | Committed data snapshot: Kalshi parquet files and the cached S&P 500 daily bars used by the hosted dashboard. |
| `render.yaml`, `requirements.txt` | Render deployment (free plan, offline stock data). |

Run the dashboard locally:

```
pip install -r requirements.txt
cd dashboard && python app.py      # http://127.0.0.1:8050
```

Update the data:

```
python kalshi_batch_download.py            # Kalshi daily / 1min / trades for all contracts
cd dashboard && python prefetch_stocks.py  # refresh the S&P 500 daily cache
git add data && git commit -m "data: refresh" && git push   # Render redeploys automatically
```

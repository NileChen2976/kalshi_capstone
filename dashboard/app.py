# -*- coding: utf-8 -*-
"""
app.py -- Kalshi contracts x S&P 500 dashboard (Plotly Dash), three tabs:

  Event pair (D vs R)   the two binary contracts of one event in D terms, cross-book checks,
                        merged trade flow                                          -> tab_pair.py
  Stocks x Kalshi       one stock vs several Kalshi series, closeness table, event-window
                        statistics, intraday view of one contract                  -> tab_stock.py
  Live books            order books of every contract refreshed every 10 s from the local
                        collector (run  python live_collector.py  separately)      -> tab_live.py

Run locally:   python app.py  ->  http://127.0.0.1:8050
Hosted:        gunicorn --chdir dashboard app:server   (see ../render.yaml; the Live tab needs the collector)
"""
from __future__ import annotations

import os

from dash import Dash, dcc, html

import tab_live
import tab_pair
import tab_replay
import tab_stock

app = Dash(__name__)
app.title = "Kalshi x S&P 500"
server = app.server            # for gunicorn

app.layout = html.Div(style={"fontFamily": "Segoe UI, Arial", "padding": "8px 16px"}, children=[
    html.H3("Kalshi contracts  ×  S&P 500 stocks", style={"margin": "4px 0 8px"}),
    dcc.Tabs(id="tabs", value="pair", children=[
        dcc.Tab(label="Event pair (D vs R)", value="pair", children=tab_pair.layout()),
        dcc.Tab(label="Stocks × Kalshi", value="stock", children=tab_stock.layout()),
        dcc.Tab(label="Live books", value="live", children=tab_live.layout()),
        dcc.Tab(label="Replay", value="replay", children=tab_replay.layout()),
    ]),
])

for module in (tab_pair, tab_stock, tab_live, tab_replay):
    module.register(app)

if __name__ == "__main__":
    # Local run: LIVE_COLLECTOR=auto (default) starts the order-book collector in a background thread
    # unless another collector is already writing data/live/latest.json (e.g. the scheduled task
    # KalshiLiveCollector, see run_collector.cmd); "1" forces it on, "0" turns it off.
    # Under gunicorn (hosted) this block never executes, so the hosted app never polls Kalshi.
    mode = os.environ.get("LIVE_COLLECTOR", "auto").strip().lower()
    if mode not in ("0", "false", "no"):
        import threading
        import time

        import live_collector
        from live_data import LATEST

        def _external_collector_alive(max_age_s: float = 45.0) -> bool:
            try:
                return (time.time() - LATEST.stat().st_mtime) < max_age_s
            except OSError:
                return False

        def _supervised():
            # start (or restart) the collector whenever nothing else is capturing
            while True:
                if mode != "1" and _external_collector_alive():
                    time.sleep(30)
                    continue
                try:
                    live_collector.run(interval=10.0, depth=100)
                except Exception as e:  # noqa: BLE001
                    print(f"live collector crashed: {type(e).__name__}: {e} -- restarting in 5 s", flush=True)
                time.sleep(5)

        threading.Thread(target=_supervised, daemon=True, name="live-collector").start()
    app.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8050")))

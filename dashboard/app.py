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
    # Local run: start the order-book collector in a background thread so one command does everything.
    # Set LIVE_COLLECTOR=0 to skip it (e.g. when live_collector.py already runs in another terminal).
    # Under gunicorn (hosted) this block never executes, so the hosted app never polls Kalshi.
    if os.environ.get("LIVE_COLLECTOR", "1").strip().lower() not in ("0", "false", "no"):
        import threading

        import live_collector

        threading.Thread(target=live_collector.run, kwargs=dict(interval=10.0, depth=100),
                         daemon=True, name="live-collector").start()
    app.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8050")))

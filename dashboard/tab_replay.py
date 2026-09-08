# -*- coding: utf-8 -*-
"""
tab_replay.py -- Tab "Replay": scrub through a captured day of order books.

Pick a capture day (UTC folder of the collector) and an event; drag the slider or press play.
The mirrored ladder (10 best levels per side, both contracts in D terms) and the metrics table
show the book at that round; the day chart below shows D mid, R-as-D mid and the cross-book sums
with a marker at the current time — click on it to jump.
"""
from __future__ import annotations

import copy

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update

from pair_data import events
from replay_data import available_days, event_series, load_day
from tab_live import C_D, C_R, C_TEXT, METRIC_COLS, TABLE_STYLE, ladder_figure, metrics_from_ladders

NOTE = {"fontSize": "12px", "color": "#555", "margin": "4px 0"}
ROW = {"display": "flex", "alignItems": "center", "gap": "6px", "margin": "6px 0", "flexWrap": "wrap"}
BTN = {"marginRight": "6px", "padding": "4px 10px"}
SPEEDS = [{"label": "1 round / tick (10 s)", "value": 1}, {"label": "6 rounds (1 min)", "value": 6},
          {"label": "30 rounds (5 min)", "value": 30}, {"label": "180 rounds (30 min)", "value": 180}]
REPLAY_COLS = [c for c in METRIC_COLS if c["id"] not in ("age_s", "latency_ms")]


def slider_marks(rd) -> dict:
    marks = {}
    last_hour = None
    for i, t in enumerate(rd.ts_et):
        if t.hour != last_hour:
            marks[i] = {"label": f"{t:%H:%M}", "style": {"fontSize": "10px"}}
            last_hour = t.hour
    return marks


def day_figure(day: str, event: str) -> go.Figure:
    rd = load_day(day)
    s = event_series(rd, event)
    fig = go.Figure()
    if "mid_D" in s:
        fig.add_trace(go.Scatter(x=s.index, y=s["mid_D"], name="D mid", line=dict(color=C_D, width=1.4, shape="hv")))
    if "mid_R" in s:
        fig.add_trace(go.Scatter(x=s.index, y=s["mid_R"], name="R mid (as D)", line=dict(color=C_R, width=1.4, shape="hv")))
    if "ask_sum_m1" in s:
        fig.add_trace(go.Scatter(x=s.index, y=s["ask_sum_m1"], name="ask_D + ask_R − 1", yaxis="y2",
                                 line=dict(color="#d62728", width=1, shape="hv")))
        fig.add_trace(go.Scatter(x=s.index, y=s["bid_sum_m1"], name="bid_D + bid_R − 1", yaxis="y2",
                                 line=dict(color="#2ca02c", width=1, shape="hv")))
    fig.update_layout(height=280, margin=dict(l=55, r=55, t=30, b=30), hovermode="x unified", font=dict(color=C_TEXT),
                      legend=dict(orientation="h", y=1.14, x=0),
                      yaxis=dict(title="P(D)", tickformat=".3f"),
                      yaxis2=dict(title="cross-book $", overlaying="y", side="right", tickformat=".3f", range=[-0.03, 0.03]))
    fig.update_xaxes(title_text=f"{day} capture, ET")
    return fig


def with_marker(base: go.Figure, t: pd.Timestamp) -> go.Figure:
    fig = copy.deepcopy(base)
    fig.add_vline(x=t, line=dict(color="#333", width=1.5, dash="dot"))
    return fig


def layout() -> html.Div:
    days = available_days()
    return html.Div([
        html.Div(style=ROW, children=[
            html.B("Capture day (UTC):"),
            dcc.Dropdown(id="rp-day", options=[{"label": d, "value": d} for d in days], value=days[-1] if days else None,
                         clearable=False, style={"width": "160px"}),
            html.B("Event:", style={"marginLeft": "12px"}),
            dcc.Dropdown(id="rp-event", options=[{"label": e, "value": e} for e in events()], value=list(events())[0],
                         clearable=False, style={"width": "200px"}),
            html.Button("◀", id="rp-prev", n_clicks=0, style=BTN, title="one round back"),
            html.Button("▶ Play", id="rp-play", n_clicks=0, style=BTN),
            html.Button("▶", id="rp-next", n_clicks=0, style=BTN, title="one round forward"),
            dcc.Dropdown(id="rp-speed", options=SPEEDS, value=6, clearable=False, style={"width": "200px"}),
            html.B(id="rp-time", style={"marginLeft": "12px"}),
        ]),
        dcc.Slider(id="rp-slider", min=0, max=1, step=1, value=0, marks={}, updatemode="mouseup",
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="rp-ladder", config={"displaylogo": False}, style={"flex": "3", "minWidth": 0}),
            html.Div(style={"flex": "2", "minWidth": 0}, children=[
                html.Div("Book at the selected round: 10 best levels per side, both contracts in D terms.", style=NOTE),
                dash_table.DataTable(id="rp-metrics", columns=REPLAY_COLS, data=[], **TABLE_STYLE),
            ]),
        ]),
        html.Div("Day overview — click to jump to that time.", style=NOTE),
        dcc.Graph(id="rp-daychart", config={"displaylogo": False}),
        dcc.Interval(id="rp-tick", interval=700, disabled=True),
        dcc.Store(id="rp-playing", data=False),
        html.Div("No capture days yet: run the collector (python app.py starts it) and come back." if not days else "",
                 style=NOTE),
    ])


def register(app) -> None:
    @app.callback(Output("rp-playing", "data"), Output("rp-tick", "disabled"), Output("rp-play", "children"),
                  Input("rp-play", "n_clicks"), State("rp-playing", "data"), prevent_initial_call=True)
    def _toggle(_n, playing):
        playing = not playing
        return playing, (not playing), ("❚❚ Pause" if playing else "▶ Play")

    @app.callback(
        Output("rp-slider", "value"), Output("rp-slider", "max"), Output("rp-slider", "marks"),
        Input("rp-day", "value"), Input("rp-tick", "n_intervals"), Input("rp-prev", "n_clicks"), Input("rp-next", "n_clicks"),
        Input("rp-daychart", "clickData"),
        State("rp-slider", "value"), State("rp-speed", "value"), State("rp-playing", "data"),
    )
    def _position(day, _tick, _p, _n, click, cur, speed, playing):
        if not day:
            return 0, 1, {}
        rd = load_day(day)
        n = len(rd.rounds)
        trig = ctx.triggered_id
        if trig in (None, "rp-day"):                 # first load or a new day: jump to the last round
            return n - 1, n - 1, slider_marks(rd)
        if trig == "rp-daychart":                    # click on the day overview
            if not click:
                return no_update, no_update, no_update
            t = pd.Timestamp(click["points"][0]["x"])
            i = int(rd.ts_et.searchsorted(t))
            return rd.index_at(i), no_update, no_update
        cur = int(cur or 0)
        if trig == "rp-tick":
            if not playing:
                return no_update, no_update, no_update
            nxt = cur + int(speed or 1)
            return (nxt if nxt < n else 0), no_update, no_update
        if trig == "rp-prev":
            return rd.index_at(cur - 1), no_update, no_update
        if trig == "rp-next":
            return rd.index_at(cur + 1), no_update, no_update
        return no_update, no_update, no_update

    @app.callback(
        Output("rp-ladder", "figure"), Output("rp-metrics", "data"), Output("rp-time", "children"), Output("rp-daychart", "figure"),
        Input("rp-slider", "value"), Input("rp-event", "value"), State("rp-day", "value"),
    )
    def _show(i, event, day):
        if not day:
            empty = go.Figure()
            return empty, [], "", empty
        rd = load_day(day)
        i = rd.index_at(int(i or 0))
        ev = events()[event]
        books = rd.books_at(i)
        frames = {side: books.get(tk) for side, tk in ev.items()}
        fig = ladder_figure(event, frames, title=rd.label(i))
        rows = metrics_from_ladders(event, frames)
        return fig, rows, rd.label(i), with_marker(_day_base(day, event), rd.ts_et[i])


_DAY_CACHE: dict[tuple[str, str], go.Figure] = {}


def _day_base(day: str, event: str) -> go.Figure:
    key = (day, event)
    if key not in _DAY_CACHE:
        _DAY_CACHE[key] = day_figure(day, event)
    return _DAY_CACHE[key]

# -*- coding: utf-8 -*-
"""
tab_live.py -- Tab "Live books": the -D and -R order books of every event, refreshed from
data/live/latest.json every 10 s (written by live_collector.py, which must run separately).

Per event: a mirrored ladder (10 best levels per side, both contracts in D terms, bids to the
left, asks to the right), a metrics table (best bid/ask, spread, top sizes, cross-book sums),
and the cross-book sums over the last 60 minutes.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

from live_data import age_seconds, history, ladder, latest, top, top_levels
from pair_data import events

C_D, C_R, C_TEXT = "#2a78d6", "#eb6834", "#52514e"
ET = ZoneInfo("America/New_York")
NOTE = {"fontSize": "12px", "color": "#555", "margin": "4px 0"}
TABLE_STYLE = dict(style_cell={"fontSize": "12px", "padding": "2px 6px", "textAlign": "right"},
                   style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"})
DEPTH_SHOWN = 10


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_ladder(event: str, books: dict) -> go.Figure:
    """Mirrored ladder from the collector's latest records."""
    ev = events()[event]
    frames = {}
    for side in ("D", "R"):
        rec = books.get(ev[side])
        if rec:
            frames[side] = top_levels(ladder(rec, ev[side]), DEPTH_SHOWN)
    return ladder_figure(event, frames)


def ladder_figure(event: str, frames: dict[str, pd.DataFrame], title: str | None = None) -> go.Figure:
    """Mirrored ladder from ladders already in D terms: frames = {'D': DataFrame(side, price, qty), 'R': ...}."""
    ev = events()[event]
    fig = go.Figure()
    frames = {k: top_levels(v, DEPTH_SHOWN) for k, v in frames.items() if v is not None and len(v)}
    if not frames:
        fig.add_annotation(text="no data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=520, margin=dict(l=60, r=20, t=30, b=30))
        return fig
    prices = sorted({p for lv in frames.values() for p in lv["price"].round(4)})
    cats = [f"{p:.3f}" for p in prices]
    for side, color in (("D", C_D), ("R", C_R)):
        lv = frames.get(side)
        if lv is None:
            continue
        label = ev[side] if side == "D" else f"{ev[side]} (as D)"
        for bs, sign, alpha in (("bid", -1, 0.9), ("ask", 1, 0.5)):
            g = lv[lv["side"] == bs]
            fig.add_trace(go.Bar(
                y=[f"{p:.3f}" for p in g["price"].round(4)], x=sign * g["qty"], orientation="h",
                name=f"{label} {bs}", marker_color=_rgba(color, alpha), marker_line_width=0, offsetgroup=side,
                customdata=g["qty"], hovertemplate="%{y}  " + bs + " %{customdata:,.0f}<extra>" + label + "</extra>",
            ))
    fig.update_layout(
        height=520, margin=dict(l=60, r=20, t=30, b=30), barmode="group", bargap=0.15, bargroupgap=0.05,
        legend=dict(orientation="h", y=1.08, x=0), font=dict(color=C_TEXT), plot_bgcolor="#fcfcfb",
        yaxis=dict(type="category", categoryorder="array", categoryarray=cats, title="price (D terms)"),
        xaxis=dict(title="resting contracts   ◀ bids   |   asks ▶", tickformat=",.0f"),
        title=dict(text=title or "", x=0.5, font=dict(size=13)),
    )
    fig.add_vline(x=0, line=dict(color="#999", width=1))
    return fig


def metrics_from_ladders(event: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Best bid/ask, spread, top sizes per contract (D terms) and the cross-book sums, from ladders."""
    ev = events()[event]
    rows, tp = [], {}
    for side in ("D", "R"):
        lad = frames.get(side)
        if lad is None or lad.empty:
            continue
        t = top(lad)
        tp[side] = t
        rows.append({"contract": ev[side] + ("" if side == "D" else " (as D)"), "best_bid": t["best_bid"], "best_ask": t["best_ask"],
                     "spread": t["spread"], "bid_top": t["bid_top"], "ask_top": t["ask_top"]})
    if "D" in tp and "R" in tp and tp["D"]["best_bid"] is not None and tp["R"]["best_bid"] is not None:
        # R in D terms: R yes-ask = 1 - R best_bid(D terms); R yes-bid = 1 - R best_ask(D terms)
        rows.append({"contract": "ask_D + ask_R − 1", "spread": round(tp["D"]["best_ask"] - tp["R"]["best_bid"], 4)})
        rows.append({"contract": "bid_D + bid_R − 1", "spread": round(tp["D"]["best_bid"] - tp["R"]["best_ask"], 4)})
    return rows


def metrics_rows(event: str, books: dict) -> list[dict]:
    ev = events()[event]
    rows, raw = [], {}
    for side in ("D", "R"):
        rec = books.get(ev[side])
        if not rec:
            continue
        tp = top(ladder(rec, ev[side]))
        ob = rec["orderbook_fp"]
        yes = [float(p) for p, _ in (ob.get("yes_dollars") or [])]
        no = [float(p) for p, _ in (ob.get("no_dollars") or [])]
        raw[side] = {"yes_bid": max(yes) if yes else None, "yes_ask": (1 - max(no)) if no else None}
        rows.append({"contract": ev[side] + ("" if side == "D" else " (as D)"), "best_bid": tp["best_bid"], "best_ask": tp["best_ask"],
                     "spread": tp["spread"], "bid_top": tp["bid_top"], "ask_top": tp["ask_top"],
                     "age_s": round(age_seconds(rec), 1), "latency_ms": rec.get("latency_ms")})
    if "D" in raw and "R" in raw and None not in raw["D"].values() and None not in raw["R"].values():
        rows.append({"contract": "ask_D + ask_R − 1", "spread": round(raw["D"]["yes_ask"] + raw["R"]["yes_ask"] - 1, 4)})
        rows.append({"contract": "bid_D + bid_R − 1", "spread": round(raw["D"]["yes_bid"] + raw["R"]["yes_bid"] - 1, 4)})
    return rows


METRIC_COLS = [
    {"name": "Contract / check", "id": "contract"}, {"name": "Best bid", "id": "best_bid"}, {"name": "Best ask", "id": "best_ask"},
    {"name": "Spread / value", "id": "spread"}, {"name": "Bid top size", "id": "bid_top"}, {"name": "Ask top size", "id": "ask_top"},
    {"name": "Age (s)", "id": "age_s"}, {"name": "Latency (ms)", "id": "latency_ms"},
]


def build_history(event: str, minutes: int = 60) -> go.Figure:
    ev = events()[event]
    fig = go.Figure()
    hd, hr = history(ev["D"], minutes), history(ev["R"], minutes)
    if hd.empty or hr.empty:
        fig.add_annotation(text="no history yet", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=220, margin=dict(l=55, r=20, t=30, b=30))
        return fig
    # R history is already in D terms: R best_bid (D terms) = 1 - R yes-ask, R best_ask = 1 - R yes-bid
    a = hd.set_index("ts")[["best_bid", "best_ask"]].rename(columns={"best_bid": "bidD", "best_ask": "askD"})
    b = hr.set_index("ts")[["best_bid", "best_ask"]].rename(columns={"best_bid": "bidRd", "best_ask": "askRd"})
    j = pd.merge_asof(a.sort_index(), b.sort_index(), left_index=True, right_index=True, direction="nearest",
                      tolerance=pd.Timedelta(seconds=15)).dropna()
    j.index = j.index.tz_convert(ET).tz_localize(None)          # plot in ET
    # ask_D + ask_R - 1 = askD - bidRd ;  bid_D + bid_R - 1 = bidD - askRd
    fig.add_trace(go.Scatter(x=j.index, y=j["askD"] - j["bidRd"], name="ask_D + ask_R − 1", line=dict(color="#d62728", width=1.4, shape="hv")))
    fig.add_trace(go.Scatter(x=j.index, y=j["bidD"] - j["askRd"], name="bid_D + bid_R − 1", line=dict(color="#2ca02c", width=1.4, shape="hv")))
    fig.add_trace(go.Scatter(x=j.index, y=(j["bidD"] + j["askD"]) / 2, name="D mid", line=dict(color=C_D, width=1, dash="dot", shape="hv"), yaxis="y2"))
    fig.add_hline(y=0, line=dict(color="#bbb", width=1))
    mid = (j["bidD"] + j["askD"]) / 2
    fig.update_layout(height=260, margin=dict(l=55, r=55, t=30, b=30), legend=dict(orientation="h", y=1.12, x=0),
                      hovermode="x unified", font=dict(color=C_TEXT),
                      yaxis=dict(title="$", tickformat=".3f"),
                      yaxis2=dict(title="D mid", overlaying="y", side="right", tickformat=".3f",
                                  range=[float(mid.min()) - 0.01, float(mid.max()) + 0.01]))
    fig.update_xaxes(title_text=f"last {minutes} min (ET)")
    return fig


# ----------------------------------------------------------------------------- layout
def layout() -> html.Div:
    blocks = [html.Div(id="live-status", style={"fontSize": "13px", "margin": "4px 0 8px"})]
    for ev in events():
        blocks.append(html.H4(ev, style={"margin": "12px 0 4px"}))
        blocks.append(html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id=f"live-ladder-{ev}", config={"displaylogo": False}, style={"flex": "3", "minWidth": 0}),
            html.Div(style={"flex": "2", "minWidth": 0}, children=[
                html.Div(f"Top {DEPTH_SHOWN} levels per side shown; 100 per side are saved.", style=NOTE),
                dash_table.DataTable(id=f"live-metrics-{ev}", columns=METRIC_COLS, data=[], **TABLE_STYLE),
                dcc.Graph(id=f"live-hist-{ev}", config={"displaylogo": False}),
            ]),
        ]))
    return html.Div([
        html.Div("Refreshes every 10 s from data/live/latest.json. Start the collector in another terminal: "
                 "python live_collector.py", style=NOTE),
        dcc.Interval(id="live-interval", interval=10_000, n_intervals=0),
        *blocks,
    ])


def register(app) -> None:
    outputs = [Output("live-status", "children")]
    for ev in events():
        outputs += [Output(f"live-ladder-{ev}", "figure"), Output(f"live-metrics-{ev}", "data"), Output(f"live-hist-{ev}", "figure")]

    @app.callback(outputs, Input("live-interval", "n_intervals"))
    def _refresh(_n):
        j = latest()
        if not j:
            empty = go.Figure()
            empty.add_annotation(text="collector not running", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return ["⚠ No live data: run  python live_collector.py  in the dashboard folder."] + [empty, [], empty] * len(events())
        books = j["books"]
        rt = dt.datetime.fromisoformat(j["round_ts"])
        age = (dt.datetime.now(dt.timezone.utc) - rt).total_seconds()
        status = (f"Collector round {rt.astimezone(ET):%Y-%m-%d %H:%M:%S} ET ({rt:%H:%M:%S} UTC), "
                  f"{age:.0f} s ago, {len(books)} books")
        if age > 60:
            status = "⚠ stale: " + status
        out = [status]
        for ev in events():
            out += [build_ladder(ev, books), metrics_rows(ev, books), build_history(ev)]
        return out

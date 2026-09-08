# -*- coding: utf-8 -*-
"""
tab_pair.py -- Tab "Event pair": the -D and -R contracts of one Kalshi event side by side, in D terms.

Top figure (daily, shared x):
    row 1  D mid, R mid mapped to 1 - price, event P(D)
    row 2  cross-book checks: D mid - R-implied mid, ask_D + ask_R - 1, bid_D + bid_R - 1
    row 3  daily volume D vs R          row 4  trades per day D vs R
Bottom figure (one ET day): 1-min mid/bid/ask of both contracts in D terms, merged trades
(colour = pro-D / pro-R, symbol = contract), cumulative net flow. Trade table follows the view.
"""
from __future__ import annotations

import bisect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, Patch, ctx, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

from kalshi_data import all_days, trade_days
from pair_data import day_flow, events, merged_trades, pair_1min, pair_daily

C_D, C_R, C_PD = "#2a78d6", "#eb6834", "#52514e"
C_PRO_D, C_PRO_R = "#2ca02c", "#d62728"
BTN = {"marginRight": "6px", "padding": "4px 10px"}
ROW = {"display": "flex", "alignItems": "center", "gap": "6px", "margin": "6px 0", "flexWrap": "wrap"}
NOTE = {"fontSize": "12px", "color": "#555", "margin": "4px 0"}
TABLE_STYLE = dict(style_cell={"fontSize": "12px", "padding": "2px 6px", "textAlign": "right"},
                   style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"})


def _events() -> list[str]:
    return list(events().keys())


def _days(event: str) -> tuple[list[str], list[str]]:
    """(all covered days, days with trades) for the event; the default day is the last day
    on which BOTH contracts traded, so the merged view opens with something to show."""
    ev = events()[event]
    days = sorted(set(all_days(ev["D"])) | set(all_days(ev["R"])))
    both = sorted(set(trade_days(ev["D"])) & set(trade_days(ev["R"])))
    either = sorted(set(trade_days(ev["D"])) | set(trade_days(ev["R"])))
    return days, (both or either)


# ----------------------------------------------------------------------------- figures
def build_daily(event: str) -> go.Figure:
    p = pair_daily(event)
    ev = events()[event]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.42, 0.2, 0.19, 0.19], vertical_spacing=0.035,
                        subplot_titles=("Mid in D terms", "Cross-book checks", "Daily volume (contracts)", "Trades per day"))
    x = p["date"]
    fig.add_trace(go.Scatter(x=x, y=p["mid_D"], name=f"{ev['D']} mid", line=dict(color=C_D, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=p["mid_Rd"], name=f"{ev['R']} as 1 - mid", line=dict(color=C_R, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=p["p_d"], name="event P(D)", line=dict(color=C_PD, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=p["residual"], name="D mid − R-implied mid", line=dict(color=C_PD, width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=p["ask_sum_m1"], name="ask_D + ask_R − 1", line=dict(color=C_PRO_R, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=p["bid_sum_m1"], name="bid_D + bid_R − 1", line=dict(color=C_PRO_D, width=1)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#bbb", width=1), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=p["vol_D"], name="volume D", marker_color=C_D, marker_line_width=0), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=p["vol_R"], name="volume R", marker_color=C_R, marker_line_width=0), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=p["trades_D"], name="trades D", marker_color=C_D, marker_line_width=0), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=p["trades_R"], name="trades R", marker_color=C_R, marker_line_width=0), row=4, col=1)
    fig.update_layout(height=790, margin=dict(l=55, r=20, t=30, b=70), hovermode="x unified", barmode="group",
                      legend=dict(orientation="h", y=-0.06, x=0), bargap=0.2, uirevision=event)
    fig.update_yaxes(title_text="P(D)", row=1, col=1)
    fig.update_yaxes(title_text="$", tickformat=".3f", row=2, col=1)
    fig.update_xaxes(type="date", row=4, col=1)
    return fig


def build_intraday(event: str, day: str, view: str):
    ev = events()[event]
    m = pair_1min(event, day)
    tr = merged_trades(event, day)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        subplot_titles=("Quotes (D terms) and trades", "Cumulative net flow: pro-D − pro-R contracts"))
    for side, color in (("D", C_D), ("R", C_R)):
        if view != "merged" and view != side:
            continue
        f = m.get(side)
        if f is None or f.empty:
            continue
        lbl = ev[side] if side == "D" else f"{ev[side]} (as D)"
        fig.add_trace(go.Scatter(x=f["t"], y=f["ask"], name=f"{lbl} ask", line=dict(color=color, width=0.8, dash="dot", shape="hv"), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=f["t"], y=f["bid"], name=f"{lbl} bid", line=dict(color=color, width=0.8, dash="dot", shape="hv"), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=f["t"], y=f["mid"], name=f"{lbl} mid", line=dict(color=color, width=1.6, shape="hv")), row=1, col=1)
    if not tr.empty:
        sub = (tr if view == "merged" else tr[tr["contract"] == view]).reset_index(drop=True)
        size = np.clip(4 + 2.2 * np.log1p(sub["count"].to_numpy()), 4, 18)   # positional, so sub needs a fresh index
        for direction, color in (("pro-D", C_PRO_D), ("pro-R", C_PRO_R)):
            for contract, symbol in (("D", "circle"), ("R", "diamond")):
                g = sub[(sub["direction"] == direction) & (sub["contract"] == contract)]
                if g.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=g["t"], y=g["price_d"], mode="markers", name=f"{direction} via {contract}",
                    marker=dict(color=color, symbol=symbol, size=size[g.index], line=dict(width=0.5, color="white"), opacity=0.85),
                    customdata=g[["count", "yes_price", "taker_side", "ticker", "trade_id"]].to_numpy(),
                    hovertemplate="%{x|%H:%M:%S.%L}  D-price %{y:.3f}<br>%{customdata[3]} yes %{customdata[1]:.3f} taker %{customdata[2]}<br>"
                                  "contracts %{customdata[0]:,.2f}<br>%{customdata[4]}<extra>" + f"{direction} via {contract}" + "</extra>",
                ), row=1, col=1)
        signed = np.where(sub["direction"] == "pro-D", sub["count"], -sub["count"])
        fig.add_trace(go.Scatter(x=sub["t"], y=np.cumsum(signed), name="net flow", line=dict(color=C_PD, width=1.6, shape="hv"),
                                 fill="tozeroy", fillcolor="rgba(82,81,78,0.12)"), row=2, col=1)
    if all(f is None or f.empty for f in m.values()) and tr.empty:
        fig.add_annotation(text=f"No data for {event} on {day}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    d0 = pd.Timestamp(day)
    fig.update_layout(height=600, margin=dict(l=55, r=20, t=40, b=95), hovermode="closest", dragmode="pan",
                      legend=dict(orientation="h", y=-0.14, x=0))
    fig.update_xaxes(range=[d0, d0 + pd.Timedelta(days=1)])
    fig.update_yaxes(title_text="P(D)", row=1, col=1)
    fig.update_yaxes(title_text="contracts", row=2, col=1)
    fig.update_xaxes(title_text="ET", row=2, col=1)
    return fig, tr


def trade_rows(tr: pd.DataFrame) -> list[dict]:
    if tr.empty:
        return []
    out = pd.DataFrame({
        "id": tr["trade_id"], "ts": tr["t"].astype(str),
        "time": tr["t"].dt.strftime("%H:%M:%S.%f").str[:-3], "contract": tr["contract"],
        "direction": tr["direction"], "price_d": tr["price_d"].round(4), "yes_price": tr["yes_price"],
        "taker_side": tr["taker_side"], "count": tr["count"].round(2),
    })
    return out.to_dict("records")


TRADE_COLS = [
    {"name": "Time (ET)", "id": "time"}, {"name": "Contract", "id": "contract"}, {"name": "Direction", "id": "direction"},
    {"name": "D price", "id": "price_d"}, {"name": "Own yes px", "id": "yes_price"}, {"name": "Taker", "id": "taker_side"},
    {"name": "Contracts", "id": "count"},
]


def flow_text(tr: pd.DataFrame, event: str, day: str) -> str:
    if tr.empty:
        return f"{event} {day}: no trades"
    g = day_flow(tr)
    parts = [f"{r.contract} {r.direction}: {int(r.trades)} trades / {r.contracts:,.0f} contracts" for r in g.itertuples()]
    net = tr.loc[tr["direction"] == "pro-D", "count"].sum() - tr.loc[tr["direction"] == "pro-R", "count"].sum()
    return f"{event} {day}: {len(tr)} trades, net flow {net:+,.0f} contracts (pro-D − pro-R)  |  " + "  |  ".join(parts)


# ----------------------------------------------------------------------------- layout
def layout() -> html.Div:
    ev0 = _events()[0]
    days, tdays = _days(ev0)
    return html.Div([
        html.Div(style=ROW, children=[
            html.B("Event:"),
            dcc.Dropdown(id="pair-event", options=[{"label": e, "value": e} for e in _events()], value=ev0,
                         clearable=False, style={"width": "220px"}),
            html.Span("R contract shown as 1 − price, so both books sit on one P(D) axis. "
                      "pro-D = buys D-yes or R-no; pro-R = buys R-yes or D-no.", style={"fontSize": "12px", "color": "#888"}),
        ]),
        dcc.Graph(id="pair-daily", config={"displaylogo": False}),

        html.Hr(),
        html.Div(style=ROW, children=[
            html.B("Intraday (ET day):"),
            html.Button("◀ Prev day", id="pair-prev-d", n_clicks=0, style=BTN),
            dcc.DatePickerSingle(id="pair-day", date=tdays[-1] if tdays else days[-1], min_date_allowed=days[0],
                                 max_date_allowed=days[-1], display_format="YYYY-MM-DD"),
            html.Button("Next day ▶", id="pair-next-d", n_clicks=0, style=BTN),
            html.Button("◀ Prev day with trades", id="pair-prev-td", n_clicks=0, style=BTN),
            html.Button("Next day with trades ▶", id="pair-next-td", n_clicks=0, style=BTN),
            dcc.RadioItems(id="pair-view", value="merged", inline=True, style={"marginLeft": "12px"},
                           options=[{"label": "merged", "value": "merged"}, {"label": "D only", "value": "D"}, {"label": "R only", "value": "R"}]),
            html.Span("(click a day in the daily chart to jump)", style={"fontSize": "12px", "color": "#888"}),
        ]),
        html.Div(id="pair-flow", style={"fontSize": "12px", "color": "#333", "margin": "4px 0"}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="pair-intraday", config={"scrollZoom": True, "displaylogo": False}, style={"flex": "3", "minWidth": 0}),
            html.Div(style={"flex": "2", "minWidth": 0}, children=[
                html.Div("Merged trades — filtered to the visible range; click a row to centre the chart", style=NOTE),
                dash_table.DataTable(
                    id="pair-trades", columns=TRADE_COLS, data=[], page_size=15, page_action="native", sort_action="native",
                    **TABLE_STYLE,
                    style_data_conditional=[
                        {"if": {"filter_query": "{direction} = pro-D", "column_id": "direction"}, "color": C_PRO_D},
                        {"if": {"filter_query": "{direction} = pro-R", "column_id": "direction"}, "color": C_PRO_R},
                        {"if": {"filter_query": "{contract} = D", "column_id": "contract"}, "color": C_D},
                        {"if": {"filter_query": "{contract} = R", "column_id": "contract"}, "color": C_R},
                    ]),
            ]),
        ]),
        dcc.Store(id="pair-day-trades", data={"day": None, "rows": []}),
    ])


# ----------------------------------------------------------------------------- callbacks
def register(app) -> None:
    @app.callback(Output("pair-daily", "figure"), Input("pair-event", "value"))
    def _daily(event):
        return build_daily(event)

    @app.callback(
        Output("pair-day", "date"), Output("pair-day", "min_date_allowed"), Output("pair-day", "max_date_allowed"),
        Input("pair-event", "value"), Input("pair-daily", "clickData"),
        Input("pair-prev-d", "n_clicks"), Input("pair-next-d", "n_clicks"),
        Input("pair-prev-td", "n_clicks"), Input("pair-next-td", "n_clicks"),
        State("pair-day", "date"), prevent_initial_call=True,
    )
    def _nav(event, click, _p, _n, _pt, _nt, cur):
        trig = ctx.triggered_id
        days, tdays = _days(event)
        cur = str(cur)[:10]
        if trig == "pair-event":
            return (tdays[-1] if tdays else days[-1]), days[0], days[-1]
        nu = no_update
        if trig == "pair-daily":
            if not click:
                return nu, nu, nu
            d = str(click["points"][0]["x"])[:10]
            return (d if days[0] <= d <= days[-1] else nu), nu, nu
        if trig in ("pair-prev-d", "pair-next-d"):
            d = (date.fromisoformat(cur) + timedelta(days=1 if trig == "pair-next-d" else -1)).isoformat()
            return min(max(d, days[0]), days[-1]), nu, nu
        if trig == "pair-next-td":
            i = bisect.bisect_right(tdays, cur)
            return (tdays[i] if i < len(tdays) else nu), nu, nu
        if trig == "pair-prev-td":
            i = bisect.bisect_left(tdays, cur) - 1
            return (tdays[i] if i >= 0 else nu), nu, nu
        return nu, nu, nu

    @app.callback(
        Output("pair-intraday", "figure"), Output("pair-day-trades", "data"), Output("pair-trades", "data"),
        Output("pair-trades", "page_current"), Output("pair-flow", "children"),
        Input("pair-day", "date"), Input("pair-event", "value"), Input("pair-view", "value"),
    )
    def _intraday(day, event, view):
        day = str(day)[:10]
        fig, tr = build_intraday(event, day, view)
        rows = trade_rows(tr if view == "merged" or tr.empty else tr[tr["contract"] == view])
        return fig, {"day": day, "rows": rows}, rows, 0, flow_text(tr, event, day)

    @app.callback(
        Output("pair-trades", "data", allow_duplicate=True), Output("pair-trades", "page_current", allow_duplicate=True),
        Input("pair-intraday", "relayoutData"), State("pair-day-trades", "data"), State("pair-day", "date"),
        prevent_initial_call=True,
    )
    def _filter(relayout, store, day):
        if not relayout or not store or store.get("day") != str(day)[:10]:
            return no_update, no_update
        rows = store["rows"]
        lo = hi = None
        for ax in ("xaxis", "xaxis2"):
            r0, r1 = relayout.get(f"{ax}.range[0]"), relayout.get(f"{ax}.range[1]")
            if r0 is not None and r1 is not None:
                lo, hi = pd.Timestamp(r0), pd.Timestamp(r1)
                break
        if lo is None:
            if any(relayout.get(f"{ax}.autorange") for ax in ("xaxis", "xaxis2")):
                return rows, 0
            return no_update, no_update
        return [r for r in rows if lo <= pd.Timestamp(r["ts"]) <= hi], 0

    @app.callback(
        Output("pair-intraday", "figure", allow_duplicate=True),
        Input("pair-trades", "active_cell"), State("pair-day-trades", "data"), prevent_initial_call=True,
    )
    def _center(cell, store):
        if not cell or not store:
            return no_update
        hit = next((r for r in store["rows"] if r["id"] == cell.get("row_id")), None)
        if hit is None:
            return no_update
        ts = pd.Timestamp(hit["ts"])
        rng = [str(ts - pd.Timedelta(minutes=10)), str(ts + pd.Timedelta(minutes=10))]
        patch = Patch()
        patch["layout"]["xaxis"]["range"] = rng
        patch["layout"]["xaxis2"]["range"] = rng
        return patch

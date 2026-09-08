# -*- coding: utf-8 -*-
"""
tab_stock.py -- Tab "Stocks x Kalshi": one stock against several Kalshi series.

Panel A   Daily. Left axis: the selected Kalshi series (contracts and/or event P(D); R contracts
          shown as 1 - price when "D terms" is on). Right axis: the stock's backward-adjusted close.
          Stock chosen by ticker or by paging through the S&P 500 (optionally inside one sector).
          Clicking a day moves panel B to that day.
Closeness One row per selected series: correlations of 16:00-ET-aligned daily changes with the
          stock return (same day, lead/lag), best lag, and event-day hit rate / lift.
Events    Event-window statistics for one selected series (bars + tables). Click an event row:
          panel A zooms to it, panel B jumps to that day.
Panel B   Intraday view of one contract on one ET day (quotes, trades, 1-min volume) with the trade table.
"""
from __future__ import annotations

import bisect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, Patch, ctx, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

from closeness import closeness_table
from event_study import EVENT_PREFIX, describe, run_event_study, series_label
from kalshi_data import DEFAULT_KALSHI_TICKER, all_days, kalshi_1min, kalshi_daily, kalshi_tickers, kalshi_trades, trade_days
from pair_data import events, is_r, pair_daily
from stock_data import get_stock_daily, load_universe, normalize_ticker, universe_info

# ----------------------------------------------------------------------------- data
UNI = load_universe()
SECTORS = sorted(UNI["sector"].dropna().unique().tolist())
KALSHI_TICKERS = kalshi_tickers()
SERIES_KEYS = KALSHI_TICKERS + [f"{EVENT_PREFIX}{e}" for e in events()]
DEFAULT_TICKER = "AAPL"
ALL_SECTORS = "ALL"
_ALL_DAYS = sorted({d for k in KALSHI_TICKERS for d in all_days(k)})
DAY_MIN, DAY_MAX = _ALL_DAYS[0], _ALL_DAYS[-1]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # fixed order
C_STOCK = "#333333"
C_UP, C_DOWN, C_BASE, C_TEXT = "#e34948", "#2a78d6", "#8a8987", "#52514e"


def universe_list(sector: str | None) -> list[str]:
    if not sector or sector == ALL_SECTORS:
        return UNI["ticker"].tolist()
    return UNI.loc[UNI["sector"] == sector, "ticker"].tolist()


def default_day(kalshi_ticker: str) -> str:
    td = trade_days(kalshi_ticker)
    return td[-1] if td else all_days(kalshi_ticker)[-1]


def series_daily(key: str, d_terms: bool) -> pd.DataFrame:
    """date, mid, bid, ask for a series key (bid/ask NaN for event P(D))."""
    if key.startswith(EVENT_PREFIX):
        p = pair_daily(key[len(EVENT_PREFIX):])
        return pd.DataFrame({"date": p["date"], "mid": p["p_d"], "bid": np.nan, "ask": np.nan})
    kd = kalshi_daily(key)
    if d_terms and is_r(key):
        return pd.DataFrame({"date": kd["date"], "mid": 1 - kd["mid_close"], "bid": 1 - kd["yes_ask_close"], "ask": 1 - kd["yes_bid_close"]})
    return pd.DataFrame({"date": kd["date"], "mid": kd["mid_close"], "bid": kd["yes_bid_close"], "ask": kd["yes_ask_close"]})


# ----------------------------------------------------------------------------- figures
def build_fig_a(ticker: str, keys: list[str], d_terms: bool):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    keys = keys or [DEFAULT_KALSHI_TICKER]
    for i, key in enumerate(keys):
        s = series_daily(key, d_terms)
        color = PALETTE[i % len(PALETTE)]
        label = series_label(key, d_terms)
        if len(keys) == 1 and s["bid"].notna().any():
            fig.add_trace(go.Scatter(x=s["date"], y=s["ask"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=s["date"], y=s["bid"], name="bid/ask band", fill="tonexty",
                                     fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s["date"], y=s["mid"], name=label, line=dict(color=color, width=1.6),
                                 hovertemplate="%{y:.3f}<extra>" + label + "</extra>"))
    err = None
    try:
        st = get_stock_daily(ticker)
        fig.add_trace(go.Scatter(
            x=st["date"], y=st["close_hfq"], name=f"{ticker} close (backward-adjusted)", line=dict(color=C_STOCK, width=1.6),
            customdata=st[["close", "volume"]].to_numpy(),
            hovertemplate="adj %{y:.2f}  (raw close %{customdata[0]:.2f}, vol %{customdata[1]:,.0f})<extra>" + ticker + "</extra>",
        ), secondary_y=True)
    except Exception as e:  # noqa: BLE001
        err = str(e)
    fig.update_layout(
        height=430, margin=dict(l=55, r=55, t=30, b=20), hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(type="date", rangeslider=dict(visible=True, thickness=0.07)),
        uirevision="A",
    )
    fig.update_yaxes(title_text="P(yes) / P(D)", secondary_y=False)
    fig.update_yaxes(title_text=f"{ticker} adjusted close ($)", secondary_y=True)
    return fig, err


def build_fig_b(kalshi_ticker: str, day: str):
    m = kalshi_1min(kalshi_ticker, day)
    t = kalshi_trades(kalshi_ticker, day)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)
    if not m.empty:
        fig.add_trace(go.Scatter(x=m["t"], y=m["yes_ask_close"], name="ask",
                                 line=dict(color="rgba(214,39,40,0.55)", width=1, shape="hv")), row=1, col=1)
        fig.add_trace(go.Scatter(x=m["t"], y=m["yes_bid_close"], name="bid",
                                 line=dict(color="rgba(44,160,44,0.55)", width=1, shape="hv")), row=1, col=1)
        fig.add_trace(go.Scatter(x=m["t"], y=m["mid_close"], name="mid",
                                 line=dict(color="#2a78d6", width=1.6, shape="hv")), row=1, col=1)
        fig.add_trace(go.Bar(x=m["t"], y=m["volume"], name="1-min volume",
                             marker_color="rgba(110,110,110,0.65)"), row=2, col=1)
    if not t.empty:
        size = np.clip(4 + 2.2 * np.log1p(t["count"].to_numpy()), 4, 18)
        for side, color, label in (("yes", "#2ca02c", "taker buys yes"), ("no", "#d62728", "taker buys no")):
            g = t[t["taker_side"] == side]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g["t"], y=g["yes_price"], mode="markers", name=label,
                marker=dict(color=color, size=size[g.index], line=dict(width=0.5, color="white"), opacity=0.85),
                customdata=g[["count", "no_price", "taker_book_side", "trade_id"]].to_numpy(),
                hovertemplate="%{x|%H:%M:%S.%L}  yes %{y:.3f} / no %{customdata[1]:.3f}<br>"
                              "contracts %{customdata[0]:,.2f}  book %{customdata[2]}<br>%{customdata[3]}<extra>" + label + "</extra>",
            ), row=1, col=1)
    if m.empty and t.empty:
        fig.add_annotation(text=f"No data for {kalshi_ticker} on {day}", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=16))
    d0 = pd.Timestamp(day)
    fig.update_layout(height=520, margin=dict(l=55, r=20, t=30, b=20), hovermode="closest", dragmode="pan",
                      legend=dict(orientation="h", y=1.06, x=0), bargap=0)
    fig.update_xaxes(range=[d0, d0 + pd.Timedelta(days=1)])     # shared axes: set on both
    fig.update_yaxes(title_text="P(yes)", row=1, col=1)
    fig.update_yaxes(title_text="contracts", row=2, col=1)
    fig.update_xaxes(title_text="ET", row=2, col=1)
    return fig, m, t


def build_fig_events(summary: pd.DataFrame, s_thr: float) -> go.Figure:
    fig = go.Figure()
    if summary is None or summary.empty:
        fig.add_annotation(text="No events", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    x = [f"{int(o):+d}" for o in summary["offset"]]
    for col, name, color, ncol, mcol in (("hit_up", "Kalshi up events", C_UP, "n_up", "mean_ret_up"),
                                         ("hit_down", "Kalshi down events", C_DOWN, "n_down", "mean_ret_down")):
        fig.add_trace(go.Bar(
            x=x, y=summary[col], name=name, marker_color=color, marker_line_width=0,
            customdata=np.c_[summary[ncol], summary[mcol]],
            hovertemplate="offset %{x} days<br>share with stock |r| ≥ " + f"{s_thr:.1%}" +
                          ": %{y:.1%}<br>events %{customdata[0]}  mean return %{customdata[1]:+.2%}<extra>%{fullData.name}</extra>",
        ))
    base = float(summary["baseline"].iloc[0])
    fig.add_hline(y=base, line=dict(color=C_BASE, width=2, dash="dash"),
                  annotation_text=f"unconditional baseline {base:.1%}", annotation_position="top right",
                  annotation_font=dict(color=C_TEXT, size=11))
    fig.update_layout(height=300, margin=dict(l=55, r=20, t=30, b=40), barmode="group", bargap=0.35, bargroupgap=0.08,
                      legend=dict(orientation="h", y=1.12, x=0), plot_bgcolor="#fcfcfb", paper_bgcolor="white",
                      font=dict(color=C_TEXT))
    fig.update_xaxes(title_text="trading-day offset from the event day", showgrid=False)
    fig.update_yaxes(title_text="share of events with a big stock move", tickformat=".0%", gridcolor="#eeeeea", rangemode="tozero")
    return fig


# ----------------------------------------------------------------------------- tables
def table_rows(t: pd.DataFrame) -> list[dict]:
    if t.empty:
        return []
    out = pd.DataFrame({
        "id": t["trade_id"], "ts": t["t"].astype(str),
        "time": t["t"].dt.strftime("%H:%M:%S.%f").str[:-3],
        "yes_price": t["yes_price"], "no_price": t["no_price"],
        "count": t["count"].round(2), "taker_side": t["taker_side"],
        "taker_book_side": t["taker_book_side"], "yes_notional": t["yes_notional"].round(2),
    })
    return out.to_dict("records")


TRADE_COLS = [
    {"name": "Time (ET)", "id": "time"}, {"name": "Yes px", "id": "yes_price"}, {"name": "No px", "id": "no_price"},
    {"name": "Contracts", "id": "count"}, {"name": "Taker", "id": "taker_side"}, {"name": "Book side", "id": "taker_book_side"},
    {"name": "Yes notional $", "id": "yes_notional"},
]
PCT1 = {"type": "numeric", "format": {"specifier": ".1%"}}
PCT2S = {"type": "numeric", "format": {"specifier": "+.2%"}}
F3 = {"type": "numeric", "format": {"specifier": "+.3f"}}
CLOSENESS_COLS = [
    {"name": "Series", "id": "label"}, {"name": "Days", "id": "n_days"},
    {"name": "Corr same day", "id": "corr_same_day", **F3},
    {"name": "Corr Kalshi leads 1d", "id": "corr_kalshi_leads_1d", **F3},
    {"name": "Corr stock leads 1d", "id": "corr_stock_leads_1d", **F3},
    {"name": "Best lag", "id": "best_lag"}, {"name": "Best lag corr", "id": "best_lag_corr", **F3},
    {"name": "Events", "id": "n_events"}, {"name": "Hit rate day 0", "id": "hit0", **PCT1},
    {"name": "Lift day 0", "id": "lift0", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Baseline", "id": "baseline", **PCT1},
]


def event_table_columns(K: int) -> list[dict]:
    cols = [{"name": "Date", "id": "date"}, {"name": "Dir", "id": "direction"},
            {"name": "Before", "id": "kalshi_prev"}, {"name": "After", "id": "kalshi_mid"},
            {"name": "Kalshi ret", "id": "kalshi_ret", "type": "numeric", "format": {"specifier": "+.1%"}}]
    for o in range(-K, K + 1):
        cols.append({"name": f"Stock r {o:+d}", "id": f"ret_{o:+d}", **PCT2S})
    cols += [{"name": "Big move in window", "id": "any_big_in_window"},
             {"name": "Window max |r|", "id": "max_abs_ret", "type": "numeric", "format": {"specifier": ".2%"}}]
    return cols


SUMMARY_COLS = [
    {"name": "Offset", "id": "offset"},
    {"name": "Up n", "id": "n_up"}, {"name": "Up hit rate", "id": "hit_up", **PCT1}, {"name": "Up mean r", "id": "mean_ret_up", **PCT2S},
    {"name": "Down n", "id": "n_down"}, {"name": "Down hit rate", "id": "hit_down", **PCT1}, {"name": "Down mean r", "id": "mean_ret_down", **PCT2S},
    {"name": "All hit rate", "id": "hit_all", **PCT1}, {"name": "Baseline", "id": "baseline", **PCT1},
    {"name": "Lift", "id": "lift_all", "type": "numeric", "format": {"specifier": ".2f"}},
]
TABLE_STYLE = dict(style_cell={"fontSize": "12px", "padding": "2px 6px", "textAlign": "right"},
                   style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"})
BTN = {"marginRight": "6px", "padding": "4px 10px"}
ROW = {"display": "flex", "alignItems": "center", "gap": "6px", "margin": "6px 0", "flexWrap": "wrap"}
NOTE = {"fontSize": "12px", "color": "#555", "margin": "4px 0"}


# ----------------------------------------------------------------------------- layout
def layout() -> html.Div:
    return html.Div([
        html.Div(style=ROW, children=[
            html.B("Kalshi series:"),
            dcc.Dropdown(id="kalshi-keys", multi=True, clearable=False, value=[DEFAULT_KALSHI_TICKER],
                         options=[{"label": series_label(k, True), "value": k} for k in SERIES_KEYS], style={"width": "560px"}),
            dcc.Checklist(id="d-terms", value=["d"], options=[{"label": " show R contracts as P(D) = 1 − price", "value": "d"}],
                          style={"fontSize": "13px"}),
        ]),
        html.Div(style=ROW, children=[
            html.B("Stock:"),
            dcc.Input(id="ticker-input", value=DEFAULT_TICKER, debounce=True, placeholder="ticker, then Enter",
                      style={"width": "110px", "textTransform": "uppercase"}),
            html.Button("Go", id="go", n_clicks=0, style=BTN),
            html.B("Sector:"),
            dcc.Dropdown(id="sector", options=[{"label": "All sectors", "value": ALL_SECTORS}] + [{"label": s, "value": s} for s in SECTORS],
                         value=ALL_SECTORS, clearable=False, style={"width": "240px"}),
            html.Button("◀ Prev", id="prev-t", n_clicks=0, style=BTN),
            html.Button("Next ▶", id="next-t", n_clicks=0, style=BTN),
            html.Span(id="ticker-pos", style={"color": "#555"}),
        ]),
        html.Div(id="ticker-info", style={"fontSize": "14px", "fontWeight": "bold", "margin": "2px 0"}),
        html.Div(id="a-status", style=NOTE),
        dcc.Graph(id="fig-a", config={"displaylogo": False}),

        html.Div("Closeness — daily changes aligned at 16:00 ET; Kalshi in probability points, stock in returns. "
                 "Positive lag = Kalshi moves first. Sorted by |corr same day|.", style=NOTE),
        dash_table.DataTable(id="closeness-table", columns=CLOSENESS_COLS, data=[], **TABLE_STYLE),

        html.Hr(),
        html.Div(style=ROW, children=[
            html.B("Event windows for:"),
            dcc.Dropdown(id="ev-series", clearable=False, style={"width": "300px"}),
            html.Span("Kalshi move threshold ±", style={"marginLeft": "12px"}),
            dcc.Input(id="k-thr", type="number", value=3, min=0.1, step=0.5, debounce=True, style={"width": "60px"}), html.Span("%"),
            dcc.RadioItems(id="k-mode", value="pct", inline=True,
                           options=[{"label": "relative return", "value": "pct"}, {"label": "probability points", "value": "pp"}],
                           style={"marginLeft": "6px"}),
            html.Span("Stock |r| ≥", style={"marginLeft": "12px"}),
            dcc.Input(id="s-thr", type="number", value=3, min=0.1, step=0.5, debounce=True, style={"width": "60px"}), html.Span("%"),
            html.Span("Window ±", style={"marginLeft": "12px"}),
            dcc.Input(id="win-k", type="number", value=2, min=1, max=10, step=1, debounce=True, style={"width": "50px"}),
            html.Span("trading days"),
        ]),
        html.Pre(id="ev-text", style={"fontSize": "12px", "color": "#333", "margin": "4px 0", "whiteSpace": "pre-wrap"}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="fig-ev", config={"displaylogo": False}, style={"flex": "2", "minWidth": 0}),
            html.Div(style={"flex": "3", "minWidth": 0}, children=[
                html.Div("Per offset: hit rate = share of events whose stock |r| exceeds the threshold on that day; "
                         "lift = all-events hit rate / baseline", style=NOTE),
                dash_table.DataTable(id="ev-summary", columns=SUMMARY_COLS, data=[], **TABLE_STYLE),
            ]),
        ]),
        html.Div("Events — click a row: panel A zooms to the event, panel B jumps to that day. "
                 "Highlighted cells are stock returns beyond the threshold.", style=NOTE),
        dash_table.DataTable(id="events-table", columns=[], data=[], page_size=10, page_action="native",
                             sort_action="native", **TABLE_STYLE),
        dcc.Store(id="events-store", data={"rows": []}),

        html.Hr(),
        html.Div(style=ROW, children=[
            html.B("Intraday contract:"),
            dcc.Dropdown(id="intra-ticker", options=[{"label": k, "value": k} for k in KALSHI_TICKERS],
                         value=DEFAULT_KALSHI_TICKER, clearable=False, style={"width": "220px"}),
            html.Button("◀ Prev day", id="prev-d", n_clicks=0, style=BTN),
            dcc.DatePickerSingle(id="day", date=default_day(DEFAULT_KALSHI_TICKER), min_date_allowed=DAY_MIN,
                                 max_date_allowed=DAY_MAX, display_format="YYYY-MM-DD"),
            html.Button("Next day ▶", id="next-d", n_clicks=0, style=BTN),
            html.Button("◀ Prev day with trades", id="prev-td", n_clicks=0, style=BTN),
            html.Button("Next day with trades ▶", id="next-td", n_clicks=0, style=BTN),
            html.Span(id="day-info", style={"color": "#555"}),
            html.Span("(wheel = zoom, drag = pan; click a day in panel A to jump)", style={"fontSize": "12px", "color": "#888"}),
        ]),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="fig-b", config={"scrollZoom": True, "displaylogo": False}, style={"flex": "3", "minWidth": 0}),
            html.Div(style={"flex": "2", "minWidth": 0}, children=[
                html.Div("Trades — filtered to the visible range; click a row to centre the chart on it", style=NOTE),
                dash_table.DataTable(
                    id="trades-table", columns=TRADE_COLS, data=[], page_size=15, page_action="native",
                    sort_action="native", **TABLE_STYLE,
                    style_data_conditional=[
                        {"if": {"filter_query": "{taker_side} = yes", "column_id": "taker_side"}, "color": "#2ca02c"},
                        {"if": {"filter_query": "{taker_side} = no", "column_id": "taker_side"}, "color": "#d62728"},
                    ]),
            ]),
        ]),
        dcc.Store(id="ticker-store", data=DEFAULT_TICKER),
        dcc.Store(id="day-trades", data={"day": None, "rows": []}),
    ])


# ----------------------------------------------------------------------------- callbacks
def register(app) -> None:
    @app.callback(
        Output("ticker-store", "data"), Output("ticker-pos", "children"), Output("ticker-input", "value"),
        Output("ticker-info", "children"),
        Input("go", "n_clicks"), Input("ticker-input", "n_submit"),
        Input("prev-t", "n_clicks"), Input("next-t", "n_clicks"), Input("sector", "value"),
        State("ticker-input", "value"), State("ticker-store", "data"),
    )
    def nav_ticker(_go, _sub, _prev, _next, sector, typed, current):
        trig = ctx.triggered_id
        cur = normalize_ticker(current or DEFAULT_TICKER)
        lst = universe_list(sector)
        if trig in ("go", "ticker-input"):
            new = normalize_ticker(typed) or cur
        elif trig in ("prev-t", "next-t"):
            base = lst if cur in lst else sorted(set(lst) | {cur})
            new = base[(base.index(cur) + (1 if trig == "next-t" else -1)) % len(base)]
        elif trig == "sector":
            new = cur if cur in lst else (lst[0] if lst else cur)
        else:
            new = cur
        scope = "S&P 500" if not sector or sector == ALL_SECTORS else sector
        pos = (f"{lst.index(new) + 1} / {len(lst)} in {scope} (sorted by ticker)" if new in lst
               else f"{new} is not in {scope}; loading it anyway")
        info = universe_info(new)
        label = f"{new}  ·  {info['name']}  ·  {info['sector']}" if info["name"] else f"{new}  ·  not in the S&P 500 list"
        return new, pos, new, label

    @app.callback(Output("fig-a", "figure"), Output("a-status", "children"),
                  Input("ticker-store", "data"), Input("kalshi-keys", "value"), Input("d-terms", "value"))
    def update_fig_a(ticker, keys, dterms):
        fig, err = build_fig_a(ticker, keys or [], bool(dterms))
        msg = (f"⚠ {ticker}: {err}" if err else
               f"{ticker}: daily bars from 2024-06 (yfinance, backward-adjusted). Kalshi: daily mid = (bid + ask) / 2 at the "
               f"daily close; event P(D) = mean of D mid and 1 − R mid.")
        return fig, msg

    @app.callback(Output("closeness-table", "data"),
                  Input("ticker-store", "data"), Input("kalshi-keys", "value"), Input("d-terms", "value"),
                  Input("k-thr", "value"), Input("k-mode", "value"), Input("s-thr", "value"))
    def update_closeness(ticker, keys, dterms, k_thr, mode, s_thr):
        if not keys:
            return []
        df = closeness_table(ticker, keys, k_thr=float(k_thr or 3) / 100, s_thr=float(s_thr or 3) / 100,
                             mode=mode, d_terms=bool(dterms))
        return df.round(4).to_dict("records")

    @app.callback(Output("ev-series", "options"), Output("ev-series", "value"),
                  Input("kalshi-keys", "value"), Input("d-terms", "value"), State("ev-series", "value"))
    def sync_event_series(keys, dterms, cur):
        keys = keys or [DEFAULT_KALSHI_TICKER]
        opts = [{"label": series_label(k, bool(dterms)), "value": k} for k in keys]
        return opts, (cur if cur in keys else keys[0])

    @app.callback(
        Output("ev-text", "children"), Output("fig-ev", "figure"), Output("ev-summary", "data"),
        Output("events-table", "columns"), Output("events-table", "data"), Output("events-table", "page_current"),
        Output("events-table", "style_data_conditional"), Output("events-store", "data"),
        Input("ticker-store", "data"), Input("ev-series", "value"), Input("d-terms", "value"),
        Input("k-thr", "value"), Input("k-mode", "value"), Input("s-thr", "value"), Input("win-k", "value"),
    )
    def update_events(ticker, key, dterms, k_thr, mode, s_thr, K):
        if not key:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        try:
            k_thr, s_thr, K = float(k_thr or 3) / 100, float(s_thr or 3) / 100, int(K or 2)
            res = run_event_study(ticker, key, k_thr=k_thr, s_thr=s_thr, K=K, mode=mode, d_terms=bool(dterms))
        except Exception as e:  # noqa: BLE001
            return f"⚠ {ticker}: {e}", go.Figure(), [], [], [], 0, [], {"rows": []}
        ev = res["events"].copy()
        rows = []
        if not ev.empty:
            ev["id"] = ev["date"]
            ev["any_big_in_window"] = ev["any_big_in_window"].map({True: "yes", False: ""})
            rows = ev.to_dict("records")
        styles = []
        for o in range(-K, K + 1):
            c = f"ret_{o:+d}"
            styles.append({"if": {"filter_query": f"{{{c}}} >= {s_thr} || {{{c}}} <= {-s_thr}", "column_id": c},
                           "backgroundColor": "#fff1c2", "fontWeight": "bold"})
        styles += [{"if": {"filter_query": "{direction} = up", "column_id": "direction"}, "color": C_UP},
                   {"if": {"filter_query": "{direction} = down", "column_id": "direction"}, "color": C_DOWN}]
        return (describe(res), build_fig_events(res["summary"], s_thr), res["summary"].round(4).to_dict("records"),
                event_table_columns(K), rows, 0, styles, {"rows": rows})

    @app.callback(Output("fig-a", "figure", allow_duplicate=True),
                  Input("events-table", "active_cell"), State("events-store", "data"), prevent_initial_call=True)
    def zoom_fig_a_to_event(cell, store):
        if not cell or not store or not cell.get("row_id"):
            return no_update
        d = pd.Timestamp(cell["row_id"])
        patch = Patch()
        patch["layout"]["xaxis"]["range"] = [str(d - pd.Timedelta(days=45)), str(d + pd.Timedelta(days=45))]
        patch["layout"]["uirevision"] = f"evt-{d:%Y%m%d}"
        return patch

    @app.callback(
        Output("day", "date"),
        Input("fig-a", "clickData"), Input("prev-d", "n_clicks"), Input("next-d", "n_clicks"),
        Input("prev-td", "n_clicks"), Input("next-td", "n_clicks"), Input("events-table", "active_cell"),
        Input("intra-ticker", "value"),
        State("day", "date"), prevent_initial_call=True,
    )
    def nav_day(click, _p, _n, _pt, _nt, ev_cell, kalshi_ticker, cur):
        trig = ctx.triggered_id
        cur = str(cur)[:10]
        days, tdays = all_days(kalshi_ticker), trade_days(kalshi_ticker)
        if trig == "intra-ticker":
            return default_day(kalshi_ticker)
        if trig == "events-table":
            d = (ev_cell or {}).get("row_id")
            return d if d and days[0] <= d <= days[-1] else no_update
        if trig == "fig-a":
            if not click:
                return no_update
            d = str(click["points"][0]["x"])[:10]
            return d if days[0] <= d <= days[-1] else no_update
        if trig in ("prev-d", "next-d"):
            d = (date.fromisoformat(cur) + timedelta(days=1 if trig == "next-d" else -1)).isoformat()
            return min(max(d, days[0]), days[-1])
        if trig == "next-td":
            i = bisect.bisect_right(tdays, cur)
            return tdays[i] if i < len(tdays) else no_update
        if trig == "prev-td":
            i = bisect.bisect_left(tdays, cur) - 1
            return tdays[i] if i >= 0 else no_update
        return no_update

    @app.callback(
        Output("fig-b", "figure"), Output("day-trades", "data"),
        Output("trades-table", "data"), Output("trades-table", "page_current"), Output("day-info", "children"),
        Input("day", "date"), Input("intra-ticker", "value"),
    )
    def update_day(day, kalshi_ticker):
        day = str(day)[:10]
        fig, m, t = build_fig_b(kalshi_ticker, day)
        rows = table_rows(t)
        n_contracts = float(t["count"].sum()) if not t.empty else 0.0
        return fig, {"day": day, "rows": rows}, rows, 0, \
            f"{kalshi_ticker} {day}: {len(m)} one-minute bars, {len(t)} trades, {n_contracts:,.0f} contracts"

    @app.callback(
        Output("trades-table", "data", allow_duplicate=True), Output("trades-table", "page_current", allow_duplicate=True),
        Input("fig-b", "relayoutData"), State("day-trades", "data"), State("day", "date"), prevent_initial_call=True,
    )
    def filter_table_by_view(relayout, store, day):
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

    @app.callback(Output("fig-b", "figure", allow_duplicate=True),
                  Input("trades-table", "active_cell"), State("day-trades", "data"), prevent_initial_call=True)
    def center_on_trade(cell, store):
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

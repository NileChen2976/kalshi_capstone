# -*- coding: utf-8 -*-
"""
event_study.py -- event-window statistics: big moves in a Kalshi contract vs. large moves in a stock.

Definitions (all configurable):
  * Kalshi return: change of the contract mid between two consecutive stock trading days at
    16:00 ET.  mode="pct": mid_t / mid_{t-1} - 1 (relative, default);  mode="pp": mid_t - mid_{t-1}
    (probability points, 0.03 = 3 points).  The 16:00 ET mid comes from an as-of lookup on a
    timeline built from the 1-min candles plus the daily 24:00 closes, so the Kalshi return and
    the stock close-to-close return cover the same interval.
  * Event: return >= +k_thr -> "up", <= -k_thr -> "down".
  * Big stock move: |daily hfq close return| >= s_thr.
  * Event window: K trading days on each side of the event day, offset = -K..+K.

Outputs:
  events  : one row per event, with the stock return / big-move flag at every offset
  summary : one row per offset, hit rates (up / down / all) vs the unconditional baseline,
            plus the mean signed stock return by event direction
"""
from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np
import pandas as pd

from kalshi_data import DATA_DIR, DEFAULT_KALSHI_TICKER, kalshi_daily
from stock_data import get_stock_daily


# --------------------------------------------------------------------------- Kalshi 16:00 ET series
@lru_cache(maxsize=8)
def kalshi_timeline(kalshi_ticker: str = DEFAULT_KALSHI_TICKER) -> pd.DataFrame:
    """Every known mid observation (1-min bar ends + daily 24:00 closes), ascending. Columns: t, mid."""
    d = kalshi_daily(kalshi_ticker)
    parts = [pd.DataFrame({"t": d["date"] + pd.Timedelta(days=1), "mid": d["mid_close"]})]
    for f in sorted(glob.glob(str(DATA_DIR / "1min" / "*.parquet"))):
        m = pd.read_parquet(f, columns=["ticker", "period_end_et", "mid_close"])
        m = m[m["ticker"] == kalshi_ticker]
        if m.empty:
            continue
        parts.append(pd.DataFrame({"t": m["period_end_et"].dt.tz_localize(None), "mid": m["mid_close"]}))
    tl = pd.concat(parts).dropna().sort_values("t").drop_duplicates("t", keep="last")
    return tl.reset_index(drop=True)


def kalshi_close_on(dates: pd.Series, kalshi_ticker: str, time_et: str = "16:00") -> pd.Series:
    """Kalshi mid at time_et on each given date (as-of: last observation at or before that time)."""
    tl = kalshi_timeline(kalshi_ticker)
    hh, mm = map(int, time_et.split(":"))
    q = pd.DataFrame({"t": pd.to_datetime(dates) + pd.Timedelta(hours=hh, minutes=mm)})
    out = pd.merge_asof(q.sort_values("t"), tl, on="t", direction="backward")
    return pd.Series(out["mid"].to_numpy(), index=q.index)


# --------------------------------------------------------------------------- main
def run_event_study(
    ticker: str,
    kalshi_ticker: str = DEFAULT_KALSHI_TICKER,
    k_thr: float = 0.03,
    s_thr: float = 0.03,
    K: int = 2,
    mode: str = "pct",
    time_et: str = "16:00",
) -> dict:
    s = get_stock_daily(ticker)[["date", "close_hfq"]].copy()
    s = s.sort_values("date").reset_index(drop=True)
    s["kalshi_mid"] = kalshi_close_on(s["date"], kalshi_ticker, time_et)
    s = s.dropna(subset=["kalshi_mid"]).reset_index(drop=True)      # keep trading days after Kalshi data starts
    if len(s) < 2 * K + 3:
        raise ValueError(f"{ticker}: not enough trading days overlapping with {kalshi_ticker}")

    s["stock_ret"] = s["close_hfq"].pct_change()
    s["kalshi_ret"] = s["kalshi_mid"].pct_change() if mode == "pct" else s["kalshi_mid"].diff()
    s["stock_big"] = s["stock_ret"].abs() >= s_thr
    s["direction"] = np.where(s["kalshi_ret"] >= k_thr, "up", np.where(s["kalshi_ret"] <= -k_thr, "down", ""))

    offsets = list(range(-K, K + 1))
    baseline = float(s["stock_big"].iloc[1:].mean())            # unconditional daily probability of a big move
    win_any = s["stock_big"].astype(int).rolling(2 * K + 1, center=True).max().dropna()
    baseline_any = float(win_any.mean()) if len(win_any) else np.nan   # random window of the same length

    ev_idx = [i for i in s.index[s["direction"] != ""].tolist() if i - K >= 1 and i + K < len(s)]  # full window only
    rows = []
    for i in ev_idx:
        r = {
            "date": s.at[i, "date"].strftime("%Y-%m-%d"),
            "direction": s.at[i, "direction"],
            "kalshi_prev": round(float(s.at[i - 1, "kalshi_mid"]), 4),
            "kalshi_mid": round(float(s.at[i, "kalshi_mid"]), 4),
            "kalshi_ret": round(float(s.at[i, "kalshi_ret"]), 4),
        }
        big_any, max_abs = False, 0.0
        for o in offsets:
            ret = float(s.at[i + o, "stock_ret"])
            big = bool(s.at[i + o, "stock_big"])
            r[f"ret_{o:+d}"] = round(ret, 4)
            r[f"big_{o:+d}"] = big
            big_any |= big
            max_abs = max(max_abs, abs(ret))
        r["any_big_in_window"] = big_any
        r["max_abs_ret"] = round(max_abs, 4)
        rows.append(r)
    events = pd.DataFrame(rows)

    srows = []
    for o in offsets:
        rec = {"offset": o}
        for tag in ("up", "down", "all"):
            sub = events if tag == "all" or events.empty else events[events["direction"] == tag]
            n = len(sub)
            rec[f"n_{tag}"] = n
            rec[f"hit_{tag}"] = float(sub[f"big_{o:+d}"].mean()) if n else np.nan
            rec[f"mean_ret_{tag}"] = float(sub[f"ret_{o:+d}"].mean()) if n else np.nan
        rec["baseline"] = baseline
        rec["lift_all"] = rec["hit_all"] / baseline if baseline and not np.isnan(rec["hit_all"]) else np.nan
        srows.append(rec)
    summary = pd.DataFrame(srows)

    n_up = int((events["direction"] == "up").sum()) if len(events) else 0
    n_down = int((events["direction"] == "down").sum()) if len(events) else 0
    any_rate = float(events["any_big_in_window"].mean()) if len(events) else np.nan
    return {
        "ticker": ticker, "kalshi_ticker": kalshi_ticker,
        "params": dict(k_thr=k_thr, s_thr=s_thr, K=K, mode=mode, time_et=time_et),
        "n_days": int(len(s) - 1), "n_up": n_up, "n_down": n_down,
        "baseline": baseline, "baseline_any": baseline_any, "any_rate": any_rate,
        "events": events, "summary": summary, "aligned": s,
    }


def describe(res: dict) -> str:
    p = res["params"]
    unit = "relative return" if p["mode"] == "pct" else "probability points"
    return "\n".join([
        f"{res['ticker']} vs {res['kalshi_ticker']}  |  {res['n_days']} trading days  |  "
        f"Kalshi event threshold ±{p['k_thr']:.1%} ({unit})  |  big stock move |r| ≥ {p['s_thr']:.1%}  |  window ±{p['K']} days",
        f"Events: {res['n_up']} up, {res['n_down']} down  |  unconditional daily probability of a big stock move {res['baseline']:.1%}",
        f"At least one big move inside the window: event windows {res['any_rate']:.1%}  vs  random windows {res['baseline_any']:.1%}",
    ])


if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    ktk = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_KALSHI_TICKER
    res = run_event_study(tk, ktk)
    pd.set_option("display.width", 200)
    print(describe(res))
    print()
    print(res["summary"].round(4).to_string(index=False))
    print()
    K = res["params"]["K"]
    cols = ["date", "direction", "kalshi_prev", "kalshi_mid", "kalshi_ret"] + \
           [f"ret_{o:+d}" for o in range(-K, K + 1)] + ["any_big_in_window"]
    print(res["events"][cols].to_string(index=False))

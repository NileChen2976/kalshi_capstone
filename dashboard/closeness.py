# -*- coding: utf-8 -*-
"""
closeness.py -- how closely does each Kalshi series track a stock? (used by the Stocks x Kalshi tab)

All series are aligned at 16:00 ET (see event_study.aligned_frame). Kalshi changes are in
probability points, stock changes are simple returns; correlations are scale-free.

Columns of closeness_table():
    series / label          key ("CONTROLS-2026-D", "EVENT:CONTROLS-2026", ...) and its display label
    n_days                  overlapping trading days
    corr_same_day           corr(dKalshi_t, stock_ret_t)
    corr_kalshi_leads_1d    corr(dKalshi_{t-1}, stock_ret_t)   (Kalshi moved yesterday, stock today)
    corr_stock_leads_1d     corr(dKalshi_t, stock_ret_{t-1})   (stock moved yesterday, Kalshi today)
    best_lag / best_lag_corr    lag in -3..3 with the largest |corr| (positive lag = Kalshi leads)
    n_events / hit0 / lift0     event-window statistics at offset 0 for the given thresholds

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_study import run_event_study, series_label


def closeness_table(stock: str, keys: list[str], k_thr: float = 0.03, s_thr: float = 0.03,
                    mode: str = "pct", d_terms: bool = True, max_lag: int = 3) -> pd.DataFrame:
    rows = []
    for key in keys:
        try:
            res = run_event_study(stock, key, k_thr=k_thr, s_thr=s_thr, K=1, mode=mode, d_terms=d_terms)
        except Exception as e:  # noqa: BLE001
            rows.append({"series": key, "label": series_label(key, d_terms), "error": str(e)})
            continue
        a = res["aligned"]
        x = a["kalshi_mid"].diff()            # probability points regardless of the event-study mode
        y = a["stock_ret"]
        lags = {k: x.shift(k).corr(y) for k in range(-max_lag, max_lag + 1)}
        best = max(lags.items(), key=lambda kv: -1 if np.isnan(kv[1]) else abs(kv[1]))
        row0 = res["summary"].loc[res["summary"]["offset"] == 0].iloc[0]
        rows.append({
            "series": key, "label": res["kalshi_label"], "n_days": res["n_days"],
            "corr_same_day": lags[0], "corr_kalshi_leads_1d": lags[1], "corr_stock_leads_1d": lags[-1],
            "best_lag": best[0], "best_lag_corr": best[1],
            "n_events": res["n_up"] + res["n_down"], "hit0": row0["hit_all"], "lift0": row0["lift_all"],
            "baseline": res["baseline"],
        })
    df = pd.DataFrame(rows)
    if "corr_same_day" in df:
        df = df.sort_values("corr_same_day", key=lambda s: s.abs(), ascending=False, na_position="last")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from pair_data import events
    keys = [t for ev in events().values() for t in ev.values()] + [f"EVENT:{e}" for e in events()]
    pd.set_option("display.width", 200)
    print(closeness_table("AAPL", keys).round(3).to_string(index=False))

"""
Score past forecasts once the actuals arrive.

    python track_predictions.py

Roadmap Phase 7.4. predict_next_day.py writes a row the evening before, with
`actual_mw` blank because the answer does not exist yet. Grid-India publishes
it two days later. This fills in the blanks and reports live operational
performance.

Why this is the number that matters
-----------------------------------
Every other figure in this project is a backtest. A backtest reconstructs what
you *would* have known, and however careful the reconstruction, it is still
assembled after the fact by someone who already knows the answer. This file
accumulates forecasts that were committed to disk before the outcome existed.
When the paper claims an operational MAPE, this is the only table that cannot
be accused of hindsight.

It will be a small sample for a long time, so coverage and MAE here are
reported with their sample size and never rounded into a headline claim until
the log covers a March-June heat season.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import PEAK_TARGET, PREDICTIONS_LOG, RAW_PEAK, RESULTS, SUMMER_MONTHS


def main():
    if not PREDICTIONS_LOG.exists():
        print(f"no {PREDICTIONS_LOG} yet - nothing issued. Run "
              f"predict_next_day.py in the evening.")
        return

    log = pd.read_csv(PREDICTIONS_LOG)
    log["target_date"] = pd.to_datetime(log["target_date"])
    actual = (pd.read_csv(RAW_PEAK, parse_dates=["date"])
              .dropna(subset=[PEAK_TARGET])
              .drop_duplicates("date", keep="last")
              .set_index("date")[PEAK_TARGET])

    # fill actuals that have landed since the last run
    log["actual_mw"] = pd.to_numeric(log.get("actual_mw"), errors="coerce")
    landed = actual.reindex(log["target_date"]).to_numpy()
    newly = log["actual_mw"].isna() & pd.notna(landed)
    log.loc[newly, "actual_mw"] = landed[newly.to_numpy()]
    log["error_mw"] = (log["predicted_mw"] - log["actual_mw"]).round(1)
    log.to_csv(PREDICTIONS_LOG, index=False)

    n_new = int(newly.sum())
    scored = log.dropna(subset=["actual_mw"]).copy()
    pending = log["actual_mw"].isna().sum()

    print(f"=== live forecast performance ===")
    print(f"  issued      {len(log)}")
    print(f"  scored      {len(scored)}  ({n_new} newly matured this run)")
    print(f"  awaiting    {pending}  (Grid-India publishes ~2 days behind)")

    if scored.empty:
        print("\n  nothing scored yet - check back once actuals land.")
        return

    e = scored["error_mw"].to_numpy(float)
    y = scored["actual_mw"].to_numpy(float)
    mae = float(np.abs(e).mean())
    mape = float(np.mean(np.abs(e / y)) * 100)
    bias = float(e.mean())

    print(f"\n  span        {scored['target_date'].min():%Y-%m-%d} -> "
          f"{scored['target_date'].max():%Y-%m-%d}")
    print(f"  MAE         {mae:,.1f} MW")
    print(f"  MAPE        {mape:.2f}%")
    print(f"  bias        {bias:+,.1f} MW")
    print(f"  worst       {np.abs(e).max():,.0f} MW")

    cov = {}
    for lv in (90, 95):
        lo, hi = scored.get(f"lo{lv}"), scored.get(f"hi{lv}")
        if lo is None or lo.isna().all():
            continue
        inside = (y >= lo.to_numpy(float)) & (y <= hi.to_numpy(float))
        cov[lv] = float(inside.mean())
        print(f"  {lv}% PI     covers {inside.mean() * 100:.1f}% "
              f"(nominal {lv}%), mean width "
              f"{(hi - lo).mean():,.0f} MW")

    summer = scored[scored["target_date"].dt.month.isin(SUMMER_MONTHS)]
    print(f"\n  March-June in sample: {len(summer)} days", end="")
    if len(summer) < 20:
        print(" - too few to judge heat-season behaviour, which is where")
        print("  this model is hardest pressed. Treat the figures above as")
        print("  provisional until a real summer is in the log.")
    else:
        se = summer["error_mw"].to_numpy(float)
        print(f"  MAE {np.abs(se).mean():,.1f} MW  "
              f"bias {se.mean():+,.1f} MW")

    print(f"\n  reference: backtested operational expectation is ~524 MW MAE")
    print(f"  / ~3.0% MAPE (results/operational_gap.json). A live figure far")
    print(f"  from that is a signal, not noise - check the weather logger and")
    print(f"  whether H still holds before trusting the model.")

    recent = scored.nlargest(min(5, len(scored)), "target_date")
    print(f"\n  most recent:")
    print(f"  {'date':<12}{'pred':>8}{'actual':>9}{'err':>8}")
    for r in recent.sort_values("target_date").itertuples():
        print(f"  {r.target_date:%Y-%m-%d}{r.predicted_mw:>8.0f}"
              f"{r.actual_mw:>9.0f}{r.error_mw:>+8.0f}")

    out = {
        "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_issued": int(len(log)), "n_scored": int(len(scored)),
        "n_pending": int(pending),
        "span": [str(scored["target_date"].min().date()),
                 str(scored["target_date"].max().date())],
        "mae_mw": round(mae, 1), "mape_pct": round(mape, 3),
        "bias_mw": round(bias, 1),
        "worst_abs_error_mw": round(float(np.abs(e).max()), 1),
        "interval_coverage": {f"{k}%": round(v, 4) for k, v in cov.items()},
        "n_summer_days": int(len(summer)),
        "is_provisional": bool(len(summer) < 20),
        "backtest_reference_mae_mw": 524,
        "note": ("live, pre-committed forecasts - the only figures here not "
                 "produced with hindsight"),
    }
    path = RESULTS / "live_performance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

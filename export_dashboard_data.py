"""
Export every number the dashboard renders into one JSON file.

    python export_dashboard_data.py

Runs at the end of both daily workflows. The dashboard is a static page: it
fetches this file and renders it, so this script is the only place where
artefacts are read and reshaped.

Two rules, both load-bearing:

1. **Never invent a number.** If a source artefact is missing or a field is
   absent, the key is omitted and the dashboard renders that panel's empty
   state. A dashboard that silently substitutes a plausible figure is worse
   than one with a gap in it.
2. **Quirks are resolved here, not in the UI.** The interim-model row and the
   `observed_filled_from_forecast` string are normalised into plain fields so
   no React component has to know about CSV sentinels.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dashboard" / "public" / "data" / "dashboard.json"
HISTORY_DAYS = 400
LEDGER_ROWS = 14

import sys
sys.path.insert(0, str(ROOT))
from config import MASTER, PEAK_TARGET, PREDICTIONS_LOG, RESULTS


def jload(name):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def clean(x):
    """NaN/NaT -> None so the JSON carries null, never the string 'nan'."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return x


def humanise_fill(raw) -> str | None:
    """'2026-11-11(lead0)' -> a sentence. Returns None when absent."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    s = str(raw).strip()
    if s in ("", "nan"):
        return None
    days = re.findall(r"\d{4}-\d{2}-\d{2}", s)
    if not days:
        return f"Yesterday's weather was taken from a forecast ({s})."
    when = ", ".join(datetime.strptime(d, "%Y-%m-%d").strftime("%-d %b")
                     for d in days)
    return (f"Weather for {when} came from a forecast — that day has not "
            f"reached the observational archive yet.")


def main():
    out: dict = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    # ---- demand history --------------------------------------------------
    master = pd.read_csv(MASTER, parse_dates=["date"])
    obs = master.dropna(subset=[PEAK_TARGET]).sort_values("date")
    hist = obs.tail(HISTORY_DAYS)
    out["history"] = [{"date": d.strftime("%Y-%m-%d"), "mw": round(float(v), 1)}
                      for d, v in zip(hist["date"], hist[PEAK_TARGET])]
    last = obs.iloc[-1]
    out["latest_actual"] = {"date": last["date"].strftime("%Y-%m-%d"),
                            "mw": round(float(last[PEAK_TARGET]), 1)}
    rec = obs.loc[obs[PEAK_TARGET].idxmax()]
    out["record"] = {"mw": round(float(rec[PEAK_TARGET]), 1),
                     "date": rec["date"].strftime("%Y-%m-%d")}
    out["dataset"] = {"rows": int(len(master)),
                      "start": master["date"].min().strftime("%Y-%m-%d"),
                      "end": master["date"].max().strftime("%Y-%m-%d")}

    # ---- forecast + ledger ----------------------------------------------
    if PREDICTIONS_LOG.exists():
        p = pd.read_csv(PREDICTIONS_LOG)
        p["target_date"] = pd.to_datetime(p["target_date"])
        for c in ("predicted_mw", "actual_mw", "error_mw", "lo90", "hi90",
                  "lo95", "hi95"):
            if c in p.columns:
                p[c] = pd.to_numeric(p[c], errors="coerce")
        p = p.sort_values("target_date")

        # The hero is the newest row that has NOT yet been scored - i.e. a
        # forecast about the future. If every row is scored, fall back to the
        # newest row so the page still has something to show.
        future = p[p["actual_mw"].isna()]
        row = (future.iloc[-1] if len(future) else p.iloc[-1])
        warns = []
        if not bool(row.get("model_is_champion")):
            warns.append("Served by the interim model, not the research "
                         "champion.")
        nf = row.get("features_nan")
        if nf is not None and str(nf).strip() not in ("", "nan"):
            warns.append(f"Some inputs were unavailable at forecast time: {nf}.")
        fill = humanise_fill(row.get("observed_filled_from_forecast"))
        if fill:
            warns.append(fill)

        out["forecast"] = {
            "target_date": row["target_date"].strftime("%Y-%m-%d"),
            "issued_at": clean(row.get("issued_at")),
            "mw": round(float(row["predicted_mw"]), 1),
            "lo90": clean(round(float(row["lo90"]), 1)) if pd.notna(row.get("lo90")) else None,
            "hi90": clean(round(float(row["hi90"]), 1)) if pd.notna(row.get("hi90")) else None,
            "lo95": clean(round(float(row["lo95"]), 1)) if pd.notna(row.get("lo95")) else None,
            "hi95": clean(round(float(row["hi95"]), 1)) if pd.notna(row.get("hi95")) else None,
            "model_version": clean(row.get("model_version")),
            "is_champion": bool(row.get("model_is_champion")),
            "weather_source": clean(row.get("weather_source")),
            "H": int(row["H"]) if pd.notna(row.get("H")) else None,
            "warnings": warns,
        }

        led = []
        for _, r in p.tail(LEDGER_ROWS).iloc[::-1].iterrows():
            act = r.get("actual_mw")
            inside = None
            if pd.notna(act) and pd.notna(r.get("lo90")):
                inside = bool(float(r["lo90"]) <= float(act) <= float(r["hi90"]))
            led.append({
                "date": r["target_date"].strftime("%Y-%m-%d"),
                "pred": round(float(r["predicted_mw"]), 1),
                "actual": round(float(act), 1) if pd.notna(act) else None,
                "error": (round(float(r["predicted_mw"]) - float(act), 1)
                          if pd.notna(act) else None),
                "lo90": round(float(r["lo90"]), 1) if pd.notna(r.get("lo90")) else None,
                "hi90": round(float(r["hi90"]), 1) if pd.notna(r.get("hi90")) else None,
                "inside90": inside,
                "is_champion": bool(r.get("model_is_champion")),
            })
        out["ledger"] = led

    # ---- live scorecard --------------------------------------------------
    if (lv := jload("live_performance.json")):
        out["live"] = lv

    # ---- backtest comparison --------------------------------------------
    # Bars are NOT a like-for-like sample: persistence is scored on the Phase 6
    # Regime A set (n=1,085) and the champion on the Phase 7 shared set
    # (n=1,066). Each bar therefore carries its own n and basis so the UI can
    # say so rather than implying one controlled comparison.
    bt = {}
    bench = jload("phase6_benchmark.json")
    if bench:
        np_ = bench.get("results", {}).get("A", {}).get("naive_persistence", {}).get("overall")
        if np_:
            bt["persistence"] = {"mae": round(np_["mae"], 1), "n": np_["n_samples"],
                                 "basis": "Yesterday's peak repeated · hindsight weather"}
    champ = json.loads((ROOT / "models" / "champion_config.json").read_text())
    vp = champ.get("validation_performance", {})
    if vp.get("regime_b_mae"):
        bt["champion_forecast_weather"] = {
            "mae": round(vp["regime_b_mae"], 1), "n": 1066,
            "basis": "This model · real day-ahead weather forecasts"}
    if vp.get("regime_a_mae"):
        bt["champion_hindsight_weather"] = {
            "mae": round(vp["regime_a_mae"], 1), "n": 1066,
            "basis": "This model · weather known only afterwards"}
    if (ivj := jload("intervals.json")):
        wf = ivj.get("walk_forward", {})
        if wf.get("mae_mw"):
            bt["walk_forward"] = {"mae": round(wf["mae_mw"], 1),
                                  "n": wf.get("n_predictions"),
                                  "span": [wf.get("start"), wf.get("end")]}
        cov = ivj.get("coverage", {})
        if cov:
            out["coverage"] = {k: {"nominal": v["nominal"],
                                   "realised": round(v["realised_coverage"], 4)}
                               for k, v in cov.items()}
    if bt:
        out["backtest"] = bt

    # ---- drivers ---------------------------------------------------------
    if (fi := jload("feature_importance.json")):
        LAB = {"weather": "Weather", "demand_lags": "Recent demand",
               "calendar_trend": "Calendar & season"}
        rows = [{"key": k, "label": LAB.get(k, k),
                 "dmae": round(v["permutation_dmae_mw"], 1)}
                for k, v in fi.get("by_block", {}).items()
                if v.get("permutation_dmae_mw") is not None]
        if rows:
            out["drivers"] = sorted(rows, key=lambda r: -r["dmae"])
            out["drivers_caveat"] = fi.get("method", {}).get("caveat")

    # ---- weather sensitivity --------------------------------------------
    gap = jload("operational_gap.json")
    if gap:
        s = gap.get("phase_5_4_sensitivity", {}).get("mw_per_degc_t_max")
        if s:
            out["sensitivity_mw_per_c"] = round(s, 1)
        if gap.get("caveats"):
            out["caveats"] = gap["caveats"]

    # ---- weak spots ------------------------------------------------------
    if (ea := jload("error_analysis.json")):
        LAB = {"overall": "All days", "working_day": "Working day",
               "weekend": "Weekend", "holiday": "Public holiday",
               "long_weekend": "Long weekend", "summer_mar_jun": "Summer (Mar–Jun)",
               "monsoon_oct_dec": "Monsoon (Oct–Dec)", "surge_top5pct": "Surge days (top 5%)",
               "hottest_decile": "Hottest 10% of days", "diwali_window": "Diwali",
               "pongal_window": "Pongal", "may_day": "May Day",
               "dussehra_ayudha": "Dussehra / Ayudha", "ramadan_eid": "Ramadan / Eid"}
        segs = []
        for k, v in ea.get("segments", {}).items():
            if k.startswith("year_") or k == "post_covid_2023":
                continue
            segs.append({"key": k, "label": LAB.get(k, k.replace("_", " ")),
                         "n": v["n"], "mae": round(v["mae"], 1),
                         "bias": round(v["bias"], 1),
                         "small": bool(v.get("anecdotal_n_lt_20"))})
        if segs:
            out["segments"] = sorted(segs, key=lambda s: -s["mae"])
        rd = ea.get("residual_diagnostics", {})
        if rd.get("lag1") is not None:
            out["residual_acf1"] = round(rd["lag1"], 3)

    # ---- publication timing ---------------------------------------------
    if (pt := jload("publish_time.json")):
        out["publish"] = {"H": pt.get("H"), "n_run_days": pt.get("n_run_days"),
                          "latest_run_clock": pt.get("run_clock", {}).get("latest")}

    out["model"] = {"name": champ.get("model_type"),
                    "feature_count": champ.get("feature_count"),
                    "frozen_at": champ.get("generated_at")}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    keys = ", ".join(sorted(out))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(json.dumps(out)) / 1024:.1f} KB · keys: {keys}")
    missing = [k for k in ("forecast", "live", "backtest", "drivers",
                           "segments", "coverage") if k not in out]
    if missing:
        print(f"  absent (panels will render empty states): {', '.join(missing)}")


if __name__ == "__main__":
    main()

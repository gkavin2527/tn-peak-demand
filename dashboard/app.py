"""
Tamil Nadu peak demand forecast dashboard.

    streamlit run dashboard/app.py

Roadmap Phase 7.3. Reads only committed artefacts - predictions_log.csv,
results/*.json, master.csv - and fits nothing. If a panel is empty, the
upstream job has not run; the dashboard is a window, not a pipeline.

Deliberately absent: the "peak hour window" panel promised in earlier drafts.
There is no hourly demand data in this project, so it was dropped at D6
rather than faked from a daily series.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (H, MASTER, PEAK_TARGET, PREDICTIONS_LOG, RESULTS,
                    SUMMER_MONTHS)

st.set_page_config(page_title="TN Peak Demand Forecast", page_icon="⚡",
                   layout="wide")


def load_json(name):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=600)
def load_master_df():
    df = pd.read_csv(MASTER, parse_dates=["date"])
    return df


@st.cache_data(ttl=60)
def load_preds():
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_LOG)
    df["target_date"] = pd.to_datetime(df["target_date"])
    for c in ("predicted_mw", "actual_mw", "error_mw", "lo90", "hi90",
              "lo95", "hi95"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("target_date")


master = load_master_df()
preds = load_preds()
gap = load_json("operational_gap.json")
live = load_json("live_performance.json")
iv = load_json("intervals.json")
pub = load_json("publish_time.json")

st.title("⚡ Tamil Nadu Next-Day Peak Demand")
st.caption(
    f"Day-ahead forecast with one-day demand-data latency (H={H}). "
    "Grid-India publishes each day's peak the following morning, so an "
    "evening forecast for tomorrow is built on demand through *yesterday*."
)

# ---------------------------------------------------------------- tomorrow
st.header("Tomorrow's forecast")

if preds.empty:
    st.info("No forecasts issued yet. Run `python predict_next_day.py` "
            "(after `fetch_weather.py forecast --slot=evening`).")
else:
    latest = preds.iloc[-1]
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.2])
    c1.metric(f"Peak forecast · {latest.target_date:%a %d %b}",
              f"{latest.predicted_mw:,.0f} MW")
    if pd.notna(latest.get("lo90")):
        c2.metric("90% interval",
                  f"{latest.lo90:,.0f}–{latest.hi90:,.0f}",
                  f"±{(latest.hi90 - latest.lo90) / 2:,.0f} MW",
                  delta_color="off")
    if pd.notna(latest.get("lo95")):
        c3.metric("95% interval",
                  f"{latest.lo95:,.0f}–{latest.hi95:,.0f}",
                  f"±{(latest.hi95 - latest.lo95) / 2:,.0f} MW",
                  delta_color="off")
    c4.metric("Anchor (newest demand)",
              f"{latest.get('anchor_mw', float('nan')):,.0f} MW",
              f"target − {int(latest.get('H', H))} days", delta_color="off")

    st.caption(
        f"Issued {latest.get('issued_at', '?')} · weather "
        f"{latest.get('weather_source', '?')} · model "
        f"{latest.get('model_version', '?')}"
        + ("" if latest.get("model_is_champion") else
           " · **stand-in model**, not Track A's champion")
    )
    # pandas reads an empty CSV cell as NaN, and str(nan) == "nan" is truthy -
    # the same trap that made holiday_name's "None" sentinel read as missing.
    nan_feats = latest.get("features_nan")
    if pd.notna(nan_feats) and str(nan_feats).strip() not in ("", "nan"):
        st.warning(f"Features missing at inference: {nan_feats}")

    filled = latest.get("observed_filled_from_forecast")
    if pd.notna(filled) and str(filled).strip() not in ("", "nan"):
        st.caption(
            f"Yesterday-side of the weather deltas filled from the same "
            f"forecast for {filled} — that day is not in the reanalysis "
            f"archive yet."
        )

# ------------------------------------------------------------- performance
st.header("Performance")
tab_live, tab_backtest = st.tabs(["Live (pre-committed)", "Backtest"])

with tab_live:
    scored = preds.dropna(subset=["actual_mw"]) if not preds.empty else pd.DataFrame()
    if scored.empty:
        st.info("No forecasts have matured yet. Actuals arrive ~2 days later; "
                "`python track_predictions.py` fills them in.")
    else:
        e = scored["error_mw"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scored forecasts", f"{len(scored)}")
        k2.metric("MAE", f"{e.abs().mean():,.0f} MW")
        k3.metric("MAPE", f"{(e.abs() / scored.actual_mw).mean() * 100:.2f}%")
        k4.metric("Bias", f"{e.mean():+,.0f} MW")

        n_summer = int(scored["target_date"].dt.month.isin(SUMMER_MONTHS).sum())
        if n_summer < 20:
            st.warning(
                f"Only {n_summer} March–June days in this sample. Peak season "
                "is where the model is hardest pressed, so treat these live "
                "figures as provisional until a real summer is in the log."
            )
        chart = scored.set_index("target_date")[["actual_mw", "predicted_mw"]]
        st.line_chart(chart, height=320)
        st.caption("These are the only numbers here not produced with "
                   "hindsight — each was written to disk before the outcome "
                   "existed.")

with tab_backtest:
    if not gap:
        st.info("No results/operational_gap.json yet. Run "
                "`python run_phase5.py`.")
    else:
        key_a = f"H{gap['H_measured']}_regime_a"
        rows = []
        for k, v in gap.get("phase_5_3_regimes", {}).items():
            if not k.startswith(f"H{gap['H_measured']}") or "gap" in k:
                continue
            o = v.get("overall", {})
            if not o:
                continue
            rows.append({
                "regime": v.get("label", k),
                "MAE (MW)": round(o["mae"], 1),
                "MAPE (%)": round(o["mape"], 2),
                "RMSE (MW)": round(o["rmse"], 1),
                "Bias (MW)": round(o["bias"], 1),
                "n": o["n"],
            })
        if rows:
            st.subheader(f"Reanalysis vs forecast weather (H={gap['H_measured']})")
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         width="stretch")
        head = gap.get("phase_5_3_regimes", {}).get(
            f"H{gap['H_measured']}_gap_lead1", {}).get("overall")
        if head:
            st.metric("Cost of not knowing tomorrow's weather",
                      f"{head['d_mae']:+,.1f} MW MAE",
                      f"{head['d_mae_pct']:+.1f}% vs reanalysis",
                      delta_color="inverse")
            st.caption(
                f"**{head['mae_b']:,.0f} MW is the number you can hit "
                f"tomorrow morning.** {head['mae_a']:,.0f} MW is the "
                "reanalysis figure — useful for comparing feature sets, "
                "not achievable in production."
            )
        sens = gap.get("phase_5_4_sensitivity")
        if sens:
            st.caption(f"Sensitivity: {sens['mw_per_degc_t_max']:+.0f} MW per "
                       f"°C of day-ahead t_max error "
                       f"(r={sens['corr_t_max_err_vs_mw_shift']:.2f}).")
        for c in gap.get("caveats", []):
            st.caption(f"· {c}")

# -------------------------------------------------------------- sandbox
st.header("Weather sandbox")
st.caption("What would the model say on a hotter day? This perturbs the most "
           "recent row's weather and refits nothing — it reads the shape of "
           "the fitted response, not a new forecast.")

if gap is None:
    st.info("Run `python run_phase5.py` first so the sandbox has a model "
            "configuration to describe.")
else:
    s1, s2 = st.columns(2)
    bump = s1.slider("Shift state-weighted t_max (°C)", -5.0, 8.0, 0.0, 0.5)
    sens = gap.get("phase_5_4_sensitivity", {})
    per_c = sens.get("mw_per_degc_t_max")
    if per_c and not preds.empty:
        base = preds.iloc[-1]["predicted_mw"]
        s2.metric(f"Forecast at {bump:+.1f} °C",
                  f"{base + per_c * bump:,.0f} MW",
                  f"{per_c * bump:+,.0f} MW")
        st.caption(
            f"Linear extrapolation at {per_c:+.0f} MW/°C, the measured "
            "sensitivity of this model to day-ahead temperature error. It is "
            "a local slope: the true demand–temperature response saturates at "
            "high temperature, so treat large shifts as indicative only."
        )
    else:
        st.info("Needs both a logged prediction and Phase 5 sensitivity output.")

# -------------------------------------------------------------- footnotes
with st.expander("Data latency, intervals, and what these numbers mean"):
    if pub:
        st.markdown(
            f"**Why H={pub['H']}.** Grid-India's report for day D was never in "
            f"hand on day D itself across {pub['n_run_days']} observed daily "
            f"runs (min gap {pub['min_gap_days']}d). So an evening forecast for "
            f"tomorrow uses demand through yesterday, and `peak_lag_1` — the "
            f"obvious feature — is not available. "
            f"[`results/publish_time.json`]"
        )
    if iv:
        st.markdown("**Intervals** are rolling split-conformal on relative "
                    "residuals, calibrated separately for March–June and the "
                    "rest of the year:")
        for season, s in iv.get("seasons", {}).items():
            widths = ", ".join(f"{lv}: ±{q * 100:.2f}%" for lv, q
                               in s["relative_halfwidth"].items())
            st.markdown(f"- `{season}` ({s['note']}): {widths}")
        if "coverage" in iv:
            st.markdown("Realised coverage on the calibration walk-forward: "
                        + ", ".join(
                            f"{float(k) * 100:.0f}% → {v['realised_coverage'] * 100:.1f}%"
                            for k, v in iv["coverage"].items()))
        st.caption(iv.get("caveat", ""))
    st.markdown(
        f"**Dataset**: {len(master):,} rows, {master['date'].min():%Y-%m-%d} to "
        f"{master['date'].max():%Y-%m-%d}. All-time peak "
        f"{master[PEAK_TARGET].max():,.0f} MW on "
        f"{master.loc[master[PEAK_TARGET].idxmax(), 'date']:%Y-%m-%d}."
    )

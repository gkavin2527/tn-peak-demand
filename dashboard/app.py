"""
Tamil Nadu Energy Intelligence — next-day electricity demand forecast.

    streamlit run dashboard/app.py

Reads only committed artefacts — predictions_log.csv, results/*.json,
master.csv — and fits nothing. If a panel is empty the upstream job has not
run; the dashboard is a window, not a pipeline.

Redesigned for a mixed audience: officials, utility staff, researchers and the
public. The rule applied throughout is that the plain-language reading comes
first and the statistical detail stays available but subordinate. Nothing is
simplified to the point of being wrong — every number still traces to an
artefact, and where a figure would be fabricated a panel shows an empty state
instead.

Deliberately absent: a "peak hour window" panel. There is no hourly demand
data in this project, so it was dropped rather than faked from a daily series.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (H, MASTER, PEAK_TARGET, PREDICTIONS_LOG, RESULTS,
                    SUMMER_MONTHS)
import theme as T
from theme import card, empty_state, icon, inject_css, note, section

st.set_page_config(page_title="Tamil Nadu Energy Intelligence",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")
inject_css()


# ---------------------------------------------------------------- data
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
    return pd.read_csv(MASTER, parse_dates=["date"])


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
feat = load_json("feature_importance.json")

obs = master.dropna(subset=[PEAK_TARGET])
newest_obs_date = obs["date"].max()
newest_obs_mw = float(obs.loc[obs["date"].idxmax(), PEAK_TARGET])
latest = preds.iloc[-1] if not preds.empty else None


def mw(x, dp=0):
    return "—" if pd.isna(x) else f"{x:,.{dp}f}"


# ---------------------------------------------------------------- header
days_stale = (pd.Timestamp.now().normalize() - newest_obs_date).days
fresh_col, fresh_txt = ((T.POSITIVE, "Current") if days_stale <= 2 else
                        (T.WARN, f"{days_stale} days behind"))
issued = (str(latest.get("issued_at", "")) if latest is not None else "")
issued_disp = issued[:16].replace("T", " ") if issued and issued != "nan" else "—"

st.html(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">{icon('zap', 20, '#FFFFFF', 2.2)}</div>
    <div>
      <div class="brand-t">Tamil Nadu Energy Intelligence</div>
      <div class="brand-s">Next-Day Electricity Demand Forecast</div>
    </div>
  </div>
  <div class="status">
    <div class="status-i"><span class="status-k">Demand data</span>
      <span class="status-v"><span class="dot" style="background:{fresh_col}"></span>
      {fresh_txt} · {newest_obs_date:%d %b %Y}</span></div>
    <div class="status-i"><span class="status-k">Forecast issued</span>
      <span class="status-v">{issued_disp}</span></div>
    <div class="status-i"><span class="status-k">Records</span>
      <span class="status-v">{len(master):,} days</span></div>
  </div>
</div>""")


# ------------------------------------------------- 1. tomorrow's forecast
section("Tomorrow's Expected Peak Demand",
        "The highest electricity demand Tamil Nadu is expected to reach.",
        "zap")

if latest is None:
    empty_state(
        "No forecast has been issued yet",
        "The evening job publishes tomorrow's forecast each day at 20:00 IST. "
        "Once it runs, the expected peak demand will appear here.")
else:
    pred = float(latest.predicted_mw)
    delta = pred - newest_obs_mw
    pct = delta / newest_obs_mw * 100 if newest_obs_mw else 0.0
    if abs(pct) < 0.5:
        chip, ico_d, word = "chip-flat", "activity", "broadly unchanged from"
    elif delta > 0:
        chip, ico_d, word = "chip-up", "trending-up", "higher than"
    else:
        chip, ico_d, word = "chip-down", "trending-down", "lower than"
    chip_col = {"chip-up": T.WARN, "chip-down": T.POSITIVE,
                "chip-flat": T.INK_SOFT}[chip]

    h1, h2 = st.columns([1.72, 1], gap="medium")
    with h1:
        st.html(f"""
<div class="hero">
  <div class="hero-eyebrow">{icon('zap', 13, T.ACCENT)}
    Forecast for {latest.target_date:%A, %d %B %Y}</div>
  <div class="hero-value">{pred:,.0f}<span class="hero-unit">MW</span></div>
  <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap">
    <span class="chip {chip}">{icon(ico_d, 13, chip_col)}
      {delta:+,.0f} MW ({pct:+.1f}%)</span>
    <span style="color:{T.INK_FAINT};font-size:.8125rem">
      vs {newest_obs_mw:,.0f} MW on {newest_obs_date:%d %b}</span>
  </div>
  <p class="hero-say">Tamil Nadu is expected to reach a peak electricity
    demand of approximately <b style="color:{T.INK}">{pred:,.0f} megawatts</b>
    on {latest.target_date:%A}, {word} the most recently published figure.</p>
</div>""")
    with h2:
        lo, hi = latest.get("lo90"), latest.get("hi90")
        band = (f"{lo:,.0f} – {hi:,.0f} MW" if pd.notna(lo) else "Not available")
        st.html(card("Likely range", band, "",
                     "9 times out of 10, demand should fall in this band.",
                     "target", "accent"))
        st.html("<div style='height:.75rem'></div>")
        st.html(card("Most recent actual", f"{newest_obs_mw:,.0f}", "MW",
                     f"Published {newest_obs_date:%d %b %Y}", "database"))

    # --- integrity notices, preserved from the original -------------------
    if not latest.get("model_is_champion"):
        note("This forecast was produced by the <b>interim serving model</b>, "
             "not the research champion model. Figures are valid but will "
             "differ slightly from the published research results.",
             "warn", "alert")

    nan_feats = latest.get("features_nan")
    if pd.notna(nan_feats) and str(nan_feats).strip() not in ("", "nan"):
        note(f"Some inputs were unavailable when this forecast was made: "
             f"<b>{nan_feats}</b>. Treat it as provisional.", "warn", "alert")

    filled = latest.get("observed_filled_from_forecast")
    if pd.notna(filled) and str(filled).strip() not in ("", "nan"):
        note(f"Yesterday's weather for {filled} was taken from a forecast — "
             f"that day has not reached the observational archive yet.",
             "quiet", "info")


# ------------------------------------------------------- 2. demand trend
section("How Demand Is Changing",
        "Recorded daily peak demand, with the latest forecast marked. "
        "Use the range selector to zoom in or out.", "trending-up")

RANGES = {"30 days": 30, "90 days": 90, "1 year": 365, "3 years": 1095,
          "All time": None}
r1, _ = st.columns([1, 3])
choice = r1.selectbox("Period", list(RANGES), index=1,
                      label_visibility="collapsed")
days = RANGES[choice]
hist = obs if days is None else obs[obs["date"] >= newest_obs_date -
                                    pd.Timedelta(days=days)]
hist = hist[["date", PEAK_TARGET]].rename(columns={PEAK_TARGET: "mw"})

if hist.empty:
    empty_state("No demand history in this period",
                "Choose a longer period to see recorded demand.")
else:
    y_lo = float(hist["mw"].min()) * 0.96
    y_hi = float(hist["mw"].max()) * 1.04
    if latest is not None:
        y_hi = max(y_hi, float(latest.predicted_mw) * 1.04)
        y_lo = min(y_lo, float(latest.predicted_mw) * 0.96)

    base = alt.Chart(hist).mark_line(
        color=T.HISTORY, strokeWidth=1.6,
        interpolate="monotone").encode(
        x=alt.X("date:T", title=None,
                axis=alt.Axis(format="%d %b", grid=False, labelColor=T.INK_FAINT,
                              tickColor=T.LINE, domainColor=T.LINE,
                              labelFontSize=11)),
        y=alt.Y("mw:Q", title=None, scale=alt.Scale(domain=[y_lo, y_hi]),
                axis=alt.Axis(grid=True, gridColor="#F1F5F9", gridDash=[3, 3],
                              labelColor=T.INK_FAINT, domainOpacity=0,
                              tickOpacity=0, labelFontSize=11,
                              format=",.0f", labelExpr="datum.value/1000+'k'")),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%A, %d %b %Y"),
                 alt.Tooltip("mw:Q", title="Peak demand (MW)", format=",.0f")],
    )
    hover = alt.selection_point(on="pointerover", nearest=True,
                                fields=["date"], empty=False)
    pts = base.mark_circle(size=52, color=T.ACCENT, opacity=0).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))).add_params(hover)
    layers = [base, pts]

    if latest is not None:
        fdf = pd.DataFrame({"date": [latest.target_date],
                            "mw": [float(latest.predicted_mw)]})
        # Dashed connector from the last actual to the forecast makes the
        # handover explicit rather than implying continuous observation.
        bridge = pd.DataFrame({"date": [newest_obs_date, latest.target_date],
                               "mw": [newest_obs_mw, float(latest.predicted_mw)]})
        layers.append(alt.Chart(bridge).mark_line(
            color=T.ACCENT, strokeWidth=2, strokeDash=[5, 4]).encode(
            x="date:T", y="mw:Q"))
        layers.append(alt.Chart(fdf).mark_point(
            shape="diamond", size=170, color=T.ACCENT, filled=True,
            stroke="#FFFFFF", strokeWidth=2).encode(
            x="date:T", y="mw:Q",
            tooltip=[alt.Tooltip("date:T", title="Forecast for",
                                 format="%A, %d %b %Y"),
                     alt.Tooltip("mw:Q", title="Expected peak (MW)",
                                 format=",.0f")]))

    st.altair_chart(
        alt.layer(*layers).properties(height=330).configure_view(
            strokeWidth=0).configure_axis(labelFont=T.FONT, titleFont=T.FONT),
        width="stretch")

    lg = (f'<span style="display:inline-flex;align-items:center;gap:.4rem">'
          f'<span style="width:16px;height:2px;background:{T.HISTORY};'
          f'display:inline-block"></span> Recorded demand</span>')
    if latest is not None:
        lg += (f'<span style="display:inline-flex;align-items:center;gap:.4rem;'
               f'margin-left:1.25rem"><span style="width:16px;height:0;'
               f'border-top:2px dashed {T.ACCENT};display:inline-block"></span>'
               f' Forecast</span>')
    st.html(f'<div style="color:{T.INK_FAINT};font-size:.75rem;'
            f'margin:-.35rem 0 .25rem">{lg}</div>')


# --------------------------------------------------- 3. forecast confidence
section("How Reliable Is This Estimate",
        "Forecasts are a range, not a single number. These bands come from "
        "how far past forecasts actually landed from reality.", "target")

if latest is None or pd.isna(latest.get("lo90")):
    empty_state("Confidence range not available yet",
                "Ranges appear once a forecast has been issued with "
                "calibrated intervals attached.", "target")
else:
    pred = float(latest.predicted_mw)
    l90, h90 = float(latest.lo90), float(latest.hi90)
    l95 = float(latest.lo95) if pd.notna(latest.get("lo95")) else l90
    h95 = float(latest.hi95) if pd.notna(latest.get("hi95")) else h90

    rng = pd.DataFrame([
        {"band": "Wider range", "lo": l95, "hi": h95, "order": 0},
        {"band": "Likely range", "lo": l90, "hi": h90, "order": 1}])
    pad = (h95 - l95) * 0.12
    xs = alt.Scale(domain=[l95 - pad, h95 + pad])
    bars = alt.Chart(rng).mark_bar(height=26, cornerRadius=4).encode(
        x=alt.X("lo:Q", scale=xs, title=None,
                axis=alt.Axis(format=",.0f", grid=False, labelColor=T.INK_FAINT,
                              domainColor=T.LINE, tickColor=T.LINE,
                              labelFontSize=11)),
        x2="hi:Q",
        y=alt.Y("band:N", title=None, sort=alt.SortField("order"),
                axis=alt.Axis(labelColor=T.INK, labelFontSize=12,
                              domainOpacity=0, tickOpacity=0, labelPadding=8)),
        color=alt.Color("band:N", scale=alt.Scale(
            domain=["Likely range", "Wider range"],
            range=[T.ACCENT, "#C7D9FF"]), legend=None),
        tooltip=[alt.Tooltip("band:N", title="Range"),
                 alt.Tooltip("lo:Q", title="From (MW)", format=",.0f"),
                 alt.Tooltip("hi:Q", title="To (MW)", format=",.0f")])
    mark = alt.Chart(pd.DataFrame({"v": [pred]})).mark_rule(
        color=T.INK, strokeWidth=2).encode(x=alt.X("v:Q", scale=xs))
    st.altair_chart(
        alt.layer(bars, mark).properties(height=110).configure_view(
            strokeWidth=0).configure_axis(labelFont=T.FONT),
        width="stretch")

    f1, f2, f3 = st.columns(3)
    f1.html(card("Expected", f"{pred:,.0f}", "MW",
                 "Our single best estimate.", "zap", "accent"))
    f2.html(card("Likely range", f"{l90:,.0f} – {h90:,.0f}", "",
                 f"±{(h90 - l90) / 2:,.0f} MW · 9 days in 10.", "target"))
    f3.html(card("Wider range", f"{l95:,.0f} – {h95:,.0f}", "",
                 f"±{(h95 - l95) / 2:,.0f} MW · 19 days in 20.", "target"))
    st.html("<div style='height:.6rem'></div>")
    note("Actual demand may vary depending on weather and electricity "
         "consumption patterns. The ranges widen in summer, when demand is "
         "hardest to predict.", "quiet", "info")


# ------------------------------------------------------ 4. weather impact
section("How Weather Affects Electricity Demand",
        "Hotter days push cooling demand up. Move the slider to see what the "
        "model estimates for a warmer or cooler day.", "thermometer")

sens = (gap or {}).get("phase_5_4_sensitivity", {})
per_c = sens.get("mw_per_degc_t_max")

if not per_c or latest is None:
    empty_state(
        "Weather scenarios not available yet",
        "This needs both an issued forecast and the measured "
        "temperature-sensitivity output from the analysis pipeline.",
        "thermometer")
else:
    w1, w2 = st.columns([1.1, 1], gap="large")
    with w1:
        bump = st.slider("Temperature change versus the forecast day (°C)",
                         -5.0, 8.0, 0.0, 0.5, format="%+.1f °C")
        base_mw = float(latest.predicted_mw)
        scen = base_mw + per_c * bump
        shift = per_c * bump
        if abs(bump) < 0.01:
            sentence = ("At the forecast temperature, the expected peak is "
                        f"<b style='color:{T.INK}'>{base_mw:,.0f} MW</b>.")
        else:
            direction = "increase" if shift > 0 else "decrease"
            sentence = (f"If temperatures were <b style='color:{T.INK}'>"
                        f"{abs(bump):.1f} °C {'warmer' if bump > 0 else 'cooler'}"
                        f"</b>, the model estimates electricity demand could "
                        f"{direction} by approximately <b style='color:{T.INK}'>"
                        f"{abs(shift):,.0f} MW</b>.")
        st.html(f'<p style="font-size:.9375rem;line-height:1.65;margin:.5rem 0 0">'
                f'{sentence}</p>')
    with w2:
        tone = "warn" if shift > 0 else ("pos" if shift < 0 else "")
        st.html(card(f"Estimated peak at {bump:+.1f} °C", f"{scen:,.0f}", "MW",
                     f"{shift:+,.0f} MW versus the {base_mw:,.0f} MW forecast",
                     "thermometer", tone))

    scn = pd.DataFrame([{"k": "Forecast", "v": base_mw},
                        {"k": f"At {bump:+.1f} °C", "v": scen}])
    lo = min(base_mw, scen) * 0.985
    hi = max(base_mw, scen) * 1.015
    st.altair_chart(
        alt.Chart(scn).mark_bar(height=30, cornerRadius=4).encode(
            x=alt.X("v:Q", scale=alt.Scale(domain=[lo, hi]), title=None,
                    axis=alt.Axis(format=",.0f", grid=False,
                                  labelColor=T.INK_FAINT, domainColor=T.LINE,
                                  tickColor=T.LINE, labelFontSize=11)),
            y=alt.Y("k:N", title=None, sort=None,
                    axis=alt.Axis(labelColor=T.INK, labelFontSize=12,
                                  domainOpacity=0, tickOpacity=0, labelPadding=8)),
            color=alt.Color("k:N", scale=alt.Scale(
                domain=list(scn["k"]), range=["#C7D9FF", T.ACCENT]), legend=None),
            tooltip=[alt.Tooltip("k:N", title="Scenario"),
                     alt.Tooltip("v:Q", title="Peak demand (MW)", format=",.0f")]
        ).properties(height=96).configure_view(strokeWidth=0)
        .configure_axis(labelFont=T.FONT), width="stretch")

    note(f"<b>This is a what-if scenario, not a forecast.</b> It applies the "
         f"model's measured sensitivity of {per_c:+.0f} MW per °C to the "
         f"existing forecast. That rate is a local slope — real demand "
         f"flattens out at high temperatures, so large shifts are indicative "
         f"only.", "quiet", "info")


# ---------------------------------------------------- 5. what drives demand
if feat:
    section("What Influences Demand",
            "Measured by how much forecast accuracy degrades when each group "
            "of inputs is scrambled — a direct test of how much the model "
            "relies on it.", "activity")
    blocks = feat.get("by_block", {})
    NAMES = {"weather": "Weather conditions",
             "demand_lags": "Recent demand history",
             "calendar_trend": "Day of week, season and long-term growth"}
    bd = pd.DataFrame([
        {"k": NAMES.get(k, k), "v": v["permutation_dmae_mw"]}
        for k, v in blocks.items() if v.get("permutation_dmae_mw")])
    if not bd.empty:
        bd = bd.sort_values("v", ascending=False)
        st.altair_chart(
            alt.Chart(bd).mark_bar(height=30, cornerRadius=4,
                                   color=T.ACCENT).encode(
                x=alt.X("v:Q", title=None, axis=alt.Axis(
                    grid=True, gridColor="#F1F5F9", gridDash=[3, 3],
                    labelColor=T.INK_FAINT, domainOpacity=0, tickOpacity=0,
                    format=",.0f", labelFontSize=11)),
                y=alt.Y("k:N", title=None, sort="-x", axis=alt.Axis(
                    labelColor=T.INK, labelFontSize=12, domainOpacity=0,
                    tickOpacity=0, labelPadding=8, labelLimit=320)),
                tooltip=[alt.Tooltip("k:N", title="Input group"),
                         alt.Tooltip("v:Q", title="Accuracy lost without it (MW)",
                                     format=",.0f")]
            ).properties(height=130).configure_view(strokeWidth=0)
            .configure_axis(labelFont=T.FONT), width="stretch")
        top = bd.iloc[0]
        st.html(
            f'<p style="font-size:.8438rem;color:{T.INK_SOFT};line-height:1.6;'
            f'margin:-.2rem 0 0"><b style="color:{T.INK}">'
            f'{top.k.lower().capitalize()}</b> carries the most information the '
            f'model cannot get anywhere else: removing it costs about '
            f'{top.v:,.0f} MW of accuracy.</p>')


# ------------------------------------------------------ 6. performance
section("How Accurate Have Forecasts Been",
        "Two different questions: how our live forecasts have done, and how "
        "the system performed when tested against years of historical data.",
        "check")

tab_live, tab_back = st.tabs(["Live forecast tracking", "Historical performance"])

with tab_live:
    st.html(f'<p style="font-size:.875rem;color:{T.INK_SOFT};line-height:1.6;'
            f'margin:.35rem 0 1rem">See how close our previous predictions were '
            f'to the electricity demand that actually occurred. These are the '
            f'only figures here produced without hindsight — each was written '
            f'down before the outcome existed.</p>')
    scored = preds.dropna(subset=["actual_mw"]) if not preds.empty else pd.DataFrame()
    if scored.empty:
        empty_state(
            "Your first forecast results are being collected",
            "Actual electricity demand becomes available for comparison once "
            "Grid-India publishes it, about two days after each forecast date. "
            "Accuracy figures will appear here automatically.")
    else:
        e = scored["error_mw"]
        m1, m2, m3, m4 = st.columns(4)
        m1.html(card("Forecasts scored", f"{len(scored)}", "",
                     "Compared against published demand.", "check"))
        m2.html(card("Typical error", f"{e.abs().mean():,.0f}", "MW",
                     "Average distance from actual demand.", "target"))
        m3.html(card("Typical error", f"{(e.abs() / scored.actual_mw).mean() * 100:.2f}", "%",
                     "As a share of actual demand.", "target"))
        bias = e.mean()
        m4.html(card("Tendency", f"{bias:+,.0f}", "MW",
                     "Over-forecast" if bias > 0 else "Under-forecast",
                     "activity", "warn" if abs(bias) > 300 else ""))

        n_summer = int(scored["target_date"].dt.month.isin(SUMMER_MONTHS).sum())
        if n_summer < 20:
            st.html("<div style='height:.9rem'></div>")
            note(f"Only {n_summer} March–June days are in this sample. Peak "
                 f"season is when the system is hardest pressed, so treat "
                 f"these live figures as provisional until a full summer has "
                 f"been recorded.", "warn", "alert")

        st.html("<div style='height:1rem'></div>")
        cmp_df = scored[["target_date", "actual_mw", "predicted_mw"]].melt(
            "target_date", var_name="series", value_name="mw")
        cmp_df["series"] = cmp_df["series"].map(
            {"actual_mw": "Actual demand", "predicted_mw": "Forecast"})
        st.altair_chart(
            alt.Chart(cmp_df).mark_line(strokeWidth=2, point=True).encode(
                x=alt.X("target_date:T", title=None, axis=alt.Axis(
                    format="%d %b", grid=False, labelColor=T.INK_FAINT,
                    domainColor=T.LINE, tickColor=T.LINE, labelFontSize=11)),
                y=alt.Y("mw:Q", title=None, scale=alt.Scale(zero=False),
                        axis=alt.Axis(grid=True, gridColor="#F1F5F9",
                                      gridDash=[3, 3], labelColor=T.INK_FAINT,
                                      domainOpacity=0, tickOpacity=0,
                                      format=",.0f", labelFontSize=11)),
                color=alt.Color("series:N", title=None, scale=alt.Scale(
                    domain=["Actual demand", "Forecast"],
                    range=[T.INK, T.ACCENT]),
                    legend=alt.Legend(orient="top", labelFont=T.FONT,
                                      labelColor=T.INK_SOFT, symbolType="stroke")),
                tooltip=[alt.Tooltip("target_date:T", title="Date",
                                     format="%d %b %Y"),
                         alt.Tooltip("series:N", title=""),
                         alt.Tooltip("mw:Q", title="MW", format=",.0f")]
            ).properties(height=300).configure_view(strokeWidth=0)
            .configure_axis(labelFont=T.FONT), width="stretch")

with tab_back:
    st.html(f'<p style="font-size:.875rem;color:{T.INK_SOFT};line-height:1.6;'
            f'margin:.35rem 0 1rem">Understand how accurately the forecasting '
            f'system performed on historical data, and what it costs to rely '
            f'on a weather <i>forecast</i> rather than the weather record we '
            f'only have afterwards.</p>')
    if not gap:
        empty_state("Historical results not generated yet",
                    "The analysis pipeline has not produced its evaluation "
                    "output for this environment.", "database")
    else:
        hm = gap.get("H_measured", H)
        head = gap.get("phase_5_3_regimes", {}).get(
            f"H{hm}_gap_lead1", {}).get("overall")
        if head:
            g1, g2, g3 = st.columns(3)
            g1.html(card("Using real weather forecasts",
                         f"{head['mae_b']:,.0f}", "MW",
                         "Typical error. This is what is achievable in "
                         "practice.", "check", "accent"))
            g2.html(card("Using the weather record",
                         f"{head['mae_a']:,.0f}", "MW",
                         "Only knowable afterwards — a benchmark, not a "
                         "forecast.", "database"))
            g3.html(card("Cost of not knowing the weather",
                         f"{head['d_mae']:+,.0f}", "MW",
                         f"{head['d_mae_pct']:+.1f}% worse when using genuine "
                         f"forecasts.", "alert", "warn"))
            st.html("<div style='height:.9rem'></div>")
            note(f"<b>{head['mae_b']:,.0f} MW is the number achievable "
                 f"tomorrow morning.</b> The {head['mae_a']:,.0f} MW figure "
                 f"assumes perfect knowledge of tomorrow's weather, which no "
                 f"forecaster has. Most published studies report the second "
                 f"number; this system measures both.", "info", "info")

        rows = []
        for k, v in gap.get("phase_5_3_regimes", {}).items():
            if not k.startswith(f"H{hm}") or "gap" in k:
                continue
            o = v.get("overall", {})
            if o:
                rows.append({"Weather used": v.get("label", k),
                             "Typical error (MW)": round(o["mae"], 1),
                             "Typical error (%)": round(o["mape"], 2),
                             "Larger errors (RMSE, MW)": round(o["rmse"], 1),
                             "Tendency (MW)": round(o["bias"], 1),
                             "Days tested": o["n"]})
        if rows:
            st.html("<div style='height:1rem'></div>")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if sens:
            st.html(f'<p style="font-size:.8125rem;color:{T.INK_FAINT};'
                    f'margin:.6rem 0 0">Each 1 °C of error in the day-ahead '
                    f'temperature forecast moves demand by about '
                    f'{sens["mw_per_degc_t_max"]:+.0f} MW '
                    f'(correlation {sens["corr_t_max_err_vs_mw_shift"]:.2f}).</p>')
        for c in gap.get("caveats", []):
            st.html(f'<p style="font-size:.8125rem;color:{T.INK_FAINT};'
                    f'margin:.25rem 0 0">· {c}</p>')


# ------------------------------------------------------ 7. transparency
section("About This Forecast",
        "Where the numbers come from, and what they do and do not mean.",
        "database")

with st.expander("Where the electricity demand data comes from"):
    st.markdown(
        f"Daily peak demand is published by **Grid-India** (formerly POSOCO) "
        f"in its Daily Power Supply Position reports. This dashboard covers "
        f"**{len(master):,} days**, from {master['date'].min():%d %B %Y} to "
        f"{master['date'].max():%d %B %Y}.\n\n"
        f"The highest demand ever recorded in this dataset is "
        f"**{master[PEAK_TARGET].max():,.0f} MW**, on "
        f"{master.loc[master[PEAK_TARGET].idxmax(), 'date']:%d %B %Y}.\n\n"
        f"Weather comes from Open-Meteo: historical reanalysis for past days, "
        f"and archived day-ahead forecasts for the forecast itself.")

with st.expander("Why the forecast is made before demand data is published"):
    if pub:
        st.markdown(
            f"Grid-India publishes a day's peak demand the **following "
            f"morning**. Across **{pub['n_run_days']} observed daily runs** of "
            f"this pipeline, same-day data was never available.\n\n"
            f"So a forecast issued this evening for tomorrow can only use "
            f"demand data up to **yesterday** — not today. That two-day gap "
            f"(written as H = {pub['H']}) is measured from the pipeline's own "
            f"records rather than assumed, and it rules out using yesterday's "
            f"demand, which would otherwise be the single most useful input.\n\n"
            f"This is a genuine operating constraint, not a modelling choice.")
    else:
        st.markdown(
            f"Grid-India publishes a day's peak demand the following morning, "
            f"so an evening forecast for tomorrow uses demand through "
            f"yesterday (H = {H}).")

with st.expander("What the confidence ranges mean"):
    st.markdown(
        "The **likely range** is calibrated so that roughly 9 out of every 10 "
        "days should fall inside it; the **wider range**, 19 out of 20.\n\n"
        "These are not guesses. They are built from how far past forecasts "
        "actually landed from reality, using a distribution-free method that "
        "makes no assumption that errors follow a bell curve — ours do not.")
    if iv:
        st.markdown("Calibrated separately for March–June and the rest of the "
                    "year, because summer is harder to predict:")
        for season, s in iv.get("seasons", {}).items():
            widths = ", ".join(f"{lv}: ±{q * 100:.2f}%" for lv, q
                               in s["relative_halfwidth"].items())
            st.markdown(f"- **{season}** ({s['note']}): {widths}")
        if "coverage" in iv:
            st.markdown("Checked against reality: " + ", ".join(
                f"{float(k) * 100:.0f}% band held {v['realised_coverage'] * 100:.1f}% "
                f"of the time" for k, v in iv["coverage"].items()))
        if iv.get("caveat"):
            st.caption(iv["caveat"])

with st.expander("When forecast accuracy can actually be measured"):
    st.markdown(
        "A forecast can only be scored once the real demand for that day has "
        "been published — about two days later.\n\n"
        "**Live tracking** counts only forecasts written down *before* the "
        "outcome was known. It is the honest test, and it starts small.\n\n"
        "**Historical performance** replays the system over years of past "
        "data. It gives far more days to judge by, but it is a reconstruction "
        "rather than a live record.\n\n"
        "Both are shown because neither alone is sufficient.")

if latest is not None:
    st.html(
        f'<div style="margin-top:2.25rem;padding-top:1rem;border-top:1px solid '
        f'{T.LINE};color:{T.INK_FAINT};font-size:.75rem;line-height:1.6">'
        f'Forecast issued {issued_disp} · weather source '
        f'{latest.get("weather_source", "—")} · model '
        f'{latest.get("model_version", "—")} · horizon H={int(latest.get("H", H))}'
        f'<br>All figures read from committed pipeline outputs. '
        f'Nothing on this page is simulated.</div>')

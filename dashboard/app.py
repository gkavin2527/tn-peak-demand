"""
Tamil Nadu Energy Intelligence — next-day electricity demand forecast.

    streamlit run dashboard/app.py

Reads only committed artefacts — predictions_log.csv, results/*.json,
master.csv — and fits nothing. If a panel is empty the upstream job has not
run; the dashboard is a window, not a pipeline.

Built for a mixed audience: officials, utility staff, researchers and the
public. The plain-language reading comes first; statistical detail stays
available but subordinate. Every number traces to an artefact — where a figure
would have to be invented, a panel shows an empty state instead.

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
from theme import empty_state, icon, inject_css, note, panel_open, sparkline

st.set_page_config(page_title="Tamil Nadu Energy Intelligence", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")
inject_css()


def axis(fmt=",.0f", grid=False):
    # Solid, recessive gridlines. Dashed grid is an anti-pattern - it adds
    # noise and reads as a data encoding.
    return alt.Axis(format=fmt, grid=grid, gridColor="#ECE9E3",
                    labelColor=T.INK_FAINT, labelFontSize=11, labelFont=T.SANS,
                    domainColor=T.LINE, tickColor=T.LINE, labelPadding=6)


def finish(ch, h=250):
    return ch.properties(height=h).configure_view(strokeWidth=0)


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
iv = load_json("intervals.json")
pub = load_json("publish_time.json")
feat = load_json("feature_importance.json")

obs = master.dropna(subset=[PEAK_TARGET])
newest_date = obs["date"].max()
newest_mw = float(obs.loc[obs["date"].idxmax(), PEAK_TARGET])
latest = preds.iloc[-1] if not preds.empty else None
recent = obs.tail(30)[["date", PEAK_TARGET]].rename(columns={PEAK_TARGET: "mw"})


# ---------------------------------------------------------------- top bar
stale = (pd.Timestamp.now().normalize() - newest_date).days
ok = stale <= 2
issued = str(latest.get("issued_at", "")) if latest is not None else ""
issued_disp = issued[:16].replace("T", " ") if issued and issued != "nan" else "—"

st.html(f"""
<div class="nav">
  <div class="nav-brand">
    <div class="nav-mark">{icon('zap', 16, '#FFFFFF', 2.2)}</div>
    <span class="nav-name">Tamil Nadu Energy</span>
    <span class="nav-ver">H={H}</span>
  </div>
  <div class="seg"><span class="on">Forecast</span><span>Accuracy</span>
    <span>Method</span></div>
  <div class="nav-r">
    <div class="nav-search">{icon('search', 14, T.INK_FAINT)}
      <em>Search demand, dates, weather…</em></div>
    <span class="pill"><span class="dot" style="background:
      {T.GREEN if ok else T.CORAL}"></span>
      {'Pipeline healthy' if ok else f'{stale} days behind'}</span>
    {icon('bell', 16, T.INK_FAINT)}
    <div class="who"><div class="who-av">TN</div>
      <div><div class="who-n">Grid-India feed</div>
      <div class="who-r">{newest_date:%d %b %Y}</div></div></div>
  </div>
</div>""")


# ---------------------------------------------------------------- hero
if latest is None:
    st.html(f"""
<div class="wash"><div class="eyebrow">
  <span class="dot" style="background:{T.CORAL}"></span>Forecast · awaiting run</div>
  <h1 class="display">No forecast <i>yet</i>.</h1>
  <p class="hero-sub">The evening job publishes tomorrow's expected peak each
  day at 20:00 IST. It will appear here once it runs.</p></div>""")
else:
    pred = float(latest.predicted_mw)
    delta = pred - newest_mw
    pct = delta / newest_mw * 100 if newest_mw else 0.0
    st.html(f"""
<div class="wash"><div class="hero-row"><div>
  <div class="eyebrow"><span class="dot" style="background:{T.CORAL}"></span>
    Forecast · {latest.target_date:%d %B %Y}</div>
  <h1 class="display">Tomorrow, <i>{pred:,.0f} MW</i>.</h1>
  <p class="hero-sub">Tamil Nadu is expected to reach a peak electricity demand
  of about {pred:,.0f} megawatts on {latest.target_date:%A} — against
  {newest_mw:,.0f} MW recorded on {newest_date:%d %b}.</p></div>
  <div style="display:flex;gap:.5rem;align-items:center">
    <span class="ghost">Last 30 days</span>
    <span class="btn-dark">{icon('database', 13, '#fff')} Export</span>
  </div></div></div>""")


# ---------------------------------------------------------------- KPI row
def kpi(label, num, unit, foot, chip_html="", spark="", warm=False):
    return (f'<div class="k{" warm" if warm else ""}">'
            f'<div class="k-top"><span class="k-label">{label}</span>{chip_html}</div>'
            f'<div class="k-num">{num}<span class="k-unit">{unit}</span></div>'
            f'<div class="k-bot"><span class="k-foot">{foot}</span>{spark}</div></div>')


if latest is not None:
    pred = float(latest.predicted_mw)
    delta = pred - newest_mw
    pct = delta / newest_mw * 100 if newest_mw else 0.0
    cls = "chip-fl" if abs(pct) < 0.5 else ("chip-dn" if delta > 0 else "chip-up")
    car = "caret-up" if delta > 0 else "caret-down"
    ccol = {"chip-up": T.GREEN, "chip-dn": T.CORAL, "chip-fl": T.INK_SOFT}[cls]
    chip = (f'<span class="chip {cls}">{icon(car, 11, ccol, 2.4)}'
            f'{pct:+.1f}%</span>')

    k1, k2, k3 = st.columns(3, gap="small")
    k1.html(kpi("Expected peak demand", f"{pred:,.0f}", "MW",
                f"vs {newest_mw:,.0f} MW previous", chip,
                sparkline(recent["mw"], T.INK)))

    lo, hi = latest.get("lo90"), latest.get("hi90")
    if pd.notna(lo):
        k2.html(kpi("Likely range", f"±{(hi - lo) / 2:,.0f}", "MW",
                    f"{lo:,.0f} – {hi:,.0f} MW · 9 days in 10",
                    f'<span class="chip chip-fl">90%</span>',
                    sparkline([lo, pred, hi], T.CAT[1]), warm=True))
    else:
        k2.html(kpi("Likely range", "—", "", "Interval not attached to this run",
                    "", "", warm=True))

    yr = obs.loc[obs["date"].dt.year == newest_date.year, PEAK_TARGET]
    k3.html(kpi("Highest this year", f"{yr.max():,.0f}", "MW",
                f"{len(yr)} days recorded in {newest_date.year}",
                f'<span class="chip chip-fl">{newest_date.year}</span>',
                sparkline(obs.tail(90)[PEAK_TARGET], T.INK_FAINT)))

    if not latest.get("model_is_champion"):
        st.html("<div style='height:.9rem'></div>")
        note("Produced by the <b>interim serving model</b>, not the research "
             "champion. Valid, but will differ slightly from published "
             "research figures.", "warn", "alert")
    nf = latest.get("features_nan")
    if pd.notna(nf) and str(nf).strip() not in ("", "nan"):
        note(f"Some inputs were unavailable: <b>{nf}</b>. Treat as provisional.",
             "warn", "alert")
    fl = latest.get("observed_filled_from_forecast")
    if pd.notna(fl) and str(fl).strip() not in ("", "nan"):
        note(f"Yesterday's weather for {fl} came from a forecast — that day is "
             f"not in the observational archive yet.", "info", "info")


# -------------------------------------------------- daily demand + drivers
st.html("<div style='height:1.15rem'></div>")
c_left, c_right = st.columns([1.42, 1], gap="small")

with c_left:
    lg = (f'<div class="lgd">'
          f'<span><i style="background:{T.SERIES["recorded"]}"></i>Recorded</span>'
          f'<span><i style="background:{T.SERIES["forecast"]}"></i>Forecast</span>'
          f'</div>')
    st.html(panel_open("Daily peak demand", lg))
    bars = recent.copy()
    bars["kind"] = "Recorded"
    if latest is not None:
        bars = pd.concat([bars, pd.DataFrame(
            [{"date": latest.target_date, "mw": float(latest.predicted_mw),
              "kind": "Forecast"}])], ignore_index=True)
    st.altair_chart(finish(
        alt.Chart(bars).mark_bar(size=13, cornerRadiusTopLeft=4,
                                 cornerRadiusTopRight=4).encode(
            x=alt.X("date:T", title=None,
                    axis=alt.Axis(format="%d", grid=False, labelColor=T.INK_FAINT,
                                  labelFontSize=10, domainColor=T.LINE,
                                  tickColor=T.LINE, labelAngle=0)),
            y=alt.Y("mw:Q", title=None, scale=alt.Scale(zero=False, nice=True),
                    axis=axis(grid=True)),
            color=alt.Color("kind:N", scale=alt.Scale(
                domain=["Recorded", "Forecast"],
                range=[T.SERIES["recorded"], T.SERIES["forecast"]]),
                legend=None),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b %Y"),
                     alt.Tooltip("mw:Q", title="Peak (MW)", format=",.0f"),
                     alt.Tooltip("kind:N", title="")]), 268), width="stretch")
    st.html("</div>")

with c_right:
    st.html(panel_open("What drives demand",
                       '<span class="ghost">Accuracy lost if removed</span>'))
    if not feat:
        st.html('<div class="empty-s" style="padding:2rem 0">Driver analysis '
                'has not been generated in this environment.</div>')
    else:
        NAMES = {"weather": "Weather", "demand_lags": "Recent demand",
                 "calendar_trend": "Calendar &amp; season"}
        COLS = T.DRIVERS   # fixed per entity, not by rank
        bl = [{"k": NAMES.get(k, k), "v": v["permutation_dmae_mw"],
               "c": COLS.get(k, T.CAT[0])}
              for k, v in feat.get("by_block", {}).items()
              if v.get("permutation_dmae_mw")]
        bl.sort(key=lambda r: -r["v"])
        tot = sum(r["v"] for r in bl) or 1
        dd = pd.DataFrame(bl)
        st.altair_chart(finish(
            alt.Chart(dd).mark_arc(innerRadius=52, outerRadius=78,
                                   stroke="#FBFAF7", strokeWidth=2).encode(
                theta=alt.Theta("v:Q", stack=True),
                color=alt.Color("k:N", scale=alt.Scale(
                    domain=[r["k"] for r in bl], range=[r["c"] for r in bl]),
                    legend=None),
                tooltip=[alt.Tooltip("k:N", title="Input"),
                         alt.Tooltip("v:Q", title="MW accuracy lost",
                                     format=",.0f")]), 172), width="stretch")
        rows = "".join(
            f'<div class="rk"><span class="rk-sw" style="background:{r["c"]}"></span>'
            f'<span class="rk-n">{r["k"]}</span>'
            f'<span class="rk-v">{r["v"]:,.0f}</span>'
            f'<span class="rk-p">{r["v"] / tot * 100:.1f}%</span></div>' for r in bl)
        st.html(rows)
    st.html("</div>")


# ------------------------------------------------------- confidence band
st.html("<div style='height:1.15rem'></div>")
if latest is not None and pd.notna(latest.get("lo90")):
    pred = float(latest.predicted_mw)
    l90, h90 = float(latest.lo90), float(latest.hi90)
    l95 = float(latest.lo95) if pd.notna(latest.get("lo95")) else l90
    h95 = float(latest.hi95) if pd.notna(latest.get("hi95")) else h90
    st.html(panel_open("How reliable is this estimate",
                       '<span class="ghost">Built from past forecast errors</span>'))
    rng = pd.DataFrame([{"b": "19 days in 20", "lo": l95, "hi": h95, "o": 0},
                        {"b": "9 days in 10", "lo": l90, "hi": h90, "o": 1}])
    pad = (h95 - l95) * .14
    xs = alt.Scale(domain=[l95 - pad, h95 + pad])
    st.altair_chart(finish(alt.layer(
        alt.Chart(rng).mark_bar(height=22, cornerRadius=5).encode(
            x=alt.X("lo:Q", scale=xs, title=None, axis=axis()), x2="hi:Q",
            y=alt.Y("b:N", title=None, sort=alt.SortField("o"),
                    axis=alt.Axis(labelColor=T.INK, labelFontSize=12,
                                  labelFont=T.SANS, domainOpacity=0,
                                  tickOpacity=0, labelPadding=10)),
            color=alt.Color("b:N", scale=alt.Scale(
                domain=["9 days in 10", "19 days in 20"],
                range=T.BAND), legend=None),
            tooltip=[alt.Tooltip("b:N", title="Range"),
                     alt.Tooltip("lo:Q", title="From", format=",.0f"),
                     alt.Tooltip("hi:Q", title="To", format=",.0f")]),
        alt.Chart(pd.DataFrame({"v": [pred]})).mark_rule(
            color=T.INK, strokeWidth=2).encode(x=alt.X("v:Q", scale=xs))), 108),
        width="stretch")
    st.html(f'<p style="font-size:.8125rem;color:{T.INK_FAINT};margin:.1rem 0 0">'
            f'Expected {pred:,.0f} MW. Actual demand varies with weather and '
            f'consumption patterns; these bands are calibrated on how far past '
            f'forecasts actually landed, and widen in summer.</p></div>')


# ------------------------------------------------------ recent forecasts
st.html("<div style='height:1.15rem'></div>")
st.html(panel_open("Recent forecasts",
                   f'<span class="ghost">{len(preds)} issued</span>'))
if preds.empty:
    st.html('<div class="empty-s" style="padding:1.6rem 0">No forecasts have '
            'been issued yet.</div>')
else:
    rows = ""
    for _, r in preds.tail(8)[::-1].iterrows():
        act = r.get("actual_mw")
        if pd.notna(act):
            err = abs(float(r.predicted_mw) - float(act))
            acc = max(0.0, 100 - err / float(act) * 100)
            # one hue, two states - green when inside the stated band
            inside = (pd.notna(r.get("lo90")) and
                      float(r.lo90) <= float(act) <= float(r.hi90))
            col = T.GREEN if inside else T.CAT[1]
            bar = (f'<div class="bar"><i style="width:{min(acc, 100):.0f}%;'
                   f'background:{col}"></i></div>')
            actual, accs = f"{float(act):,.0f}", f"{acc:.1f}%"
        else:
            bar = ('<div class="bar"><i style="width:0"></i></div>')
            actual, accs = "<span style='color:#9D9DA6'>pending</span>", "—"
        champ = ("" if r.get("model_is_champion") else
                 f' <span class="chip chip-fl" style="font-size:.625rem">'
                 f'interim</span>')
        rows += (f'<tr><td class="mono">{r.target_date:%d %b %Y}{champ}</td>'
                 f'<td class="num">{float(r.predicted_mw):,.0f}</td>'
                 f'<td class="num">{actual}</td>'
                 f'<td style="width:38%">{bar}</td>'
                 f'<td class="num">{accs}</td></tr>')
    st.html(f'<table class="tb"><thead><tr><th>Forecast date</th>'
            f'<th>Predicted (MW)</th><th>Actual (MW)</th>'
            f'<th>Accuracy</th><th></th></tr></thead><tbody>{rows}</tbody></table>')
    st.html(f'<p style="font-size:.75rem;color:{T.INK_FAINT};margin:.75rem 0 0">'
            f'Actual demand is published about two days after each forecast '
            f'date. Bars show accuracy; green means the outcome fell inside '
            f'the stated likely range.</p>')
st.html("</div>")


# ------------------------------------------------------- weather scenario
st.html("<div style='height:1.15rem'></div>")
sens = (gap or {}).get("phase_5_4_sensitivity", {})
per_c = sens.get("mw_per_degc_t_max")

st.html(panel_open("How weather affects demand",
                   '<span class="ghost">Model scenario, not a forecast</span>'))
if not per_c or latest is None:
    st.html('<div class="empty-s" style="padding:1.6rem 0">Needs an issued '
            'forecast and the measured temperature sensitivity.</div></div>')
else:
    wa, wb = st.columns([1.25, 1], gap="large")
    with wa:
        bump = st.slider("Temperature change vs the forecast day (°C)",
                         -5.0, 8.0, 0.0, 0.5, format="%+.1f °C")
        base_mw = float(latest.predicted_mw)
        scen, shift = base_mw + per_c * bump, per_c * bump
        if abs(bump) < 0.01:
            line = f"At the forecast temperature, the expected peak is {base_mw:,.0f} MW."
        else:
            line = (f"If temperatures were {abs(bump):.1f} °C "
                    f"{'warmer' if bump > 0 else 'cooler'}, demand could "
                    f"{'rise' if shift > 0 else 'fall'} by about "
                    f"{abs(shift):,.0f} MW.")
        st.html(f'<p style="font-size:.9375rem;color:{T.INK};line-height:1.6;'
                f'margin:.4rem 0 0">{line}</p>'
                f'<p style="font-size:.75rem;color:{T.INK_FAINT};margin:.5rem 0 0">'
                f'Applies the measured sensitivity of {per_c:+.0f} MW per °C. '
                f'That is a local slope — real demand flattens at high '
                f'temperature, so large shifts are indicative only.</p>')
    with wb:
        st.html(f'<div style="text-align:right"><div class="k-label">'
                f'Scenario peak</div><div class="k-num">{scen:,.0f}'
                f'<span class="k-unit">MW</span></div>'
                f'<div class="k-foot">{shift:+,.0f} MW vs the '
                f'{base_mw:,.0f} MW forecast</div></div>')
    st.html("</div>")


# ----------------------------------------------------------- performance
st.html("<div style='height:1.15rem'></div>")
st.html(panel_open("How accurate has it been"))
t_live, t_hist = st.tabs(["Live tracking", "Historical"])

with t_live:
    scored = preds.dropna(subset=["actual_mw"]) if not preds.empty else pd.DataFrame()
    if scored.empty:
        empty_state("Your first forecast results are being collected",
                    "Actual demand becomes available for comparison once "
                    "Grid-India publishes it, about two days after each "
                    "forecast date. Accuracy appears here automatically.")
    else:
        e = scored["error_mw"]
        a, b, c, d = st.columns(4)
        for col, lab, val, foot in (
                (a, "Scored", f"{len(scored)}", "forecasts with actuals"),
                (b, "Typical error", f"{e.abs().mean():,.0f}", "MW from actual"),
                (c, "As a share", f"{(e.abs() / scored.actual_mw).mean() * 100:.2f}", "% of demand"),
                (d, "Tendency", f"{e.mean():+,.0f}",
                 "over-forecast" if e.mean() > 0 else "under-forecast")):
            col.html(f'<div class="k-label">{lab}</div><div class="k-num" '
                     f'style="font-size:2rem">{val}</div>'
                     f'<div class="k-foot">{foot}</div>')
        n_sum = int(scored["target_date"].dt.month.isin(SUMMER_MONTHS).sum())
        if n_sum < 20:
            st.html("<div style='height:.8rem'></div>")
            note(f"Only {n_sum} March–June days in this sample. Peak season is "
                 f"when the system is hardest pressed — treat these as "
                 f"provisional until a full summer is recorded.", "warn", "alert")
        cmp_df = scored[["target_date", "actual_mw", "predicted_mw"]].melt(
            "target_date", var_name="s", value_name="mw")
        cmp_df["s"] = cmp_df["s"].map({"actual_mw": "Actual", "predicted_mw": "Forecast"})
        st.altair_chart(finish(
            alt.Chart(cmp_df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(
                size=70, filled=True, stroke="#FBFAF7", strokeWidth=2)).encode(
                x=alt.X("target_date:T", title=None, axis=axis("%d %b")),
                y=alt.Y("mw:Q", title=None, scale=alt.Scale(zero=False),
                        axis=axis(grid=True)),
                color=alt.Color("s:N", title=None, scale=alt.Scale(
                    domain=["Actual", "Forecast"],
                    range=[T.SERIES["recorded"], T.SERIES["forecast"]]),
                    legend=alt.Legend(orient="top", labelFont=T.SANS,
                                      labelColor=T.INK_SOFT, symbolType="stroke",
                                      labelFontSize=12)),
                tooltip=[alt.Tooltip("target_date:T", title="Date", format="%d %b %Y"),
                         alt.Tooltip("s:N", title=""),
                         alt.Tooltip("mw:Q", title="MW", format=",.0f")]), 280),
            width="stretch")

with t_hist:
    if not gap:
        empty_state("Historical results not generated",
                    "The analysis pipeline has not produced evaluation output "
                    "in this environment.", "database")
    else:
        hm = gap.get("H_measured", H)
        head = gap.get("phase_5_3_regimes", {}).get(
            f"H{hm}_gap_lead1", {}).get("overall")
        if head:
            g1, g2, g3 = st.columns(3)
            for col, lab, val, foot in (
                (g1, "With real weather forecasts", f"{head['mae_b']:,.0f}",
                 "MW typical error — achievable in practice"),
                (g2, "With the weather record", f"{head['mae_a']:,.0f}",
                 "MW — only knowable afterwards"),
                (g3, "Cost of not knowing", f"{head['d_mae']:+,.0f}",
                 f"MW · {head['d_mae_pct']:+.1f}% worse")):
                col.html(f'<div class="k-label">{lab}</div><div class="k-num" '
                         f'style="font-size:2rem">{val}</div>'
                         f'<div class="k-foot">{foot}</div>')
            st.html("<div style='height:.9rem'></div>")
            note(f"<b>{head['mae_b']:,.0f} MW is what is achievable tomorrow "
                 f"morning.</b> The {head['mae_a']:,.0f} MW figure assumes "
                 f"perfect knowledge of tomorrow's weather, which no forecaster "
                 f"has. Most published studies report the second number.")
        rows = []
        for k, v in gap.get("phase_5_3_regimes", {}).items():
            if not k.startswith(f"H{hm}") or "gap" in k:
                continue
            o = v.get("overall", {})
            if o:
                rows.append({"Weather used": v.get("label", k),
                             "Typical error (MW)": round(o["mae"], 1),
                             "Typical error (%)": round(o["mape"], 2),
                             "Larger errors (MW)": round(o["rmse"], 1),
                             "Tendency (MW)": round(o["bias"], 1),
                             "Days": o["n"]})
        if rows:
            st.html("<div style='height:.9rem'></div>")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for c in gap.get("caveats", []):
            st.html(f'<p style="font-size:.75rem;color:{T.INK_FAINT};'
                    f'margin:.2rem 0 0">· {c}</p>')
st.html("</div>")


# -------------------------------------------------------------- method
st.html("<div style='height:1.5rem'></div>")
st.html(f'<div class="p-title" style="margin-bottom:.75rem">About this '
        f'forecast</div>')

with st.expander("Where the data comes from"):
    st.markdown(
        f"Daily peak demand is published by **Grid-India** (formerly POSOCO) in "
        f"its Daily Power Supply Position reports. This covers **{len(master):,} "
        f"days**, {master['date'].min():%d %B %Y} to {master['date'].max():%d %B %Y}. "
        f"The highest demand recorded is **{master[PEAK_TARGET].max():,.0f} MW** "
        f"on {master.loc[master[PEAK_TARGET].idxmax(), 'date']:%d %B %Y}.\n\n"
        f"Weather comes from Open-Meteo — historical reanalysis for past days, "
        f"archived day-ahead forecasts for the forecast itself.")

with st.expander("Why the forecast is made before demand data is published"):
    if pub:
        st.markdown(
            f"Grid-India publishes a day's peak the **following morning**. "
            f"Across **{pub['n_run_days']} observed daily runs** of this "
            f"pipeline, same-day data was never available.\n\n"
            f"So a forecast issued this evening for tomorrow can only use "
            f"demand up to **yesterday** — not today. That two-day gap (H = "
            f"{pub['H']}) is measured from the pipeline's own records rather "
            f"than assumed, and it rules out using yesterday's demand, which "
            f"would otherwise be the single most useful input. A genuine "
            f"operating constraint, not a modelling choice.")
    else:
        st.markdown(f"Grid-India publishes a day's peak the following morning, "
                    f"so an evening forecast uses demand through yesterday "
                    f"(H = {H}).")

with st.expander("What the ranges mean"):
    st.markdown(
        "The **likely range** is calibrated so roughly 9 days in 10 fall "
        "inside it; the **wider range**, 19 in 20. These are built from how "
        "far past forecasts actually landed, using a method that makes no "
        "assumption that errors follow a bell curve — ours do not.")
    if iv:
        for season, s in iv.get("seasons", {}).items():
            w = ", ".join(f"{lv}: ±{q * 100:.2f}%"
                          for lv, q in s["relative_halfwidth"].items())
            st.markdown(f"- **{season}** ({s['note']}): {w}")
        if "coverage" in iv:
            st.markdown("Checked against reality: " + ", ".join(
                f"{float(k) * 100:.0f}% band held "
                f"{v['realised_coverage'] * 100:.1f}% of the time"
                for k, v in iv["coverage"].items()))

with st.expander("When accuracy can actually be measured"):
    st.markdown(
        "A forecast can only be scored once real demand for that day is "
        "published, about two days later.\n\n"
        "**Live tracking** counts only forecasts written down *before* the "
        "outcome was known — the honest test, and it starts small. "
        "**Historical** replays the system over years of past data: far more "
        "days to judge by, but a reconstruction rather than a live record. "
        "Both are shown because neither alone is sufficient.")

if latest is not None:
    st.html(f'<div style="margin-top:1.75rem;padding-top:.9rem;border-top:'
            f'1px solid {T.LINE};color:{T.INK_FAINT};font-size:.75rem;'
            f'line-height:1.6">Issued {issued_disp} · weather '
            f'{latest.get("weather_source", "—")} · model '
            f'{latest.get("model_version", "—")} · horizon H='
            f'{int(latest.get("H", H))}<br>All figures read from committed '
            f'pipeline outputs. Nothing on this page is simulated.</div>')

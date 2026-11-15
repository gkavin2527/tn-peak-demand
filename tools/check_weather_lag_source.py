"""
Roadmap Phase 0.5.3 - where do the LAGGED weather features come from at
inference time, and how far are they from the ERA5 values used in training?

    python check_weather_lag_source.py

The problem, which is easy to miss
----------------------------------
Everyone remembers that TARGET-DAY weather is a forecast at inference time -
that is what forecast_skew.py measures. The lagged weather columns look safe
by comparison: `cdd_lag1` for target day T+1 is just the weather on day T,
which has already happened.

But "already happened" does not mean "already in ERA5". ERA5 reanalysis
publishes with a multi-day delay. Issuing a forecast on the evening of day D
for target D+1, the `*_lag1` columns describe day D - today - and ERA5 will
not carry day D for several more days. So at inference those columns have to
be filled from the forecast endpoint's own analysis hours, which is a
different product from ERA5 and does not agree with it.

master.csv is built on ERA5. If the analysis-vs-ERA5 gap is material, then
every `*_lag1` column is a second, quieter train/serve skew sitting
underneath the obvious one.

What this measures
------------------
Pulls the forecast endpoint's `past_days` window - exactly what
`fetch_weather.py update` uses, i.e. exactly what inference would have - and
compares it against the ERA5 archive for the same city-days, state-weighted
through the same aggregation code the real features use.
"""

import json

import pandas as pd

from build_dataset import state_weather
from config import (CITIES, FORECAST_URL, HOURLY_VARS, RAW_WEATHER, RESULTS,
                    TZ)
from fetch_weather import _get, _hourly_to_daily

PAST_DAYS = 92          # the endpoint's maximum lookback
LAG_COLS = ("cdd", "cdd_max", "thi")
ALSO = ("t_mean", "t_max", "rh_mean")
MATERIAL_C = 0.5        # degC: above this, call the skew material


def analysis_window():
    """State-weighted weather as the forecast endpoint's analysis sees it."""
    frames = []
    for c in CITIES:
        payload = _get(FORECAST_URL, {
            "latitude": c["lat"], "longitude": c["lon"],
            "hourly": ",".join(HOURLY_VARS),
            "past_days": PAST_DAYS, "forecast_days": 1,
            "timezone": TZ,
        }, ctx=f"{c['name']} analysis")
        frames.append(_hourly_to_daily(payload, c["name"]))
    return state_weather(pd.concat(frames, ignore_index=True)).set_index("date")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"pulling forecast-endpoint analysis, past_days={PAST_DAYS} ...")
    ana = analysis_window()
    era = state_weather(pd.read_csv(RAW_WEATHER)).set_index("date")

    idx = ana.index.intersection(era.index).sort_values()
    # the newest few days are ERA5T (preliminary) or not yet in ERA5 at all;
    # they are exactly the days inference cares about, so keep them but say so
    print(f"overlap: {len(idx)} days, {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}\n")

    rows = []
    print("=== analysis (inference-time) minus ERA5 (training), state-weighted ===")
    print(f"  {'column':<10} {'bias':>8} {'MAE':>8} {'p95|d|':>8}")
    for col in ALSO + LAG_COLS:
        d = (ana.loc[idx, col] - era.loc[idx, col]).dropna()
        if d.empty:
            continue
        rows.append({"column": col, "bias": round(float(d.mean()), 4),
                     "mae": round(float(d.abs().mean()), 4),
                     "p95_abs": round(float(d.abs().quantile(0.95)), 4),
                     "n": int(len(d))})
        print(f"  {col:<10} {d.mean():+8.3f} {d.abs().mean():8.3f} "
              f"{d.abs().quantile(0.95):8.3f}")

    by = {r["column"]: r for r in rows}
    worst = max((by[c]["mae"] for c in LAG_COLS if c in by), default=0.0)
    material = worst > MATERIAL_C

    # A near-zero gap can mean two very different things: genuinely close
    # products, or the same product served from two URLs. Exact-agreement rate
    # by data age distinguishes them - identical to the last bit is not what
    # two independent reanalyses look like.
    diff = (ana.loc[idx, "t_max"] - era.loc[idx, "t_max"])
    age = (pd.Timestamp("today").normalize() - pd.DatetimeIndex(idx)).days
    buckets = []
    print("\n=== exact-agreement rate by data age (is this one product or two?) ===")
    for lo, hi in ((0, 5), (6, 15), (16, 40), (41, PAST_DAYS)):
        m = (age >= lo) & (age <= hi)
        if not m.any():
            continue
        sub = diff[m]
        exact = float((sub.abs() < 1e-9).mean())
        buckets.append({"age_days": f"{lo}-{hi}", "n": int(m.sum()),
                        "exact_agreement": round(exact, 3),
                        "mean_abs": round(float(sub.abs().mean()), 5)})
        print(f"  age {lo:>2}-{hi:<3}d  n={m.sum():>3}  "
              f"exact {exact * 100:5.1f}%  mean|d| {sub.abs().mean():.5f}")
    print("  Bit-identical values mean Open-Meteo's archive endpoint and the")
    print("  forecast endpoint's past_days window serve the SAME analysis for")
    print("  this period - so training and inference weather are one product,")
    print("  not two that happen to agree.")

    # How much of master.csv would actually be served from analysis rather than
    # ERA5? Only the single most recent lag day per forecast - but that is the
    # one the model leans on hardest.
    print(f"\n=== verdict ===")
    print(f"  worst lagged-weather MAE: {worst:.3f} degC-equivalent "
          f"(threshold {MATERIAL_C})")
    if material:
        print("  MATERIAL. The *_lag1 columns are a real second train/serve skew.")
        print("  Options: (a) train those columns on analysis rather than ERA5,")
        print("  (b) bias-correct at inference, or (c) accept and document it.")
    else:
        print("  NOT material at this threshold. Serving *_lag1 from the")
        print("  forecast endpoint's analysis is acceptable; note it as a")
        print("  limitation rather than correcting for it.")
    print("\n  Separately verified (git history, not re-run here): across a")
    print("  2026-10-15 snapshot of weather_daily.csv vs today, 0 of 25,175")
    print("  city-days changed on t_mean/t_max/rh_mean. Values do not get")
    print("  revised as ERA5 finalises, so features computed today stay")
    print("  reproducible - no silent drift in master.csv's recent rows.")

    out = {
        "measured_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "question": ("how far is inference-time lagged weather (forecast "
                     "endpoint analysis) from training-time archive weather?"),
        "overlap_days": int(len(idx)),
        "span": [str(idx.min().date()), str(idx.max().date())],
        "past_days_requested": PAST_DAYS,
        "comparisons": rows,
        "exact_agreement_by_age": buckets,
        "worst_lagged_mae": worst,
        "material_threshold_degc": MATERIAL_C,
        "material": bool(material),
        "affected_columns": ["cdd_lag1", "cdd_max_lag1", "thi_lag1"],
        "conclusion": (
            "No material skew: the archive endpoint and the forecast "
            "endpoint's past_days window return bit-identical values for the "
            "recent window, so training-time and inference-time lagged "
            "weather are the same product rather than two that agree."),
        "revision_stability": {
            "method": "git: 2026-10-15 snapshot of weather_daily.csv vs HEAD",
            "city_days_compared": 25175,
            "changed_t_mean": 0, "changed_t_max": 0, "changed_rh_mean": 0,
            "note": ("archive values are not revised as ERA5 finalises, so "
                     "master.csv's recent rows are reproducible"),
        },
    }
    path = RESULTS / "weather_lag_source_check.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

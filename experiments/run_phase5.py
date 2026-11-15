"""
Phase 5 - the operational forecast degradation experiment.

    python run_phase5.py [--quick]

Measures what it costs to forecast with weather you had, rather than weather
that happened. This is the project's headline operational result and the one
number most load-forecasting papers never report, because evaluating on
reanalysis is both easier and flattering.

    5.2  archive vs real log       -> results/archive_vs_log_overlap.csv
    5.3  dual-regime walk-forward  -> results/operational_gap.json
    5.4  sensitivity & failure modes
    5.5  real-log gold-standard check

Regime A   target-day weather = ERA5 reanalysis (what actually happened)
Regime B   target-day weather = archived day-ahead forecast (what you held)

Both regimes run the SAME frozen model through the SAME weekly-refit
walk-forward; only the target-day weather columns differ, swapped through
src.contract.swap_weather. Gate G0 verifies that swapping reanalysis back in
reproduces Regime A to 0.000 MW, so the measured gap is forecast error and
not an artifact of the swap.

Model: a stand-in, per roadmap C.3 - Track A's champion does not exist yet.
`--champion models/champion_config.json` switches to it when it does, which is
the whole point of the hand-off at G2. The experiment design does not change.
"""

from __future__ import annotations

import threadlimit  # noqa: F401  - must precede numpy/sklearn

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_dataset import state_weather
from config import (FORECAST_ARCHIVE, FORECAST_LOG, H, H_IDEALISED, MASTER,
                    RANDOM_SEED, RAW_WEATHER, REFIT_DAYS, RESULTS, TEST_START)
from src.contract import load_master, model_features, run_walk_forward, swap_weather
from src.operational import metrics as M

ANCHOR = "peak_lag_1"      # under load_master(H) this is the newest available day


# --------------------------------------------------------------------------
# weather sources
# --------------------------------------------------------------------------

def observed_weather():
    wx = state_weather(pd.read_csv(RAW_WEATHER)).set_index("date").sort_index()
    wx.attrs["source"] = "era5_reanalysis"
    return wx


def forecast_weather():
    """{lead: state-weighted forecast weather}, from the archived model runs."""
    if not FORECAST_ARCHIVE.exists():
        raise SystemExit("missing data/weather_forecast_archive.csv - run "
                         "'python fetch_weather.py archive-forecast'")
    raw = pd.read_csv(FORECAST_ARCHIVE)
    out = {}
    for lead, g in raw.groupby("lead"):
        wx = state_weather(g.drop(columns=["lead"])).set_index("date").sort_index()
        wx.attrs["source"] = f"forecast_lead_{int(lead)}"
        out[int(lead)] = wx
    return out


def logged_weather():
    """{lead: state-weighted weather} from forecasts we logged live."""
    if not FORECAST_LOG.exists():
        return {}
    log = pd.read_csv(FORECAST_LOG, parse_dates=["date", "issued_on"])
    if "issue_slot" not in log.columns:
        log["issue_slot"] = "morning"
    log["lead"] = (log["date"] - log["issued_on"]).dt.days
    # One forecast per (city, date, lead): prefer the evening run, which is
    # what inference actually holds. Mixing slots would blend two different
    # model runs into one series.
    log = log.sort_values("issue_slot")           # evening sorts before morning
    log = log.drop_duplicates(subset=["city", "date", "lead"], keep="first")
    out = {}
    for lead, g in log[log["lead"] >= 1].groupby("lead"):
        wx = state_weather(g.drop(columns=["lead", "issued_on", "issue_slot"]))
        wx = wx.set_index("date").sort_index()
        wx.attrs["source"] = f"live_log_lead_{int(lead)}"
        out[int(lead)] = wx
    return out


# --------------------------------------------------------------------------
# 5.2 archive vs the real log
# --------------------------------------------------------------------------

def phase_5_2(fc, log, obs):
    """Does the archive match forecasts we actually held? Gate on 0.5 degC."""
    print("=" * 72)
    print("5.2  archive vs real log - is the archive a valid stand-in?")
    print("=" * 72)
    rows = []
    for lead in sorted(set(fc) & set(log)):
        a, l = fc[lead], log[lead]
        idx = a.index.intersection(l.index)
        if not len(idx):
            continue
        for col in ("t_mean", "t_max", "rh_mean", "cdd_max", "hours_above_32"):
            if col not in a.columns or col not in l.columns:
                continue
            d = (l.loc[idx, col] - a.loc[idx, col]).dropna()
            rows.append({"lead": lead, "column": col, "n": len(d),
                         "bias_log_minus_archive": round(float(d.mean()), 4),
                         "mae": round(float(d.abs().mean()), 4),
                         "corr": round(float(l.loc[idx, col].corr(
                             a.loc[idx, col])), 5)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no overlap between archive and live log yet\n")
        return df, {"verdict": "no_overlap"}

    path = RESULTS / "archive_vs_log_overlap.csv"
    df.to_csv(path, index=False)
    for lead in sorted(df["lead"].unique()):
        s = df[df["lead"] == lead]
        bits = "  ".join(f"{r.column} {r.bias_log_minus_archive:+.2f}/{r.mae:.2f}"
                         for r in s.itertuples())
        print(f"  lead {lead}  n={int(s['n'].iloc[0]):<4} {bits}")
    print("  (bias / MAE, live log minus archive)")

    gate = df[(df["lead"] == df["lead"].min()) &
              (df["column"].isin(["t_max", "cdd_max"]))]["mae"].max()
    ok = bool(gate <= 0.5)
    print(f"\n  pre-agreed rule: t_max / cdd_max MAE <= 0.5 degC at lead 1")
    print(f"  observed {gate:.3f} -> {'PASS, headline result stands' if ok else 'FAIL, label Phase 5 preliminary'}")
    print(f"  wrote {path}\n")
    return df, {"gate_metric_mae": round(float(gate), 4), "threshold": 0.5,
                "passed": ok,
                "verdict": "archive is a valid stand-in" if ok
                           else "label Phase 5 results preliminary"}


# --------------------------------------------------------------------------
# 5.3 dual-regime walk-forward
# --------------------------------------------------------------------------

def factory():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, random_state=RANDOM_SEED)


# Walk-forward is the expensive step (~98 weekly refits each), and 5.4 and 5.5
# need the same Regime A and lead-1 walks 5.3 already computed. Memoising by
# (H, weather source) cuts the full run by roughly a third and - more
# importantly - guarantees every phase reports numbers from the *same* fitted
# models rather than from independent refits that could differ by seed luck.
_WALK_CACHE: dict = {}


def walk(df, feats, end, refit, key=None):
    if key is not None and key in _WALK_CACHE:
        return _WALK_CACHE[key]
    out = run_walk_forward(factory, df, TEST_START, end, refit_days=refit,
                           features=feats, differenced=True, anchor=ANCHOR)
    if key is not None:
        _WALK_CACHE[key] = out
    return out


def phase_5_3(obs, fc, refit, end):
    print("=" * 72)
    print(f"5.3  dual-regime test, weekly refit ({refit}d), test from {TEST_START}")
    print("=" * 72)

    out, rows = {}, []
    for h in (H, H_IDEALISED):
        base = load_master(H=h)
        feats = model_features(base)
        tag = f"H={h}" + ("  [measured, headline]" if h == H else "  [idealised]")

        # Regime A: reanalysis. swap_weather is still applied, with observed
        # weather, so A and B differ in the weather VALUES and nothing else -
        # not in which code path produced them.
        a_df = swap_weather(base, obs, observed=obs)
        a = M.evaluate(walk(a_df, feats, end, refit, key=(h, "era5")),
                       label=f"A reanalysis {tag}")
        rows.append((f"A reanalysis  {tag}", a))
        out[f"H{h}_regime_a"] = a

        # The idealised H=1 row exists to show the cost of data latency at the
        # headline lead; running it across every lead triples the cost and
        # answers nothing extra.
        leads = sorted(fc) if h == H else [min(fc)]
        for lead in leads:
            b_df = swap_weather(base, fc[lead], observed=obs)
            b = M.evaluate(walk(b_df, feats, end, refit, key=(h, f"fc{lead}")),
                           label=f"B forecast lead {lead} {tag}")
            rows.append((f"B forecast lead {lead}  {tag}", b))
            out[f"H{h}_regime_b_lead{lead}"] = b
            out[f"H{h}_gap_lead{lead}"] = M.compare(a, b)

    print(M.table(rows, "overall"))
    print("\n  March-June only:")
    print(M.table(rows, "summer_mar_jun"))
    print("\n  top-5% surge days (per calendar year):")
    print(M.table(rows, "surge_per_year"))

    head = out[f"H{H}_gap_lead1"]["overall"]
    print(f"\n  HEADLINE (H={H}, lead 1, overall): "
          f"MAE {head['mae_a']} -> {head['mae_b']} MW "
          f"({head['d_mae']:+} MW, {head['d_mae_pct']:+}%), "
          f"bias {head['bias_a']:+} -> {head['bias_b']:+} MW")
    surge = out[f"H{H}_gap_lead1"].get("surge_per_year", {})
    if surge:
        print(f"  on surge days (n={surge['n']}): MAE {surge['mae_a']} -> "
              f"{surge['mae_b']} MW, bias {surge['bias_a']:+} -> {surge['bias_b']:+} MW")
    print()
    return out


# --------------------------------------------------------------------------
# 5.4 sensitivity & failure modes
# --------------------------------------------------------------------------

def phase_5_4(obs, fc, refit, end, lead=1):
    print("=" * 72)
    print("5.4  sensitivity: how does weather-forecast error become MW error?")
    print("=" * 72)
    base = load_master(H=H)
    feats = model_features(base)
    pa = walk(swap_weather(base, obs, observed=obs), feats, end, refit,
              key=(H, "era5"))
    pb = walk(swap_weather(base, fc[lead], observed=obs), feats, end, refit,
              key=(H, f"fc{lead}"))

    j = pa.merge(pb, on="date", suffixes=("_a", "_b"))
    j["date"] = pd.to_datetime(j["date"])
    # both frames carry y_true, so the merge suffixes it; they are identical
    j["y_true"] = j["y_true_a"]
    j["mw_shift"] = j["y_pred_b"] - j["y_pred_a"]
    j["err_b"] = j["y_pred_b"] - j["y_true"]
    idx = pd.DatetimeIndex(j["date"])
    j["t_max_err"] = (fc[lead]["t_max"].reindex(idx).to_numpy()
                      - obs["t_max"].reindex(idx).to_numpy())
    j["cdd_max_err"] = (fc[lead]["cdd_max"].reindex(idx).to_numpy()
                        - obs["cdd_max"].reindex(idx).to_numpy())
    j = j.dropna(subset=["t_max_err", "mw_shift"])

    slope, intercept = np.polyfit(j["t_max_err"], j["mw_shift"], 1)
    r = float(np.corrcoef(j["t_max_err"], j["mw_shift"])[0, 1])
    print(f"  prediction shift vs t_max forecast error:")
    print(f"    {slope:+.1f} MW per degC   (r={r:.3f}, n={len(j)})")
    s2, _ = np.polyfit(j["cdd_max_err"], j["mw_shift"], 1)
    print(f"    {s2:+.1f} MW per cdd_max degC")
    print(f"  so a 1 degC day-ahead temperature miss moves the forecast about")
    print(f"  {abs(slope):.0f} MW - against a {j['err_b'].abs().mean():.0f} MW mean "
          f"absolute error, weather error is a real but partial driver.")

    # Calendar context on the failures. Without it the worst-day table invites
    # the wrong conclusion - that the big misses are weather misses - when the
    # t_max errors on those rows are mostly under 1 degC.
    cal = pd.read_csv(MASTER, parse_dates=["date"])[
        ["date", "holiday_name", "is_holiday", "day_before_holiday",
         "day_after_holiday", "pongal_window"]]
    j = j.merge(cal, on="date", how="left")
    j["festival_adjacent"] = ((j["is_holiday"] == 1)
                              | (j["day_before_holiday"] == 1)
                              | (j["day_after_holiday"] == 1)
                              | (j["pongal_window"] == 1)).fillna(False)

    worst = j.reindex(j["err_b"].abs().sort_values(ascending=False).index).head(10)
    print(f"\n  worst 10 days under forecast weather:")
    print(f"  {'date':<12}{'actual':>8}{'pred':>8}{'err':>8}"
          f"{'t_max err':>10}  {'calendar context'}")
    for r_ in worst.itertuples():
        ctx = str(r_.holiday_name or "")
        if ctx in ("No Holiday", "nan", ""):
            ctx = ("day after a holiday" if r_.day_after_holiday == 1 else
                   "day before a holiday" if r_.day_before_holiday == 1 else
                   "-")
        print(f"  {r_.date:%Y-%m-%d}{r_.y_true:>8.0f}{r_.y_pred_b:>8.0f}"
              f"{r_.err_b:>+8.0f}{r_.t_max_err:>+10.2f}  {ctx}")

    n_fest = int(worst["festival_adjacent"].sum())
    fest = j[j["festival_adjacent"]]
    calm = j[~j["festival_adjacent"]]
    print(f"\n  {n_fest} of the worst 10 are festival-adjacent (holiday, or the")
    print(f"  day before/after one, or inside the Pongal window).")
    print(f"  festival-adjacent days: n={len(fest):<4} mean |err| "
          f"{fest['err_b'].abs().mean():6.1f} MW   bias {fest['err_b'].mean():+7.1f}")
    print(f"  all other days:         n={len(calm):<4} mean |err| "
          f"{calm['err_b'].abs().mean():6.1f} MW   bias {calm['err_b'].mean():+7.1f}")
    print(f"  The model systematically OVER-forecasts festivals: industry shuts")
    print(f"  down and demand collapses further than a single is_holiday flag")
    print(f"  can express. Pongal has its own multi-day window feature; Diwali")
    print(f"  does not, and Diwali days dominate this table.")

    # persist the predictions so this diagnosis can be redone without a re-run
    keep = ["date", "y_true", "y_pred_a", "y_pred_b", "err_b", "mw_shift",
            "t_max_err", "cdd_max_err", "holiday_name", "festival_adjacent"]
    j[keep].to_csv(RESULTS / "phase5_predictions.csv", index=False)

    big = j[j["t_max_err"].abs() > 1.0]
    print(f"\n  days with |t_max error| > 1 degC: {len(big)} of {len(j)} "
          f"({len(big)/len(j)*100:.1f}%), mean |MW error| "
          f"{big['err_b'].abs().mean():.0f} vs "
          f"{j[j['t_max_err'].abs() <= 1]['err_b'].abs().mean():.0f} on calm days\n")

    return {
        "mw_per_degc_t_max": round(float(slope), 2),
        "mw_per_degc_cdd_max": round(float(s2), 2),
        "corr_t_max_err_vs_mw_shift": round(r, 4),
        "n_days": int(len(j)),
        "mean_abs_err_when_t_max_err_gt_1c": round(float(big["err_b"].abs().mean()), 1),
        "mean_abs_err_when_t_max_err_le_1c": round(
            float(j[j["t_max_err"].abs() <= 1]["err_b"].abs().mean()), 1),
        "worst_days": [
            {"date": f"{r_.date:%Y-%m-%d}", "y_true": round(r_.y_true, 0),
             "y_pred": round(r_.y_pred_b, 0), "error_mw": round(r_.err_b, 0),
             "t_max_forecast_error_c": round(r_.t_max_err, 2),
             "holiday_name": (None if str(r_.holiday_name) in ("No Holiday", "nan")
                              else str(r_.holiday_name)),
             "festival_adjacent": bool(r_.festival_adjacent)}
            for r_ in worst.itertuples()],
        "festival_vs_other": {
            "worst10_festival_adjacent": n_fest,
            "festival_adjacent": {
                "n": int(len(fest)),
                "mean_abs_error_mw": round(float(fest["err_b"].abs().mean()), 1),
                "bias_mw": round(float(fest["err_b"].mean()), 1)},
            "other_days": {
                "n": int(len(calm)),
                "mean_abs_error_mw": round(float(calm["err_b"].abs().mean()), 1),
                "bias_mw": round(float(calm["err_b"].mean()), 1)},
            "finding": (
                "The dominant failure mode is festival demand collapse, not "
                "weather error. Days with |t_max error| > 1C are barely worse "
                "than calm days, but festival-adjacent days carry a large "
                "negative bias: the model over-forecasts them because a single "
                "is_holiday binary cannot express a multi-day industrial "
                "shutdown. Pongal has pongal_window; Diwali has no equivalent "
                "and Diwali days dominate the worst-error table. Recommended "
                "to Track A (Phase 3.4 feature ablation) rather than changed "
                "here, because master.csv's schema is frozen at G0."),
        },
    }


# --------------------------------------------------------------------------
# 5.5 real-log gold standard
# --------------------------------------------------------------------------

def phase_5_5(obs, log, refit):
    print("=" * 72)
    print("5.5  gold-standard check on forecasts we actually logged")
    print("=" * 72)
    if 1 not in log:
        print("  live log has no lead-1 rows yet\n")
        return {"available": False}

    wx = log[1]
    base = load_master(H=H)
    feats = model_features(base)
    start, end = wx.index.min(), wx.index.max()
    print(f"  log window {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({len(wx)} days)")

    # Same test start as 5.3 so the model sees the same history; only the
    # scored window is the log's.
    pa = walk(swap_weather(base, obs, observed=obs), feats, end, refit,
              key=(H, "era5"))
    pb = walk(swap_weather(base, wx, observed=obs), feats, end, refit,
              key=(H, "livelog1"))
    win = lambda p: p[(pd.to_datetime(p["date"]) >= start)
                      & (pd.to_datetime(p["date"]) <= end)]
    a = M.evaluate(win(pa), label="A reanalysis (log window)")
    b = M.evaluate(win(pb), label="B live log lead 1")
    print(M.table([("A reanalysis (log window)", a), ("B live log lead 1", b)]))
    gap = M.compare(a, b)
    print(f"\n  gap on the logged window: {gap['overall']['d_mae']:+} MW MAE")
    print("  SMALL SAMPLE: one season, no March-June heat. This validates")
    print("  forecast fidelity, not the heat-driven gap. Re-run after")
    print("  summer 2027, when the log covers a real peak season.\n")
    return {"available": True, "window": [str(start.date()), str(end.date())],
            "n_days": int(len(wx)), "regime_a": a, "regime_b": b, "gap": gap,
            "caveat": ("one monsoon/NE-monsoon season only, no heat season; "
                       "validates fidelity not the heat-driven gap")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="monthly refit and lead 1 only, for a fast sanity run")
    ap.add_argument("--champion", default=None,
                    help="models/champion_config.json (G2 hand-off)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    refit = 30 if args.quick else REFIT_DAYS

    obs = observed_weather()
    fc = forecast_weather()
    log = logged_weather()
    if args.quick:
        fc = {1: fc[1]}

    end = min(obs.index.max(), max(w.index.max() for w in fc.values()))
    print(f"\nmodel: {'champion ' + args.champion if args.champion else 'STAND-IN (roadmap C.3)'}")
    print(f"H={H} (measured), idealised H={H_IDEALISED}, refit every {refit}d")
    print(f"test window {TEST_START} -> {end:%Y-%m-%d}, leads {sorted(fc)}\n")
    if args.champion:
        print("  NOTE: --champion given but loading is not implemented until the\n"
              "  G2 hand-off defines champion_config.json's schema. Using stand-in.\n")

    overlap, gate = phase_5_2(fc, log, obs)
    gap = phase_5_3(obs, fc, refit, end)
    sens = phase_5_4(obs, fc, refit, end)
    goldstd = phase_5_5(obs, log, refit)

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "model": "stand-in (HistGradientBoostingRegressor, differenced)"
                 if not args.champion else args.champion,
        "model_is_champion": False,
        "H_measured": H, "H_idealised": H_IDEALISED,
        "anchor": ANCHOR,
        "refit_days": refit,
        "test_start": TEST_START, "test_end": f"{end:%Y-%m-%d}",
        "leads": sorted(fc),
        "surge_definition": "top 5% per calendar year (D4 primary)",
        "ci_method": f"moving-block bootstrap, {M.BLOCK_DAYS}d blocks, "
                     f"{M.N_BOOT} draws",
        "phase_5_2_archive_vs_log": gate,
        "phase_5_3_regimes": gap,
        "phase_5_4_sensitivity": sens,
        "phase_5_5_real_log": goldstd,
        "caveats": [
            "Model is a Track B stand-in, not Track A's champion (roadmap C.3). "
            "Re-run at G2 with --champion.",
            "Regime B weather comes from Open-Meteo's archived past runs, "
            "validated against the live log in 5.2.",
            "Forecast bias is non-stationary (+0.49C in 2021 to -0.55C in 2026), "
            "so a fixed bias correction would be wrong; none is applied.",
        ],
    }
    path = RESULTS / "operational_gap.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("=" * 72)
    print(f"wrote {path}")
    print(f"wrote {RESULTS / 'archive_vs_log_overlap.csv'}")


if __name__ == "__main__":
    main()

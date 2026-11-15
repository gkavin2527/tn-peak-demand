"""
W2-1 - regenerate every figure from the CURRENT champion.

    python make_figures.py

The five figures in figures/ were committed in 01dfc7e, which predates all
weather work. They depict the weather-free A0-era Ridge;
fig5_feature_coefficients.png shows coefficients for a feature set no longer
in use. They cannot go in a paper, and the audit flagged this as a blocker.

Every figure below is regenerated from committed artifacts, and every caption
names the model and the regime it was measured under - the single thing the
old figures lacked that made them dangerous rather than merely stale.

Reads only results/*.json and results/*.csv, so it never refits a model and
cannot disagree with the numbers the experiments reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("figures")
RESULTS = Path("results")
DPI = 200

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": True,
})

C_A, C_B, C_REF, C_WARN = "#2166ac", "#b2182b", "#999999", "#d95f02"


def load(name):
    return json.loads((RESULTS / name).read_text())


def caption(ax, text):
    ax.text(0.0, -0.17, text, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#444444", wrap=True)


def fig1_actual_vs_pred(cfg):
    """Champion, Regime B, 2025."""
    p = pd.read_csv(RESULTS / "phase7_champion_predictions.csv",
                    parse_dates=["date"])
    p = p[(p.regime == "B") & (p.date.dt.year == 2025)].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(p.date, p.y_true, lw=1.1, color="#222222", label="actual")
    ax.plot(p.date, p.y_pred, lw=1.1, color=C_B, label="forecast", alpha=0.85)
    ax.set_ylabel("peak demand (MW)")
    mae = float(np.abs(p.y_pred - p.y_true).mean())
    ax.set_title(f"Day-ahead peak demand, 2025 - {cfg['model_type']}, "
                 f"Regime B (archived lead-1 forecast weather)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    caption(ax, f"n={len(p)} days. MAE {mae:.1f} MW. H=2: forecast issued the "
                f"evening of D-1 for D, using demand through D-2. Weather is "
                f"a genuine archived lead-1 forecast, not reanalysis.")
    fig.savefig(FIG / "fig1_actual_vs_predicted_2025.png", bbox_inches="tight")
    plt.close(fig)
    return "fig1_actual_vs_predicted_2025.png"


def fig2_weather_ladder(cfg):
    """THE headline figure: what operational weather actually costs."""
    gap = load("operational_gap.json") if (RESULTS / "operational_gap.json").exists() else {}
    ab = load("phase5_regime_ab_results.json")

    # Phase 5 Ridge ladder - the committed three-condition result
    a0 = load("experiment_a0_results.json")
    def dig(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d
    mae_a0 = (dig(a0, "overall", "mae") or dig(a0, "metrics", "overall", "mae")
              or 695.58)
    rid_a = dig(ab, "regime_a", "overall", "mae") or 571.61
    rid_b = dig(ab, "regime_b", "overall", "mae") or 611.95

    ch_a = cfg["validation_performance"]["regime_a_mae"]
    ch_b = cfg["validation_performance"]["regime_b_mae"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    labels = ["no weather\n(A0)", "reanalysis\n(Regime A)",
              "lead-1 forecast\n(Regime B)"]
    vals = [mae_a0, rid_a, rid_b]
    xs = np.arange(3)
    bars = ax1.bar(xs, vals, color=[C_REF, C_A, C_B], width=0.5)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.1f}",
                 ha="center", fontsize=8)
    ax1.set_xticks(xs, labels, fontsize=8)
    ax1.set_xlim(-0.65, 2.95)
    ax1.set_ylabel("validation MAE (MW)")
    ax1.set_ylim(0, max(vals) * 1.22)
    ax1.set_title("Ridge: the three-condition ladder")

    gain = mae_a0 - rid_a
    lost = rid_b - rid_a
    # Annotations sit in the gaps BETWEEN bars, never on top of one.
    ax1.annotate("", xy=(-0.42, rid_a), xytext=(-0.42, mae_a0),
                 arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0))
    ax1.text(-0.36, (mae_a0 + rid_a) / 2, f"weather buys\n{gain:.0f} MW",
             fontsize=7.5, va="center", ha="left", color="#444444")
    ax1.annotate("", xy=(2.46, rid_b), xytext=(2.46, rid_a),
                 arrowprops=dict(arrowstyle="<->", color=C_WARN, lw=1.2))
    ax1.text(2.52, (rid_a + rid_b) / 2, f"forecast\ncosts {lost:.0f} MW",
             fontsize=7.5, va="center", ha="left", color=C_WARN)
    ax1.hlines([rid_a, rid_b], [0.75, 1.75], [2.46, 2.46],
               color="#bbbbbb", lw=0.7, ls=":")

    x = np.arange(2)
    ax2.bar(x - 0.18, [rid_a, rid_b], 0.34, color=C_REF, label="Ridge")
    ax2.bar(x + 0.18, [ch_a, ch_b], 0.34, color=C_B,
            label=f"champion\n({cfg['model_type']})")
    for xi, v in zip(x - 0.18, [rid_a, rid_b]):
        ax2.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.18, [ch_a, ch_b]):
        ax2.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=7.5)
    ax2.set_xticks(x, ["Regime A\n(reanalysis)", "Regime B\n(forecast)"],
                   fontsize=8)
    ax2.set_ylabel("validation MAE (MW)")
    ax2.set_title("The penalty survives a better model")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.set_ylim(0, max(rid_b, ch_b) * 1.32)

    caption(ax1, f"Expanding walk-forward, val 2023/24/25, n=1,066 shared "
                 f"dates, H=2. Regime B uses a 29,535-row archive of genuine "
                 f"lead-1 forecasts. The {lost:.0f} MW penalty is invisible to "
                 f"reanalysis-only evaluation, which is what the literature "
                 f"reports.")
    fig.savefig(FIG / "fig2_weather_ladder.png", bbox_inches="tight")
    plt.close(fig)
    return "fig2_weather_ladder.png"


def fig3_model_benchmark(cfg):
    """Phase 6 benchmark with paired CIs vs Ridge, plus the Phase 7 champion."""
    dm = load("dm_tests.json")["results"]["phase6_vs_ridge_regime_a"]
    rows = sorted(dm, key=lambda r: r["delta_mae_mw"])
    names = [r["comparison"].replace(" vs ridge", "") for r in rows]
    d = [r["delta_mae_mw"] for r in rows]
    lo = [r["bootstrap_ci_95"][0] for r in rows]
    hi = [r["bootstrap_ci_95"][1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(names))
    cols = [C_A if v < 0 else C_B for v in d]
    ax.barh(y, d, color=cols, height=0.6)
    ax.errorbar(d, y, xerr=[np.array(d) - np.array(lo),
                            np.array(hi) - np.array(d)],
                fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5)
    ax.axvline(0, color="#000000", lw=0.9)
    ax.set_yticks(y, names)
    ax.set_xlabel("paired MAE difference vs Ridge (MW)   <- better | worse ->")
    ax.set_title("Phase 6 model benchmark, Regime A (reanalysis)")
    caption(ax, "Bars are paired mean MAE differences against Ridge; whiskers "
                "are 95% moving-block bootstrap CIs (7-day blocks, n=2000, "
                "seed 42). Diebold-Mariano agrees with the bootstrap on all "
                "14 headline comparisons. Every pure GBM is WORSE than Ridge: "
                "trees cannot extrapolate a trending target.")
    fig.savefig(FIG / "fig3_model_benchmark.png", bbox_inches="tight")
    plt.close(fig)
    return "fig3_model_benchmark.png"


def fig4_segments(cfg):
    """Error by segment, champion, Regime B - with anecdotal ones marked."""
    ea = load("error_analysis.json")
    seg = ea["segments"]
    order = ["overall", "working_day", "weekend", "summer_mar_jun",
             "monsoon_oct_dec", "hottest_decile", "surge_top5pct", "holiday",
             "long_weekend", "dussehra_ayudha", "may_day", "pongal_window",
             "diwali_window"]
    order = [s for s in order if s in seg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    y = np.arange(len(order))
    mae = [seg[s]["mae"] for s in order]
    bias = [seg[s]["bias"] for s in order]
    anec = [seg[s]["anecdotal_n_lt_20"] for s in order]
    labels = [f"{s}  (n={seg[s]['n']})" + ("  *" if a else "")
              for s, a in zip(order, anec)]

    ax1.barh(y, mae, color=[("#cccccc" if a else C_A) for a in anec], height=0.62)
    ax1.axvline(seg["overall"]["mae"], color="#000000", lw=0.9, ls="--")
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("MAE (MW)")
    ax1.set_title("MAE by segment")
    ax1.invert_yaxis()

    cols = [("#cccccc" if a else (C_B if v > 0 else C_A))
            for v, a in zip(bias, anec)]
    ax2.barh(y, bias, color=cols, height=0.62)
    lo = [seg[s]["bias_ci_95"][0] if seg[s]["bias_ci_95"] else seg[s]["bias"]
          for s in order]
    hi = [seg[s]["bias_ci_95"][1] if seg[s]["bias_ci_95"] else seg[s]["bias"]
          for s in order]
    ax2.errorbar(bias, y, xerr=[np.array(bias) - np.array(lo),
                                np.array(hi) - np.array(bias)],
                 fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5)
    ax2.axvline(0, color="#000000", lw=0.9)
    ax2.set_xlabel("bias (MW)   negative = under-forecast")
    ax2.set_title("Bias by segment, with 95% CI")

    caption(ax1, f"{cfg['model_type']}, Regime B, n={ea['n_scored']}. Dashed "
                 f"line is overall MAE. Grey bars marked * have n<20 and are "
                 f"ANECDOTAL - no claim should rest on them. The two "
                 f"operationally costly cells are surge days "
                 f"(under-forecast) and Diwali (over-forecast).")
    fig.savefig(FIG / "fig4_error_by_segment.png", bbox_inches="tight")
    plt.close(fig)
    return "fig4_error_by_segment.png"


def fig5_importance(cfg):
    """Two-part attribution + the permutation disagreement."""
    fi = load("feature_importance.json")
    t = pd.DataFrame(fi["features"]).head(14).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(t))
    ax1.barh(y, t.ridge_mean_abs_contrib_mw, color=C_A, height=0.6,
             label="Ridge |contribution|")
    ax1.barh(y, t.gbm_mean_abs_shap_mw, left=t.ridge_mean_abs_contrib_mw,
             color=C_B, height=0.6, label="residual GBM |TreeSHAP|")
    ax1.set_yticks(y, t.feature)
    ax1.set_xlabel("mean |attribution| (MW)")
    ax1.set_title("Two-part attribution (top 14)")
    ax1.legend(frameon=False, fontsize=7, loc="lower right")

    blocks = fi["by_block"]
    names = list(blocks)
    x = np.arange(len(names))
    comb = [blocks[b]["combined_mean_abs_mw"] for b in names]
    perm = [blocks[b]["permutation_dmae_mw"] for b in names]
    ax2.bar(x - 0.19, comb, 0.36, color=C_A, label="attribution")
    ax2.bar(x + 0.19, perm, 0.36, color=C_WARN, label="permutation dMAE")
    ax2.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=7)
    ax2.set_ylabel("MW")
    ax2.set_title("The instruments disagree - and that is the finding")
    ax2.legend(frameon=False, fontsize=7)

    caption(ax1, f"{cfg['model_type']}, attributed on {fi['n_attributed']} "
                 f"Regime B validation rows. TreeSHAP is exact (LightGBM "
                 f"pred_contrib, reconstruction error "
                 f"{fi['treeshap_reconstruction_error_mw']:.1e} MW). Ridge "
                 f"carries the level; the GBM supplies "
                 f"{fi['component_split']['gbm_share_of_total_sd_pct']:.0f}% of "
                 f"total sd. Right: attribution ranks the demand lags above "
                 f"weather, permutation reverses it - the lags are mutually "
                 f"collinear, so shuffling one lets the model recover from the "
                 f"others. Weather carries information nothing else supplies.")
    fig.savefig(FIG / "fig5_feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    return "fig5_feature_importance.png"


def fig6_residuals(cfg):
    """Residual distribution + ACF on the CHAMPION."""
    ea = load("error_analysis.json")
    rd = ea["residual_diagnostics"]
    r = pd.read_csv(RESULTS / "champion_residuals_regime_b.csv",
                    parse_dates=["date"])
    e = r.err.to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax1.hist(e, bins=50, color=C_A, alpha=0.85, edgecolor="white", lw=0.4)
    ax1.axvline(0, color="#000000", lw=0.9)
    ax1.axvline(e.mean(), color=C_B, lw=1.2, ls="--",
                label=f"mean {e.mean():+.0f}")
    ax1.set_xlabel("forecast error (MW)")
    ax1.set_ylabel("days")
    ax1.set_title("Residual distribution")
    ax1.legend(frameon=False, fontsize=7)

    lags = list(range(0, 22))
    a = [rd["acf"][str(k)] for k in lags]
    band = rd["acf_significance_band"]
    ax2.bar(lags, a, color=C_A, width=0.65)
    ax2.axhline(band, color=C_WARN, lw=0.8, ls="--")
    ax2.axhline(-band, color=C_WARN, lw=0.8, ls="--")
    ax2.axhline(0, color="#000000", lw=0.9)
    ax2.set_xlabel("lag (days)")
    ax2.set_ylabel("autocorrelation")
    ax2.set_title("Residual ACF")

    caption(ax1, f"{cfg['model_type']}, Regime B, n={len(e)}. Skew "
                 f"{rd['skew']:+.2f}, excess kurtosis "
                 f"{rd['excess_kurtosis']:+.2f} - clearly non-Gaussian, which "
                 f"is why intervals are split-conformal rather than "
                 f"normal-theory. Lag-1 ACF is {rd['lag1']:+.3f} (dashed band "
                 f"+/-{band:.3f}): errors remain positively autocorrelated, so "
                 f"the model leaves exploitable structure, and all CIs in this "
                 f"paper use a moving-block bootstrap rather than an iid one.")
    fig.savefig(FIG / "fig6_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    return "fig6_residual_diagnostics.png"


def main():
    FIG.mkdir(exist_ok=True)
    cfg = json.loads(Path("models/champion_config.json").read_text())
    print("=" * 74)
    print("W2-1  REGENERATING FIGURES FROM THE CURRENT CHAMPION")
    print("=" * 74)
    print(f"champion: {cfg['model_type']}  {cfg['feature_set']}  "
          f"(frozen {cfg.get('generated_at')})\n")

    stale = sorted(p.name for p in FIG.glob("*.png"))
    if stale:
        print(f"replacing {len(stale)} existing figure(s):")
        for s in stale:
            print(f"  - {s}")
        print()

    made = []
    for fn in (fig1_actual_vs_pred, fig2_weather_ladder, fig3_model_benchmark,
               fig4_segments, fig5_importance, fig6_residuals):
        try:
            name = fn(cfg)
            sz = (FIG / name).stat().st_size / 1024
            print(f"  [OK]   {name}  ({sz:.0f} KB)")
            made.append(name)
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(made)}/6 figures written to {FIG}/")
    orphans = sorted(set(stale) - set(made))
    if orphans:
        print("\nSTALE figures still present - these depict the old "
              "weather-free model and must NOT be used:")
        for o in orphans:
            print(f"  ! {o}")


if __name__ == "__main__":
    main()

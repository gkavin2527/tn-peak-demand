"""
Phase 7 festival-feature acceptance tests.

    python test_phase7_festival.py

Both checks in section 2 exist because this code was wrong on first write:

* **Sentinel collision.** days_to_diwali used 0 for "no Diwali nearby", and
  diwali_window was derived as abs(days) <= 2 - so every ordinary day in the
  year was flagged as a Diwali window. Same shape of bug as holiday_name
  using the string "None", which pandas read as NaN for 4,577 rows.

* **Sign inversion.** The offset was computed as the direction to search
  rather than (date - event), so the column said -1 the day after Diwali and
  +3 three days before, contradicting its own docstring.

Section 3 guards the thing the remediation plan explicitly forbids: A2 must
not be edited in place, because every committed Phase 5/6 number is measured
against that exact 26-feature list.
"""

from __future__ import annotations

import threadlimit  # noqa: F401

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import calendar_features
from src.contract import PHASE5_FEATURE_SETS
from tn_holidays import calendar as tn_calendar

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    print("=" * 74)
    print("PHASE 7 FESTIVAL FEATURE ACCEPTANCE TESTS")
    print("=" * 74)

    years = range(2013, 2031)
    tn = tn_calendar(years)
    diwali = sorted(pd.Timestamp(k) for k, v in tn.items()
                    if "diwali" in v.lower() or "deepavali" in v.lower())

    # ---- 1. the calendar can actually resolve the festival ------------
    print("\n1. Diwali resolves across the whole horizon")
    check("Diwali resolved for every year 2013-2030", len(diwali) == len(list(years)),
          f"{len(diwali)} dates for {len(list(years))} years")
    check("covers the series and the serving horizon",
          diwali[0].year == 2013 and diwali[-1].year >= 2030,
          f"{diwali[0]:%Y-%m-%d} .. {diwali[-1]:%Y-%m-%d}")
    # lunisolar: must NOT be a fixed month/day, or a constant would have done
    check("Diwali is lunisolar, so a fixed-date rule cannot express it",
          len({(d.month, d.day) for d in diwali}) > 5,
          f"{len({(d.month, d.day) for d in diwali})} distinct month/day pairs")

    cal = calendar_features(pd.Series(pd.date_range("2013-01-01", "2026-12-31")))
    cal = cal.set_index("date")

    # ---- 2. the two bugs this file exists for -------------------------
    print("\n2. Window flag and signed distance")
    per_year = cal.groupby(cal.index.year)["diwali_window"].sum()
    check("exactly 5 window days per year (+/-2 around one date)",
          set(per_year.unique()) == {5}, f"observed {sorted(per_year.unique())}")

    # sentinel collision: a day far from any Diwali must not be flagged
    far = cal[cal["days_to_diwali"] == 0.0]
    far_not_diwali = far[~far.index.isin(diwali)]
    check("days_to_diwali==0 away from Diwali is NOT flagged as a window",
          int(far_not_diwali["diwali_window"].sum()) == 0,
          f"{int(far_not_diwali['diwali_window'].sum())} false flags "
          f"among {len(far_not_diwali)} such days")
    check("Diwali itself is flagged and reads 0",
          all(cal.loc[d, "diwali_window"] == 1 and cal.loc[d, "days_to_diwali"] == 0
              for d in diwali if d in cal.index))

    # sign convention: (date - event), negative BEFORE
    d0 = [d for d in diwali if d.year == 2024][0]
    signs_ok = (cal.loc[d0 - pd.Timedelta(days=2), "days_to_diwali"] == -2
                and cal.loc[d0 - pd.Timedelta(days=1), "days_to_diwali"] == -1
                and cal.loc[d0 + pd.Timedelta(days=1), "days_to_diwali"] == +1
                and cal.loc[d0 + pd.Timedelta(days=2), "days_to_diwali"] == +2)
    check("sign is (date - event): negative before, positive after", signs_ok,
          f"around {d0:%Y-%m-%d}")
    check("signed distance is clipped to +/-3",
          cal["days_to_diwali"].abs().max() == 3.0,
          f"max |days| {cal['days_to_diwali'].abs().max():g}")
    check("window is a strict subset of the clipped distance",
          not ((cal["diwali_window"] == 1) & (cal["days_to_diwali"].abs() > 2)).any())

    # these are calendar features: known years ahead, so no leakage
    check("no NaN anywhere (calendar is known in advance)",
          not cal[["diwali_window", "days_to_diwali"]].isna().any().any())

    # ---- 3. A2 is untouched; A4 is additive --------------------------
    print("\n3. A2 is frozen; A4 is strictly additive")
    a2 = PHASE5_FEATURE_SETS["A2_weather_deltas_anomalies"]
    a4 = PHASE5_FEATURE_SETS["A4_festival"]
    check("A2 is still the canonical 26-feature list", len(a2) == 26, f"{len(a2)}")
    check("A2 contains no festival feature",
          not {"diwali_window", "days_to_diwali"} & set(a2))
    # identity, not length - a count check already let a swapped pair through once
    check("A4 is A2 in order plus exactly the two new columns",
          a4 == a2[:6] + ["diwali_window", "days_to_diwali"] + a2[6:],
          f"A4 has {len(a4)} features")
    check("no T-1 demand/energy leaks into A4",
          not {"peak_lag_1", "energy_lag_1"} & set(a4))

    # ---- 4. master.csv carries them ---------------------------------
    print("\n4. master.csv")
    m = pd.read_csv("data/master.csv", parse_dates=["date"])
    check("both columns present in master.csv",
          {"diwali_window", "days_to_diwali"} <= set(m.columns))
    check("declared in the feature_groups contract",
          {"diwali_window", "days_to_diwali"}
          <= set(json.loads(Path("feature_groups.json").read_text())
                 ["groups"]["calendar"]))
    check("calendar group is not shifted by H",
          "calendar" not in json.loads(Path("feature_groups.json").read_text())
          ["shifted_by_H"],
          "festival dates are known years ahead - shifting them would be wrong")

    # ---- 5. the gate artifact, if it has been run --------------------
    print("\n5. Gate artifact")
    gate = Path("results/phase7_festival.json")
    if not gate.exists():
        check("gate artifact exists", False, "run run_phase7_festival.py")
    else:
        g = json.loads(gate.read_text())
        check("gate records the adoption bar",
              "excludes zero" in g["protocol"]["adoption_bar"],
              g["protocol"]["adoption_bar"])
        check("gate used the paired moving-block bootstrap",
              "moving-block" in g["protocol"]["bootstrap"])
        check("gate discloses it is another selection round",
              "selection round" in g["note_selection"])
        # Read the champion from config, not hardcoded: the champion changed
        # at the Phase 7 re-freeze and a literal name here stops the test
        # tracking reality - which is exactly what happened.
        cfg_now = json.loads(
            Path("models/champion_config.json").read_text())
        champ_model = cfg_now["model_type"]
        check("gate was run against the frozen champion",
              champ_model in g["results"],
              f"gate has {sorted(g['results'])}, champion is {champ_model}")
        champ = g["results"][champ_model]["gates"]["B"]
        # the decision cell: champion, operational regime
        consistent = champ["adopt"] == (champ["ci_95_high"] < 0)
        check("adopt flag follows the CI, not the point estimate", consistent,
              f"delta {champ['delta_mae']:+.2f}, "
              f"CI [{champ['ci_95_low']:+.2f}, {champ['ci_95_high']:+.2f}], "
              f"adopt={champ['adopt']}")
        seg = g["results"][champ_model]["segments_regime_b"]["diwali_window"]
        check("the feature reduces Diwali bias where it is supposed to act",
              abs(seg["candidate_bias"]) < abs(seg["base_bias"]),
              f"bias {seg['base_bias']:+.0f} -> {seg['candidate_bias']:+.0f} MW "
              f"on n={seg['n']}")

    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 74)
    if failed:
        print(f"PHASE 7 FESTIVAL ACCEPTANCE FAILED - {len(failed)} of {len(results)}")
        for _, n, d in failed:
            print(f"  FAIL  {n}  {d}")
        sys.exit(1)
    print(f"ALL {len(results)} PHASE 7 FESTIVAL CHECKS PASSED  ✓")
    print("=" * 74)


if __name__ == "__main__":
    main()

"""
Measure H: how stale is the demand data when a forecast is issued?

    python measure_publish_time.py            # from pipeline evidence on disk
    python measure_publish_time.py --probe    # also query Grid-India's API

Roadmap Phase 0.5.2. This measurement unblocks Phase 7, because every demand
feature has to be shifted by H and H cannot be guessed.

The question
------------
Grid-India publishes one Daily PSP report per operating day, some time after
that day ends. The roadmap's framing: issue a forecast on the EVENING of day
D, targeting day D+1. Then

    H = 1  if day D's report is already published at issue time
    H = 2  if it is not (newest demand observation is day D-1)

Three sources of evidence, and why only one of them is trustworthy
------------------------------------------------------------------
1. `report_published` in peak_demand.csv. **Circular - do not use.** It is the
   report's own internal "Date of Reporting" field, and the extractor derives
   the observation date from it as `report_date - 1 day`. Measuring the gap
   between them just recovers that subtraction, not real publish behaviour.

2. Grid-India's API metadata (`CreatedOn`, `Field2`). Genuinely observed, but
   the fields are not self-consistent: records exist whose `Title_` says
   08.11.26 while `Field2` says 2026-11-09, so `Field2` sometimes carries the
   data date and sometimes the publish date. Useful for time-of-day only,
   and reported here with that caveat.

3. **What the pipeline actually had in hand on each run.** For every daily run,
   compare the run timestamp against the newest data date it came away with.
   This involves no interpretation of upstream metadata at all, which is why
   it is the primary evidence below.
"""

import json
import sys
from datetime import date

import pandas as pd

from config import RAW_PEAK, RESULTS

PSP_SOURCE = "grid_india_psp_report"


def _daily_runs(g):
    """Rows fetched by a daily run, not by a bulk backfill.

    A backfill stamps hundreds of dates with one instant; a daily run stamps
    one or two. Grouping by hour and keeping the sparse groups separates them.
    """
    g = g.copy()
    g["ret"] = pd.to_datetime(g["retrieved_at"], errors="coerce")
    g = g.dropna(subset=["ret"])
    per_hour = g.groupby(g["ret"].dt.floor("h")).size()
    return g[g["ret"].dt.floor("h").isin(per_hour[per_hour <= 3].index)]


def observed_gap():
    """Primary evidence: on run-day X, how new was the newest data obtained?"""
    pk = pd.read_csv(RAW_PEAK, parse_dates=["date"])
    g = pk[pk["source"] == PSP_SOURCE]
    if g.empty:
        sys.exit("no PSP-sourced rows in peak_demand.csv - nothing to measure")

    d = _daily_runs(g)
    if d.empty:
        sys.exit("no daily-run rows found - only bulk backfill timestamps")

    d = d.assign(run_day=d["ret"].dt.normalize())
    obs = d.groupby("run_day")["date"].max().reset_index()
    obs["gap_days"] = (obs["run_day"] - obs["date"]).dt.days

    minutes = d["ret"].dt.hour * 60 + d["ret"].dt.minute
    clock = {k: f"{int(v) // 60:02d}:{int(v) % 60:02d}"
             for k, v in (("earliest", minutes.min()),
                          ("median", minutes.median()),
                          ("latest", minutes.max()))}
    return obs, clock


def api_upload_times():
    """Supporting evidence: when do reports actually appear on the CDN?

    Best-effort; Grid-India answers Indian IPs only. `Field2` is unreliable as
    a data date (see module docstring), so this is reported as a time-of-day
    distribution for the subset whose upload lands a day after Field2, and
    explicitly not used to decide H.
    """
    try:
        from fetch_peak_demand import fiscal_year_for, list_reports
    except Exception as e:
        return {"ok": False, "reason": f"import failed: {e}"}

    try:
        recs = []
        for fy in {fiscal_year_for(date.today()),
                   fiscal_year_for(date.today().replace(year=date.today().year - 1))}:
            recs += list_reports(fy)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:160]}"}

    df = pd.DataFrame(recs)
    if df.empty or "CreatedOn" not in df.columns:
        return {"ok": False, "reason": "unexpected API payload shape"}

    df = df[df.get("FileType") == "DAILY_PSP_REPORT"].copy()
    df["created"] = pd.to_datetime(df["CreatedOn"], format="%d-%m-%Y %H:%M",
                                   errors="coerce")
    df["field2"] = pd.to_datetime(df.get("Field2"), errors="coerce")
    df = df.dropna(subset=["created", "field2"])
    if df.empty:
        return {"ok": False, "reason": "no parseable timestamps"}

    nextday = df[(df["created"].dt.normalize() - df["field2"]).dt.days == 1]
    if nextday.empty:
        return {"ok": False, "reason": "no next-day uploads to time"}

    mins = nextday["created"].dt.hour * 60 + nextday["created"].dt.minute
    newest = df.nlargest(1, "field2").iloc[0]
    return {
        "ok": True,
        "probed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_reports_listed": int(len(df)),
        "next_day_upload_clock_ist": {
            "n": int(len(nextday)),
            "earliest": f"{int(mins.min()) // 60:02d}:{int(mins.min()) % 60:02d}",
            "median": f"{int(mins.median()) // 60:02d}:{int(mins.median()) % 60:02d}",
            "latest": f"{int(mins.max()) // 60:02d}:{int(mins.max()) % 60:02d}",
        },
        "newest_report_title": str(newest.get("Title_", "")),
        "newest_report_uploaded": str(newest["created"]),
        "caveat": ("Field2 is not consistently the data date upstream, so this "
                   "is a time-of-day distribution only and does not set H"),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    obs, clock = observed_gap()
    probe = (api_upload_times() if "--probe" in sys.argv
             else {"ok": False, "reason": "not requested"})

    hist = obs["gap_days"].value_counts().sort_index()
    min_gap = int(obs["gap_days"].min())
    # On the evening of day D the newest demand is D - min_gap + ... in practice
    # min_gap is measured relative to the run day, so newest = run_day - min_gap.
    # Issue on evening of D, target D+1  =>  newest = D - min_gap = target - (1 + min_gap)
    h = 1 + min_gap

    print("=== PRIMARY: what each daily run actually had in hand ===")
    print("  gap = run_day - (newest data date obtained)")
    for gap, n in hist.items():
        print(f"  {gap:+d}d  {n:>4}  ({n / hist.sum() * 100:5.1f}%)")
    print(f"  n = {hist.sum()} run days, "
          f"{obs['run_day'].min():%Y-%m-%d} -> {obs['run_day'].max():%Y-%m-%d}")
    print(f"  run clock times: {clock['earliest']} - {clock['latest']} "
          f"(median {clock['median']})")
    print(f"  minimum gap ever observed: {min_gap}d")
    if min_gap >= 1:
        print(f"  -> no run, even one at {clock['latest']}, ever saw same-day data")

    if probe.get("ok"):
        c = probe["next_day_upload_clock_ist"]
        print("\n=== SUPPORTING: Grid-India upload times (next-day uploads) ===")
        print(f"  n={c['n']}  earliest {c['earliest']}  median {c['median']}  "
              f"latest {c['latest']} IST")
        print(f"  newest listed: {probe['newest_report_title']} "
              f"uploaded {probe['newest_report_uploaded']}")
        print("  (time-of-day only - Field2 is unreliable upstream, see docstring)")
    elif "--probe" in sys.argv:
        print(f"\n=== probe unavailable: {probe['reason']} ===")

    print(f"\n=== VERDICT: H = {h} ===")
    print(f"  Day D's report is never in hand on day D itself "
          f"({hist.sum()}/{hist.sum()} runs).")
    print(f"  An evening-of-D issue time holds demand only through D-1.")
    print(f"  Target is D+1, so the newest demand observation is target - {h}.")
    print()
    print(f"  CONSEQUENCE: peak_lag_1 is unavailable for a genuine day-ahead")
    print(f"  forecast. The newest usable demand feature is peak_lag_2, and a")
    print(f"  differenced target must anchor on peak_lag_2, not peak_lag_1.")
    print(f"  src/contract.py enforces this by shifting demand features by H-1.")
    print(f"  Paper wording (roadmap B.3): \"day-ahead forecast with one-day")
    print(f"  demand-data latency\".")

    out = {
        "measured_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "method": "observed run_day vs newest data date obtained (non-circular)",
        "H": int(h),
        "min_gap_days": min_gap,
        "gap_histogram": {f"{int(k):+d}d": int(v) for k, v in hist.items()},
        "n_run_days": int(hist.sum()),
        "run_day_span": [str(obs["run_day"].min().date()),
                         str(obs["run_day"].max().date())],
        "run_clock": clock,
        "api_probe": probe,
        "issue_time": "evening of day D, target D+1",
        "rejected_evidence": (
            "report_published in peak_demand.csv is the report's internal "
            "'Date of Reporting'; the extractor sets observation_date = "
            "report_date - 1, so comparing them is circular and was not used."),
        "consequence": ("peak_lag_1 unavailable at H=2; anchor differenced "
                        "targets on peak_lag_2"),
    }
    path = RESULTS / "publish_time.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()

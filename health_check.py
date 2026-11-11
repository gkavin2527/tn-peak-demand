"""
Verify the dataset is healthy. Exits non-zero on failure so the GitHub Action
turns red and you get an email instead of quietly collecting broken data.

    python health_check.py
"""

import sys
import time
from datetime import date

import pandas as pd

from config import (FORECAST_DAYS, FORECAST_LOG, H_BY_GROUP, LEAKY_SAMEDAY,
                    MASTER, PEAK_TARGET, RAW_DEMAND, RAW_PEAK, RAW_POSOCO,
                    RAW_WEATHER, TARGET_COL)

MAX_DEMAND_LAG = 5      # upstream CSV normally sits ~1 day behind
MAX_PEAK_LAG = 5
MAX_WEATHER_LAG = 3
MAX_MASTER_LAG = 1      # master is rebuilt from demand on every run
DEAD_SOURCE_LAG = 14    # past this, upstream is not late - it has stopped
MIN_ROWS = 3000
MAX_PROVISIONAL_DAYS = 10   # ERA5 lands ~5 days late; more means reconcile is not running


def load(path, date_col, label, problems):
    """Read a dated CSV. Returns (df, latest, lag_in_days), or Nones."""
    if not path.exists():
        problems.append(f"{label}: file missing ({path.name})")
        return None, None, None

    df = pd.read_csv(path)
    if date_col not in df.columns:
        problems.append(f"{label}: no '{date_col}' column")
        return None, None, None

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    latest = df[date_col].max()
    lag = (pd.Timestamp(date.today()) - latest).days
    return df, latest, lag


def report(label, df, latest, lag, status):
    print(f"  {label:<16} {len(df):>7,} rows   latest {latest:%Y-%m-%d}   "
          f"{lag}d behind   {status}")


def upstream_latest():
    """
    Latest dated row in the raw upstream download, or None if it isn't on disk.

    fetch_demand.py rewrites this file every run, so in CI it is what the
    source served us minutes ago. Comparing it to our slice separates "the
    source hasn't published yet" from "we downloaded new rows and then lost
    them" - identical staleness, completely different problem. It is
    gitignored, so a standalone local run may not have it.
    """
    if not RAW_POSOCO.exists():
        return None

    # Only a download from this run can be compared against the slice. A
    # leftover from an old local run says nothing about what the source holds
    # today, and against a freshly pulled demand_daily.csv it would read as
    # rows going missing when nothing is wrong.
    if time.time() - RAW_POSOCO.stat().st_mtime > 24 * 3600:
        return None
    try:
        raw = pd.read_csv(RAW_POSOCO, usecols=["yyyymmdd", TARGET_COL])
    except (ValueError, pd.errors.ParserError):
        return None    # schema drift; fetch_demand.py is the one that reports it

    d = pd.to_datetime(raw.dropna(subset=[TARGET_COL])["yyyymmdd"],
                       format="%Y%m%d", errors="coerce")
    return None if d.isna().all() else d.max()


def demand_status(latest, lag, upstream, problems, warnings):
    """Freshness verdict for the demand slice. Returns the status string."""
    if upstream is not None and upstream > latest:
        dropped = (upstream - latest).days
        problems.append(
            f"demand: upstream published through {upstream:%Y-%m-%d} but "
            f"demand_daily.csv stops at {latest:%Y-%m-%d} - {dropped} day(s) of "
            f"available data is not landing, so the slice or the commit is broken")
        return "BROKEN"

    if lag <= MAX_DEMAND_LAG:
        return "ok"

    # Past the limit, but we hold everything the source has published. The
    # delay is theirs: Grid India's reports slip over weekends and national
    # holidays, and the republication runs on its own cadence. Going red every
    # morning for a third party's publishing schedule only teaches you to
    # ignore a red run - and the rows are not lost, they arrive when upstream
    # catches up. So warn, until the silence is long enough that "late" stops
    # being the likely explanation.
    if lag > DEAD_SOURCE_LAG:
        problems.append(
            f"demand: nothing published since {latest:%Y-%m-%d}, {lag} days ago "
            f"(limit {DEAD_SOURCE_LAG}) - assume the source has moved or died "
            f"rather than that it is running late")
        return "DEAD"

    why = ("upstream is level with us"
           if upstream is not None else
           "raw download absent, cannot tell whose delay this is")
    warnings.append(f"demand: {lag} days behind - {why}; "
                    f"fails at {DEAD_SOURCE_LAG}d")
    return "STALE"


def main():
    problems, warnings = [], []

    print("data freshness")
    demand, d_latest, d_lag = load(RAW_DEMAND, "date", "demand", problems)
    if demand is not None:
        status = demand_status(d_latest, d_lag, upstream_latest(), problems, warnings)
        report("demand", demand, d_latest, d_lag, status)

    weather, w_latest, w_lag = load(RAW_WEATHER, "date", "weather", problems)
    if weather is not None:
        report("weather", weather, w_latest, w_lag,
               "ok" if w_lag <= MAX_WEATHER_LAG else "STALE")
        if w_lag > MAX_WEATHER_LAG:
            problems.append(f"weather: {w_lag} days behind (limit {MAX_WEATHER_LAG})")

        # Provenance. 'update' rows are the forecast model's own analysis, not
        # ERA5 reanalysis, and the two differ by ~0.9 C. A handful at the tail
        # is normal - ERA5 lands ~5 days late. A pile of them means reconcile
        # has stopped running and a temperature step is accumulating inside the
        # history every model is trained on.
        if "source" in weather.columns:
            prov = int((weather["source"] == "update").sum())
            cities = max(len(weather["city"].unique()), 1) if "city" in weather else 1
            print(f"  {'  provisional':<16} {prov:>7,} city-days "
                  f"(~{prov / cities:.0f} days awaiting ERA5)")
            if prov / cities > MAX_PROVISIONAL_DAYS:
                warnings.append(
                    f"weather: {prov / cities:.0f} days still provisional "
                    f"(limit {MAX_PROVISIONAL_DAYS}) - run "
                    f"'python fetch_weather.py reconcile'")

    peak, p_latest, p_lag = load(RAW_PEAK, "date", "peak demand", problems)
    if peak is not None:
        report("peak demand", peak, p_latest, p_lag,
               "ok" if p_lag <= MAX_PEAK_LAG else "STALE")
        if p_lag > MAX_PEAK_LAG:
            warnings.append(f"peak demand: {p_lag} days behind (limit {MAX_PEAK_LAG}d)")

    master, m_latest, m_lag = load(MASTER, "date", "master", problems)
    if master is not None:
        # Judged against available inputs (peak demand and weather), not against today
        inputs = [("peak demand", p_latest)]
        if weather is not None:
            inputs.append(("weather", w_latest))
        valid_inputs = [(name, dt) for name, dt in inputs if dt is not None]
        if valid_inputs:
            ref_name, ref_latest = min(valid_inputs, key=lambda x: x[1])
        elif d_latest is not None:
            ref_name, ref_latest = "demand", d_latest
        else:
            ref_name, ref_latest = "today", None
        if ref_latest is None:
            behind, ref = m_lag, "today"
        else:
            behind, ref = (ref_latest - m_latest).days, ref_name
        stale = behind > MAX_MASTER_LAG
        report("master", master, m_latest, m_lag, "STALE" if stale else "ok")
        if stale:
            problems.append(
                f"master: {behind} days behind {ref} (latest {m_latest:%Y-%m-%d}) "
                f"- build_dataset.py has not rebuilt it from the latest data")

    if FORECAST_LOG.exists():
        fl = pd.read_csv(FORECAST_LOG)
        issued = pd.to_datetime(fl["issued_on"]).dt.date
        days = issued.nunique()
        lag = (date.today() - issued.max()).days
        print(f"  {'forecast log':<16} {days:>7,} days   latest {issued.max()}   "
              f"{lag}d behind   {'ok' if lag <= 2 else 'STALE'}")
        if lag > 2:
            problems.append(f"forecast log: not written for {lag} days "
                            f"(this data cannot be recovered later)")

        # Horizon coverage: the log is only useful for evaluating an h-day
        # forecast if it actually contains h-day-ahead rows. Logging fewer
        # horizons than FORECAST_DAYS silently caps what you can ever measure.
        h = (pd.to_datetime(fl["date"]) - pd.to_datetime(fl["issued_on"])).dt.days
        reach = int(h.max()) if len(h) else -1
        if reach < FORECAST_DAYS - 1:
            warnings.append(
                f"forecast log reaches only h={reach} but FORECAST_DAYS="
                f"{FORECAST_DAYS} asks for h={FORECAST_DAYS - 1}; "
                f"longer horizons cannot be backfilled later")

        # Interior gaps matter as much as staleness: a run that failed three
        # weeks ago leaves a permanent hole no archive can fill, and the log
        # would still look "fresh" today. Warn rather than fail - the hole is
        # already unrecoverable, so going red every day after would only
        # train you to ignore this check.
        uniq = pd.Series(sorted(issued.unique()))
        if len(uniq) > 1:
            d = uniq.diff().dropna().dt.days
            gaps = [(uniq[i], int(g)) for i, g in d.items() if g > 2]
            if gaps:
                total = sum(g - 1 for _, g in gaps)
                warnings.append(
                    f"forecast log has {len(gaps)} gap(s) over 2 days "
                    f"({total} issue-dates missing, unrecoverable): "
                    + ", ".join(f"{g}d before {dt}" for dt, g in gaps[:5])
                    + ("..." if len(gaps) > 5 else ""))
    else:
        warnings.append("forecast log missing - start it, it cannot be backfilled")

    if master is not None:
        print("\nintegrity")
        if len(master) < MIN_ROWS:
            problems.append(f"master: only {len(master):,} rows (expected >{MIN_ROWS:,})")

        span = pd.date_range(master["date"].min(), master["date"].max(), freq="D")
        gaps = span.difference(pd.DatetimeIndex(master["date"]))
        print(f"  missing days   {len(gaps)}")

        if PEAK_TARGET not in master.columns:
            problems.append(f"master: no '{PEAK_TARGET}' column - primary target is missing")
        else:
            y = master[PEAK_TARGET]
            print(f"  peak MW range  {y.min():,.0f} - {y.max():,.0f} MW")
            if y.min() < 3000 or y.max() > 35000:
                problems.append(f"peak target out of plausible range: {y.min():.0f}-{y.max():.0f} MW")

            if master.tail(30)[PEAK_TARGET].nunique() == 1:
                problems.append("peak target constant over last 30 rows - upstream may be broken")

            if "cdd" in master.columns:
                corr = y.corr(master["cdd"])
                print(f"  cdd-peak corr  {corr:.3f}")
                if abs(corr) < 0.2:
                    problems.append(f"weather-peak correlation collapsed to {corr:.3f} "
                                    f"- the join may be misaligned")

        # Source-staleness drift. config.H_BY_GROUP encodes how far behind
        # each upstream publication runs: demand/energy come from the PSP
        # report, the sr_* grid columns from a slice that lands a day later.
        # Those offsets are what load_master() and predict_next_day.py shift
        # by, so if upstream changes its cadence the declared offsets go stale
        # and features silently turn up NaN on the row being predicted -
        # present in training, absent at serving, with nothing to warn you.
        # This is the daily-CI half of the same check test_contract.py makes.
        if peak is not None and demand is not None:
            grid_col = next((c for c in ("sr_max_demand_mw", "sr_demand_met_mw")
                             if c in demand.columns), None)
            if grid_col is not None:
                newest_demand = peak.dropna(subset=[PEAK_TARGET])["date"].max()
                newest_grid = demand.dropna(subset=[grid_col])["date"].max()
                observed = int((newest_demand - newest_grid).days)
                declared = H_BY_GROUP["grid"] - H_BY_GROUP["demand"]
                print(f"  grid lag       {observed}d behind demand "
                      f"(H_BY_GROUP declares {declared}d)")
                if observed != declared:
                    warnings.append(
                        f"grid source is {observed} day(s) behind demand but "
                        f"config.H_BY_GROUP declares {declared} - sr_*_lag1 "
                        f"will be "
                        f"{'NaN at inference' if observed > declared else 'staler than necessary'}"
                        f". Update H_BY_GROUP and re-run test_contract.py")

        # Leak guard. Every column in master.csv must be knowable before the
        # target day begins. These are published the morning after, so their
        # presence means a feature-engineering change has quietly reintroduced
        # the leak that makes a backtest look ~24% better than reality.
        reintroduced = [c for c in LEAKY_SAMEDAY if c in master.columns]
        print(f"  same-day leaks {len(reintroduced)}")
        if reintroduced:
            problems.append(
                "master: same-day columns present that are published after the "
                f"target day: {', '.join(reintroduced)} - these are not "
                "available at prediction time; lag them instead")

        # Energy lives in the raw slice now, not in master
        if demand is not None and "tn_energy_gwh" in demand.columns:
            e = demand["tn_energy_gwh"].dropna()
            if len(e):
                print(f"  energy range   {e.min():,.0f} - {e.max():,.0f} GWh")

        # Calendar drift. Counted on the CALENDAR, not on master's rows: an
        # earlier version counted rows and flagged 2013 at 16 holidays against
        # a median of 23, which was really the 82-day Q1 2013 row gap swallowing
        # 8 holidays. Counting rows measures the gaps; counting the calendar
        # measures what this check is actually for - whether a holidays-package
        # upgrade moved the calendar under a feature the model relies on.
        try:
            from tn_holidays import coverage
            years = range(int(master["date"].dt.year.min()),
                          int(master["date"].dt.year.max()) + 1)
            cov = coverage(years)
            print(f"  holidays/year  {cov.min()}-{cov.max()} (median {cov.median():.0f})")
            if cov.min() < cov.median() - 5:
                thin = cov[cov < cov.median() - 5]
                warnings.append(
                    f"holiday calendar thin in {list(thin.index)} "
                    f"({thin.min()} vs median {cov.median():.0f}) - a festival "
                    f"series may have lost coverage; run python tn_holidays.py")
        except ValueError as e:
            problems.append(f"holiday calendar: {e}")
        except ImportError:
            warnings.append("tn_holidays.py not importable - is_holiday may be "
                            "inconsistent across years")

    print()
    for w in warnings:
        print(f"WARNING  {w}")
    for p in problems:
        print(f"FAIL     {p}")

    if problems:
        print(f"\n{len(problems)} problem(s) found")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

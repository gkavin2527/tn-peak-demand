"""
Open-Meteo weather pipeline.

Four modes:
    python fetch_weather.py backfill [--force]        # history -> weather_daily.csv
    python fetch_weather.py update                    # recent actuals -> weather_daily.csv
    python fetch_weather.py reconcile [--days N]      # replace provisional rows with ERA5
    python fetch_weather.py forecast [--slot=evening] # -> weather_forecast_log.csv
    python fetch_weather.py archive-forecast [--force] # -> weather_forecast_archive.csv

At prediction time you do not have tomorrow's actual weather, only a forecast,
and the two differ systematically. Two files cover that gap:

  weather_daily.csv             observed weather. Rows are ERA5 reanalysis
                                where available and provisional analysis from
                                the forecast endpoint for the last few days,
                                tracked in a `source` column and reconciled to
                                reanalysis once ERA5 catches up. The two differ
                                by ~0.9 C, so mixing them silently would put a
                                step in the middle of every temperature series.
  weather_forecast_log.csv      what WE logged, live, from 2026-08-04 on.
                                Ground truth about our own pipeline, but short.
  weather_forecast_archive.csv  Open-Meteo's archived past model runs, resolved
                                by lead time, back to mid-2021. Long enough for
                                an operational backtest across real seasons.

The live log is still the one that cannot be reconstructed later, so it runs
first in the daily job. The archive is a one-time backfill.
"""

import random
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from config import (ARCHIVE_URL, CACHE, CITIES, FORECAST_ARCHIVE,
                    FORECAST_DAYS, FORECAST_LOG, FORECAST_URL, HOURLY_VARS,
                    MAX_LEAD, PREVIOUS_RUNS_START, PREVIOUS_RUNS_URL,
                    RAW_WEATHER, START_DATE, TZ)

TIMEOUT = 120
BACKOFF = [5, 15, 45, 120]      # seconds between attempts, jittered


def _get(url, params, ctx=""):
    """GET with retries on the failures that actually happen to this pipeline.

    A read timeout on one of 60 backfill requests used to kill the whole run,
    so transient network faults and 5xx are retried alongside the documented
    429. A non-429 4xx is a bad request and will not fix itself, so it raises
    immediately rather than burning three minutes of backoff.
    """
    where = f" [{ctx}]" if ctx else ""
    attempts = len(BACKOFF) + 1        # 5 tries, 4 waits between them
    last = None

    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)

            if r.status_code == 200:
                payload = r.json()
                # a throttle/error body is valid JSON with no 'hourly' key;
                # without this check it becomes a KeyError deep in the caller
                if "hourly" in payload:
                    return payload
                last = f"no 'hourly' in response: {str(payload)[:200]}"
            elif r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            else:
                # 4xx other than 429 is a bad request; it will not fix itself
                r.raise_for_status()

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last = f"{type(e).__name__}: {e}"

        if attempt <= len(BACKOFF):
            sleep = BACKOFF[attempt - 1] * random.uniform(0.8, 1.2)
            print(f"      retry {attempt}/{len(BACKOFF)} after {last} "
                  f"- sleeping {sleep:.0f}s")
            time.sleep(sleep)

    raise RuntimeError(f"open-meteo request failed{where} "
                       f"after {attempts} attempts: {last}")


def _hourly_to_daily(payload, city):
    """Collapse hourly readings into daily aggregates for one city."""
    h = pd.DataFrame(payload["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.date

    g = h.groupby("date")
    out = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(),
        "t_max": g["temperature_2m"].max(),
        "t_min": g["temperature_2m"].min(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "app_t_mean": g["apparent_temperature"].mean(),
        "app_t_max": g["apparent_temperature"].max(),
        # hours spent above a comfort threshold - captures duration of heat,
        # which daily mean temperature throws away
        "hours_above_32": g["temperature_2m"].apply(lambda s: (s > 32).sum()),
    }).reset_index()

    out["city"] = city
    return out


def _chunk_is_complete(path, y_end):
    """True if a cached chunk already covers its whole year.

    Only the current year can be incomplete: it was cached mid-year and the
    days since are still missing. Past years are closed and always complete.
    """
    try:
        cached = pd.read_csv(path)
        return not cached.empty and str(cached["date"].max()) >= y_end
    except Exception:
        return False        # unreadable/truncated cache - just re-fetch it


def backfill(start=START_DATE, end=None, force=False):
    """Pull archive weather city-by-city, year-by-year, into data/_cache/.

    Each chunk is written the instant it succeeds. An earlier version held all
    60 in memory and wrote once at the end, so a timeout on request 20 threw
    away nineteen good ones. Re-running now costs only the chunks still missing.
    """
    end = end or (date.today() - timedelta(days=1)).isoformat()
    years = range(int(start[:4]), int(end[:4]) + 1)

    jobs = []
    for c in CITIES:
        for year in years:
            y_start = max(f"{year}-01-01", start)
            y_end = min(f"{year}-12-31", end)
            if y_start <= y_end:
                jobs.append((c, year, y_start, y_end))

    failed = []
    for i, (c, year, y_start, y_end) in enumerate(jobs, start=1):
        tag = f"[{i}/{len(jobs)}] {c['name']} {year}"
        path = CACHE / f"{c['name']}_{year}.csv"

        if not force and path.exists() and _chunk_is_complete(path, y_end):
            print(f"{tag} ... skipping (cached)")
            continue

        print(f"{tag} ... ", end="", flush=True)
        try:
            payload = _get(ARCHIVE_URL, {
                "latitude": c["lat"], "longitude": c["lon"],
                "start_date": y_start, "end_date": y_end,
                "hourly": ",".join(HOURLY_VARS),
                "timezone": TZ,
            }, ctx=f"{c['name']} {year}")
        except (RuntimeError, requests.exceptions.RequestException) as e:
            # one bad city-year must not cost us the other 59
            print(f"FAILED\n      {e}")
            failed.append(f"{c['name']} {year}")
            continue

        chunk = _hourly_to_daily(payload, c["name"])
        chunk["source"] = "archive"
        chunk.to_csv(path, index=False)
        print(f"ok ({len(chunk)} days)")
        time.sleep(1)

    return _consolidate(failed)


def _consolidate(failed=()):
    """Concat every cached chunk into weather_daily.csv."""
    files = sorted(CACHE.glob("*.csv"))
    if not files:
        sys.exit("no cached chunks - nothing to consolidate")

    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    long = (long.drop_duplicates(subset=["date", "city"], keep="last")
                .sort_values(["date", "city"])
                .reset_index(drop=True))
    long.to_csv(RAW_WEATHER, index=False)
    print(f"\nwrote {len(long):,} city-days from {len(files)} chunks "
          f"-> {RAW_WEATHER}")
    print(f"  range {long['date'].min()} -> {long['date'].max()}")

    if failed:
        # partial data beats no data, but the run is not a success
        print(f"\n{len(failed)} chunk(s) still missing after retries:")
        for f in failed:
            print(f"  {f}")
        print("re-run 'python fetch_weather.py backfill' to retry just these")
        sys.exit(1)
    return long


def update(past_days=7):
    """Append recent ACTUAL weather to weather_daily.csv.

    Uses the forecast endpoint with past_days rather than the archive, because
    the archive is a reanalysis product and runs several days behind. The
    forecast endpoint's past_days window carries observed values and is current
    to yesterday, which is what we need to stay in step with the demand data.
    """
    if not RAW_WEATHER.exists():
        sys.exit("no weather_daily.csv yet - run 'backfill' first")

    today = date.today()
    frames = []

    for c in CITIES:
        payload = _get(FORECAST_URL, {
            "latitude": c["lat"], "longitude": c["lon"],
            "hourly": ",".join(HOURLY_VARS),
            "past_days": past_days, "forecast_days": 1,
            "timezone": TZ,
        })
        df = _hourly_to_daily(payload, c["name"])
        # keep only complete past days; today is partial, tomorrow is a forecast
        df = df[pd.to_datetime(df["date"]).dt.date < today]
        # provisional: this is the forecast model's own analysis, which runs
        # ~0.9 C warmer than ERA5. 'reconcile' replaces these once ERA5 lands.
        df["source"] = "update"
        frames.append(df)
        time.sleep(1)

    new = pd.concat(frames, ignore_index=True)
    old = pd.read_csv(RAW_WEATHER)

    merged = pd.concat([old, new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    before = len(old)
    merged = (merged.drop_duplicates(subset=["date", "city"], keep="last")
                    .sort_values(["date", "city"]))
    if "source" not in merged.columns:
        merged["source"] = "archive"
    merged["source"] = merged["source"].fillna("archive")
    merged.to_csv(RAW_WEATHER, index=False)

    prov = int((merged["source"] == "update").sum())
    print(f"weather updated: {before:,} -> {len(merged):,} city-days "
          f"(latest {merged['date'].max()}, {prov:,} provisional)")


def reconcile(days=45):
    """Replace provisional `update` rows with ERA5 reanalysis where available.

    `update` reads the forecast endpoint's past_days window, which is the
    forecast model's own analysis - current to yesterday, but ~0.9 C warmer
    than the ERA5 reanalysis the rest of the history is built from. Left alone
    it puts a step in every temperature series exactly where the model is
    scored, which shows up as a phantom "forecast cold bias" of -0.75 C that is
    really just two incompatible definitions of "actual".

    ERA5 lands about 5 days behind, so this reruns over a trailing window and
    overwrites what it can. The most recent days legitimately stay provisional.
    """
    if not RAW_WEATHER.exists():
        sys.exit("no weather_daily.csv yet - run 'backfill' first")

    old = pd.read_csv(RAW_WEATHER)
    if "source" not in old.columns:
        old["source"] = "archive"
    old["date"] = pd.to_datetime(old["date"]).dt.date

    start = (date.today() - timedelta(days=days)).isoformat()
    end = (date.today() - timedelta(days=1)).isoformat()
    frames = []

    for c in CITIES:
        try:
            payload = _get(ARCHIVE_URL, {
                "latitude": c["lat"], "longitude": c["lon"],
                "start_date": start, "end_date": end,
                "hourly": ",".join(HOURLY_VARS), "timezone": TZ,
            }, ctx=f"{c['name']} reconcile")
        except (RuntimeError, requests.exceptions.RequestException) as e:
            print(f"  {c['name']}: FAILED - {e}")
            continue
        df = _hourly_to_daily(payload, c["name"])
        # the archive pads the tail with nulls until ERA5 publishes
        df = df.dropna(subset=["t_mean"])
        df["source"] = "archive"
        frames.append(df)
        time.sleep(1)

    if not frames:
        sys.exit("reconcile: no archive data retrieved")

    new = pd.concat(frames, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"]).dt.date

    was = old[old["source"] == "update"]
    replaced = pd.merge(was[["date", "city"]], new[["date", "city"]],
                        on=["date", "city"]).shape[0]

    # new last => archive wins over any provisional row for the same city-day
    merged = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates(subset=["date", "city"], keep="last")
                .sort_values(["date", "city"]))
    merged.to_csv(RAW_WEATHER, index=False)

    left = int((merged["source"] == "update").sum())
    print(f"reconciled {replaced:,} provisional city-days to ERA5 reanalysis; "
          f"{left:,} still provisional (ERA5 has not published them yet)")
    return merged


def _previous_runs_to_daily(payload, city, lead):
    """Collapse one lead time's archived forecast into daily aggregates.

    The previous-runs API returns columns suffixed _previous_day{lead}; strip
    the suffix so the frame matches the reanalysis schema exactly and the two
    can be compared column-for-column.
    """
    sfx = f"_previous_day{lead}"
    hourly = {k[: -len(sfx)]: v for k, v in payload["hourly"].items()
              if k.endswith(sfx)}
    if not hourly or all(all(x is None for x in v) for v in hourly.values()):
        return None

    hourly["time"] = payload["hourly"]["time"]
    out = _hourly_to_daily({"hourly": hourly}, city)
    out["lead"] = lead
    return out


def archive_forecast(start=PREVIOUS_RUNS_START, end=None, force=False):
    """Backfill real past forecasts, by lead time, into FORECAST_ARCHIVE.

    One request per city-year returns every lead at once, so this is 5 cities x
    ~6 years of requests rather than one per day. Chunks are cached exactly
    like the reanalysis backfill so an interrupted run resumes.
    """
    end = end or (date.today() - timedelta(days=1)).isoformat()
    years = range(int(start[:4]), int(end[:4]) + 1)
    leads = range(1, MAX_LEAD + 1)
    variables = ",".join(f"{v}_previous_day{d}" for v in HOURLY_VARS for d in leads)

    jobs = []
    for c in CITIES:
        for year in years:
            y_start = max(f"{year}-01-01", start)
            y_end = min(f"{year}-12-31", end)
            if y_start <= y_end:
                jobs.append((c, year, y_start, y_end))

    failed = []
    for i, (c, year, y_start, y_end) in enumerate(jobs, start=1):
        tag = f"[{i}/{len(jobs)}] {c['name']} {year}"
        path = CACHE / f"fc_{c['name']}_{year}.csv"

        if not force and path.exists() and _chunk_is_complete(path, y_end):
            print(f"{tag} ... skipping (cached)")
            continue

        print(f"{tag} ... ", end="", flush=True)
        try:
            payload = _get(PREVIOUS_RUNS_URL, {
                "latitude": c["lat"], "longitude": c["lon"],
                "start_date": y_start, "end_date": y_end,
                "hourly": variables, "timezone": TZ,
            }, ctx=f"{c['name']} {year} past-runs")
        except (RuntimeError, requests.exceptions.RequestException) as e:
            print(f"FAILED\n      {e}")
            failed.append(f"{c['name']} {year}")
            continue

        parts = [d for d in (_previous_runs_to_daily(payload, c["name"], l)
                             for l in leads) if d is not None]
        if not parts:
            print("no data (before archive start)")
            continue

        chunk = pd.concat(parts, ignore_index=True)
        chunk.to_csv(path, index=False)
        print(f"ok ({len(chunk):,} city-day-leads)")
        time.sleep(1)

    files = sorted(CACHE.glob("fc_*.csv"))
    if not files:
        sys.exit("no forecast chunks cached - nothing to consolidate")

    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    long = (long.dropna(subset=["t_mean"])
                .drop_duplicates(subset=["date", "city", "lead"], keep="last")
                .sort_values(["date", "city", "lead"])
                .reset_index(drop=True))
    long.to_csv(FORECAST_ARCHIVE, index=False)
    print(f"\nwrote {len(long):,} city-day-leads -> {FORECAST_ARCHIVE}")
    print(f"  range {long['date'].min()} -> {long['date'].max()}   "
          f"leads {sorted(long['lead'].unique())}")

    if failed:
        print(f"\n{len(failed)} chunk(s) still missing after retries:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    return long


def log_forecast(slot=None):
    """Pull FORECAST_DAYS of forecast for each city and append with a stamp of
    when it was issued. issued_on is what makes this leak-free later.

    The horizon matters more than it looks. master.csv is built on reanalysis,
    but at prediction time you substitute a forecast, and the two differ
    systematically rather than just noisily. This log is the only way to
    measure that, and it cannot be backfilled from any archive, so log one
    horizon more than you think you need.

    `slot` ('morning' or 'evening') records WHICH run produced the forecast,
    and is part of the dedup key. Without it the evening run would overwrite
    the morning's rows for the same issued_on - losing half the data and, worse,
    making the forecast actually used at inference indistinguishable from the
    one logged twelve hours earlier. Inference runs in the evening (H=2, see
    config.H), so the evening slot is the operationally honest one; the morning
    slot is kept because it is what the first 98 days of this log contain.
    """
    issued_on = date.today().isoformat()
    if slot is None:
        slot = "morning" if datetime.now().hour < 14 else "evening"
    frames = []

    for c in CITIES:
        payload = _get(FORECAST_URL, {
            "latitude": c["lat"], "longitude": c["lon"],
            "hourly": ",".join(HOURLY_VARS),
            "forecast_days": FORECAST_DAYS,
            "timezone": TZ,
        })
        df = _hourly_to_daily(payload, c["name"])
        df["issued_on"] = issued_on
        df["issue_slot"] = slot
        frames.append(df)

    new = pd.concat(frames, ignore_index=True)

    if FORECAST_LOG.exists():
        old = pd.read_csv(FORECAST_LOG)
        # the first 98 days of this log predate the slot split and all came
        # from the 07:00 IST cron
        if "issue_slot" not in old.columns:
            old["issue_slot"] = "morning"
        new = pd.concat([old, new], ignore_index=True)

    # normalise before deduping: rows read back from CSV carry date strings
    # while fresh rows carry datetime.date objects, and the two never compare
    # equal - so re-running on the same day used to append a second copy
    # instead of replacing the first
    new["date"] = pd.to_datetime(new["date"]).dt.date
    new["issued_on"] = pd.to_datetime(new["issued_on"]).dt.date
    new["issue_slot"] = new["issue_slot"].fillna("morning")
    new = (new.drop_duplicates(subset=["issued_on", "issue_slot", "city", "date"],
                               keep="last")
              .sort_values(["issued_on", "issue_slot", "city", "date"])
              .reset_index(drop=True))

    new.to_csv(FORECAST_LOG, index=False)
    print(f"logged {slot} forecast issued {issued_on} -> {FORECAST_LOG} "
          f"({len(new):,} rows)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "backfill"
    if mode == "backfill":
        backfill(force="--force" in sys.argv)
    elif mode == "update":
        update()
    elif mode == "forecast":
        slot = next((a.split("=")[1] for a in sys.argv if a.startswith("--slot=")),
                    None)
        log_forecast(slot)
    elif mode == "reconcile":
        n = [a for a in sys.argv if a.startswith("--days=")]
        reconcile(days=int(n[0].split("=")[1]) if n else 45)
    elif mode == "archive-forecast":
        archive_forecast(force="--force" in sys.argv)
    else:
        sys.exit("usage: fetch_weather.py "
                 "[backfill|update|reconcile|forecast|archive-forecast]")

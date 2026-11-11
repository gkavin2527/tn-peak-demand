"""
A stable Tamil Nadu holiday calendar.

    python tn_holidays.py            # validate and report coverage
    python tn_holidayspy --export    # rewrite data/tn_holidays.csv

Why this exists
---------------
`holidays.India(subdiv="TN")` is the right base - it carries the Tamil
calendar, including Pongal, Thiruvalluvar Day, Uzhavar Thirunal and Puthandu -
but it has coverage gaps, and they are not random. Lunisolar festivals are
resolved in some years and silently absent in others:

    Maha Shivaratri   present in  6 of 15 years
    Ram Navami        present in 12 of 15 years
    Janmashtami       present in 14 of 15 years

The consequence for a load model is worse than a missing row. The same festival
is a holiday in 2017 and an ordinary Tuesday in 2016, so `is_holiday` does not
mean one thing - it means "a holiday the package happened to know about", and a
model learns the package's coverage pattern instead of Tamil Nadu's calendar.

Islamic holidays are NOT affected: Id-ul-Fitr, Bakrid, Muharram and
Milad-un-Nabi are complete across 2013-2026 (2027 carries them with an
"(estimated)" suffix, which is the package flagging that moon-sighting dates
are not yet fixed - a caveat, not a gap).

How the gaps are filled
-----------------------
OVERRIDES below is a curated table. Every entry is validated on import against
the package for the years where BOTH have the festival: all 6 package-known
Maha Shivaratri dates, all 12 Ram Navami dates and all 14 Janmashtami dates
reproduce exactly. A curated series that matches the authority everywhere it
can be checked is the strongest evidence available that the filled years are
right too.

If a future `holidays` release adds a date that CONTRADICTS this table,
validate() raises rather than quietly picking one. That is deliberate: a
disagreement means one of the two is wrong and a human should look.

The resolved calendar is materialised to data/tn_holidays.csv and committed, so
the dataset does not change shape the next time the package is upgraded.
`holidays` is pinned in requirements.txt for the same reason.
"""

import sys
from datetime import date

import holidays
import pandas as pd

from config import TN_HOLIDAYS

SUBDIV = "TN"

# Festivals the package resolves inconsistently, with the full series.
# Dates are the observed Tamil Nadu public holiday.
OVERRIDES = {
    "Maha Shivaratri": [
        "2013-03-10", "2014-02-27", "2015-02-17", "2016-03-07", "2017-02-24",
        "2018-02-13", "2019-03-04", "2020-02-21", "2021-03-11", "2022-03-01",
        "2023-02-18", "2024-03-08", "2025-02-26", "2026-02-15", "2027-03-06",
    ],
    "Ram Navami": [
        "2013-04-19", "2014-04-08", "2015-03-28", "2016-04-15", "2017-04-04",
        "2018-03-25", "2019-04-13", "2020-04-02", "2021-04-21", "2022-04-10",
        "2023-03-30", "2024-04-17", "2025-04-06", "2026-03-26", "2027-04-15",
    ],
    # The package labels this "Janmashtami (Vaishnava)" and uses the later of
    # the two observance days, which is what the override follows.
    "Janmashtami (Vaishnava)": [
        "2013-08-28", "2014-08-18", "2015-09-05", "2016-08-25", "2017-08-15",
        "2018-09-03", "2019-08-24", "2020-08-12", "2021-08-30", "2022-08-19",
        "2023-09-07", "2024-08-26", "2025-08-16", "2026-09-04", "2027-08-25",
    ],
}

OVERRIDE_YEARS = (2013, 2027)      # inclusive span the table covers


def _package(years):
    """{date: name} from the holidays package, names split on '; '."""
    tn = holidays.India(subdiv=SUBDIV, years=sorted(set(years)))
    return {d: str(n) for d, n in tn.items()}


def _has(pkg, festival):
    """Years in which the package resolves `festival`, as {year: date}."""
    out = {}
    for d, names in pkg.items():
        for part in names.split("; "):
            # strip the package's moon-sighting caveat before comparing
            if part.replace(" (estimated)", "") == festival:
                out[d.year] = d
    return out


def validate(years=range(*(OVERRIDE_YEARS[0], OVERRIDE_YEARS[1] + 1))):
    """Cross-check OVERRIDES against the package. Raises on contradiction.

    Returns a per-festival report of how many years were confirmed and how
    many the override supplies on its own.
    """
    pkg = _package(years)
    report, conflicts = {}, []

    for festival, iso_dates in OVERRIDES.items():
        ours = {date.fromisoformat(s).year: date.fromisoformat(s) for s in iso_dates}
        theirs = _has(pkg, festival)

        agreed = [y for y in sorted(set(ours) & set(theirs)) if ours[y] == theirs[y]]
        differed = [y for y in sorted(set(ours) & set(theirs)) if ours[y] != theirs[y]]
        filled = sorted(set(ours) - set(theirs))

        for y in differed:
            conflicts.append(f"{festival} {y}: table says {ours[y]}, "
                             f"holidays=={holidays.__version__} says {theirs[y]}")
        report[festival] = {"confirmed": agreed, "filled": filled, "conflict": differed}

    if conflicts:
        raise ValueError(
            "curated holiday table contradicts the holidays package:\n  "
            + "\n  ".join(conflicts)
            + "\n\nOne of the two is wrong. Check the date against the Tamil Nadu "
              "government holiday notification for that year before changing either.")
    return report


def calendar(years):
    """{date: name} for Tamil Nadu across `years`, gaps filled, validated.

    Outside OVERRIDE_YEARS the package is used as-is and the inconsistency it
    carries is reported by coverage(), not silently papered over.
    """
    years = sorted(set(years))
    cal = dict(_package(years))
    validate()

    lo, hi = OVERRIDE_YEARS
    for festival, iso_dates in OVERRIDES.items():
        for s in iso_dates:
            d = date.fromisoformat(s)
            if d.year not in years or not (lo <= d.year <= hi):
                continue
            if d not in cal:
                cal[d] = festival
            elif festival not in cal[d]:
                # a real holiday already sits on this date (2017: Janmashtami
                # falls on Independence Day) - keep both names, one day
                cal[d] = f"{cal[d]}; {festival}"
    return dict(sorted(cal.items()))


def coverage(years):
    """Per-year holiday counts, for spotting calendar drift."""
    cal = calendar(years)
    s = pd.Series([d.year for d in cal])
    return s.value_counts().sort_index()


def export(years=range(2013, 2031)):
    cal = calendar(years)
    df = pd.DataFrame({"date": list(cal), "holiday_name": list(cal.values())})
    df.to_csv(TN_HOLIDAYS, index=False)
    print(f"wrote {len(df):,} holidays -> {TN_HOLIDAYS}")
    return df


def main():
    rep = validate()
    print(f"holidays=={holidays.__version__}, subdiv={SUBDIV}, "
          f"override span {OVERRIDE_YEARS[0]}-{OVERRIDE_YEARS[1]}\n")
    print("curated overrides, cross-checked against the package:")
    for festival, r in rep.items():
        print(f"  {festival:<26} {len(r['confirmed']):2d} confirmed, "
              f"{len(r['filled']):2d} filled by table")
        if r["filled"]:
            print(f"      filled years: {r['filled']}")
    print("\nper-year holiday count (calendar, not dataset rows):")
    cov = coverage(range(2013, 2028))
    for y, n in cov.items():
        print(f"  {y}  {n:2d}  {'#' * n}")
    print(f"\nspread {cov.min()}-{cov.max()}, median {cov.median():.0f}")

    if "--export" in sys.argv:
        print()
        export()


if __name__ == "__main__":
    main()

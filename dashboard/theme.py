"""Design tokens and UI primitives.

Warm editorial palette: cream surfaces on a soft neutral canvas, a display
serif for headline figures, and a restrained accent set (coral for attention,
green for healthy, muted blue for secondary series). One place for colour,
type and spacing so panels cannot drift apart.

No new dependencies - Streamlit ships Altair, and icons are inline SVG.
"""

from __future__ import annotations

import streamlit as st

# ---- palette -------------------------------------------------------------
INK        = "#16161A"   # headlines, big figures
INK_SOFT   = "#6B6B73"   # body
INK_FAINT  = "#9D9DA6"   # labels, axes
LINE       = "#E7E4DE"   # hairlines
CANVAS     = "#F2F1ED"   # page
SURFACE    = "#FBFAF7"   # cards
SURFACE_W  = "#FDF8F2"   # warm card (centre of a KPI row)

# Categorical series colours. Fixed slot order, assigned per ENTITY and never
# cycled or re-assigned by rank. Validated with the dataviz skill's checker
# against surface #FBFAF7: lightness band, chroma floor, CVD separation
# (worst adjacent dE 9.2 deutan) and normal-vision floor all PASS.
#
# CAT[2] carries a contrast WARN (2.7:1), which obliges visible labels - the
# ranked list beside the donut supplies them. Do not use CAT[2] anywhere a
# label is absent.
CAT        = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua
SERIES     = {"recorded": CAT[0], "forecast": CAT[1]}
DRIVERS    = {"demand_lags": CAT[0], "weather": CAT[1], "calendar_trend": CAT[2]}

# Interface accents - these dress chrome and text, never a data series.
CORAL      = "#DE7553"   # brand accent in nav/eyebrow only
GREEN      = "#46A077"   # status: healthy
GREEN_BG   = "#E7F2EB"
BAND       = ["#eb6834", "#F6D5C4"]   # one hue, two steps: nested ranges
CHIP_BG    = "#EFEDE8"

SERIF = ("'Instrument Serif', 'Newsreader', Georgia, 'Times New Roman', serif")
SANS  = ("Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
         "'Helvetica Neue', Arial, sans-serif")
MONO  = ("'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace")

_LUCIDE = {
    "zap":          "M4 14h6v8l10-12h-6V2z",
    "trending-up":  "M16 7h6v6 M22 7l-8.5 8.5-5-5L2 17",
    "trending-down":"M16 17h6v-6 M22 17l-8.5-8.5-5 5L2 7",
    "activity":     "M22 12h-4l-3 9L9 3l-3 9H2",
    "thermometer":  "M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z",
    "target":       "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Z M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z",
    "info":         "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z M12 16v-4 M12 8h.01",
    "clock":        "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z M12 6v6l4 2",
    "check":        "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z m-3-10 2 2 4-4",
    "alert":        "M12 9v4 M12 17h.01 M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L12.7 3.9a2 2 0 0 0-3.4 0Z",
    "database":     "M12 8c4.4 0 8-1.3 8-3s-3.6-3-8-3-8 1.3-8 3 3.6 3 8 3Z M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5 M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
    "search":       "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z M21 21l-4.3-4.3",
    "bell":         "M10.3 21a1.9 1.9 0 0 0 3.4 0 M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9",
    "caret-up":     "M18 15l-6-6-6 6",
    "caret-down":   "M6 9l6 6 6-6",
}


def icon(name: str, size: int = 16, color: str = INK_SOFT, stroke: float = 1.9) -> str:
    segs = _LUCIDE.get(name, _LUCIDE["info"]).split(" M")
    paths = "".join(f'<path d="{s if i == 0 else "M" + s}"/>'
                    for i, s in enumerate(segs))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex:none;vertical-align:-2px">{paths}</svg>')


def sparkline(values, color=INK, w=92, h=26) -> str:
    """Tiny inline SVG trend line. Returns '' when there is nothing to draw."""
    v = [float(x) for x in values if x == x]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    step = w / (len(v) - 1)
    pts = " ".join(f"{i * step:.1f},{h - (x - lo) / rng * (h - 4) - 2:.1f}"
                   for i, x in enumerate(v))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" fill="none" '
            f'style="display:block"><polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="1.5" stroke-linecap="round" '
            f'stroke-linejoin="round"/></svg>')


def inject_css() -> None:
    st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stApp {{ font-family: {SANS}; }}
.stApp {{ background: {CANVAS}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding: .9rem 1.6rem 4rem; max-width: 1420px; }}
p, span, div, li, label {{ color: {INK_SOFT}; }}

/* ---------- top bar ---------- */
.nav {{ display:flex; align-items:center; gap:1.1rem; padding:.55rem .85rem;
       background:{SURFACE}; border:1px solid {LINE}; border-radius:14px;
       margin-bottom:1.15rem; flex-wrap:wrap; }}
.nav-brand {{ display:flex; align-items:center; gap:.55rem; }}
.nav-mark {{ width:30px;height:30px;border-radius:9px;background:{INK};
            display:flex;align-items:center;justify-content:center;flex:none; }}
.nav-name {{ font-size:.9375rem;font-weight:600;color:{INK};letter-spacing:-.01em; }}
.nav-ver {{ font-size:.6875rem;color:{INK_FAINT};background:{CHIP_BG};
           padding:.14rem .4rem;border-radius:5px;font-weight:500; }}
.seg {{ display:flex;gap:.15rem;background:{CHIP_BG};padding:.2rem;border-radius:10px; }}
.seg span {{ font-size:.8125rem;font-weight:500;color:{INK_SOFT};
            padding:.33rem .8rem;border-radius:8px;white-space:nowrap; }}
.seg span.on {{ background:{INK};color:#fff; }}
.nav-r {{ margin-left:auto;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap; }}
.nav-search {{ display:flex;align-items:center;gap:.5rem;background:{CANVAS};
              border:1px solid {LINE};border-radius:9px;padding:.4rem .7rem;
              min-width:210px; }}
.nav-search em {{ font-style:normal;color:{INK_FAINT};font-size:.8125rem; }}
.pill {{ display:inline-flex;align-items:center;gap:.4rem;background:{SURFACE};
        border:1px solid {LINE};border-radius:20px;padding:.32rem .7rem;
        font-size:.8125rem;color:{INK_SOFT};font-weight:500;white-space:nowrap; }}
.dot {{ width:6px;height:6px;border-radius:50%;flex:none; }}
.who {{ display:flex;align-items:center;gap:.5rem; }}
.who-av {{ width:30px;height:30px;border-radius:50%;background:{CORAL};color:#fff;
          display:flex;align-items:center;justify-content:center;flex:none;
          font-size:.6875rem;font-weight:600;letter-spacing:.02em; }}
.who-n {{ font-size:.8125rem;font-weight:500;color:{INK};line-height:1.25; }}
.who-r {{ font-size:.6875rem;color:{INK_FAINT};line-height:1.25; }}

/* ---------- hero ---------- */
.wash {{ position:relative;border-radius:16px;overflow:hidden;
        padding:1.5rem 1.6rem 1.7rem;margin-bottom:1.15rem;
        background:
          radial-gradient(1100px 230px at 28% -40%, #F6D9C0 0%, rgba(246,217,192,0) 62%),
          radial-gradient(900px 220px at 88% -35%, #CBD9EC 0%, rgba(203,217,236,0) 60%),
          {CANVAS}; }}
.eyebrow {{ display:flex;align-items:center;gap:.45rem;font-size:.6875rem;
           font-weight:600;letter-spacing:.11em;text-transform:uppercase;
           color:{INK_SOFT};margin-bottom:.5rem; }}
.display {{ font-family:{SERIF};font-size:3.1rem;line-height:1.04;color:{INK};
           font-weight:400;letter-spacing:-.018em;margin:0; }}
.display i {{ font-style:italic; }}
.hero-sub {{ font-size:.875rem;color:{INK_SOFT};margin:.5rem 0 0;max-width:46rem;
            line-height:1.55; }}
.hero-row {{ display:flex;align-items:flex-start;justify-content:space-between;
            gap:1.25rem;flex-wrap:wrap; }}
.btn-dark {{ background:{INK};color:#fff;border-radius:9px;padding:.46rem .95rem;
            font-size:.8125rem;font-weight:500;white-space:nowrap;
            display:inline-flex;align-items:center;gap:.4rem; }}

/* ---------- cards ---------- */
.k {{ background:{SURFACE};border:1px solid {LINE};border-radius:14px;
     padding:1.05rem 1.15rem 1rem;height:100%;display:flex;flex-direction:column; }}
.k.warm {{ background:{SURFACE_W}; }}
.k-top {{ display:flex;align-items:flex-start;justify-content:space-between;gap:.6rem; }}
.k-label {{ font-size:.8125rem;color:{INK_SOFT};font-weight:500; }}
.k-num {{ font-family:{SERIF};font-size:2.85rem;line-height:1;color:{INK};
         font-weight:400;letter-spacing:-.02em;margin:.5rem 0 0; }}
.k-unit {{ font-family:{SANS};font-size:.9375rem;color:{INK_SOFT};font-weight:500;
          margin-left:.22rem;letter-spacing:0; }}
.k-bot {{ display:flex;align-items:flex-end;justify-content:space-between;
         gap:.75rem;margin-top:auto;padding-top:.85rem; }}
.k-foot {{ font-size:.75rem;color:{INK_FAINT};line-height:1.45; }}
.chip {{ display:inline-flex;align-items:center;gap:.22rem;padding:.2rem .5rem;
        border-radius:7px;font-size:.75rem;font-weight:600;white-space:nowrap; }}
.chip-up {{ background:{GREEN_BG};color:{GREEN}; }}
.chip-dn {{ background:#FBEDE8;color:{CORAL}; }}
.chip-fl {{ background:{CHIP_BG};color:{INK_SOFT}; }}

/* ---------- panels ---------- */
.p {{ background:{SURFACE};border:1px solid {LINE};border-radius:14px;
     padding:1.05rem 1.2rem 1.1rem;height:100%; }}
.p-head {{ display:flex;align-items:center;justify-content:space-between;
          gap:.75rem;margin-bottom:.9rem;flex-wrap:wrap; }}
.p-title {{ display:flex;align-items:center;gap:.45rem;font-size:.875rem;
           font-weight:600;color:{INK}; }}
.p-title:before {{ content:"";width:5px;height:5px;border-radius:50%;
                  background:{INK};flex:none; }}
.lgd {{ display:flex;gap:.85rem;font-size:.75rem;color:{INK_FAINT}; }}
.lgd i {{ width:8px;height:8px;border-radius:2px;display:inline-block;
         margin-right:.35rem;font-style:normal; }}
.ghost {{ font-size:.75rem;color:{INK_SOFT};background:{CANVAS};
         border:1px solid {LINE};border-radius:8px;padding:.3rem .65rem;
         white-space:nowrap; }}

/* ranked list beside the donut */
.rk {{ display:flex;align-items:center;gap:.6rem;padding:.44rem 0;
      border-bottom:1px solid {LINE}; }}
.rk:last-child {{ border-bottom:0; }}
.rk-sw {{ width:9px;height:9px;border-radius:2px;flex:none; }}
.rk-n {{ font-size:.8125rem;color:{INK};flex:1;white-space:nowrap;
        overflow:hidden;text-overflow:ellipsis; }}
.rk-v {{ font-size:.8125rem;color:{INK};font-weight:600;
        font-variant-numeric:tabular-nums; }}
.rk-p {{ font-size:.75rem;color:{INK_FAINT};min-width:42px;text-align:right;
        font-variant-numeric:tabular-nums; }}

/* table */
.tb {{ width:100%;border-collapse:collapse; }}
.tb th {{ font-size:.6875rem;font-weight:600;color:{INK_FAINT};
         text-transform:uppercase;letter-spacing:.07em;text-align:left;
         padding:0 .6rem .6rem 0;border-bottom:1px solid {LINE};white-space:nowrap; }}
.tb td {{ padding:.62rem .6rem .62rem 0;border-bottom:1px solid {LINE};
         font-size:.8125rem;color:{INK};vertical-align:middle; }}
.tb tr:last-child td {{ border-bottom:0; }}
.tb .num {{ font-variant-numeric:tabular-nums;white-space:nowrap; }}
.tb .mono {{ font-family:{MONO};font-size:.78125rem; }}
.bar {{ height:5px;border-radius:3px;background:{CHIP_BG};overflow:hidden;
       min-width:90px; }}
.bar i {{ display:block;height:100%;border-radius:3px; }}

/* notices + empty */
.note {{ border-radius:10px;padding:.75rem .95rem;font-size:.8125rem;
        line-height:1.55;display:flex;gap:.55rem;align-items:flex-start;
        border:1px solid; }}
.note-info {{ background:{SURFACE};border-color:{LINE};color:{INK_SOFT}; }}
.note-warn {{ background:#FDF4EF;border-color:#F3DCCF;color:#8A4B2A; }}
.empty {{ background:{SURFACE};border:1px dashed {LINE};border-radius:14px;
         padding:2.1rem 1.4rem;text-align:center; }}
.empty-t {{ color:{INK};font-weight:600;font-size:.9375rem;margin:.6rem 0 .3rem; }}
.empty-s {{ color:{INK_FAINT};font-size:.8125rem;line-height:1.55;
           max-width:29rem;margin:0 auto; }}

/* streamlit bits */
.stTabs [data-baseweb="tab-list"] {{ gap:.2rem;background:{CHIP_BG};padding:.2rem;
                                    border-radius:10px;border:0;width:fit-content; }}
.stTabs [data-baseweb="tab"] {{ font-size:.8125rem;font-weight:500;color:{INK_SOFT};
                               padding:.33rem .85rem;border-radius:8px; }}
.stTabs [aria-selected="true"] {{ background:{INK};color:#fff; }}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {{ display:none; }}
div[data-testid="stExpander"] details {{ border:1px solid {LINE};border-radius:12px;
                                        background:{SURFACE}; }}
div[data-testid="stExpander"] summary {{ font-size:.8438rem;font-weight:500;color:{INK}; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{INK}; }}

@media (max-width:1024px) {{
  .display {{ font-size:2.45rem; }} .k-num {{ font-size:2.3rem; }}
  .block-container {{ padding:.75rem 1rem 3rem; }}
}}
@media (max-width:640px) {{
  .display {{ font-size:1.95rem; }} .k-num {{ font-size:2rem; }}
  .nav {{ gap:.7rem; }} .nav-search {{ display:none; }} .seg {{ order:3;width:100%; }}
  .seg span {{ flex:1;text-align:center; }} .nav-r {{ margin-left:0; }}
  .wash {{ padding:1.15rem 1rem 1.25rem; }}
}}
</style>""")


def panel_open(title: str, right: str = "") -> str:
    return (f'<div class="p"><div class="p-head"><div class="p-title">{title}</div>'
            f'{right}</div>')


def empty_state(title: str, body: str, ico: str = "clock") -> None:
    st.html(f'<div class="empty">{icon(ico, 24, INK_FAINT, 1.5)}'
            f'<div class="empty-t">{title}</div>'
            f'<div class="empty-s">{body}</div></div>')


def note(text: str, kind: str = "info", ico: str = "info") -> None:
    c = CORAL if kind == "warn" else INK_FAINT
    st.html(f'<div class="note note-{kind}">{icon(ico, 14, c)}<div>{text}</div></div>')

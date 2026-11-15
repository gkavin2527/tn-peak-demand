"""Design tokens and shared UI primitives for the dashboard.

One place for colour, type and spacing so panels cannot drift apart. The
palette is deliberately restrained: a single accent, two state colours, and a
neutral ramp. Everything else is spacing and hierarchy.

No new dependencies - Streamlit ships Altair, and icons are inline SVG.
"""

from __future__ import annotations

import streamlit as st

# ---- tokens --------------------------------------------------------------
INK          = "#0F172A"   # primary text, deep navy
INK_SOFT     = "#475569"   # secondary text
INK_FAINT    = "#94A3B8"   # tertiary / labels
LINE         = "#E2E8F0"   # borders
SURFACE      = "#FFFFFF"   # cards
CANVAS       = "#F6F7F9"   # page background
ACCENT       = "#2563EB"   # electric blue
ACCENT_SOFT  = "#EFF4FF"
POSITIVE     = "#059669"   # green
POSITIVE_SOFT= "#ECFDF5"
WARN         = "#D97706"   # amber
WARN_SOFT    = "#FFFBEB"
HISTORY      = "#94A3B8"   # actual demand on charts

FONT = ("Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif")

# ---- Lucide icons (inline SVG, 24x24 stroke) -----------------------------
_LUCIDE = {
    "zap":        "M4 14h6v8l10-12h-6V2z",
    "trending-up":"M16 7h6v6 M22 7l-8.5 8.5-5-5L2 17",
    "trending-down":"M16 17h6v-6 M22 17l-8.5-8.5-5 5L2 7",
    "activity":   "M22 12h-4l-3 9L9 3l-3 9H2",
    "thermometer":"M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z",
    "target":     "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Z M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z",
    "info":       "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z M12 16v-4 M12 8h.01",
    "clock":      "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z M12 6v6l4 2",
    "check":      "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z m-3-10 2 2 4-4",
    "alert":      "M12 9v4 M12 17h.01 M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L12.7 3.9a2 2 0 0 0-3.4 0Z",
    "database":   "M12 8c4.4 0 8-1.3 8-3s-3.6-3-8-3-8 1.3-8 3 3.6 3 8 3Z M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5 M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
}


def icon(name: str, size: int = 16, color: str = INK_SOFT, stroke: float = 2) -> str:
    """Inline Lucide-style SVG. Returns markup for embedding in st.html."""
    d = _LUCIDE.get(name, _LUCIDE["info"])
    segs = d.split(" M")
    paths = "".join(
        f'<path d="{s if i == 0 else "M" + s}"/>' for i, s in enumerate(segs))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex:none;vertical-align:-2px">{paths}</svg>')


def inject_css() -> None:
    """Global styling. Scoped to Streamlit's containers, light theme only."""
    st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stApp {{ font-family: {FONT}; }}
.stApp {{ background: {CANVAS}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding: 1.25rem 2rem 4rem; max-width: 1340px; }}

h1, h2, h3, h4 {{ color: {INK}; font-weight: 600; letter-spacing: -0.015em; }}
p, span, div, li {{ color: {INK_SOFT}; }}

/* section heading */
.sec {{ display:flex; align-items:center; gap:.55rem; margin: 2.25rem 0 .35rem; }}
.sec h2 {{ font-size: 1.0625rem; margin:0; font-weight:600; }}
.sec-sub {{ color:{INK_FAINT}; font-size:.8125rem; margin:0 0 1rem; line-height:1.5; }}

/* cards */
.card {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:10px;
        padding:1.125rem 1.25rem; box-shadow:0 1px 2px rgba(15,23,42,.04); height:100%; }}
.card-label {{ color:{INK_FAINT}; font-size:.6875rem; font-weight:600;
              text-transform:uppercase; letter-spacing:.07em; margin-bottom:.5rem;
              display:flex; align-items:center; gap:.4rem; }}
.card-value {{ color:{INK}; font-size:1.75rem; font-weight:600; line-height:1.1;
              letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.card-unit {{ font-size:.9375rem; font-weight:500; color:{INK_SOFT}; margin-left:.2rem; }}
.card-foot {{ color:{INK_FAINT}; font-size:.75rem; margin-top:.4rem; line-height:1.45; }}

/* hero */
.hero {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:12px;
        padding:1.75rem 2rem; box-shadow:0 1px 3px rgba(15,23,42,.05); }}
.hero-eyebrow {{ color:{ACCENT}; font-size:.75rem; font-weight:600;
                text-transform:uppercase; letter-spacing:.08em;
                display:flex; align-items:center; gap:.4rem; }}
.hero-value {{ color:{INK}; font-size:3.75rem; font-weight:700; line-height:1;
              letter-spacing:-.035em; margin:.6rem 0 .1rem;
              font-variant-numeric:tabular-nums; }}
.hero-unit {{ font-size:1.5rem; font-weight:600; color:{INK_SOFT}; margin-left:.35rem; }}
.hero-say {{ color:{INK_SOFT}; font-size:.9375rem; line-height:1.6; margin-top:.9rem;
            max-width:46rem; }}

/* delta chip */
.chip {{ display:inline-flex; align-items:center; gap:.3rem; padding:.22rem .55rem;
        border-radius:6px; font-size:.8125rem; font-weight:600;
        font-variant-numeric:tabular-nums; }}
.chip-up {{ background:{WARN_SOFT}; color:{WARN}; }}
.chip-down {{ background:{POSITIVE_SOFT}; color:{POSITIVE}; }}
.chip-flat {{ background:#F1F5F9; color:{INK_SOFT}; }}

/* header bar */
.topbar {{ display:flex; align-items:center; justify-content:space-between;
          gap:1rem; padding:.9rem 0 1.1rem; border-bottom:1px solid {LINE};
          margin-bottom:.5rem; flex-wrap:wrap; }}
.brand {{ display:flex; align-items:center; gap:.75rem; }}
.brand-mark {{ width:38px; height:38px; border-radius:9px; background:{ACCENT};
              display:flex; align-items:center; justify-content:center; flex:none; }}
.brand-t {{ font-size:1.0625rem; font-weight:700; color:{INK}; line-height:1.2;
           letter-spacing:-.015em; }}
.brand-s {{ font-size:.8125rem; color:{INK_FAINT}; line-height:1.3; }}
.status {{ display:flex; align-items:center; gap:1.25rem; flex-wrap:wrap; }}
.status-i {{ display:flex; flex-direction:column; gap:.1rem; }}
.status-k {{ font-size:.6875rem; color:{INK_FAINT}; font-weight:600;
            text-transform:uppercase; letter-spacing:.06em; }}
.status-v {{ font-size:.8125rem; color:{INK}; font-weight:500;
            display:flex; align-items:center; gap:.35rem; }}
.dot {{ width:7px; height:7px; border-radius:50%; flex:none; }}

/* notices */
.note {{ border-radius:8px; padding:.8rem 1rem; font-size:.8438rem; line-height:1.55;
        display:flex; gap:.6rem; align-items:flex-start; border:1px solid; }}
.note-info {{ background:{ACCENT_SOFT}; border-color:#DBE6FF; color:#1E40AF; }}
.note-warn {{ background:{WARN_SOFT}; border-color:#FDE9C8; color:#92400E; }}
.note-quiet {{ background:#FAFBFC; border-color:{LINE}; color:{INK_SOFT}; }}
.note b {{ color:inherit; }}

/* empty state */
.empty {{ background:{SURFACE}; border:1px dashed {LINE}; border-radius:10px;
         padding:2.25rem 1.5rem; text-align:center; }}
.empty-t {{ color:{INK}; font-weight:600; font-size:.9375rem; margin:.65rem 0 .3rem; }}
.empty-s {{ color:{INK_FAINT}; font-size:.8438rem; line-height:1.55;
           max-width:30rem; margin:0 auto; }}

/* streamlit overrides */
[data-testid="stMetricValue"] {{ font-size:1.5rem; color:{INK}; font-weight:600; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-size:.875rem; font-weight:500; color:{INK_SOFT};
                               padding:.55rem .9rem; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT}; font-weight:600; }}
div[data-testid="stExpander"] details {{ border:1px solid {LINE}; border-radius:9px;
                                        background:{SURFACE}; }}
div[data-testid="stExpander"] summary {{ font-size:.875rem; font-weight:500; color:{INK}; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{ACCENT}; }}

/* responsive */
@media (max-width: 1024px) {{
  .block-container {{ padding:1rem 1.25rem 3rem; }}
  .hero-value {{ font-size:3rem; }}
}}
@media (max-width: 640px) {{
  .block-container {{ padding:.75rem .9rem 2.5rem; }}
  .hero {{ padding:1.25rem 1.1rem; }}
  .hero-value {{ font-size:2.5rem; }}
  .topbar {{ flex-direction:column; align-items:flex-start; gap:.85rem; }}
  .status {{ gap:1rem; }}
  .card-value {{ font-size:1.5rem; }}
}}
</style>""")


# ---- primitives ----------------------------------------------------------
def section(title: str, sub: str = "", ico: str = "activity") -> None:
    st.html(f'<div class="sec">{icon(ico, 17, ACCENT)}<h2>{title}</h2></div>'
            + (f'<p class="sec-sub">{sub}</p>' if sub else ""))


def card(label: str, value: str, unit: str = "", foot: str = "",
         ico: str = "", tone: str = "") -> str:
    col = {"pos": POSITIVE, "warn": WARN, "accent": ACCENT}.get(tone, INK)
    ic = icon(ico, 13, INK_FAINT) if ico else ""
    return (f'<div class="card"><div class="card-label">{ic}{label}</div>'
            f'<div class="card-value" style="color:{col}">{value}'
            f'{f"<span class=card-unit>{unit}</span>" if unit else ""}</div>'
            f'{f"<div class=card-foot>{foot}</div>" if foot else ""}</div>')


def note(text: str, kind: str = "info", ico: str = "info") -> None:
    colr = {"info": ACCENT, "warn": WARN, "quiet": INK_FAINT}.get(kind, ACCENT)
    st.html(f'<div class="note note-{kind}">{icon(ico, 15, colr)}'
            f'<div>{text}</div></div>')


def empty_state(title: str, body: str, ico: str = "clock") -> None:
    st.html(f'<div class="empty">{icon(ico, 26, INK_FAINT, 1.6)}'
            f'<div class="empty-t">{title}</div>'
            f'<div class="empty-s">{body}</div></div>')

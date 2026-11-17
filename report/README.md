# Project report

`TN_Peak_Demand_Project_Report.docx` is the SRM minor-project report for
submission. `TN_Peak_Demand_Project_Report.original.docx` is the version it
was derived from, kept so the edits below can be re-derived or reversed.

## What was changed

**Bonafide certificate.** The supervisor and head-of-department details were
authored as two columns but written as plain paragraphs, so they rendered
run together - `Dr. RAGUPATHI TDr. V. KAVITHA`. Rebuilt as a borderless
two-column table with signature space above.

**Page numbering.** The file was a single section numbered 1-33 straight
through. It is now two sections: the cover is counted but not printed, the
front matter runs ii-ix in lower-case roman, and Chapter 1 restarts the
count at arabic 1. Chapter 1 already carried a `pageBreakBefore`, so that
break was converted into the section break rather than a new one being
added - otherwise a blank page appears between the two.

**Contents.** The table of contents carried no page numbers at all; it
still held the note "Page numbers to be generated from the final Word file".
All 56 entries now have numbers with dot leaders, and the same was applied
to the 12 list-of-tables and 4 list-of-figures entries.

## How the page numbers were derived

They were read from the original PDF's rendered layout, then shifted for the
new scheme (front matter to roman, `arabic = physical page - 9`). They are
static text, not a Word TOC field.

This matters: **they are correct for the document as it stands and will not
update themselves.** If content is added or removed, re-derive them rather
than trusting what is on the page. The alternative - a live `TOC` field -
was not used because it renders as "Right-click to update" until someone
presses F9, which is worse on a file that gets submitted as-is.

## Dot leaders

The contents dots are literal `.` characters, not tab leaders.

Tab leaders were tried first and are the correct construction - a right tab
carrying `w:leader="dot"`, which is what Word's own TOC styles use. They
rendered as blank space anyway. The same render showed `5` where the front
matter should read `v`, so the converter in use flattens section page
numbering and tab leaders alike.

Literal dots cannot be dropped by a converter. They are sized from the real
Times New Roman and Times New Roman Bold metrics - chapter lines are bold and
measurably wider - targeting 447.6pt against 453.6pt of usable width. The 6pt
of slack is deliberate: an overflowing line wraps the page number onto its own
line, which is worse than dots stopping slightly short. The right tab stop is
kept so the numbers stay in a clean column.

**If you edit a contents line, the dots will no longer fit.** They are sized
for the text as it stands, not recomputed on the fly.

The roman front matter has no equivalent fallback - it depends on section
support. If your viewer shows `1-33` straight through, open the file in Word.

## Roman front-matter numerals

Each front-matter page is its own section carrying its numeral as literal
footer text, not a page-number field.

The field version was correct and rendered properly in Word. Google Docs has
no roman page-number format, so it rendered every section decimal - the front
matter came out 2,3,4 instead of ii,iii,iv. Only the arabic restart at
Chapter 1 survived. Literal text renders the same everywhere.

**These numerals are frozen.** Add or remove a front-matter page and they
will be wrong, with nothing to catch it. The section-break paragraphs are
collapsed to 1pt for the same reason the dots are sized the way they are: at
normal height each consumes a line, and on the full contents page that pushes
the last entry over and invalidates the page numbers listed on it.

To go back to proper field-based numbering - correct in Word, wrong in Google
Docs - the pre-change file is at commit b8c8b34.

/** Formatting helpers. Numbers always carry thousands separators and are
 *  rendered with tabular figures by the .num class, so columns align. */
export const mw = (n: number, dp = 0) =>
  n.toLocaleString('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp })

export const signed = (n: number, dp = 0) =>
  (n > 0 ? '+' : n < 0 ? '−' : '') + mw(Math.abs(n), dp)

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const DAYS = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

export const d = (iso: string) => new Date(iso + 'T00:00:00')
export const shortDate = (iso: string) => {
  const t = d(iso); return `${t.getDate()} ${MONTHS[t.getMonth()]}`
}
export const longDate = (iso: string) => {
  const t = d(iso)
  return `${DAYS[t.getDay()]} ${t.getDate()} ${MONTHS[t.getMonth()]} ${t.getFullYear()}`
}
export const dayName = (iso: string) => DAYS[d(iso).getDay()]
export const daysBetween = (a: string, b: string) =>
  Math.round((d(b).getTime() - d(a).getTime()) / 86400000)

/** '2026-11-15T19:54:11' -> '15 Nov 19:54'. Returns '—' for null. */
export const stamp = (iso: string | null) => {
  if (!iso) return '—'
  const [date, time] = iso.split('T')
  return `${shortDate(date)} ${(time ?? '').slice(0, 5)}`
}

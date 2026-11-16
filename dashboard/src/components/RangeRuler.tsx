/** Horizontal range ruler: 95% band, 90% band, point marker, and a tick for
 *  the latest actual. Replaces reading two intervals as four numbers.
 *
 *  Pure SVG so it scales without a chart library. Carries a text alternative
 *  because colour and position alone are not an accessible encoding. */
import { mw, shortDate } from '../lib/format'

interface Props {
  point: number
  lo90: number; hi90: number
  lo95: number; hi95: number
  actual?: { mw: number; date: string }
}

export default function RangeRuler({ point, lo90, hi90, lo95, hi95, actual }: Props) {
  const lo = Math.min(lo95, actual?.mw ?? lo95)
  const hi = Math.max(hi95, actual?.mw ?? hi95)
  const pad = (hi - lo) * 0.12 || 100
  const min = lo - pad, max = hi + pad
  const x = (v: number) => ((v - min) / (max - min)) * 100

  const label =
    `Forecast ${mw(point)} megawatts. The model's 90% range is ` +
    `${mw(lo90)} to ${mw(hi90)}, and its 95% range is ${mw(lo95)} to ${mw(hi95)}.` +
    (actual ? ` The latest recorded actual was ${mw(actual.mw)}.` : '')

  return (
    <figure className="ruler" aria-label={label}>
      <svg viewBox="0 0 100 34" preserveAspectRatio="none" className="ruler-svg"
           role="presentation" focusable="false">
        <rect x={x(lo95)} width={x(hi95) - x(lo95)} y="11" height="12"
              fill="var(--band-95)" rx="1.4" />
        <rect x={x(lo90)} width={x(hi90) - x(lo90)} y="11" height="12"
              fill="var(--band-90)" rx="1.4" />
        {actual && (
          <line x1={x(actual.mw)} x2={x(actual.mw)} y1="7" y2="27"
                stroke="var(--actual)" strokeWidth="0.55" />
        )}
        <line x1={x(point)} x2={x(point)} y1="4" y2="30"
              stroke="var(--forecast)" strokeWidth="0.9" />
      </svg>
      <div className="ruler-keys">
        <span><i className="sw" style={{ background: 'var(--forecast)' }} />
          Forecast {mw(point)}</span>
        <span><i className="sw" style={{ background: 'var(--band-90)' }} />
          90% range {mw(lo90)}–{mw(hi90)}</span>
        <span><i className="sw" style={{ background: 'var(--band-95)' }} />
          95% {mw(lo95)}–{mw(hi95)}</span>
        {actual && (
          <span><i className="sw" style={{ background: 'var(--actual)' }} />
            Actual {mw(actual.mw)} · {shortDate(actual.date)}</span>
        )}
      </div>
    </figure>
  )
}

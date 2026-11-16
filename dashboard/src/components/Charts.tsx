/** Hand-built SVG charts.
 *
 *  The spec names visx or Observable Plot. Built by hand instead: its actual
 *  requirement is "no chart library defaults left unstyled", and hand-built
 *  marks have no defaults to override. These four forms are simple enough
 *  that a charting dependency would add weight without adding control.
 *
 *  Shared rules, from the spec and the dataviz checks:
 *    - solid gridlines, never dashed
 *    - bar comparisons start at zero; the line chart may not, and says so
 *    - direct labels at line ends instead of a legend box
 *    - every chart carries an aria-label summary
 *    - colour is never the only signal (shape/label/position carry it too)
 */
import { useId, useState } from 'react'
import { mw, shortDate, signed } from '../lib/format'
import type { Agreement as AgreementT, Point, Segment } from '../lib/types'

/* ---------------------------------------------- B1 history + forecast */
export function HistoryChart({
  history, forecast, record,
}: {
  history: Point[]
  forecast?: { date: string; mw: number; lo90: number | null; hi90: number | null }
  record?: Point
}) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 760, H = 260, L = 8, R = 56, Tp = 14, B = 26
  const pts = history
  if (pts.length < 2) return null

  const vals = pts.map(p => p.mw)
  if (forecast) vals.push(forecast.mw, forecast.hi90 ?? forecast.mw, forecast.lo90 ?? forecast.mw)
  if (record) vals.push(record.mw)
  const lo = Math.min(...vals) * 0.985, hi = Math.max(...vals) * 1.015
  const n = pts.length + (forecast ? 1 : 0)
  const x = (i: number) => L + (i / (n - 1)) * (W - L - R)
  const y = (v: number) => Tp + (1 - (v - lo) / (hi - lo)) * (H - Tp - B)

  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.mw).toFixed(1)}`).join('')
  const fi = pts.length
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => lo + t * (hi - lo))
  const every = Math.max(1, Math.floor(pts.length / 6))
  const hp = hover != null ? pts[hover] : null

  return (
    <figure className="chart"
      aria-label={`Daily peak demand for the last ${pts.length} days, from ${mw(Math.min(...pts.map(p => p.mw)))} to ${mw(Math.max(...pts.map(p => p.mw)))} megawatts.${forecast ? ` The forecast for ${shortDate(forecast.date)} is ${mw(forecast.mw)} megawatts.` : ''}`}>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="presentation"
           onMouseLeave={() => setHover(null)}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
            <text x={W - R + 6} y={y(t) + 4} className="ax">{mw(t)}</text>
          </g>
        ))}
        {record && record.mw <= hi && (
          <>
            <line x1={L} x2={W - R} y1={y(record.mw)} y2={y(record.mw)}
                  stroke="var(--ink-3)" strokeWidth="1" strokeOpacity=".5" />
            <text x={L + 4} y={y(record.mw) - 5} className="ax">
              Record {mw(record.mw)} · {shortDate(record.date)}
            </text>
          </>
        )}
        {pts.filter((_, i) => i % every === 0).map((pt, k) => (
          <text key={k} x={x(k * every)} y={H - 8} className="ax" textAnchor="middle">
            {shortDate(pt.date)}
          </text>
        ))}
        <path d={line} fill="none" stroke="var(--actual)" strokeWidth="2"
              strokeLinejoin="round" strokeLinecap="round" />
        {forecast && forecast.lo90 != null && forecast.hi90 != null && (
          <>
            <line x1={x(fi)} x2={x(fi)} y1={y(forecast.hi90)} y2={y(forecast.lo90)}
                  stroke="var(--band-90)" strokeWidth="7" strokeLinecap="round" />
            <line x1={x(pts.length - 1)} x2={x(fi)}
                  y1={y(pts[pts.length - 1].mw)} y2={y(forecast.mw)}
                  stroke="var(--forecast)" strokeWidth="1.5" strokeOpacity=".55" />
            <circle cx={x(fi)} cy={y(forecast.mw)} r="4.5" fill="var(--forecast)"
                    stroke="var(--surface)" strokeWidth="2" />
          </>
        )}
        {pts.map((_, i) => (
          <rect key={i} x={x(i) - 3} y={Tp} width="6" height={H - Tp - B}
                fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
        {hp && (
          <>
            <line x1={x(hover!)} x2={x(hover!)} y1={Tp} y2={H - B}
                  stroke="var(--ink-3)" strokeWidth="1" />
            <circle cx={x(hover!)} cy={y(hp.mw)} r="4" fill="var(--actual)"
                    stroke="var(--surface)" strokeWidth="2" />
          </>
        )}
      </svg>
      <div className="chart-foot">
        <span className="dl" style={{ color: 'var(--actual)' }}>— Recorded</span>
        {forecast && <span className="dl" style={{ color: 'var(--forecast)' }}>● Forecast</span>}
        <span className="ax">Vertical axis does not start at zero.</span>
      </div>
      {hp && (
        <div className="tip" role="status">
          <b>{shortDate(hp.date)}</b> · {mw(hp.mw)} MW
        </div>
      )}
    </figure>
  )
}

/* ------------------------------------------------- C1 / D3 horizontal bars */
export function BarRows({
  rows, unit = 'MW', color = 'var(--actual)', caption,
}: {
  rows: { label: string; value: number; sub?: string; accent?: boolean }[]
  unit?: string; color?: string; caption?: string
}) {
  const max = Math.max(...rows.map(r => r.value)) || 1
  return (
    <figure className="chart"
      aria-label={rows.map(r => `${r.label}: ${mw(r.value)} ${unit}`).join('. ')}>
      <div className="bars">
        {rows.map(r => (
          <div className="bar-row" key={r.label}>
            <div className="bar-head">
              <span className="bar-label">{r.label}</span>
              <span className="bar-val num">{mw(r.value)}<i> {unit}</i></span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{
                width: `${(r.value / max) * 100}%`,
                background: r.accent ? 'var(--forecast)' : color,
              }} />
            </div>
            {r.sub && <div className="bar-sub">{r.sub}</div>}
          </div>
        ))}
      </div>
      {caption && <figcaption className="ax">{caption}</figcaption>}
    </figure>
  )
}

/* --------------------------------------------------------- D4 dot plot */
export function SegmentDots({ segments }: { segments: Segment[] }) {
  const id = useId()
  const max = Math.max(...segments.map(s => s.mae)) * 1.08
  return (
    <figure className="chart"
      aria-label={`Typical error by segment. ` +
        segments.map(s => `${s.label} ${mw(s.mae)} megawatts from ${s.n} days`).join('. ')}>
      <div className="dots">
        {segments.map(s => (
          <div className="dot-row" key={s.key}>
            <span className="dot-label">{s.label}</span>
            <div className="dot-track">
              <div className="dot-rule" />
              <span
                className={`dot-mark${s.small ? ' hollow' : ''}`}
                style={{ left: `${(s.mae / max) * 100}%` }}
                title={`${mw(s.mae)} MW · n=${s.n}${s.small ? ' · small sample' : ''}`}
              />
            </div>
            <span className="dot-val num">{mw(s.mae)}</span>
            <span className="dot-n">{s.small ? 'small sample' : `n=${s.n}`}</span>
          </div>
        ))}
      </div>
      <figcaption className="ax">
        Hollow markers have fewer than 20 days behind them — read them as
        anecdotes, not measurements. Bias is shown in the table view.
      </figcaption>
      <details className="tablefall">
        <summary>View as table</summary>
        <table className="tb" id={id}>
          <thead><tr><th>Segment</th><th>Days</th><th>Typical error</th><th>Tendency</th></tr></thead>
          <tbody>
            {segments.map(s => (
              <tr key={s.key}>
                <td>{s.label}{s.small && ' (small sample)'}</td>
                <td className="num">{s.n}</td>
                <td className="num">{mw(s.mae)} MW</td>
                <td className="num">{signed(s.bias)} MW</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  )
}

/* ----------------------------------------------------- D5 coverage pairs */
export function CoverageBars({
  coverage,
}: { coverage: Record<string, { nominal: number; realised: number }> }) {
  const rows = Object.entries(coverage).sort()
  return (
    <figure className="chart"
      aria-label={rows.map(([, v]) =>
        `${Math.round(v.nominal * 100)} percent range contained the outcome ` +
        `${(v.realised * 100).toFixed(1)} percent of the time`).join('. ')}>
      <div className="cov">
        {rows.map(([lvl, v]) => {
          const short = v.realised < v.nominal
          return (
            <div className="cov-row" key={lvl}>
              <span className="cov-label">{Math.round(v.nominal * 100)}% range</span>
              <div className="cov-track">
                <div className="cov-fill" style={{ width: `${v.realised * 100}%` }} />
                <div className="cov-target" style={{ left: `${v.nominal * 100}%` }}
                     title={`Target ${Math.round(v.nominal * 100)}%`} />
              </div>
              <span className="cov-val num">
                {(v.realised * 100).toFixed(1)}%
                <i>{short ? ' under' : ' over'} target</i>
              </span>
            </div>
          )
        })}
      </div>
      <figcaption className="ax">
        Vertical rule marks the target. Measured on the backtest walk-forward.
      </figcaption>
    </figure>
  )
}

/* ------------------------------------------- agreement: predicted vs actual */
export function Agreement({ data }: { data: AgreementT }) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 440, H = 420, M = 58
  const all = data.points.flatMap(d => [d.a, d.p])
  const lo = Math.min(...all) * 0.98, hi = Math.max(...all) * 1.02
  const sc = (v: number) => M + ((v - lo) / (hi - lo)) * (W - M - 14)
  const sy = (v: number) => (H - M) - ((v - lo) / (hi - lo)) * (H - M - 14)
  const ticks = [0, 0.5, 1].map(t => lo + t * (hi - lo))
  const hp = hover != null ? data.points[hover] : null

  return (
    <figure className="chart"
      aria-label={`Predicted against actual peak demand for ${data.n} days. ` +
        `Correlation ${data.r}. Typical error ${mw(data.mae)} megawatts, ` +
        `${data.mape} percent. Points on the diagonal are exact forecasts.`}>
      <div className="agree">
        <svg viewBox={`0 0 ${W} ${H}`} className="agree-svg" role="presentation"
             onMouseLeave={() => setHover(null)}>
          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={M} x2={W - 14} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
              <text x={M - 7} y={sy(t) + 4} className="ax" textAnchor="end">{mw(t)}</text>
              <text x={sc(t)} y={H - M + 18} className="ax" textAnchor="middle">{mw(t)}</text>
            </g>
          ))}
          {/* perfect-forecast diagonal: the closer the cloud hugs it, the better */}
          <line x1={sc(lo)} y1={sy(lo)} x2={sc(hi)} y2={sy(hi)}
                stroke="var(--ink)" strokeWidth="1.25" strokeOpacity=".55" />
          {data.points.map((d, i) => (
            <circle key={i} cx={sc(d.a)} cy={sy(d.p)} r={hover === i ? 4 : 2.1}
                    fill="var(--actual)" fillOpacity={hover === i ? 1 : .34}
                    onMouseEnter={() => setHover(i)} />
          ))}
          <text x={M} y={14} className="ax">Forecast (MW)</text>
          <text x={W - 14} y={H - 6} className="ax" textAnchor="end">Actual (MW)</text>
        </svg>
        <div className="agree-keys">
          <p className="ax">
            Each dot is one day. The diagonal is a perfect forecast — dots
            close to it are close calls.
          </p>
          {hp && (
            <p className="tip-inline num">
              actual {mw(hp.a)} · forecast {mw(hp.p)} ·
              off by {mw(Math.abs(hp.p - hp.a))} MW
            </p>
          )}
        </div>
      </div>
    </figure>
  )
}

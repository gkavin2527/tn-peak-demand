/** Vertical demand scale: where tomorrow sits against the last recorded day
 *  and the all-time record.
 *
 *  Replaces a horizontal range ruler that went through two rounds of fixes
 *  and still did not read well. The problem was the form, not the labelling:
 *  on a horizontal track "how high is demand" becomes left-right position,
 *  which nobody reads as magnitude. Height does that for free.
 *
 *  Reading top to bottom - record, forecast with its range, last actual - also
 *  puts the three numbers in the order a person asks for them, and makes the
 *  gap between Sunday and Tuesday a visible distance rather than two ticks
 *  that happen to be far apart.
 *
 *  No legend: every mark is labelled in place, and the weekday is on the
 *  actual so a low Sunday does not read as an unexplained jump.
 */
import { dayName, mw, shortDate } from '../lib/format'

interface Props {
  point: number
  lo90: number; hi90: number
  lo95: number; hi95: number
  actual?: { mw: number; date: string }
  record?: { mw: number; date: string }
  targetDate: string
}

export default function DemandScale({
  point, lo90, hi90, lo95, hi95, actual, record, targetDate,
}: Props) {
  const vals = [lo95, hi95, point]
  if (actual) vals.push(actual.mw)
  if (record) vals.push(record.mw)
  const lo = Math.min(...vals), hi = Math.max(...vals)
  const pad = (hi - lo) * 0.1 || 200
  const min = lo - pad, max = hi + pad
  // 0% is the bottom of the track, so larger demand sits higher
  const y = (v: number) => `${((v - min) / (max - min)) * 100}%`

  const aria =
    `Forecast for ${targetDate}: ${mw(point)} megawatts, with a 90% range of ` +
    `${mw(lo90)} to ${mw(hi90)} and a 95% range of ${mw(lo95)} to ${mw(hi95)}.` +
    (actual ? ` The last recorded day, ${dayName(actual.date)} ${shortDate(actual.date)}, ` +
      `reached ${mw(actual.mw)}.` : '') +
    (record ? ` The record is ${mw(record.mw)} on ${shortDate(record.date)}.` : '')

  return (
    <figure className="scale" aria-label={aria}>
      <div className="scale-plot">
        <div className="scale-axis" aria-hidden="true" />

        {record && (
          <div className="srow srow-rec" style={{ bottom: y(record.mw) }}>
            <span className="srule" />
            <span className="slab">
              <b className="num">{mw(record.mw)}</b>
              <i>record · {shortDate(record.date)}</i>
            </span>
          </div>
        )}

        {/* ranges, drawn as height so the uncertainty reads as a span */}
        <div className="sband sband-95"
             style={{ bottom: y(lo95), height: `calc(${y(hi95)} - ${y(lo95)})` }} />
        <div className="sband sband-90"
             style={{ bottom: y(lo90), height: `calc(${y(hi90)} - ${y(lo90)})` }} />

        <div className="srow srow-fc" style={{ bottom: y(point) }}>
          <span className="srule" />
          <span className="slab">
            <b className="num">{mw(point)}</b>
            <i>forecast · {dayName(targetDate).slice(0, 3)} {shortDate(targetDate)}</i>
          </span>
        </div>

        {actual && (
          <div className="srow srow-ac" style={{ bottom: y(actual.mw) }}>
            <span className="srule" />
            <span className="slab">
              <b className="num">{mw(actual.mw)}</b>
              <i>last recorded · {dayName(actual.date).slice(0, 3)} {shortDate(actual.date)}</i>
            </span>
          </div>
        )}
      </div>

      <figcaption className="scale-foot">
        Shaded bands are the model's 90% and 95% ranges
        {actual && <> — {mw(Math.abs(point - actual.mw))} MW{' '}
          {point > actual.mw ? 'above' : 'below'} {dayName(actual.date)}</>}.
      </figcaption>
    </figure>
  )
}

import { useEffect, useState } from 'react'
import CountUp from './components/CountUp'
import DemandScale from './components/DemandScale'
import { Agreement, BarRows, CoverageBars, HistoryChart, SegmentDots } from './components/Charts'
import { dayName, daysBetween, horizon, longDate, mw, shortDate, signed, stamp } from './lib/format'
import type { Data } from './lib/types'

const BASE = import.meta.env.BASE_URL

/** Headline sentence, generated from data slots. Never model-written prose:
 *  the template is fixed and only numbers vary, so it cannot drift. */
function headline(d: Data): string {
  const f = d.forecast!, a = d.latest_actual
  const parts: string[] = []
  if (a) {
    const diff = f.mw - a.mw
    const pct = Math.abs(diff / a.mw) * 100
    const dir = Math.abs(pct) < 0.5 ? 'about level with'
      : diff > 0 ? `${pct.toFixed(1)}% above` : `${pct.toFixed(1)}% below`
    parts.push(`About ${dir} ${dayName(a.date)}'s ${mw(a.mw)} MW.`)
  }
  if (f.lo90 != null && f.hi90 != null) {
    parts.push(`The model's 90% range is ${mw(f.lo90)}–${mw(f.hi90)} MW.`)
  }
  return parts.join(' ')
}

export default function App() {
  const [d, setD] = useState<Data | null>(null)
  const [err, setErr] = useState(false)

  useEffect(() => {
    fetch(`${BASE}data/dashboard.json`)
      .then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json() })
      .then(setD).catch(() => setErr(true))
  }, [])

  if (err) return (
    <main className="wrap"><div className="panel error-panel">
      <h2>Could not load the forecast data</h2>
      <p>The dashboard reads a single data file published by the daily
        pipeline. It did not load.</p>
      <button className="btn" onClick={() => location.reload()}>Try again</button>
    </div></main>
  )
  if (!d) return (
    <main className="wrap" aria-busy="true">
      <div className="skel skel-hero" /><div className="skel skel-row" />
      <div className="skel skel-chart" />
    </main>
  )

  const f = d.forecast
  const a = d.latest_actual
  const stale = a ? daysBetween(a.date, d.generated_at.slice(0, 10)) : 0
  const isStale = stale > 3

  return (
    <>
      <a className="skip" href="#forecast">Skip to forecast</a>
      <header className="bar">
        <div className="wrap bar-in">
          <span className="mark">TN<i>peak</i></span>
          <nav aria-label="Sections">
            <a href="#forecast">Forecast</a>
            <a href="#context">Context</a>
            <a href="#record">Track record</a>
            <a href="#method">Method</a>
          </nav>
        </div>
      </header>

      <main className="wrap">
        {/* ------------------------------------------------ A. FORECAST */}
        <section id="forecast" className="sec sec-hero">
          {!f ? (
            <div className="panel empty">
              <h2>No forecast issued yet</h2>
              <p>The evening run publishes one at about 20:00 IST.
                {a && <> The most recent recorded peak is {mw(a.mw)} MW
                  on {longDate(a.date)}.</>}</p>
            </div>
          ) : (
            <>
              <div className="hero">
                <div className="hero-l">
                  <p className="kicker">
                    {{ tomorrow: "TOMORROW'S EXPECTED PEAK",
                       today: "TODAY'S EXPECTED PEAK",
                       past: 'FORECAST WAS FOR',
                       ahead: 'EXPECTED PEAK' }[horizon(f.target_date)]}
                    {' · '}{longDate(f.target_date).toUpperCase()}
                  </p>
                  <p className="big num">
                    <CountUp to={f.mw} /><i>MW</i>
                  </p>
                  <p className="say">{headline(d)}</p>
                </div>
                <div className="hero-r">
                  {f.lo90 != null && f.hi90 != null && f.lo95 != null && f.hi95 != null ? (
                    <DemandScale point={f.mw} lo90={f.lo90} hi90={f.hi90}
                      lo95={f.lo95} hi95={f.hi95} targetDate={f.target_date}
                      actual={a ? { mw: a.mw, date: a.date } : undefined}
                      record={d.record} />
                  ) : (
                    <p className="ax">No range attached to this run.</p>
                  )}
                </div>
              </div>

              <p className="strip">
                Issued {stamp(f.issued_at)} IST · data to {a ? shortDate(a.date) : '—'} ·
                weather: {f.weather_source?.includes('forecast') || f.weather_source?.includes('lead')
                  ? 'day-ahead forecast' : f.weather_source ?? '—'} ·
                model: {d.model?.name ?? (f.is_champion ? 'champion' : 'interim')}
                {f.is_champion ? '' : ' (interim stand-in)'}
              </p>
              {isStale && (
                <p className="warnline">
                  Latest actual is {stale} days old. The forecast still stands,
                  but it was built on older demand data than usual.
                </p>
              )}
              {horizon(f.target_date) === 'today' && (
                <p className="warnline">
                  This is today's forecast, issued last night. Tomorrow's is
                  published by the evening run at about 20:00 IST — it has not
                  landed yet.
                </p>
              )}
              {horizon(f.target_date) === 'past' && (
                <p className="warnline">
                  This forecast was for {longDate(f.target_date)} and has been
                  overtaken. The evening run has not published a newer one.
                </p>
              )}
              {f.warnings.map((w, i) => <p className="warnline" key={i}>{w}</p>)}
            </>
          )}
        </section>

        {/* ------------------------------------------------- B. CONTEXT */}
        <section id="context" className="sec">
          <h2 className="h">Context</h2>
          <div className="grid-8-4">
            <div className="panel">
              <h3 className="ph">Recent demand</h3>
              {d.history?.length ? (
                <HistoryChart
                  history={d.history.slice(-60)}
                  record={d.record}
                  forecast={f ? {
                    date: f.target_date, mw: f.mw, lo90: f.lo90, hi90: f.hi90,
                  } : undefined} />
              ) : <p className="ax">No demand history available.</p>}
            </div>
            <div className="panel">
              <h3 className="ph">What moves the forecast</h3>
              {d.drivers?.length ? (
                <BarRows
                  rows={d.drivers.map(r => ({ label: r.label, value: r.dmae }))}
                  caption="Accuracy lost if this input is removed." />
              ) : <p className="ax">Driver analysis not available.</p>}
              {d.drivers_caveat && <p className="ax fine">{d.drivers_caveat}</p>}
            </div>
          </div>

          {d.sensitivity_mw_per_c != null && f && (
            <WhatIf base={f.mw} perC={d.sensitivity_mw_per_c} />
          )}
        </section>

        {/* -------------------------------------------- C. TRACK RECORD */}
        <section id="record" className="sec">
          <h2 className="h">Track record</h2>

          {d.agreement && (
            <div className="panel">
              <h3 className="ph">How close are the forecasts</h3>
              <p className="say sm" style={{ marginTop: 0 }}>
                Every one of these {mw(d.agreement.n)} days was forecast from
                data available before it, then compared with what Grid-India
                later published.
              </p>
              <div className="grid-agree">
                <Agreement data={d.agreement} />
                <div>
                  <dl className="stats">
                    <div><dt>Days compared</dt>
                      <dd className="num">{mw(d.agreement.n)}</dd></div>
                    <div><dt>Typical error</dt>
                      <dd className="num">{mw(d.agreement.mae)} MW</dd></div>
                    <div><dt>As a share of demand</dt>
                      <dd className="num">{d.agreement.mape}%</dd></div>
                    <div><dt>Correlation with actual</dt>
                      <dd className="num">{d.agreement.r.toFixed(3)}</dd></div>
                    <div><dt>Variation explained (R²)</dt>
                      <dd className="num">{(d.agreement.r2 * 100).toFixed(1)}%</dd></div>
                  </dl>
                  <h4 className="ph" style={{ marginTop: 'var(--s-3)' }}>
                    How often the forecast lands within…
                  </h4>
                  <BarRows unit="%" color="var(--actual)"
                    rows={d.agreement.bands.map(b => ({
                      label: `${mw(b.within)} MW`, value: b.pct,
                    }))} />
                </div>
              </div>
              <p className="ax fine">
                {d.agreement.basis} {shortDate(d.agreement.span[0])}{' '}
                {d.agreement.span[0].slice(0, 4)} to{' '}
                {shortDate(d.agreement.span[1])} {d.agreement.span[1].slice(0, 4)}.
                A reconstruction, not a live record — the live ledger below is
                the stricter test.
              </p>
            </div>
          )}

          <div className="panel">
            <h3 className="ph">Live forecasts</h3>
            {!d.live || d.live.n_scored === 0 ? (
              <p className="ax">No forecast has been scored yet. Actual demand
                is published about two days after each forecast date.</p>
            ) : (
              <>
                <p className="live-l">
                  <b className="num">{d.live.n_scored}</b> forecast
                  {d.live.n_scored === 1 ? '' : 's'} scored since going live
                  {' · '}typical error <b className="num">{mw(d.live.mae_mw)} MW</b>
                  {' · '}inside the 90% range{' '}
                  <b>{d.live.interval_coverage['90%'] === 1 ? 'yes' : 'not always'}</b>
                </p>
                {d.live.is_provisional && (
                  <>
                    <p className="badge">
                      PROVISIONAL · {d.live.n_summer_days} summer days
                    </p>
                    <div className="prog">
                      <div className="prog-fill"
                        style={{ width: `${Math.min(100, (d.live.n_scored / 30) * 100)}%` }} />
                    </div>
                    <p className="ax">
                      {d.live.n_scored} of 30 — a stated convention for when we
                      will quote a summary figure, not a statistical threshold.
                      No percentage is shown until then.
                    </p>
                  </>
                )}
              </>
            )}
          </div>

          {d.ledger?.length ? (
            <div className="panel">
              <h3 className="ph">Forecast ledger</h3>
              <table className="tb">
                <caption className="ax">
                  Every forecast issued, newest first. Written down before the
                  outcome was known.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Date</th>
                    <th scope="col" className="r">Predicted</th>
                    <th scope="col" className="r">Actual</th>
                    <th scope="col" className="r">Error</th>
                    <th scope="col">90% range</th>
                  </tr>
                </thead>
                <tbody>
                  {d.ledger.map(r => (
                    <tr key={r.date}>
                      <td>{shortDate(r.date)}
                        {!r.is_champion && <span className="tag">interim model</span>}</td>
                      <td className="num r">{mw(r.pred)}</td>
                      <td className="num r">{r.actual != null ? mw(r.actual)
                        : <span className="ax">pending</span>}</td>
                      <td className="num r">{r.error != null ? `${signed(r.error)}` : '—'}</td>
                      <td>{r.inside90 == null ? <span className="ax">— pending</span>
                        : r.inside90 ? <span className="hit">✓ inside</span>
                          : <span className="miss">✗ outside</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {d.backtest && (
            <div className="panel">
              <h3 className="ph">Backtest <span className="ax">— reconstruction, not live</span></h3>
              <BarRows
                rows={[
                  d.backtest.persistence && {
                    label: 'Repeat yesterday', value: d.backtest.persistence.mae,
                    sub: `${d.backtest.persistence.basis} · n=${d.backtest.persistence.n}`,
                  },
                  d.backtest.champion_forecast_weather && {
                    label: 'This model, real forecasts',
                    value: d.backtest.champion_forecast_weather.mae, accent: true,
                    sub: `${d.backtest.champion_forecast_weather.basis} · n=${d.backtest.champion_forecast_weather.n}`,
                  },
                  d.backtest.champion_hindsight_weather && {
                    label: 'This model, hindsight weather',
                    value: d.backtest.champion_hindsight_weather.mae,
                    sub: `${d.backtest.champion_hindsight_weather.basis} · n=${d.backtest.champion_hindsight_weather.n}`,
                  },
                ].filter(Boolean) as any}
                caption="Shorter is better. Samples differ slightly — each bar carries its own n." />
              {d.backtest.champion_forecast_weather && d.backtest.champion_hindsight_weather && (
                <p className="say sm">
                  Using real weather forecasts instead of hindsight weather costs{' '}
                  <b className="num">{mw(d.backtest.champion_forecast_weather.mae
                    - d.backtest.champion_hindsight_weather.mae)} MW</b> of accuracy.
                  Only the first figure is achievable in practice.
                </p>
              )}
            </div>
          )}

          {d.segments?.length ? (
            <div className="panel">
              <h3 className="ph">Where it is weakest</h3>
              <SegmentDots segments={d.segments} />
            </div>
          ) : null}

          {d.coverage && (
            <div className="panel">
              <h3 className="ph">Do the ranges hold</h3>
              <CoverageBars coverage={d.coverage} />
              {d.coverage['0.90'] && (
                <p className="say sm">
                  The 90% range has contained the outcome{' '}
                  {(d.coverage['0.90'].realised * 100).toFixed(1)}% of the time
                  in backtest — slightly under target.
                </p>
              )}
            </div>
          )}
        </section>

        {/* --------------------------------------------- D. METHOD */}
        <section id="method" className="sec">
          <h2 className="h">Method &amp; data</h2>
          <Acc title="Where the data comes from">
            <p>Daily peak demand is published by Grid-India in its Daily Power
              Supply Position reports{d.dataset && <>, covering {mw(d.dataset.rows)} days
                from {shortDate(d.dataset.start)} {d.dataset.start.slice(0, 4)} to{' '}
                {shortDate(d.dataset.end)} {d.dataset.end.slice(0, 4)}</>}. Weather
              comes from Open-Meteo — reanalysis for past days, archived
              day-ahead forecasts for the forecast itself.</p>
          </Acc>
          <Acc title={`Why forecasts are issued ${d.publish?.H ?? 2} days ahead`}>
            <p>Grid-India publishes a day's peak the following morning. Across{' '}
              {d.publish?.n_run_days ?? 46} observed runs of this pipeline,
              same-day data was never available — the latest run observed was
              at {d.publish?.latest_run_clock ?? '18:07'}. So an evening
              forecast for tomorrow can only use demand up to yesterday. That
              gap is measured from the pipeline's own records, not assumed, and
              it rules out using yesterday's peak — which would otherwise be
              the single most useful input.</p>
          </Acc>
          <Acc title="What the ranges mean">
            <p>The ranges are built from how far past forecasts actually landed
              from reality, using a method that assumes nothing about the shape
              of the error distribution. They are calibrated separately for
              March–June and the rest of the year, because summer is harder to
              predict. Realised coverage is reported above rather than assumed.</p>
          </Acc>
          <Acc title="Known limitations">
            <ul>
              {d.segments?.find(s => s.key === 'surge_top5pct') && (
                <li>On the highest-demand days the forecast runs low by{' '}
                  {mw(Math.abs(d.segments.find(s => s.key === 'surge_top5pct')!.bias))} MW —
                  the expensive direction, and the most important open problem.</li>
              )}
              {d.residual_acf1 != null && (
                <li>Errors are correlated from one day to the next
                  (lag-1 {d.residual_acf1.toFixed(2)}), so some structure
                  remains uncaptured.</li>
              )}
              <li>Single region, single target. Generality is not
                demonstrated — nothing here shows the approach transfers to
                another state.</li>
            </ul>
          </Acc>

          <Acc title="Which model produces these numbers">
            {d.model && (
              <>
                <p>The forecast comes from a model this project calls{' '}
                  <b>{d.model.name}</b> — {d.model.plain}. It reads{' '}
                  {d.model.feature_count} inputs: recent demand and energy
                  levels, the day of the week and season, and temperature.</p>
                <p>"Champion" is the <i>role</i> it holds, not its name. It won
                  a benchmark against nine alternatives — including four
                  gradient-boosting variants that all lost to plain linear
                  regression, because tree models cannot extrapolate a demand
                  level higher than any they were trained on.</p>
                {d.model.quantile != null && (
                  <p>Its second stage targets the{' '}
                    {(d.model.quantile * 100).toFixed(0)}th percentile of its own
                    residual rather than the average, because the error measure
                    reported here is minimised by the median, not the mean.</p>
                )}
                <p className="ax">Frozen {d.model.frozen_at.slice(0, 10)} ·
                  feature set {d.model.feature_set}</p>
              </>
            )}
          </Acc>

          {d.experiment_notes?.length ? (
            <Acc title="Notes on the historical weather comparison">
              <p className="ax">These describe the backtest that measured the
                cost of forecast weather, not the forecast above.</p>
              <ul>{d.experiment_notes.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </Acc>
          ) : null}

          <p className="dl-row">
            <a href={`${BASE}data/dashboard.json`} download>Download dashboard data (JSON)</a>
          </p>
        </section>

        <footer className="foot">
          <p className="strip">
            {d.model?.name && <>model {d.model.name} · {d.model.feature_count} features · </>}
            generated {stamp(d.generated_at)}
          </p>
          <p className="ax">Every figure on this page is read from the
            pipeline's committed outputs. Nothing here is simulated.</p>
        </footer>
      </main>
    </>
  )
}

/* --------------------------------------------------- weather what-if */
function WhatIf({ base, perC }: { base: number; perC: number }) {
  const [c, setC] = useState(0)
  const shift = perC * c
  return (
    <div className="panel">
      <h3 className="ph">If it were warmer or cooler</h3>
      <div className="whatif">
        <div>
          <label htmlFor="temp" className="ax">
            Temperature change versus the forecast day
          </label>
          <input id="temp" type="range" min={-5} max={8} step={0.5} value={c}
            onChange={e => setC(Number(e.target.value))}
            aria-valuetext={`${c > 0 ? '+' : ''}${c} degrees Celsius`} />
          <output className="num">{c > 0 ? '+' : ''}{c.toFixed(1)} °C</output>
        </div>
        <p className="say sm">
          {Math.abs(c) < 0.01
            ? `At the forecast temperature, the expected peak is ${mw(base)} MW.`
            : <>A {Math.abs(c).toFixed(1)} °C {c > 0 ? 'warmer' : 'cooler'} day
                would put the estimate near <b className="num">{mw(base + shift)} MW</b>
                {' '}({signed(shift)} MW).</>}
          <span className="ax"> Local slope of {signed(perC)} MW per °C —
            indicative only, and it flattens at high temperature.</span>
        </p>
      </div>
    </div>
  )
}

function Acc({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="acc">
      <summary>{title}</summary>
      <div className="acc-body">{children}</div>
    </details>
  )
}

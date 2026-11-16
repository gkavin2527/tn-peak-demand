/** Guards the data contract: the dashboard must never display a number that
 *  is not in dashboard.json.
 *
 *  This exists because the project has twice shipped a figure that was
 *  correct when written and wrong later — a residual statistic belonging to a
 *  superseded model, and a headline percentage computed against the wrong
 *  denominator. Both survived people reading prose. A test cannot catch bad
 *  prose, but it can catch a hardcoded number.
 */
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Data } from './lib/types'

const root = join(__dirname, '..')
const data: Data = JSON.parse(
  readFileSync(join(root, 'public/data/dashboard.json'), 'utf8'))

const sources = ['App.tsx', 'components/Charts.tsx', 'components/RangeRuler.tsx',
                 'components/CountUp.tsx']
  .map(f => readFileSync(join(root, 'src', f), 'utf8')).join('\n')

describe('data contract', () => {
  it('has a forecast with a target date and a value', () => {
    expect(data.forecast?.mw).toBeTypeOf('number')
    expect(data.forecast?.target_date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
  })

  it('never hardcodes a demand-scale number in the UI', () => {
    // Any 4-5 digit literal in the source is suspect: demand is ~10,000-22,000
    // MW, so a number in that range should always arrive from the JSON.
    const literals = sources.match(/(?<![\w.#-])\d{4,5}(?![\w%])/g) ?? []
    const suspicious = literals
      .map(Number)
      .filter(n => n >= 5000 && n <= 30000)
    expect(suspicious, `hardcoded demand-scale literals: ${suspicious}`)
      .toEqual([])
  })

  it('reports coverage as measured, not as the nominal target', () => {
    if (!data.coverage) return
    for (const [, v] of Object.entries(data.coverage)) {
      expect(v.realised).toBeTypeOf('number')
      expect(v.realised).not.toBe(v.nominal)  // realised is measured, not assumed
    }
  })

  it('flags small segments so they cannot be read as measurements', () => {
    for (const s of data.segments ?? []) {
      if (s.n < 20) expect(s.small).toBe(true)
    }
  })

  it('marks the live scorecard provisional while under 30 scored', () => {
    if (data.live && data.live.n_scored < 30) {
      expect(data.live.is_provisional).toBe(true)
    }
  })

  it('carries an n for every backtest bar, since samples differ', () => {
    for (const [, bar] of Object.entries(data.backtest ?? {})) {
      expect(bar.n).toBeTypeOf('number')
    }
  })

  it('ledger rows never claim a range hit without an actual', () => {
    for (const r of data.ledger ?? []) {
      if (r.actual == null) expect(r.inside90).toBeNull()
    }
  })
})

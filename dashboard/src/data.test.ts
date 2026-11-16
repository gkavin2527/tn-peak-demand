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
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Data } from './lib/types'

const root = join(__dirname, '..')
const data: Data = JSON.parse(
  readFileSync(join(root, 'public/data/dashboard.json'), 'utf8'))

/** Every UI source, discovered rather than listed.
 *
 *  This was a hardcoded array of four filenames. Renaming or deleting a
 *  component orphaned it: the test file then failed to load and vitest
 *  reported "no tests", which reads like a pass. It does exit 1, so CI would
 *  have caught it - but a guard that breaks whenever the code it guards is
 *  refactored is a guard people learn to work around.
 *
 *  Globbing also widens the net: a number hardcoded into a NEW component is
 *  now caught, which the fixed list could never do. */
function uiSources(): { file: string; text: string }[] {
  const dir = join(root, 'src')
  return readdirSync(dir, { recursive: true, encoding: 'utf8' })
    .filter(f => /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f))
    .map(f => ({ file: f, text: readFileSync(join(dir, f), 'utf8') }))
}

const files = uiSources()
const sources = files.map(f => f.text).join('\n')

describe('data contract', () => {
  it('has a forecast with a target date and a value', () => {
    expect(data.forecast?.mw).toBeTypeOf('number')
    expect(data.forecast?.target_date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
  })

  it('actually found the UI sources to scan', () => {
    // A glob that matches nothing would make the next test pass vacuously -
    // exactly the silent-pass this rewrite exists to remove.
    expect(files.length, 'no .tsx sources found under src/').toBeGreaterThan(3)
    expect(files.map(f => f.file)).toContain('App.tsx')
  })

  it('never hardcodes a demand-scale number in the UI', () => {
    // Any 4-5 digit literal in the source is suspect: demand is ~10,000-22,000
    // MW, so a number in that range should always arrive from the JSON.
    const offenders: string[] = []
    for (const { file, text } of files) {
      const hits = (text.match(/(?<![\w.#-])\d{4,5}(?![\w%])/g) ?? [])
        .map(Number)
        .filter(n => n >= 5000 && n <= 30000)
      if (hits.length) offenders.push(`${file}: ${hits.join(', ')}`)
    }
    expect(offenders, `hardcoded demand-scale literals\n  ${offenders.join('\n  ')}`)
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

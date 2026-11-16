/** Mirrors export_dashboard_data.py. Every field optional: a missing key is
 *  the signal for a panel to render its empty state, never a reason to
 *  substitute a plausible number. */
export interface Point { date: string; mw: number }
export interface Forecast {
  target_date: string; issued_at: string | null; mw: number
  lo90: number | null; hi90: number | null
  lo95: number | null; hi95: number | null
  model_version: string | null; is_champion: boolean
  weather_source: string | null; H: number | null; warnings: string[]
}
export interface LedgerRow {
  date: string; pred: number; actual: number | null; error: number | null
  lo90: number | null; hi90: number | null; inside90: boolean | null
  is_champion: boolean
}
export interface Live {
  n_issued: number; n_scored: number; n_pending: number; span: [string, string]
  mae_mw: number; mape_pct: number; bias_mw: number
  interval_coverage: Record<string, number>
  n_summer_days: number; is_provisional: boolean
  backtest_reference_mae_mw?: number
}
export interface Bar { mae: number; n: number; basis: string }
export interface Segment {
  key: string; label: string; n: number; mae: number; bias: number; small: boolean
}
export interface Agreement {
  n: number; span: [string, string]; mae: number; mape: number
  r: number; r2: number
  bands: { within: number; pct: number }[]
  points: { a: number; p: number }[]
  basis: string
}
export interface Data {
  generated_at: string
  latest_actual?: Point
  record?: Point
  dataset?: { rows: number; start: string; end: string }
  forecast?: Forecast
  history?: Point[]
  ledger?: LedgerRow[]
  live?: Live
  backtest?: Record<string, Bar & { span?: [string, string] }>
  coverage?: Record<string, { nominal: number; realised: number }>
  drivers?: { key: string; label: string; dmae: number }[]
  drivers_caveat?: string
  sensitivity_mw_per_c?: number
  agreement?: Agreement
  segments?: Segment[]
  residual_acf1?: number
  publish?: { H: number; n_run_days: number; latest_run_clock: string }
  model?: { name: string; feature_count: number; frozen_at: string }
  caveats?: string[]
}

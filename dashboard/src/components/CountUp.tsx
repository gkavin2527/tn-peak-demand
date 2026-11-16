/** Hero number reveal.
 *
 *  DEVIATION FROM SPEC §6, on data-integrity grounds.
 *
 *  The spec allows a count-up on the hero figure, and react-bits supplies
 *  one. Built and screenshotted, it fails badly here: for the whole animation
 *  the page displays a NUMBER THAT IS NOT TRUE. Captures taken mid-flight
 *  read "4,819 MW" and "11,500 MW" when the forecast is 17,744 MW. On a page
 *  whose entire purpose is one accurate figure — and which ships a test
 *  forbidding hardcoded demand-scale numbers — animating through false values
 *  to reach a true one is the wrong trade. Anyone screenshotting the page, on
 *  a slow device or with a stalled frame, captures a wrong forecast.
 *
 *  So the value is correct in the DOM from first paint, and only its opacity
 *  and position animate. The spec's intent — the hero arrives with a beat —
 *  is kept; the risk of displaying a false number is removed.
 *
 *  Honours prefers-reduced-motion by skipping the transition entirely.
 */
import { mw } from '../lib/format'

export default function CountUp({ to }: { to: number; ms?: number }) {
  return <span className="reveal">{mw(to)}</span>
}

/**
 * Tabbable, not merely focusable — the distinction #52's whole fix turns on, since
 * `tabindex="-1"` is reachable by script and never by Tab.
 *
 * Shared because it is one sequence read three ways across two suites (#115): which stops exist
 * and what order a reader meets them in (`AppA11y.test.tsx`), and what the roving index holds
 * the whole page to with the full grid rendered (`Matrix.test.tsx`). A per-file copy is how the
 * two counts would come to disagree about what counts as a stop.
 */
export const TABBABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]',
]
  .map((selector) => `${selector}:not([tabindex="-1"])`)
  .join(', ');

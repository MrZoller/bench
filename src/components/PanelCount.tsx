import type { ReactNode } from 'react';

/**
 * The "N of M <noun>" summary that sits opposite a panel heading.
 *
 * One component because there were three copies — the Matrix's "combinations run", the Envelope's
 * "comfortable", the Workloads' "workloads" — and all three carried a blanket
 * `whitespace-nowrap`. That class is an absolute floor on the element's width, and the
 * `flex-wrap` on the header above only decides which line the sentence lands on, not how wide it
 * is. So at a scaled root font size the longest of them took the whole document sideways: 409/320
 * at a 32px root, which is what browser text-only zoom at 200% does (#35).
 *
 * **The nowrap is still right, just not around the whole sentence.** "12 of 425" broken across a
 * line reads as two unrelated numbers, and that is the part worth protecting. The noun after it is
 * ordinary prose and can break like prose. So the unbreakable unit is the numeral pair alone and
 * the sentence wraps before the noun — which is the difference between a floor of the whole line
 * and a floor of ten characters.
 *
 * Worth stating because it is easy to "simplify" back: dropping the inner `whitespace-nowrap`
 * passes every reflow assertion in `e2e/reflow.spec.ts` and silently gives up the thing the class
 * was for. The inner span is mutation-checked there.
 */
export function PanelCount({
  count,
  total,
  children,
}: {
  count: number;
  total: number;
  children: ReactNode;
}) {
  return (
    <p className="text-sm text-[var(--color-text-muted)]">
      <span className="whitespace-nowrap">
        <span className="tabular text-[var(--color-text)]">{count}</span> of {total}
      </span>{' '}
      {children}
    </p>
  );
}

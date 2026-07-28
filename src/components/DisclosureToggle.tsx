import type { ReactNode } from 'react';

/**
 * The "show this as a table" toggle that sits under a chart.
 *
 * One component because there were three byte-identical copies — under the budget bar, under the
 * Envelope's canvas, under the workload list — and all three were **16px tall on a coarse
 * pointer** (#29). They are `text-xs` with no padding, so the button box was exactly the line box.
 *
 * **Why these get the 44px bar rather than the 24px floor.** WCAG 2.5.8 (AA) asks for 24px, and
 * the Matrix's measure toggles are held to that deliberately — a row of three labelled controls
 * with room around them is not a crowded target. These are different in kind: two of the three are
 * the *only* route to the textual equivalent of a canvas. The Envelope's table is how a
 * screen-reader or low-vision user reads the field at all, so a target that is hard to hit on the
 * accessibility affordance fails the people the affordance exists for. `marks.hitTarget` is the
 * bar this repo already declares for the Matrix grid, and it is the right one here.
 *
 * The spacing exception was the alternative and is deliberately not taken. It might well cover
 * these — each sits alone under a chart — but it has to be argued per button and re-argued
 * whenever the layout moves, where a minimum is checked by `e2e/touch-targets.spec.ts` on every
 * run.
 *
 * `inline-flex` rather than a bare `min-height`, because a min-height on the default inline-block
 * grows the box downward and leaves the label riding at the top of a 44px target — the text and
 * the underline would sit outside where a thumb aims.
 *
 * **`any-pointer: coarse`, not `pointer: coarse`** (#43). `pointer` describes the *primary*
 * pointing device, so on a touchscreen laptop, a Surface, or an iPad with a mouse attached it
 * reports `fine` — and these buttons dropped straight back to 16px for a user who can still put a
 * thumb on the screen. `any-pointer` asks whether *any* available pointer is coarse, which is the
 * condition the rule actually cares about.
 *
 * The Matrix grid deliberately stays on `pointer: coarse`, and the asymmetry is the decision
 * rather than an oversight. Widening it there gives 44px rows to every laptop that happens to have
 * a touchscreen, multiplied across hundreds of cells — a real desktop cost. Here the cost is 28px
 * once per panel, and these are the accessibility affordance, so the trade goes the other way.
 */
export function DisclosureToggle({
  expanded,
  onToggle,
  controls,
  children,
}: {
  expanded: boolean;
  onToggle: () => void;
  /** The id of the region this reveals, where one exists to point at. */
  controls?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={expanded}
      aria-controls={controls}
      className="mt-4 inline-flex items-center text-xs text-[var(--color-accent)] underline underline-offset-2 [@media(any-pointer:coarse)]:min-h-11"
    >
      {children}
    </button>
  );
}

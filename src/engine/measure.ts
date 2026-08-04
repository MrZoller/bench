/**
 * The three readings a grid can be coloured by, and the one place their direction lives.
 *
 * Capacity, decode speed and time-to-first-token are the product's three separate answers — the
 * thesis is that collapsing them into "will it fit" is what every other calculator gets wrong — so
 * a surface that draws a grid of configurations has three pictures to draw, not one. Both grids
 * offer the same three, because they are the same three questions asked over different axes: the
 * Matrix over model × device, the Envelope over context × concurrent users.
 *
 * Held here rather than in either grid because the *direction* is the part that drifts. Two of the
 * three are better when larger and one is better when smaller, and a second hand-written copy of
 * that convention is how one surface comes to paint its best cell the colour the other paints its
 * worst. Which cells are on the scale at all is the caller's question and deliberately not here —
 * see `measureValue` in `matrix.ts` and the field fill in `Envelope.tsx`, which answer it
 * differently for good reasons.
 */
export type Measure = 'fit' | 'decode' | 'ttft';

/** Which end of a measure's own scale the ramp should paint brightest. */
export type MeasureDirection = 'higher' | 'lower';

/**
 * Which way each measure reads — the thing that used to be expressed by inverting the value, and
 * the reason that was wrong ([#97](https://github.com/MrZoller/headroom/issues/97)).
 *
 * `measureOf` returned `1 / ttftSeconds` so that every measure could be treated as larger-is-better
 * by one comparison. That is true of the *ordering* and false of the *spacing*, which is what a
 * ramp is made of: `magnitudeFill` places its argument with `log1p`, and above about a second
 * `log1p(1/t) ≈ 1/t`, so the logarithm did nothing and the scale was harmonic. On the Envelope's
 * default scenario — 46 graded cells spanning 1.30s to 729.1s, three orders of magnitude — that put
 * **29 cells on one step of seven and left two interior steps unused**, on the panel whose entire
 * subject is shape. The Matrix was worse: 1,025 of 1,269 cells on the bottom step, because its
 * domain was floored at zero, which for a reciprocal reduces the placement to `t_fastest / t`.
 *
 * So the reading stays in its own units and the *direction* travels beside it, which is the only
 * form in which the logarithm is taken of the quantity a reader is actually comparing. Data rather
 * than a branch, because three surfaces read it and a hand-written copy of "ttft is backwards" is
 * how one grid comes to paint its best cell the colour the other paints its worst — the failure
 * this module was created to prevent, one level down from where it was first stated.
 */
export const MEASURE_DIRECTION = {
  fit: 'higher',
  decode: 'higher',
  ttft: 'lower',
} as const satisfies Record<Measure, MeasureDirection>;

/** The readings every scored cell carries, whichever grid it came from. */
export interface Readings {
  /** Used bytes over allocatable. Above 1 means it did not fit resident. */
  utilization: number;
  tokensPerSec: number;
  ttftSeconds: number;
}

/**
 * What a measure reads off a cell, in the measure's own units.
 *
 * **Not oriented, which is the change #97 made.** Every value here is the physical quantity —
 * headroom as a fraction, tokens per second, seconds — and {@link MEASURE_DIRECTION} says which end
 * of it is good. A caller that ranks or ramps has to consult both; a caller that only wants the
 * figure gets the figure. That also removes the round trip the Matrix legend used to make, where
 * recovering seconds from the ramp value meant `1 / (1 / t)`.
 *
 * Normalising the *direction* here and the *domain* at the call site is deliberate: a ramp is
 * scaled against the cells in front of the reader, and only the caller knows which those are.
 * Whether a cell is on the scale at all is the caller's question too — see `measureValue` in
 * `matrix.ts`, which is where "never timed" is turned away rather than given a sentinel. A zero
 * that means "no reading" cannot live in the same channel as a zero that means "no seconds": under
 * the old orientation both were the worst value and under this one they are opposite ends.
 */
export function measureOf(cell: Readings, measure: Measure): number {
  switch (measure) {
    case 'fit':
      // Headroom, so more is better. Floored at zero: a placement over the ceiling has no room
      // left rather than negative room, and the two are the same answer to this question.
      return Math.max(0, 1 - cell.utilization);
    case 'decode':
      return cell.tokensPerSec;
    case 'ttft':
      return cell.ttftSeconds;
  }
}

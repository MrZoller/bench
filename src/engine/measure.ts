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

/** The readings every scored cell carries, whichever grid it came from. */
export interface Readings {
  /** Used bytes over allocatable. Above 1 means it did not fit resident. */
  utilization: number;
  tokensPerSec: number;
  ttftSeconds: number;
}

/**
 * What a measure reads off a cell, oriented so larger is always better.
 *
 * Normalising the *direction* here and the *domain* at the call site is deliberate: a ramp is
 * scaled against the cells in front of the reader, and only the caller knows which those are.
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
      // Inverted: less time is better. A zero is a cell that was never timed, which is the worst
      // reading rather than an infinitely fast one.
      return cell.ttftSeconds > 0 ? 1 / cell.ttftSeconds : 0;
  }
}

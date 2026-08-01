import { describe, expect, it } from 'vitest';
import { MEASURE_DIRECTION, measureOf, type Measure } from './measure';

/**
 * The direction convention, asserted because it is what a second copy would get wrong: two of the
 * three measures are better when larger and one is better when smaller, and both grids paint the
 * same ramp brightest-is-best from this one function.
 */
describe('reading a measure off a cell', () => {
  const cell = (over: Partial<Parameters<typeof measureOf>[0]> = {}) => ({
    utilization: 0.5,
    tokensPerSec: 20,
    ttftSeconds: 4,
    ...over,
  });

  it('reads fit as room left, so a fuller machine scores lower', () => {
    expect(measureOf(cell({ utilization: 0.25 }), 'fit')).toBeGreaterThan(
      measureOf(cell({ utilization: 0.75 }), 'fit')
    );
    expect(measureOf(cell({ utilization: 0.25 }), 'fit')).toBeCloseTo(0.75, 10);
  });

  it('floors a placement past the ceiling at no room rather than negative room', () => {
    expect(measureOf(cell({ utilization: 4 }), 'fit')).toBe(0);
  });

  it('reads decode and latency in their own units, neither of them inverted', () => {
    expect(measureOf(cell({ tokensPerSec: 40 }), 'decode')).toBe(40);
    /*
     * Seconds, not `1 / seconds`, and this is the assertion #97 turns on. Inverting here made every
     * reading larger-is-better with one comparison and made the ramp harmonic, because
     * `magnitudeFill` then took `log1p` of a reciprocal — `log1p(1/t) ≈ 1/t` past about a second, so
     * the log did nothing. The direction lives in `MEASURE_DIRECTION` instead.
     */
    expect(measureOf(cell({ ttftSeconds: 12.5 }), 'ttft')).toBe(12.5);
    expect(measureOf(cell({ ttftSeconds: 1 }), 'ttft')).toBeLessThan(
      measureOf(cell({ ttftSeconds: 100 }), 'ttft')
    );
  });

  it('states a direction for every measure, and only latency runs the other way', () => {
    // `satisfies Record<Measure, …>` makes the coverage a compile-time claim; this is the runtime
    // half — that the one measure which reads backwards is the one that actually does.
    expect(MEASURE_DIRECTION).toEqual({ fit: 'higher', decode: 'higher', ttft: 'lower' });
  });

  it('reports an untimed cell as zero seconds and leaves the caller to reject it', () => {
    /*
     * `over` and `unsupported` cells carry zeros, and the polarity of that zero inverted with #97:
     * under the old orientation it was the worst reading by accident, and in seconds it is the best
     * one. Neither grid may put such a cell on the scale — `measureValue` in `matrix.ts` turns it
     * away, and the Envelope never grades a state that carries one — so the value here is simply the
     * field, and this test exists to say that the guard moved rather than that it disappeared.
     */
    expect(measureOf(cell({ ttftSeconds: 0 }), 'ttft')).toBe(0);
    for (const measure of ['fit', 'decode', 'ttft'] as Measure[]) {
      expect(Number.isFinite(measureOf(cell({ ttftSeconds: 0, tokensPerSec: 0 }), measure))).toBe(
        true
      );
    }
  });
});

import { describe, expect, it } from 'vitest';
import { measureOf, type Measure } from './measure';

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

  it('reads decode as it stands and latency inverted', () => {
    expect(measureOf(cell({ tokensPerSec: 40 }), 'decode')).toBeGreaterThan(
      measureOf(cell({ tokensPerSec: 4 }), 'decode')
    );
    expect(measureOf(cell({ ttftSeconds: 1 }), 'ttft')).toBeGreaterThan(
      measureOf(cell({ ttftSeconds: 100 }), 'ttft')
    );
  });

  it('treats an untimed cell as the worst latency, not an infinite one', () => {
    // `over` and `unsupported` cells carry zeros. Dividing by one of them paints the cell the best
    // colour on the grid, which is the wrong answer in the most misleading direction.
    expect(measureOf(cell({ ttftSeconds: 0 }), 'ttft')).toBe(0);
    for (const measure of ['fit', 'decode', 'ttft'] as Measure[]) {
      expect(Number.isFinite(measureOf(cell({ ttftSeconds: 0, tokensPerSec: 0 }), measure))).toBe(
        true
      );
    }
  });
});

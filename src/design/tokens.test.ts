import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { colors, magnitudeFill, magnitudeRamp, sequential } from './tokens';

/**
 * The tokens exist twice: as TypeScript for canvas code, which cannot resolve `var(--color-*)`,
 * and as CSS custom properties for components. Duplication is forced, so drift has to be caught
 * rather than trusted — a palette validated in one file and edited in the other is exactly how
 * a colourblind-unsafe pair gets shipped.
 */
describe('design tokens', () => {
  const css = readFileSync(resolve(__dirname, '../index.css'), 'utf8');

  /** `surfaceRaised` -> `--color-surface-raised`. */
  const cssName = (key: string) => `--color-${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;

  it.each(Object.entries(colors))('%s matches its CSS custom property', (key, hex) => {
    const declaration = new RegExp(`${cssName(key)}:\\s*${hex};`, 'i');
    expect(css).toMatch(declaration);
  });

  it('keeps the validated series colours distinct from the accent', () => {
    // The accent marks what is interactive and must never read as data. Guarding the exact
    // hexes because the separation was computed, not chosen — see the header of tokens.ts.
    for (const series of [colors.weights, colors.kvCache, colors.overhead]) {
      expect(series).not.toBe(colors.accent);
    }
  });
});

/**
 * The ramp both grids paint their magnitudes with, and the placement onto it.
 *
 * Two surfaces read this now, so the properties are asserted here rather than inferred from
 * whichever picture someone happened to look at. The domain floor is the one that had been wrong
 * everywhere: zero-floored, a grid whose values all sit in the top of the range gets a handful of
 * steps and reads as flat — which is #65 in the Envelope, where headroom runs 18% to 49% because
 * the weights are in every cell.
 */
describe('placing a magnitude on the ramp', () => {
  const index = (
    value: number,
    domain: { min: number; max: number },
    direction: 'higher' | 'lower' = 'higher'
  ) => (magnitudeRamp as readonly string[]).indexOf(magnitudeFill(value, domain, direction));

  it('runs brightest-is-best, which is the reverse of the ramp as authored', () => {
    // `sequential` is light-to-dark for a light surface; this chassis is dark, so the darkest step
    // recedes into the panel and cannot be the good end.
    expect(magnitudeRamp).toHaveLength(sequential.length);
    expect(magnitudeRamp[magnitudeRamp.length - 1]).toBe(sequential[0]);
    expect(magnitudeRamp[0]).toBe(sequential[sequential.length - 1]);
  });

  it('never paints a larger value a darker step', () => {
    const domain = { min: 0, max: 300 };
    let previous = -1;
    for (let value = 0; value <= 300; value += 1.5) {
      const step = index(value, domain);
      expect(step).toBeGreaterThanOrEqual(previous);
      previous = step;
    }
  });

  it('spends the ramp on what varies, not on the constant underneath it', () => {
    /*
     * The Envelope's own default scenario: 46 runnable cells whose headroom runs 0.183 to 0.488,
     * because 60 GiB of gpt-oss weights sits in every one of them. Zero-floored, the bottom of that
     * range lands on step 3 of 7 and the field spends nothing below it — measured, and the reason
     * the domain is a pair rather than a maximum.
     */
    const domain = { min: 0.183, max: 0.488 };

    expect(index(domain.min, domain)).toBe(0);
    expect(index(domain.max, domain)).toBe(magnitudeRamp.length - 1);
    // And the zero-floored reading of the same range, which is what the assertions above rule out.
    expect(index(domain.min, { min: 0, max: domain.max })).toBeGreaterThan(0);
  });

  it('gives a domain with nothing in it the top step rather than NaN', () => {
    // Unreachable on either grid today — every axis carries at least two stops and the cache grows
    // along both — but a ramp indexed by NaN paints `undefined`, which is a blank cell.
    expect(magnitudeFill(4, { min: 4, max: 4 }, 'higher')).toBe(
      magnitudeRamp[magnitudeRamp.length - 1]
    );
    expect(magnitudeFill(0, { min: 0, max: 0 }, 'higher')).toBe(
      magnitudeRamp[magnitudeRamp.length - 1]
    );
    // Both directions, since a degenerate domain has no good end for the reflection to swap.
    expect(magnitudeFill(4, { min: 4, max: 4 }, 'lower')).toBe(
      magnitudeRamp[magnitudeRamp.length - 1]
    );
  });

  it('clamps a value outside the domain rather than indexing past the ramp', () => {
    const domain = { min: 1, max: 10 };
    expect(index(0, domain)).toBe(0);
    expect(index(1000, domain)).toBe(magnitudeRamp.length - 1);
  });

  it('reads a lower-is-better domain from the other end, exactly', () => {
    const domain = { min: 1.3, max: 729 };
    // The reflection is the whole of what `direction` does, so it is asserted as an identity rather
    // than as "the fast one is brighter" — which is true of the collapsed ramp below as well.
    for (const value of [1.3, 4, 12, 40, 120, 400, 729]) {
      expect(index(value, domain, 'lower')).toBe(
        magnitudeRamp.length - 1 - index(value, domain, 'higher')
      );
    }
    expect(index(domain.min, domain, 'lower')).toBe(magnitudeRamp.length - 1);
    expect(index(domain.max, domain, 'lower')).toBe(0);
  });

  /**
   * The defect [#97](https://github.com/MrZoller/bench/issues/97) was filed for, asserted in both
   * directions so the fix cannot be silently reverted.
   *
   * The old code did not pass a direction — it inverted the *value*, handing this function
   * `1 / seconds` and letting the larger-is-better branch do the rest. That composition is the bug:
   * `log1p(1/t) ≈ 1/t` for any latency past about a second, so the logarithm does nothing and the
   * placement is harmonic. This population is the Envelope's own default scenario — 46 graded cells
   * from 1.30s to 729.1s, log-spaced across the 2.75 decades those axes really produce — and the two
   * halves below are what that field measured before and after.
   *
   * The bar is not "the fast cell is brighter", which passes on the collapsed ramp. It is that the
   * ramp is *spent*: no step holding a majority, and no interior step unused.
   */
  it('spends the ramp on a latency population instead of collapsing it', () => {
    const fastest = 1.3;
    const slowest = 729.1;
    const cells = Array.from({ length: 46 }, (_, i) => fastest * (slowest / fastest) ** (i / 45));

    const spread = (steps: number[]) => ({
      busiest: Math.max(...magnitudeRamp.map((_, s) => steps.filter((x) => x === s).length)),
      unused: magnitudeRamp.map((_, s) => s).filter((s) => !steps.includes(s)),
    });

    const fixed = spread(cells.map((t) => index(t, { min: fastest, max: slowest }, 'lower')));
    expect(fixed.unused, 'a step of the ramp is never used').toEqual([]);
    expect(fixed.busiest / cells.length, 'one step holds most of the field').toBeLessThan(0.5);

    /*
     * And the same population through the expression this replaced: reciprocal first, then the log,
     * against a domain in reciprocal units. Kept as a live assertion rather than a comment because a
     * revert would restore exactly this and every other test in this file would stay green.
     */
    const collapsed = spread(
      cells.map((t) => index(1 / t, { min: 1 / slowest, max: 1 / fastest }, 'higher'))
    );
    expect(collapsed.busiest / cells.length).toBeGreaterThan(0.6);
    // Only the majority is asserted of the old expression, and the omission is deliberate: an
    // evenly log-spaced population still reaches every step under it, while the real field — which
    // is not evenly spaced — left two of seven empty. The share is the part that holds either way.
  });
});

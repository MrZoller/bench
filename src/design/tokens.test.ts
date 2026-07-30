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
  const index = (value: number, domain: { min: number; max: number }) =>
    (magnitudeRamp as readonly string[]).indexOf(magnitudeFill(value, domain));

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
    expect(magnitudeFill(4, { min: 4, max: 4 })).toBe(magnitudeRamp[magnitudeRamp.length - 1]);
    expect(magnitudeFill(0, { min: 0, max: 0 })).toBe(magnitudeRamp[magnitudeRamp.length - 1]);
  });

  it('clamps a value outside the domain rather than indexing past the ramp', () => {
    const domain = { min: 1, max: 10 };
    expect(index(0, domain)).toBe(0);
    expect(index(1000, domain)).toBe(magnitudeRamp.length - 1);
  });
});

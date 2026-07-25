import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { colors } from './tokens';

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

import { describe, expect, it } from 'vitest';
import { contextStopsFor } from './stops';
import { gib, params, percent, rate, seconds, tokens, uniqueLabels } from './format';

/**
 * Formatting is where a correct number becomes a wrong statement. These guard the cases where
 * rounding changes the meaning rather than merely the precision.
 */
describe('percentages', () => {
  it('never reports a real quantity as none', () => {
    // Printed "0% of weights would spill" for a rig that is over budget and genuinely spilling.
    expect(percent(0.0043)).toBe('<1%');
    expect(percent(0.0001)).toBe('<1%');
  });

  it('keeps zero for actual zero', () => {
    expect(percent(0)).toBe('0%');
  });

  it('rounds normally above the floor', () => {
    expect(percent(0.93)).toBe('93%');
    expect(percent(1)).toBe('100%');
  });
});

describe('non-finite input', () => {
  it.each([
    ['gib', gib],
    ['params', params],
    ['tokens', tokens],
    ['rate', rate],
    ['seconds', seconds],
    ['percent', percent],
  ])('%s renders an em dash rather than NaN', (_name, fn) => {
    expect(fn(Number.NaN)).toBe('—');
    expect(fn(Number.POSITIVE_INFINITY)).toBe('—');
  });
});

describe('units stay readable as magnitude changes', () => {
  it('switches duration units without losing the number', () => {
    expect(seconds(0.45)).toBe('450 ms');
    expect(seconds(1.5)).toBe('1.5 s');
    expect(seconds(42)).toBe('42 s');
    expect(seconds(900)).toBe('15 min');
  });

  it('says context lengths the way people do', () => {
    expect(tokens(4096)).toBe('4K');
    expect(tokens(131072)).toBe('128K');
    expect(tokens(1048576)).toBe('1M');
    expect(tokens(1_048_576)).toBe('1M');
  });
});

/**
 * Two axis columns carried the header "32K" for 32,768 and 33,000 — different scenarios, the
 * same label, including the one the "you are here" marker was meant to identify. The store
 * accepts any integer a URL supplies, so a label has to belong to exactly one number.
 */
describe('token labels are unique per value', () => {
  it('keeps a decimal for a context that is not a whole number of K', () => {
    expect(tokens(32768)).toBe('32K');
    expect(tokens(33000)).not.toBe(tokens(32768));
    expect(tokens(33000)).toBe('32.2K');
  });

  it('still says the round numbers roundly', () => {
    expect(tokens(2048)).toBe('2K');
    expect(tokens(40960)).toBe('40K');
    expect(tokens(1048576)).toBe('1M');
  });

  it('does not pretend to guarantee uniqueness on its own', () => {
    // 131,073 is 0.0008K away from 131,072, and no fixed precision survives that. `tokens` sees
    // one number at a time, so this is not a promise it can keep — `uniqueLabels` keeps it.
    expect(tokens(131073)).toBe(tokens(131072));
  });

  it('prints a whole million roundly and a near-million honestly', () => {
    expect(tokens(1048576)).toBe('1M');
    // Was "0.95M": a decimal-million threshold divided by a binary mebibyte.
    expect(tokens(1_000_000)).toBe('976.6K');
  });
});

/**
 * An axis holds the whole set, so it can promise what `tokens` cannot: every column header
 * identifies exactly one column.
 */
describe('axis labels', () => {
  it('never labels two distinct control stops the same way', () => {
    for (const [maxContext, stored] of [
      [40960, 33000],
      [163840, 131073],
      [131072, 131071],
      [32768, 32768],
    ] as const) {
      const values = contextStopsFor(maxContext, stored);
      const labels = uniqueLabels(values);
      expect(new Set(labels).size).toBe(values.length);
    }
  });

  it('falls back to the exact count only for the columns that actually clash', () => {
    const labels = uniqueLabels([2048, 131072, 131073]);
    expect(labels[0]).toBe('2K');
    expect(labels.slice(1)).toEqual(['131,072', '131,073']);
  });
});

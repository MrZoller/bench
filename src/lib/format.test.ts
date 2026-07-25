import { describe, expect, it } from 'vitest';
import { gib, params, percent, rate, seconds, tokens } from './format';

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
    expect(tokens(1_048_576)).toBe('1M');
  });
});

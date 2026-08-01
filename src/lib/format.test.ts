import { describe, expect, it } from 'vitest';
import { contextStopsFor } from './stops';
import {
  compact,
  gib,
  multiple,
  optionLabel,
  params,
  percent,
  rate,
  seconds,
  sentences,
  tokens,
  uniqueLabels,
} from './format';

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

/**
 * The budget bar's overshoot, which is a ratio said the way people say ratios. `percent` is the
 * wrong form here: "1446% of the ceiling" and "14x the ceiling" are the same arithmetic and only
 * one of them reads as fourteen times.
 */
describe('multiples', () => {
  it('keeps a decimal only while it still means something', () => {
    // The issue's own case: 448 GiB against a 31 GiB ceiling.
    expect(multiple(448.1 / 31)).toBe('14x');
    expect(multiple(4.492)).toBe('4.5x');
    expect(multiple(10)).toBe('10x');
  });

  it('does not imply precision it rounded away', () => {
    // A trailing `.0` claims a tenth that was measured. Same rule as `tokens`.
    expect(multiple(3)).toBe('3x');
    expect(multiple(1.987)).toBe('2x');
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
    // A zero ceiling makes the budget bar's overshoot `Infinity`, which is the case that reaches
    // this formatter with something that is not a number.
    ['multiple', multiple],
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

  /**
   * Every cutoff in this module tests the figure as it will print, not the raw value (#126).
   * Rounding on the display side of a raw-value test printed "600 s" astride the minutes cutoff,
   * "1000K" astride the M cutoff, and "10.0 s" where the next band says "10 s". TTFT sweeps
   * through these bands continuously as the prompt slider moves, so each one is reachable.
   */
  it('never rounds a figure across the cutoff it was tested against', () => {
    expect(seconds(599.7)).toBe('10 min');
    expect(seconds(9.97)).toBe('10 s');
    expect(seconds(0.9996)).toBe('1.0 s');
    expect(rate(9.97)).toBe('10');
    expect(gib(9.97 * 2 ** 30)).toBe('10');
    expect(tokens(1048570)).toBe('1M');
    expect(compact(999_500)).toBe('1M');
  });

  it('and the guard fires only where the display would actually cross', () => {
    // The near side of every boundary above: a value whose rounded display stays under the
    // cutoff keeps its unit and its precision. A guard rounding at the wrong granularity turns
    // the whole 9.5–9.94 band into "10" — the first draft of this fix did exactly that.
    expect(seconds(599.4)).toBe('599 s');
    expect(seconds(9.7)).toBe('9.7 s');
    expect(seconds(0.994)).toBe('994 ms');
    expect(rate(9.7)).toBe('9.7');
    expect(gib(9.7 * 2 ** 30)).toBe('9.7');
    expect(tokens(1048524)).toBe('1023.9K');
    expect(compact(999_499)).toBe('999K');
  });

  it('says context lengths the way people do', () => {
    expect(tokens(4096)).toBe('4K');
    expect(tokens(131072)).toBe('128K');
    expect(tokens(1048576)).toBe('1M');
    expect(tokens(1_048_576)).toBe('1M');
  });

  it('says parameter counts the way people do, tenths included', () => {
    // gpt-oss-120b's real count — the docblock promised "116.8B" while the code printed "117B".
    expect(params(116_829_156_672)).toBe('116.8B');
    expect(params(671_000_000_000)).toBe('671B');
    expect(params(8_030_000_000)).toBe('8.03B');
    // 99.96B prints as the "100B" people would say, not a "100.0B" claiming a measured tenth.
    expect(params(99_960_000_000)).toBe('100B');
  });

  it('drops the trailing .0 from a rounded download count', () => {
    // Both live in the shipped catalog today: Llama 3.1 8B and 70B Instruct.
    expect(compact(8_022_692)).toBe('8M');
    expect(compact(1_009_219)).toBe('1M');
    expect(compact(1_250_000)).toBe('1.3M');
    expect(compact(890_000)).toBe('890K');
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

/**
 * Prose assembled from optional clauses, which is where a correct set of facts becomes an
 * unreadable one (#68).
 *
 * The picker note was `[warning, ceiling, curatedNote].filter(Boolean).join(' ')`, and neither of
 * the first two ended in punctuation — so nine Hardware rows read "raiseable to 240 GiB The
 * allocation ceiling reserves 16 GiB for macOS", and on the rumoured Mac the sentence that ran on
 * was the rumour warning. The fragments now terminate themselves *and* go through this, because
 * half of them come from a hand-curated JSON file where the convention is a habit rather than an
 * invariant.
 */
describe('composing prose from independent fragments', () => {
  it('starts a new sentence where the previous fragment did not end one', () => {
    // The exact failure from the issue, minus the 55 words of derivation that followed it.
    expect(
      sentences('384 GiB allocatable, raiseable to 480 GiB', 'The ceiling reserves 32 GiB')
    ).toBe('384 GiB allocatable, raiseable to 480 GiB. The ceiling reserves 32 GiB');
  });

  it('leaves a fragment that already ends a sentence alone', () => {
    expect(sentences('Rumoured — specs may change.', 'Unreleased.')).toBe(
      'Rumoured — specs may change. Unreleased.'
    );
    // Every mark that ends a sentence, including the two that hand over rather than stop.
    for (const mark of ['.', '!', '?', '…', '—']) {
      expect(sentences(`a${mark}`, 'B')).toBe(`a${mark} B`);
    }
    // And behind a closing delimiter, so "(as the datasheet says.)" is not given a second stop.
    expect(sentences('a (as stated.)', 'B')).toBe('a (as stated.) B');
  });

  it('leaves the last fragment exactly as the curator wrote it', () => {
    // Nothing follows it, so there is nothing to run into — and a curator who ended on a clause
    // rather than a sentence is not overruled here.
    expect(sentences('First.', 'trailing clause')).toBe('First. trailing clause');
    expect(sentences('only fragment')).toBe('only fragment');
  });

  it('drops the clauses that do not apply, without leaving their spaces behind', () => {
    expect(sentences(undefined, 'Only this.', false, null, '')).toBe('Only this.');
    expect(sentences('  padded  ', ' and trimmed ')).toBe('padded. and trimmed');
  });

  it('returns undefined when nothing applies, so no description points at an empty node', () => {
    expect(sentences()).toBeUndefined();
    expect(sentences(undefined, false, '', '   ')).toBeUndefined();
  });
});

/**
 * The same composition one layer over, for the string a picker shows a reader who has not chosen yet
 * (#69).
 *
 * `sentences` above composes the *note*, which `Select` renders for the selected option only. So a
 * caveat that decides whether a row is real — the rumoured Mac Studio's specs, a runtime that cannot
 * drive the selected machine — was unreachable until after the choice it was meant to inform. An
 * `<option>` renders no children, so the marker has to be in the text, and this is where it is joined.
 */
describe('marking an option a reader has not chosen yet', () => {
  it('leaves a label with nothing to say about it exactly as it was', () => {
    // Most rows are ordinary, and a separator with nothing after it is a label that looks truncated.
    expect(optionLabel('RTX 5090 — 32 GiB')).toBe('RTX 5090 — 32 GiB');
    expect(optionLabel('RTX 5090 — 32 GiB', undefined, false, null, '', '   ')).toBe(
      'RTX 5090 — 32 GiB'
    );
  });

  it('separates the marker from a label that has already spent its punctuation', () => {
    // The row the issue names. A comma or a bracket would read as part of the spec: the name has a
    // parenthetical, the capacity has an em dash and a unit.
    expect(optionLabel('Mac Studio M5 Ultra (512 GB) — 512 GiB', 'rumoured')).toBe(
      'Mac Studio M5 Ultra (512 GB) — 512 GiB · rumoured'
    );
  });

  it('composes more than one marker without a second joiner somewhere else', () => {
    expect(optionLabel('vLLM', false, 'does not run on this hardware')).toBe(
      'vLLM · does not run on this hardware'
    );
    expect(optionLabel('a', 'one', undefined, ' two ')).toBe('a · one · two');
  });
});

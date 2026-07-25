import { describe, expect, it } from 'vitest';
import {
  DECODE_FAST,
  DECODE_USABLE,
  TTFT_TOLERABLE,
  classifyDecode,
  classifyTtft,
} from './verdicts';
import { rate, seconds } from './format';

/**
 * These exist because every surface used to hold its own copy of a threshold, and the copies
 * drifted. What matters is not the numbers themselves — they are judgement, and arguable — but
 * that one number is classified one way wherever it appears, and that the classification agrees
 * with the figure printed beside it.
 */
describe('shared classification', () => {
  it('judges decode on the figure it prints, not the raw estimate', () => {
    // 29.7 rounds to "30", which the tile shows. Judging the raw value called it Usable while
    // the printed number sat on the Fast threshold.
    const judged = classifyDecode(29.7);
    expect(rate(29.7)).toBe('30');
    expect(judged.shown).toBe('30');
    expect(judged.word).toBe('Fast');
  });

  it('judges first-token latency the same way', () => {
    // 10.27s prints as "10 s" and was labelled "Slow start" against a visible threshold of 10.
    const judged = classifyTtft(10.27);
    expect(seconds(10.27)).toBe('10 s');
    expect(judged.shown).toBe('10 s');
    expect(judged.word).toBe('Noticeable');
  });

  it('reads its own units back before comparing', () => {
    // `seconds` switches to ms and min, so a naive parseFloat would read "500 ms" as 500.
    expect(classifyTtft(0.5).word).toBe('Responsive');
    expect(classifyTtft(700).word).toBe('Slow start');
  });

  it('agrees with itself at every boundary', () => {
    expect(classifyDecode(DECODE_FAST).word).toBe('Fast');
    expect(classifyDecode(DECODE_USABLE).word).toBe('Usable');
    expect(classifyTtft(TTFT_TOLERABLE).word).toBe('Noticeable');
    expect(classifyTtft(TTFT_TOLERABLE + 1).word).toBe('Slow start');
  });

  it('never returns a tone that contradicts its own word', () => {
    for (const n of [0.1, 5, 14.9, 15, 29.9, 30, 200]) {
      const { word, tone, isFast } = classifyDecode(n);
      expect(isFast).toBe(word === 'Fast');
      expect(tone).toBe(word === 'Fast' ? 'good' : word === 'Usable' ? 'warning' : 'serious');
    }
    for (const n of [0.05, 1.9, 2, 9.9, 10, 60, 700]) {
      const { word, tone, isResponsive } = classifyTtft(n);
      expect(isResponsive).toBe(word === 'Responsive');
      expect(tone).toBe(
        word === 'Responsive' ? 'good' : word === 'Noticeable' ? 'warning' : 'critical'
      );
    }
  });
});

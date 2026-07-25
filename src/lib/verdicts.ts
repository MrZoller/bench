import type { StatusTone } from '@/design/tokens';
import { rate } from './format';

/**
 * The thresholds and classifications every surface judges against.
 *
 * Out of the components on purpose. These began as constants inside the decode tile, and each
 * time another surface needed one it took a copy — the Bench's teaching aside, the Envelope's
 * cell states, the Matrix. Every copy drifted, and one of them drifted twice.
 *
 * They are reading speed and patience rather than benchmarks: below ~10 tok/s you are watching a
 * cursor, ~15 keeps pace with reading, past ~30 it outruns most people.
 */

export const DECODE_FAST = 30;
export const DECODE_USABLE = 15;

/** Share of the allocatable ceiling past which a fit counts as tight rather than comfortable. */
export const CAPACITY_TIGHT = 0.9;

/**
 * The one decode classification.
 *
 * A function rather than bare thresholds, because sharing the numbers was not enough: the tile
 * classified the figure it *prints* while the teaching aside compared the raw estimate, so at
 * 29.7 tok/s the tile read "Fast" and the aside denied it in the same breath. Everything asks
 * the same question of the same number now.
 */
export function classifyDecode(perUserTokensPerSec: number): {
  shown: string;
  word: 'Fast' | 'Usable' | 'Slow';
  tone: StatusTone;
  isFast: boolean;
} {
  const shown = rate(perUserTokensPerSec);
  const value = Number(shown);
  const word = value >= DECODE_FAST ? 'Fast' : value >= DECODE_USABLE ? 'Usable' : 'Slow';
  return {
    shown,
    word,
    tone: word === 'Fast' ? 'good' : word === 'Usable' ? 'warning' : 'serious',
    isFast: word === 'Fast',
  };
}

import type { StatusTone } from '@/design/tokens';
import { rate, seconds } from './format';

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
 * Time to first token, in seconds. Under two the answer starts while you are still letting go
 * of the return key; past ten you have gone to do something else.
 */
export const TTFT_RESPONSIVE = 2;
export const TTFT_TOLERABLE = 10;

/**
 * What a spilled placement does not promise.
 *
 * `planPlacement` sizes the spill and has no host-RAM input at all, so a configuration needing
 * hundreds of GiB of it is reported exactly like one needing two. Any surface saying a spilled
 * configuration "loads" is promising something the engine never checked, and every one of them has
 * to carry the qualifier.
 *
 * One sentence rather than a hand-maintained near-copy per panel — the two that existed had already
 * drifted to "the spilled *part*" and "the spilled *weights*". Each surface appends its own second
 * sentence, because a legend swatch and a diagnostic tile want different amounts of explanation
 * after the fact they share. If `planPlacement` ever gains a host-RAM input, this is the one place
 * that has to change, and its callers are what needs re-reading.
 */
export const HOST_RAM_UNCHECKED =
  'Loads only if the host has RAM for the spilled weights, which is not checked here.';

/**
 * The one name for the over-budget state a setting could fix.
 *
 * `raisingCeilingWouldHelp` splits `impossible` into a wall and a default, and every surface that
 * renders the split has to use one word for it: the Envelope's legend and table cells did, the
 * Matrix's `blockedBy` does, and the capacity tile said "Will not run" over a detail explaining
 * the ceiling can be raised — a flat refusal above the sentence refuting it (#121). The tile's
 * `verdict` field exists for exactly this override, and the word it takes is this one.
 *
 * `engine/matrix.ts` states the same phrase for its `blockedBy` and cannot import it — the engine
 * imports nothing outside `src/engine/` — so `matrix.test.ts` asserts the two spellings agree
 * rather than leaving them agreeing by coincidence.
 */
export const PAST_DEFAULT_ALLOCATION = 'Past the default allocation';

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

/**
 * The one first-token-latency classification, for the same reason `classifyDecode` exists.
 *
 * Judged on the printed figure: `seconds()` rounds, so 10.27s prints as "10 s" and would
 * otherwise be labelled "Slow start" against a visible threshold of 10.
 */
export function classifyTtft(ttftSeconds: number): {
  shown: string;
  word: 'Responsive' | 'Noticeable' | 'Slow start';
  tone: StatusTone;
  isResponsive: boolean;
} {
  const shown = seconds(ttftSeconds);
  const value = parseDisplayedSeconds(shown, ttftSeconds);
  const word =
    value <= TTFT_RESPONSIVE ? 'Responsive' : value <= TTFT_TOLERABLE ? 'Noticeable' : 'Slow start';
  return {
    shown,
    word,
    tone: word === 'Responsive' ? 'good' : word === 'Noticeable' ? 'warning' : 'critical',
    isResponsive: word === 'Responsive',
  };
}

/** `seconds()` switches units, so its output has to be read back before it can be compared. */
export function parseDisplayedSeconds(shown: string, raw: number): number {
  const value = Number.parseFloat(shown);
  if (!Number.isFinite(value)) return raw;
  if (shown.endsWith('ms')) return value / 1000;
  if (shown.endsWith('min')) return value * 60;
  return value;
}

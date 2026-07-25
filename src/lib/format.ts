import { GIB } from '@/engine/types';

/**
 * Display formatting.
 *
 * Two conventions held consistently, because mixing them is how a tool loses trust:
 *   - **Memory is binary and labelled GiB.** A "32GB" card holds 32 GiB, and calling that 34.4
 *     GB — technically correct — makes every figure look wrong to someone reading a spec sheet.
 *   - **Rates are decimal**, as vendors and benchmarks quote them.
 */

/** Memory, in GiB, with precision that falls away as the number grows. */
export function gib(bytes: number): string {
  if (!Number.isFinite(bytes)) return '—';
  const value = bytes / GIB;
  if (value === 0) return '0';
  if (value < 10) return value.toFixed(1);
  return Math.round(value).toString();
}

export function gibLabel(bytes: number): string {
  return `${gib(bytes)} GiB`;
}

/** Parameter counts, as people say them: 8B, 116.8B, 671B. */
export function params(count: number): string {
  if (!Number.isFinite(count)) return '—';
  const b = count / 1e9;
  if (b >= 100) return `${Math.round(b)}B`;
  if (b >= 10) return `${b.toFixed(1)}B`;
  return `${b.toFixed(2).replace(/\.?0+$/, '')}B`;
}

/** Context lengths, as people say them: 4K, 32K, 128K, 1M. */
export function tokens(count: number): string {
  if (!Number.isFinite(count)) return '—';
  // Trailing zeros dropped: 1,048,576 is "1M", not "1.0M". The modulo test that used to guard
  // this only caught exact multiples of a million, which a binary context length never is.
  if (count >= 1e6) return `${trim((count / 1e6).toFixed(1))}M`;
  if (count >= 1024) return `${Math.round(count / 1024)}K`;
  return String(count);
}

/** Drop a trailing `.0`, so a rounded figure does not imply precision it lacks. */
function trim(value: string): string {
  return value.replace(/\.0$/, '');
}

/** Throughput. Sub-10 keeps a decimal, because 3.2 and 4.0 tok/s are different lives. */
export function rate(tokensPerSec: number): string {
  if (!Number.isFinite(tokensPerSec)) return '—';
  if (tokensPerSec >= 100) return Math.round(tokensPerSec).toString();
  if (tokensPerSec >= 10) return tokensPerSec.toFixed(0);
  return tokensPerSec.toFixed(1);
}

/** Latency, switching units so the number stays small enough to read at a glance. */
export function seconds(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value >= 600) return `${Math.round(value / 60)} min`;
  if (value >= 10) return `${Math.round(value)} s`;
  if (value >= 1) return `${value.toFixed(1)} s`;
  return `${Math.round(value * 1000)} ms`;
}

/**
 * A percentage, with a floor so a real quantity never reads as nothing.
 *
 * Rounding alone printed "0% of weights would spill" for a configuration that is over budget and
 * genuinely spilling — 0.43% on a two-card MI355X rig. Zero has to stay reserved for zero.
 */
export function percent(fraction: number): string {
  if (!Number.isFinite(fraction)) return '—';
  if (fraction > 0 && fraction < 0.005) return '<1%';
  return `${Math.round(fraction * 100)}%`;
}

/** Download counts for the model picker: 1.2M, 890K. */
export function compact(count: number): string {
  if (!Number.isFinite(count)) return '—';
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${Math.round(count / 1e3)}K`;
  return String(count);
}

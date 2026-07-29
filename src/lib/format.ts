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

/**
 * Context lengths, as people say them: 4K, 32K, 128K, 1M.
 *
 * Rounds only where rounding is lossless. The round figures people name are exact multiples of
 * 1,024, so those print clean; a value that is not a whole K keeps a decimal, so 33,000 reads
 * "32.2K" rather than colliding with 32,768's "32K".
 *
 * That narrows collisions but cannot close them — 131,073 still prints as "128K", and no fixed
 * precision survives an arbitrary integer from a URL. Uniqueness is not a property this function
 * can promise, because it sees one number at a time. Where it actually matters — an axis, whose
 * labels sit side by side and must each identify one column — the caller holds the whole set and
 * disambiguates against it. See `uniqueLabels`.
 */
export function tokens(count: number): string {
  if (!Number.isFinite(count)) return '—';
  // Trailing zeros dropped: 1,048,576 is "1M", not "1.0M". The modulo test that used to guard
  // this only caught exact multiples of a million, which a binary context length never is.
  // Threshold in mebibytes, not decimal millions: dividing a 1e6 cutoff by 1,048,576 printed
  // 1,000,000 tokens as "0.95M".
  if (count >= 1048576) {
    const m = count / 1048576;
    return Number.isInteger(m) ? `${m}M` : `${trim(m.toFixed(1))}M`;
  }
  if (count >= 1024) {
    const k = count / 1024;
    return Number.isInteger(k) ? `${k}K` : `${trim(k.toFixed(1))}K`;
  }
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

/**
 * A ratio said as a multiple: 14x, 3.1x.
 *
 * `percent` is the wrong form once a ratio leaves the neighbourhood of 1. "1446% of the ceiling"
 * is the same arithmetic as "14x the ceiling" and nobody reads the first as fourteen times — which
 * matters where the multiple is carrying a magnitude a picture has stopped conveying.
 *
 * Precision falls away as the number grows, the same rule `gib` and `rate` already follow: at 14x
 * the tenth is noise, and near 3x it is the difference between a bar that still shows the gap it is
 * measuring and one that does not.
 */
export function multiple(ratio: number): string {
  if (!Number.isFinite(ratio)) return '—';
  return `${ratio >= 10 ? Math.round(ratio) : trim(ratio.toFixed(1))}x`;
}

/** Download counts for the model picker: 1.2M, 890K. */
export function compact(count: number): string {
  if (!Number.isFinite(count)) return '—';
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${Math.round(count / 1e3)}K`;
  return String(count);
}

/**
 * Axis labels that each identify exactly one value.
 *
 * `tokens()` sees one number at a time and so cannot guarantee this: 131,072 and 131,073 both
 * round to "128K", and the second is reachable from a hand-edited URL. An axis does hold the
 * whole set, so the guarantee belongs here — any value whose short label is shared falls back to
 * its exact count, which is long but unambiguous, and only for the columns that actually clash.
 */
export function uniqueLabels(values: readonly number[]): string[] {
  const short = values.map(tokens);
  const seen = new Map<string, number>();
  for (const label of short) seen.set(label, (seen.get(label) ?? 0) + 1);
  return short.map((label, i) =>
    seen.get(label)! > 1 ? values[i].toLocaleString('en-US') : label
  );
}

/**
 * The marks that end a sentence, optionally behind a closing quote or bracket.
 *
 * `—` is in the set because a fragment that trails off into an em dash is already handing over to
 * whatever follows it, and `…` for the same reason. A closing delimiter is allowed *after* the
 * mark so that a fragment ending `(as the datasheet says.)` is recognised as finished.
 */
const TERMINAL = /[.!?…—][»”’"')\]]?$/;

/**
 * Independent fragments composed into prose, each one ending as its own sentence.
 *
 * `[a, b, c].filter(Boolean).join(' ')` is the obvious way to assemble a note out of clauses that
 * may or may not apply, and it is wrong the moment a clause does not end in punctuation. Nine
 * Hardware rows read "192 GiB allocatable by default, raiseable to 240 GiB The allocation ceiling
 * reserves 16 GiB for macOS…" — and on the one rumoured machine the sentence that ran on was the
 * rumour warning, fused to a capacity figure (#68).
 *
 * **Both halves of the fix, deliberately.** The clauses the app generates now carry their own full
 * stops, because a fragment that reads as a sentence where it is written is a fragment a future
 * caller cannot misuse. This function is the guarantee behind that convention: half of these
 * fragments come from `devices.json`, which is curated by hand, so source discipline there is a
 * habit and not an invariant — and the next note added is exactly where the habit lapses.
 *
 * **The last fragment is left exactly as written.** There is nothing after it to run into, and
 * appending a full stop would overrule a curator who chose to end on a question or an ellipsis.
 *
 * Returns `undefined` rather than `''` for an empty composition, because every caller feeds this
 * to an optional `note` and an empty string would emit an `aria-describedby` pointing at nothing.
 */
export function sentences(...fragments: (string | false | null | undefined)[]): string | undefined {
  const present = fragments
    .filter((f): f is string => typeof f === 'string' && f.trim() !== '')
    .map((f) => f.trim());
  if (present.length === 0) return undefined;

  return present
    .map((fragment, i) =>
      i === present.length - 1 || TERMINAL.test(fragment) ? fragment : `${fragment}.`
    )
    .join(' ');
}

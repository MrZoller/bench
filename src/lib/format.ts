import { GIB } from '@/engine/types';

/**
 * Display formatting.
 *
 * Two conventions held consistently, because mixing them is how a tool loses trust:
 *   - **Memory is binary and labelled GiB.** A "32GB" card holds 32 GiB, and calling that 34.4
 *     GB — technically correct — makes every figure look wrong to someone reading a spec sheet.
 *   - **Rates are decimal**, as vendors and benchmarks quote them.
 *
 * And one rule about thresholds: **a unit or precision cutoff tests the figure as it will print,
 * not the raw value.** Rounding happens on the display side of every branch, so testing the raw
 * value lets a number round across the cutoff it was tested against — 599.7 failed `>= 600` and
 * then printed "600 s", 999,500 failed `>= 1e6` and printed "1000K" (#126).
 */

/**
 * Memory, in GiB, with precision that falls away as the number grows.
 *
 * Zero stays reserved for zero, the doctrine `percent` states below — so a positive quantity
 * under half a tenth reports as an upper bound rather than rounding to "0.0" (#119). Without the
 * floor, an over-budget verdict could quantify itself as nothing: a URL-borne context can land
 * used arbitrarily close above the ceiling, and the banner read "Over the ceiling by 0.0 GiB —
 * <1% of weights would spill", with the sibling clause applying the floor to the same kind of
 * tiny quantity one formatter over. Exact zero keeps its own spelling, so "0" and "<0.1" still
 * read as the different claims they are.
 */
export function gib(bytes: number): string {
  if (!Number.isFinite(bytes)) return '—';
  const value = bytes / GIB;
  if (value === 0) return '0';
  if (value > 0 && value < 0.05) return '<0.1';
  // Round-then-test at the decimal branch's own granularity: 9.97 would print "10.0" astride
  // the cutoff, while 9.7 still keeps its tenth.
  if (Math.round(value * 10) >= 100) return Math.round(value).toString();
  return value.toFixed(1);
}

export function gibLabel(bytes: number): string {
  return `${gib(bytes)} GiB`;
}

/**
 * Parameter counts, as people say them: 8B, 116.8B, 671B.
 *
 * One decimal from 10B up, trimmed where it is a round `.0` — people do say "116.8B", and the
 * branch that rounded it to "117B" made that promise unkeepable (#126). Under 10B two decimals
 * still matter: 1.24B and 1.2B are different models.
 */
export function params(count: number): string {
  if (!Number.isFinite(count)) return '—';
  const b = count / 1e9;
  if (b >= 10) return `${trim(b.toFixed(1))}B`;
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
    // A non-integer K can still round to the M cutoff on the display side: 1,048,570 is
    // 1023.994K, which prints "1024.0K" → "1024K". Rounded to the cutoff means the next unit up.
    if (Math.round(k * 10) >= 10240) return `${trim((count / 1048576).toFixed(1))}M`;
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
  // Round-then-test at the decimal branch's own granularity: 9.97 would print "10.0" astride
  // the cutoff, while 9.7 still keeps its tenth.
  if (Math.round(tokensPerSec * 10) >= 100) return tokensPerSec.toFixed(0);
  return tokensPerSec.toFixed(1);
}

/** Latency, switching units so the number stays small enough to read at a glance. */
export function seconds(value: number): string {
  if (!Number.isFinite(value)) return '—';
  // Every cutoff tests the figure at the granularity the branch below it prints: 599.7 must not
  // print "600 s", 9.97 must not print "10.0 s", and 0.9996 must not print "1000 ms" — while
  // 599.4 stays "599 s" and 9.7 stays "9.7 s".
  if (Math.round(value) >= 600) return `${Math.round(value / 60)} min`;
  if (Math.round(value * 10) >= 100) return `${Math.round(value)} s`;
  if (Math.round(value * 1000) >= 1000) return `${value.toFixed(1)} s`;
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
  // Round-then-test (999,500 rounds to "1000K"), and the M figure goes through `trim` like every
  // other formatter here — 8,022,692 downloads is "8M", not a "8.0M" implying measured tenths.
  if (count >= 1e6 || Math.round(count / 1e3) >= 1000) {
    return `${trim((count / 1e6).toFixed(1))}M`;
  }
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

/**
 * An `<option>`'s own text, plus any caveat a reader needs *before* the choice rather than after it.
 *
 * The sibling of `sentences` above, for the other half of what a picker says — and it exists because
 * a `Select` renders only the **selected** option's note. A warning that lives there is unreachable
 * until the choice it was meant to inform has already been made: the rumoured Mac Studio's
 * "Rumoured — specs may change." sat one line under the control and appeared only once someone had
 * picked the machine, so in the open list it was indistinguishable from real hardware with real
 * measured bandwidth (#69). An `<option>` renders no children, so its text is the only place a
 * marker can go.
 *
 * ` · ` rather than a comma, a bracket or a dash, because the labels this is appended to have spent
 * all three: "Mac Studio M5 Ultra (512 GB) — 512 GiB" is a parenthetical, an em dash and a unit
 * already, and a fourth kind of separator is what keeps the marker from reading as part of the spec.
 *
 * Variadic and falsy-tolerant like `sentences`, for the same two reasons: a call site composes
 * conditions instead of nesting ternaries, and a row that earns a second marker does not need a
 * second joiner written somewhere else.
 *
 * **Markers are short and lower case, and that is a rule about where they are.** They are not
 * sentences — `sentences` is for the note, which is prose. This is a tag inside a line of tabular
 * label text, and a full stop in the middle of one reads as the end of the option.
 */
export function optionLabel(
  label: string,
  ...markers: (string | false | null | undefined)[]
): string {
  const present = markers
    .filter((m): m is string => typeof m === 'string' && m.trim() !== '')
    .map((m) => m.trim());
  return [label, ...present].join(' · ');
}

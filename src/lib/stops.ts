import type { KvPrecision, RuntimeSpec } from '@/engine/types';
// The scenario *shape*, not the store: `scenario.ts` deliberately depends on nothing but engine
// types so that everything needing the shape can have it without a cycle. Type-only, so it erases.
import type { Config } from '@/store/scenario';

/**
 * The values the controls can actually produce.
 *
 * One definition, because two surfaces read the same scenario and disagreed about its shape:
 * the Bench offered concurrency up to 128 and context up to a model's own ceiling, while the
 * Envelope drew a grid stopping at 64 users and 128K. The region was therefore answering "how
 * much room is left" over a smaller domain than the one you can steer into — the columns that
 * would have gone red were simply not drawn.
 *
 * Log-spaced rather than linear. The interesting jumps in context are 4K → 32K → 128K, and a
 * linear range would spend most of its travel in a region nobody is deciding between.
 */

export const CONTEXT_STOPS = [
  2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
] as const;
export const CONCURRENCY_STOPS = [1, 2, 4, 8, 16, 32, 64, 128] as const;
export const PROMPT_STOPS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;
export const DEVICE_COUNT_STOPS = [1, 2, 4, 8] as const;

/**
 * What each setting is called, wherever it is *labelled*.
 *
 * One entry per `Config` field, because every field is something a control sets and a picture can
 * then draw. The Envelope draws two of them as its axes — so its axis titles, its table's caption
 * and its row-header column read from here rather than restating them. They had already drifted
 * three ways for one setting: the slider said "Concurrent users", the table header said "Users",
 * and the field's own y axis named it nowhere at all (#81).
 *
 * Same reasoning as `kvLabel` below, one level up: two surfaces naming one setting differently is
 * the failure this repo keeps hitting, and it is cheaper to remove than to remember. The four setup
 * settings are here for the same reason and not because a second surface reads them yet — the
 * Matrix's row axis is `<span class="sr-only">Model</span>`, which agrees with the control by
 * coincidence today, exactly the coincidence recorded on `kvLabel`.
 *
 * `satisfies Record<keyof Config, string>` is what makes the keying a claim rather than a comment:
 * rename a `Config` field, or add one, and this fails to compile instead of silently keeping a
 * label for a setting that no longer exists.
 *
 * **Prose is not a label and does not read from here.** "Currently at 32K context and 1 user" is a
 * sentence about a cell, and the Envelope's subhead ("context against concurrent users") is a
 * sentence about the panel; forcing a control's name into either produces "Currently at 32K Context
 * per sequence", which is worse English than the drift it would prevent. The test of which side a
 * surface falls on is whether it *names a setting* — a control label, an axis title, a caption or a
 * column header — or whether it says something about the state in a sentence.
 */
export const SETTING_LABELS = {
  modelId: 'Model',
  deviceId: 'Hardware',
  quantId: 'Quantization',
  runtimeId: 'Runtime',
  contextTokens: 'Context per sequence',
  concurrency: 'Concurrent users',
  promptTokens: 'Prompt length',
  kvPrecision: 'KV precision',
  deviceCount: 'Device count',
} as const satisfies Record<keyof Config, string>;

/**
 * The cache precisions a control can offer, with the name to use when a runtime has none of
 * its own.
 */
export const KV_PRECISIONS: readonly { value: KvPrecision; label: string }[] = [
  { value: 'fp16', label: 'FP16' },
  { value: 'q8', label: 'Q8' },
  { value: 'q4', label: 'Q4' },
];

const KV_FALLBACK_LABELS = new Map(KV_PRECISIONS.map((k) => [k.value, k.label]));

/**
 * What a runtime calls a cache precision.
 *
 * `KvPrecision` is an internal width, not a name anyone types: vLLM's one-byte cache is FP8 with
 * no integer-Q8 option at all, so the catalog gives it a `kvLabels` entry and the control names
 * something the user could actually pass on a command line.
 *
 * One function because there were two resolutions and they disagreed about the fallback — the
 * Bench control read this table while the Matrix heading upper-cased the raw value. They agree on
 * all three current precisions by coincidence, and the first one whose display name is not its id
 * in capitals (`fp8_e5m2`, `q4_0`) would have had the two surfaces printing different names for
 * one setting. That is the failure this repo keeps hitting, and it is cheaper to remove than to
 * remember.
 */
export function kvLabel(runtime: RuntimeSpec, precision: KvPrecision): string {
  return (
    runtime.kvLabels?.[precision] ?? KV_FALLBACK_LABELS.get(precision) ?? precision.toUpperCase()
  );
}

/**
 * The fixed stops plus whatever is currently stored.
 *
 * `coerce` accepts any integer in range, so a value arriving from a URL — `?u=3` — has to be
 * offered too, or the control displays 2 while the engine evaluates 3.
 */
export function withStored(values: readonly number[], stored: number): number[] {
  return [...new Set([...values, stored])].sort((a, b) => a - b);
}

/**
 * Contexts this model can reach, plus its exact ceiling, plus whatever is selected.
 *
 * The ceiling is included as its own stop because it is rarely a power of two — Qwen3 stops at
 * 40,960 — and `coerce` clamps to it. Without it the largest offered value was 32,768 while the
 * model would hold a quarter more, and the Envelope's rightmost column understated the room.
 */
export function contextStopsFor(maxContext: number, stored: number): number[] {
  const within = CONTEXT_STOPS.filter((t) => t < maxContext);
  const stops = new Set([...within, maxContext, Math.min(stored, maxContext)]);
  return [...stops].filter((t) => t <= maxContext).sort((a, b) => a - b);
}

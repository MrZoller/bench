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

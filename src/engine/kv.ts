import type { AttentionSpec, KvPrecision, ModelSpec, RuntimeSpec } from './types';
import { KV_BYTES } from './types';

/**
 * Bytes one cached element really costs under this runtime.
 *
 * Nominal width unless the runtime declares otherwise — see `RuntimeSpec.kvBytesPerElement`.
 * Taken through one function so placement and decode cannot end up charging different figures
 * for the same cache.
 */
export function kvElementBytes(precision: KvPrecision, runtime?: RuntimeSpec): number {
  return runtime?.kvBytesPerElement?.[precision] ?? KV_BYTES[precision];
}

/**
 * KV cache sizing.
 *
 * The formula nearly every calculator uses — `2 * layers * kv_heads * head_dim * bytes` —
 * is right only for plain GQA models with full attention on every layer. Applied blindly it
 * overestimates two families badly enough to change buying decisions:
 *
 *   - **MLA** (DeepSeek V3/V4) caches a single compressed latent per token per layer. No
 *     factor of two, no head multiplier. ~70 KB/token where the naive formula predicts
 *     several times that.
 *   - **Sliding-window layers** (gpt-oss, Gemma, Ministral) stop growing once the context
 *     passes the window, so those layers contribute a constant. gpt-oss-120b at 128K holds
 *     ~4.8 GB of KV, not the ~9.7 GB a uniform formula claims.
 *
 * Both errors point the same way: telling someone they need hardware they don't.
 */

/**
 * The bound on what this layer attends over, or `null` for the whole context.
 *
 * One function because three callers now dispatch on it and they must agree about what an absent
 * entry means: `null` and an absent `layerWindows` pattern are both "attends over everything", and
 * a reader that treated a missing entry as a window of zero would report a model with no cache.
 */
function windowOf(attention: AttentionSpec, layerIndex: number): number | null {
  return attention.layerWindows?.[layerIndex] ?? null;
}

/**
 * Whether this layer's attention is bounded — the property that makes a hybrid model's layers
 * non-interchangeable, and therefore the property a per-device layer *set* has to be read against.
 *
 * Exported for `launch.ts`, which says what bench packed onto each card: on Gemma at 128K one of
 * these caches ~128x what a windowed one does, so "nine layers" and "nine layers, two of them
 * unbounded" are different statements about the same card.
 */
export function isSlidingLayer(model: ModelSpec, layerIndex: number): boolean {
  return windowOf(model.attention, layerIndex) !== null;
}

/**
 * Tokens this layer actually holds at a context — the whole of it, or its window.
 *
 * `…Of` rather than `cachedTokens`, which is the name of the local `layerBytes` computes from the
 * same two lines: a module-level function shadowed by a `const` in the one place it would most
 * naturally be called is a trap rather than a coincidence of naming.
 */
function cachedTokensOf(model: ModelSpec, layerIndex: number, contextTokens: number): number {
  const window = windowOf(model.attention, layerIndex);
  return window === null ? contextTokens : Math.min(contextTokens, window);
}

/**
 * Whether every layer caches the same amount at this context — which is what makes a layer *count*
 * a complete description of a packing, and is a property of the scenario rather than of the model.
 *
 * `hasSlidingLayers` is the model-level question and it is the wrong one to gate a per-device split
 * on. A hybrid model whose context has not reached its shortest window has no expensive layers yet:
 * every layer holds `contextTokens` and any assignment with the same counts has the same load. Read
 * as "is this model hybrid" instead, `launch.ts` refused `-ts` on Gemma at a 1,024-token context —
 * where the flag is exact — and then explained the refusal with an imbalance that does not exist at
 * that context.
 *
 * Cached tokens rather than bytes, because every other factor in `layerBytes` is model-wide: two
 * layers cache the same bytes exactly when they cache the same tokens, so this is the same
 * comparison one multiplication earlier and needs no precision or runtime.
 */
export function layersCacheAlike(model: ModelSpec, contextTokens: number): boolean {
  const first = cachedTokensOf(model, 0, contextTokens);
  for (let layer = 1; layer < model.layers; layer++) {
    if (cachedTokensOf(model, layer, contextTokens) !== first) return false;
  }
  return true;
}

/** Bytes of KV held for a single layer at a given context length. */
function layerBytes(
  attention: AttentionSpec,
  layerIndex: number,
  contextTokens: number,
  elemBytes: number
): number {
  const window = windowOf(attention, layerIndex);
  // `null` (or an absent pattern) means this layer attends over everything.
  const cachedTokens = window === null ? contextTokens : Math.min(contextTokens, window);

  switch (attention.core.kind) {
    case 'gqa':
      // Keys and values, one set per KV head.
      return 2 * attention.core.kvHeads * attention.core.headDim * elemBytes * cachedTokens;
    case 'mla':
      // One joint latent plus the decoupled RoPE part. Cached once, not once per head.
      return (attention.core.kvLoraRank + attention.core.qkRopeHeadDim) * elemBytes * cachedTokens;
  }
}

/**
 * KV bytes for one sequence at `contextTokens`.
 *
 * Summed per layer rather than multiplied by layer count, because hybrid models have
 * different windows on different layers and the difference compounds with context.
 */
export function kvBytesPerSequence(
  model: ModelSpec,
  contextTokens: number,
  precision: KvPrecision,
  runtime?: RuntimeSpec
): number {
  const elemBytes = kvElementBytes(precision, runtime);
  let total = 0;
  for (let layer = 0; layer < model.layers; layer++) {
    total += layerBytes(model.attention, layer, contextTokens, elemBytes);
  }
  return total;
}

/**
 * KV cost of one token when every layer is caching — the headline "bytes per token" figure
 * used to compare architectures. Constant, and correct only below the shortest window.
 */
export function kvBytesPerToken(
  model: ModelSpec,
  precision: KvPrecision,
  runtime?: RuntimeSpec
): number {
  return kvBytesPerSequence(model, 1, precision, runtime);
}

/**
 * What one *more* token costs at the current context — the honest answer to "what does
 * another 1K of context buy me".
 *
 * Differs from {@link kvBytesPerToken} for hybrid models, because layers past their window
 * have stopped growing and contribute nothing further. For gpt-oss-120b at 128K the marginal
 * cost is half the headline figure, since only the 18 full-attention layers are still
 * accumulating.
 */
export function marginalKvBytesPerToken(
  model: ModelSpec,
  contextTokens: number,
  precision: KvPrecision,
  runtime?: RuntimeSpec
): number {
  const at = Math.max(1, contextTokens);
  return (
    kvBytesPerSequence(model, at + 1, precision, runtime) -
    kvBytesPerSequence(model, at, precision, runtime)
  );
}

/** Total KV across every concurrent sequence. */
export function kvBytesTotal(
  model: ModelSpec,
  contextTokens: number,
  concurrency: number,
  precision: KvPrecision,
  runtime?: RuntimeSpec
): number {
  return kvBytesPerSequence(model, contextTokens, precision, runtime) * concurrency;
}

/**
 * KV bytes actually read per generated token.
 *
 * Decode re-reads the whole cache for the sequence on every step, so at long context this
 * term rivals or beats the weight-reading term — which is why a model that "fits" can still
 * crawl. Routinely omitted from tok/s estimates elsewhere.
 */
export function kvReadBytesPerToken(
  model: ModelSpec,
  contextTokens: number,
  precision: KvPrecision,
  runtime?: RuntimeSpec
): number {
  return kvBytesPerSequence(model, contextTokens, precision, runtime);
}

/**
 * Whether any layer uses a bounded attention window — drives the explain layer.
 *
 * Asked through {@link isSlidingLayer} over the model's own layer range, rather than over
 * `layerWindows` directly, so this and the per-layer readers cannot diverge. `some((w) => w !==
 * null)` diverges in two ways, both outside the declared `readonly (number | null)[]` and both
 * silent: an explicit `undefined` entry counts as a window here while `windowOf` reads it as full
 * attention, and an entry past `model.layers` counts as a window belonging to a layer that does not
 * exist. Neither is reachable from the shipped catalog — every row that states a pattern states
 * exactly `layers` entries — which is what makes agreement by construction cheaper than a check.
 */
export function hasSlidingLayers(model: ModelSpec): boolean {
  for (let layer = 0; layer < model.layers; layer++) {
    if (isSlidingLayer(model, layer)) return true;
  }
  return false;
}

/**
 * Query-key pairs a prefill pass actually computes, summed over every layer.
 *
 * Two corrections to the naive `layers * N^2` live here, and they compound.
 *
 * **Causality.** These are decoder-only models: prompt token `i` attends over positions 0..i and
 * never over later ones, so a full-attention layer computes `N * (N + 1) / 2` pairs rather than
 * `N^2`. Charging the square nearly doubles the attention term at long prompts, which is enough
 * on its own to make the tile claim attention dominates a pass where it does not.
 *
 * **Sliding windows.** A sliding layer attends over at most its window however long the prompt
 * gets, so its cost is linear rather than quadratic — and it is causal too, so the first `W`
 * positions are still a triangle before the band becomes uniform. Ignoring windows entirely
 * overstates gpt-oss prefill FLOPs by roughly 19% at a 16K prompt; ignoring causality overstates
 * every model by nearly 2x. The same mistake on the compute side that the naive KV formula makes
 * on the memory side.
 *
 * **A cached prefix.** `cachedPrefixTokens` is the scenario every multi-turn archetype actually
 * describes and that this function could not express: `n` *new* tokens attending against a
 * `P`-token prefix already resident in the cache. The new tokens are still read and projected —
 * that is the linear term, and it stays on `n` — but each of them attends over the prefix as well
 * as over itself and the new tokens before it. So a full-attention layer computes
 * `n * P + causal(n)` rather than `causal(n)`.
 *
 * This makes the cached case **more** work than the standalone one, not less, which is the
 * counter-intuitive part: 16K attending against a resident 64K session is nine times the pairs of
 * 16K attending against itself. What a prefix cache buys is not having to *re-read* the prefix; it
 * does not make the new tokens cheaper to attend.
 *
 * Sliding layers cap the prefix at their window, which is the same dispatch the rest of this
 * function already makes — a token at absolute position `P + i` attends over
 * `min(W, P + i + 1)` positions, so the filling triangle is whatever part of the window the prefix
 * has not already used up.
 *
 * At `P = 0` every branch reduces to the expression it replaced, exactly rather than approximately.
 * That matters more than it looks: the published anchors are single-prompt, and a calibration that
 * moved because a new parameter was threaded through would stop being evidence of anything.
 */
export function attentionPairs(
  model: ModelSpec,
  promptTokens: number,
  cachedPrefixTokens = 0
): number {
  const n = Math.max(0, promptTokens);
  const prefix = Math.max(0, cachedPrefixTokens);
  /** Every position attending over itself and everything before it. */
  const causal = (span: number) => (span * (span + 1)) / 2;
  /** `n` new tokens over a resident prefix, unbounded. At `prefix = 0` this is `causal(n)`. */
  const withPrefix = n * prefix + causal(n);

  const windows = model.attention.layerWindows;
  if (!windows) return model.layers * withPrefix;

  let pairs = 0;
  for (let layer = 0; layer < model.layers; layer++) {
    const window = windows[layer];
    if (window === null || window === undefined || prefix + n <= window) {
      // Full attention, or a window the whole working set still fits inside.
      pairs += withPrefix;
    } else {
      // `k` new tokens are still inside the filling triangle — none of them, once the prefix alone
      // has filled the window — and the rest attend over a band of exactly `window`.
      const k = Math.min(Math.max(window - prefix, 0), n);
      pairs += k * prefix + causal(k) + (n - k) * window;
    }
  }
  return pairs;
}

/** KV bytes one layer caches for one sequence — the unit a layer split hands out. */
export function layerKvBytes(
  model: ModelSpec,
  layerIndex: number,
  contextTokens: number,
  precision: KvPrecision,
  runtime?: RuntimeSpec
): number {
  return layerBytes(model.attention, layerIndex, contextTokens, kvElementBytes(precision, runtime));
}

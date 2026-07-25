import type { AttentionSpec, KvPrecision, ModelSpec } from './types';
import { KV_BYTES } from './types';

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

/** Bytes of KV held for a single layer at a given context length. */
function layerBytes(
  attention: AttentionSpec,
  layerIndex: number,
  contextTokens: number,
  elemBytes: number
): number {
  const window = attention.layerWindows?.[layerIndex];
  // `null` (or an absent pattern) means this layer attends over everything.
  const cachedTokens =
    window === null || window === undefined ? contextTokens : Math.min(contextTokens, window);

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
  precision: KvPrecision
): number {
  const elemBytes = KV_BYTES[precision];
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
export function kvBytesPerToken(model: ModelSpec, precision: KvPrecision): number {
  return kvBytesPerSequence(model, 1, precision);
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
  precision: KvPrecision
): number {
  const at = Math.max(1, contextTokens);
  return kvBytesPerSequence(model, at + 1, precision) - kvBytesPerSequence(model, at, precision);
}

/** Total KV across every concurrent sequence. */
export function kvBytesTotal(
  model: ModelSpec,
  contextTokens: number,
  concurrency: number,
  precision: KvPrecision
): number {
  return kvBytesPerSequence(model, contextTokens, precision) * concurrency;
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
  precision: KvPrecision
): number {
  return kvBytesPerSequence(model, contextTokens, precision);
}

/** Whether any layer uses a bounded attention window — drives the explain layer. */
export function hasSlidingLayers(model: ModelSpec): boolean {
  return model.attention.layerWindows?.some((w) => w !== null) ?? false;
}

/**
 * Summed attention span across layers: how many key positions each query token attends over,
 * totalled over the network.
 *
 * Prefill attention cost is quadratic only on full-attention layers. A sliding layer attends
 * over at most its window however long the prompt is, making its cost linear. Ignoring that
 * overstates gpt-oss prefill FLOPs by roughly 19% at a 16K prompt — the same mistake on the
 * compute side that the naive KV formula makes on the memory side.
 */
export function attentionSpanPerToken(model: ModelSpec, promptTokens: number): number {
  const windows = model.attention.layerWindows;
  if (!windows) return model.layers * promptTokens;

  let span = 0;
  for (let layer = 0; layer < model.layers; layer++) {
    const window = windows[layer];
    span += window === null || window === undefined ? promptTokens : Math.min(promptTokens, window);
  }
  return span;
}

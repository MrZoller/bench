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

/** Whether any layer uses a bounded attention window — drives the explain layer. */
export function hasSlidingLayers(model: ModelSpec): boolean {
  return model.attention.layerWindows?.some((w) => w !== null) ?? false;
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
 */
export function attentionPairs(model: ModelSpec, promptTokens: number): number {
  const n = Math.max(0, promptTokens);
  /** Every position attending over itself and everything before it. */
  const causal = (span: number) => (span * (span + 1)) / 2;

  const windows = model.attention.layerWindows;
  if (!windows) return model.layers * causal(n);

  let pairs = 0;
  for (let layer = 0; layer < model.layers; layer++) {
    const window = windows[layer];
    if (window === null || window === undefined || n <= window) {
      pairs += causal(n);
    } else {
      // A triangle while the window is still filling, then a band of constant width.
      pairs += causal(window) + (n - window) * window;
    }
  }
  return pairs;
}

/**
 * KV held by the busiest device once whole layers are handed out.
 *
 * A layer count is not a KV divisor on a hybrid model. Gemma 3 27B has ten full-attention layers
 * among 62, and at 128K context each of those caches about 128 times what a sliding layer does —
 * so a card that lands two of them holds far more than 2/62 of the total. Dividing by the layer
 * count reported roughly 13.4 GiB per card on an eight-way split where the busiest really needs
 * about 18.6, which is the difference between fitting a 15 GiB card and not.
 *
 * Assignment is longest-processing-time first: sort the layers by size and give each to the
 * lightest device so far. That is the best balance a scheduler could reasonably achieve, so the
 * figure it returns is a *lower* bound on what the busiest card holds — llama.cpp's default
 * contiguous split can be worse. Erring toward the optimistic side of a bound is acceptable;
 * erring toward it by 128x, as a flat divisor does, is not.
 */
export function kvBytesBusiestDevice(
  model: ModelSpec,
  contextTokens: number,
  concurrency: number,
  precision: KvPrecision,
  shards: number,
  runtime?: RuntimeSpec
): number {
  const elemBytes = kvElementBytes(precision, runtime);
  const sizes = Array.from({ length: model.layers }, (_, i) =>
    layerBytes(model.attention, i, contextTokens, elemBytes)
  ).sort((a, b) => b - a);

  const devices = new Array(Math.max(1, Math.floor(shards))).fill(0) as number[];
  for (const size of sizes) {
    let lightest = 0;
    for (let d = 1; d < devices.length; d++) {
      if (devices[d] < devices[lightest]) lightest = d;
    }
    devices[lightest] += size;
  }
  return Math.max(...devices) * Math.max(1, concurrency);
}

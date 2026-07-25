import { describe, expect, it } from 'vitest';
import {
  attentionPairs,
  hasSlidingLayers,
  kvBytesPerSequence,
  kvBytesPerToken,
  kvBytesTotal,
  kvElementBytes,
} from './kv';
import {
  DEEPSEEK_V3,
  GPT_OSS_120B,
  GPT_OSS_20B,
  LLAMA_31_8B,
  LLAMA_CPP,
  QWEN3_32B,
  VLLM,
} from './fixtures';
import { GIB } from './types';

const KIB = 1024;

describe('GQA — the case everyone gets right', () => {
  it('matches a hand computation for Llama 3.1 8B', () => {
    // 2 (K and V) * 8 kv heads * 128 head dim * 2 bytes = 4096 B per layer per token,
    // over 32 layers = 128 KiB/token.
    expect(kvBytesPerToken(LLAMA_31_8B, 'fp16')).toBe(128 * KIB);
  });

  it('scales linearly with context and concurrency', () => {
    // 128 KiB/token * 8192 tokens lands on exactly 1 GiB, which makes the arithmetic checkable.
    expect(kvBytesPerSequence(LLAMA_31_8B, 8192, 'fp16')).toBe(1 * GIB);
    expect(kvBytesTotal(LLAMA_31_8B, 8192, 4, 'fp16')).toBe(4 * GIB);
  });

  it('halves with Q8 KV and quarters with Q4', () => {
    const fp16 = kvBytesPerToken(LLAMA_31_8B, 'fp16');
    expect(kvBytesPerToken(LLAMA_31_8B, 'q8')).toBe(fp16 / 2);
    expect(kvBytesPerToken(LLAMA_31_8B, 'q4')).toBe(fp16 / 4);
  });

  it('lands Qwen3-32B in the 192-328 KB/token band reported for GQA models', () => {
    // 64 layers * 2 * 8 * 128 * 2 B = 256 KiB/token.
    expect(kvBytesPerToken(QWEN3_32B, 'fp16')).toBe(256 * KIB);
  });
});

describe('MLA — where the naive formula overestimates by multiples', () => {
  /**
   * DeepSeek-V3 is the published anchor: ~70 KB/token, against 192-328 KB/token for
   * comparable GQA models. 61 layers * (512 + 64) * 2 bytes = 70,272 B.
   */
  it('reproduces the published ~70 KB/token for DeepSeek V3', () => {
    const perToken = kvBytesPerToken(DEEPSEEK_V3, 'fp16');
    expect(perToken).toBe(70_272);
    expect(perToken / 1000).toBeCloseTo(70.3, 1);
  });

  it('caches several times less than a GQA model with more layers would', () => {
    // The point of the fixture: DeepSeek has almost as many layers as Qwen3-32B but a
    // fraction of the cache. Anything that regressed MLA to the GQA path fails here.
    expect(kvBytesPerToken(DEEPSEEK_V3, 'fp16')).toBeLessThan(
      kvBytesPerToken(QWEN3_32B, 'fp16') / 3
    );
  });

  it('has no factor of two — a single latent, not a key and a value', () => {
    // Written out independently of the implementation so a stray *2 can't pass.
    const expected = 61 * (512 + 64) * 2;
    expect(kvBytesPerToken(DEEPSEEK_V3, 'fp16')).toBe(expected);
  });
});

describe('sliding-window layers — where the cache stops growing', () => {
  it('flags gpt-oss as hybrid and Llama as not', () => {
    expect(hasSlidingLayers(GPT_OSS_120B)).toBe(true);
    expect(hasSlidingLayers(LLAMA_31_8B)).toBe(false);
  });

  it('holds ~4.5 GiB at 128K, roughly half what a uniform formula predicts', () => {
    const actual = kvBytesPerSequence(GPT_OSS_120B, 131072, 'fp16');

    // 18 full layers over the whole context, 18 sliding layers pinned at 128 tokens.
    const perLayerPerToken = 2 * 8 * 64 * 2;
    const expected = 18 * perLayerPerToken * 131072 + 18 * perLayerPerToken * 128;
    expect(actual).toBe(expected);
    expect(actual / GIB).toBeCloseTo(4.5, 1);

    // What a calculator that ignored `layer_types` would report.
    const naive = 36 * perLayerPerToken * 131072;
    expect(naive / actual).toBeCloseTo(2.0, 1);
  });

  it('stops charging for sliding layers once context passes the window', () => {
    const atWindow = kvBytesPerSequence(GPT_OSS_120B, 128, 'fp16');
    const past = kvBytesPerSequence(GPT_OSS_120B, 256, 'fp16');

    // Doubling context past the window doubles only the full-attention half, so the total
    // grows by less than 2x. A uniform model would return exactly 2x.
    expect(past / atWindow).toBeLessThan(2);
    expect(past / atWindow).toBeCloseTo(1.5, 2);
  });

  it('is identical to the uniform result below the window, where nothing is clipped', () => {
    const ctx = 64; // under the 128-token window, so every layer caches everything
    const perLayerPerToken = 2 * 8 * 64 * 2;
    expect(kvBytesPerSequence(GPT_OSS_120B, ctx, 'fp16')).toBe(36 * perLayerPerToken * ctx);
  });
});

/**
 * Two things the naive `layers * N^2` gets wrong, and they compound.
 */
describe('prefill attention pairs', () => {
  it('counts a causal triangle, not a square', () => {
    const n = 1024;
    // Query i attends over positions 0..i, so the count is N(N+1)/2 per layer.
    expect(attentionPairs(LLAMA_31_8B, n)).toBe(LLAMA_31_8B.layers * ((n * (n + 1)) / 2));
    // Which is very nearly half the square the old model charged.
    expect((LLAMA_31_8B.layers * n * n) / attentionPairs(LLAMA_31_8B, n)).toBeCloseTo(2, 2);
  });

  it('bands a sliding layer once its window is full, and keeps it causal before that', () => {
    const w = 128;
    // Below the window, a sliding layer is indistinguishable from a full one.
    expect(attentionPairs(GPT_OSS_20B, 64)).toBe(GPT_OSS_20B.layers * ((64 * 65) / 2));

    // Above it, the sliding half is a triangle plus a band while the full half keeps growing.
    const n = 4096;
    const full = 12 * ((n * (n + 1)) / 2);
    const sliding = 12 * ((w * (w + 1)) / 2 + (n - w) * w);
    expect(attentionPairs(GPT_OSS_20B, n)).toBe(full + sliding);
  });

  it('grows quadratically on full attention and sub-quadratically with a window', () => {
    // Doubling the prompt quadruples a dense model's attention work.
    const dense = attentionPairs(LLAMA_31_8B, 16384) / attentionPairs(LLAMA_31_8B, 8192);
    expect(dense).toBeCloseTo(4, 2);

    // gpt-oss's sliding half is linear, so it must grow strictly more slowly — though only
    // just, at these lengths: a 128-token window against an 8K prompt is a thin band beside
    // twelve full-attention triangles, so the full layers dominate the total.
    const windowed = attentionPairs(GPT_OSS_20B, 16384) / attentionPairs(GPT_OSS_20B, 8192);
    expect(windowed).toBeLessThan(dense);
    expect(windowed).toBeGreaterThan(2);

    // The window's effect is on the *absolute* cost, where it is large: half the layers stop
    // scaling with the prompt at all.
    const uniform = 24 * ((16384 * 16385) / 2);
    expect(uniform / attentionPairs(GPT_OSS_20B, 16384)).toBeGreaterThan(1.9);
  });

  it('is zero for an empty prompt rather than negative', () => {
    expect(attentionPairs(LLAMA_31_8B, 0)).toBe(0);
  });

  it('matches a position-by-position count, not just its own closed form', () => {
    // The tests above assert `causal(W) + (N - W) * W`, which is the expression the
    // implementation evaluates — so an off-by-one in that formula would pass all of them. This
    // one counts what each query position is actually permitted to see.
    const bruteForce = (layers: readonly (number | null)[], n: number) => {
      let pairs = 0;
      for (const window of layers) {
        for (let i = 0; i < n; i++) {
          pairs += window === null ? i + 1 : Math.min(i + 1, window);
        }
      }
      return pairs;
    };

    const windows = GPT_OSS_20B.attention.layerWindows!;
    // Around the window boundary, where an off-by-one would hide.
    for (const n of [0, 1, 2, 127, 128, 129, 300, 1024]) {
      expect(attentionPairs(GPT_OSS_20B, n)).toBe(bruteForce(windows, n));
    }

    const dense = Array.from({ length: LLAMA_31_8B.layers }, () => null);
    for (const n of [0, 1, 2, 513]) {
      expect(attentionPairs(LLAMA_31_8B, n)).toBe(bruteForce(dense, n));
    }
  });
});

/**
 * A cache is charged what the runtime really stores, which is not always the nominal width.
 */
describe('effective KV element width', () => {
  it('charges llama.cpp for its block scales', () => {
    // q8_0 stores 32 elements plus a 2-byte scale.
    expect(kvElementBytes('q8', LLAMA_CPP)).toBeCloseTo(34 / 32, 6);
    expect(kvElementBytes('q4', LLAMA_CPP)).toBeCloseTo(18 / 32, 6);
    // FP16 is a plain float and costs exactly two.
    expect(kvElementBytes('fp16', LLAMA_CPP)).toBe(2);
  });

  it('charges vLLM the nominal width, because FP8 really is one byte', () => {
    expect(kvElementBytes('q8', VLLM)).toBe(1);
  });

  it('falls back to nominal with no runtime at all', () => {
    expect(kvElementBytes('q8')).toBe(1);
  });

  it('makes a llama.cpp cache measurably larger than the nominal figure', () => {
    const nominal = kvBytesTotal(QWEN3_32B, 32768, 8, 'q8');
    const real = kvBytesTotal(QWEN3_32B, 32768, 8, 'q8', LLAMA_CPP);
    expect(real / nominal).toBeCloseTo(34 / 32, 4);
  });
});

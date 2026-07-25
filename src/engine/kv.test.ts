import { describe, expect, it } from 'vitest';
import { hasSlidingLayers, kvBytesPerSequence, kvBytesPerToken, kvBytesTotal } from './kv';
import { DEEPSEEK_V3, GPT_OSS_120B, LLAMA_31_8B, QWEN3_32B } from './fixtures';
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

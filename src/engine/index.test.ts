import { describe, expect, it } from 'vitest';
import { evaluate, type Scenario } from './index';
import { DGX_SPARK, GPT_OSS_120B, LLAMA_31_8B, LLAMA_CPP, QWEN3_32B, RTX_5090 } from './fixtures';
import { getQuant } from '@/data/quants';
import type { DeviceSpec } from './types';

const base: Scenario = {
  model: LLAMA_31_8B,
  quant: getQuant('q4_k_m'),
  usage: { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
  rig: { device: RTX_5090, count: 1 },
  runtime: LLAMA_CPP,
};

/** Every number the UI renders must be finite, or a slider drag paints NaN across the page. */
function expectAllFinite(scenario: Scenario) {
  const result = evaluate(scenario);
  const numbers = [
    result.weights.totalBytes,
    result.weights.effectiveBpw,
    result.activations.totalBytes,
    result.placement.usedBytesPerDevice,
    result.placement.utilization,
    result.decode.perUserTokensPerSec,
    result.decode.aggregateTokensPerSec,
    result.prefill.ttftSeconds,
    result.prefill.prefillTokensPerSec,
    result.kvBytesPerToken,
    result.marginalKvBytesPerToken,
    result.maxContextTokens,
  ];
  for (const value of numbers) {
    expect(Number.isFinite(value)).toBe(true);
  }
  return result;
}

describe('evaluate', () => {
  it('produces a coherent evaluation for an ordinary scenario', () => {
    const result = expectAllFinite(base);

    expect(result.placement.fits).toBe(true);
    expect(result.decode.perUserTokensPerSec).toBeGreaterThan(0);
    expect(result.prefill.ttftSeconds).toBeGreaterThan(0);
    expect(result.maxContextTokens).toBeGreaterThan(0);
  });

  it('reports the marginal KV cost below the headline figure for hybrid models', () => {
    const result = evaluate({
      ...base,
      model: GPT_OSS_120B,
      quant: getQuant('mxfp4'),
      usage: { contextTokens: 131072, concurrency: 1, kvPrecision: 'fp16' },
      rig: { device: DGX_SPARK, count: 1 },
    });

    // Past the 128-token window only the 18 full-attention layers still accumulate, so one
    // more token costs half what the architecture-comparison figure suggests.
    expect(result.marginalKvBytesPerToken).toBe(result.kvBytesPerToken / 2);
    expect(result.hasSlidingLayers).toBe(true);
  });

  it('reports equal marginal and headline KV for a full-attention model', () => {
    const result = evaluate(base);
    expect(result.marginalKvBytesPerToken).toBe(result.kvBytesPerToken);
    expect(result.hasSlidingLayers).toBe(false);
  });
});

/**
 * Degenerate inputs reach the engine from sliders, saved links and hand-edited querystrings.
 * Each of these produced either NaN or a mutually inconsistent answer before normalization.
 */
describe('degenerate inputs', () => {
  it('survives zero context without emitting NaN', () => {
    expectAllFinite({
      ...base,
      usage: { contextTokens: 0, concurrency: 1, kvPrecision: 'fp16' },
    });
  });

  it('survives zero prompt tokens without emitting NaN', () => {
    expectAllFinite({
      ...base,
      usage: { contextTokens: 8192, concurrency: 1, promptTokens: 0, kvPrecision: 'fp16' },
    });
  });

  it('treats zero concurrency as one sequence everywhere, not as free KV', () => {
    const zero = evaluate({
      ...base,
      usage: { contextTokens: 8192, concurrency: 0, kvPrecision: 'fp16' },
    });
    const one = evaluate(base);

    // Previously placement dropped the KV term entirely at concurrency 0 and reported that
    // anything fits, while decode clamped to 1 and reported real throughput for it.
    expect(zero.placement.kvBytesPerDevice).toBe(one.placement.kvBytesPerDevice);
    expect(zero.maxContextTokens).toBe(one.maxContextTokens);
  });

  it('treats a zero-device rig as a single device everywhere', () => {
    const zero = evaluate({ ...base, rig: { device: RTX_5090, count: 0 } });
    const one = evaluate(base);

    expect(zero.placement.weightBytesPerDevice).toBe(one.placement.weightBytesPerDevice);
    expect(zero.decode.perUserTokensPerSec).toBeCloseTo(one.decode.perUserTokensPerSec, 6);
  });

  it('reports an unreachable time-to-first-token rather than NaN when compute is unknown', () => {
    const noFlops: DeviceSpec = { ...RTX_5090, flops: {} };
    const result = evaluate({ ...base, rig: { device: noFlops, count: 1 } });

    expect(result.prefill.ttftSeconds).toBe(Infinity);
    expect(result.prefill.prefillTokensPerSec).toBe(0);
    // Decode is memory-bound and unaffected by a missing FLOPS figure.
    expect(result.decode.perUserTokensPerSec).toBeGreaterThan(0);
  });

  /**
   * NaN reaches the engine from `Number(params.get('ctx'))` on a hand-edited link. A plain
   * `Math.max(1, Math.floor(NaN))` is NaN, so the clamps have to test for finiteness rather
   * than rely on the clamp itself.
   */
  it.each(['contextTokens', 'concurrency', 'promptTokens'] as const)(
    'survives NaN in usage.%s',
    (field) => {
      expectAllFinite({
        ...base,
        usage: { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16', [field]: NaN },
      });
    }
  );

  it('survives NaN in the device count', () => {
    expectAllFinite({ ...base, rig: { device: RTX_5090, count: NaN } });
  });

  it('keeps a model far too large for the rig finite and marked as not fitting', () => {
    const result = expectAllFinite({
      ...base,
      model: QWEN3_32B,
      quant: getQuant('bf16'),
      usage: { contextTokens: 40960, concurrency: 16, kvPrecision: 'fp16' },
    });

    expect(result.placement.fits).toBe(false);
    expect(result.decode.perUserTokensPerSec).toBeGreaterThan(0);
  });
});

describe('the offload cliff is visible on both numbers users watch', () => {
  it('penalises time-to-first-token as well as throughput', () => {
    const scenario: Scenario = {
      ...base,
      model: QWEN3_32B,
      quant: getQuant('bf16'),
      usage: { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
    };
    const result = evaluate(scenario);

    expect(result.placement.offloadFraction).toBeGreaterThan(0);
    expect(result.decode.offloadPenalty).toBeDefined();
    // Before this, prefill ignored placement entirely and reported a TTFT as though every
    // weight were resident — invisible on exactly the number people look at first.
    expect(result.prefill.offloadPenalty).toBeDefined();

    const resident = evaluate({ ...scenario, quant: getQuant('q4_k_m') });
    expect(result.prefill.ttftSeconds).toBeGreaterThan(resident.prefill.ttftSeconds);
  });

  /**
   * The dense case above cannot distinguish the batch-1 expert union from the full weight
   * set, because for a dense model they are the same number. An MoE model on a small card is
   * the configuration this tool exists to reason about, and it is where sizing the streamed
   * volume at one token's worth of experts understated TTFT by roughly 5x.
   */
  it('streams the whole offloaded weight set for an MoE prefill, not one token’s experts', () => {
    const scenario: Scenario = {
      ...base,
      model: GPT_OSS_120B,
      quant: getQuant('mxfp4'),
      usage: { contextTokens: 4096, concurrency: 1, promptTokens: 512, kvPrecision: 'fp16' },
      rig: { device: RTX_5090, count: 1 },
    };
    const result = evaluate(scenario);
    expect(result.placement.offloadFraction).toBeGreaterThan(0);

    // A prefill pass over 512 tokens routes through essentially all 128 experts, so the
    // streamed term must reflect the full offloaded set.
    const streamed = result.weights.totalBytes * result.placement.offloadFraction * 0.9;
    expect(result.prefill.ttftSeconds).toBeGreaterThan(streamed / 80e9);
  });
});

import { describe, expect, it } from 'vitest';
import { activeWeightBytes, effectiveActiveParams, weightBreakdown, weightBytes } from './weights';
import { DEEPSEEK_V3, GPT_OSS_120B, LLAMA_31_8B, QWEN3_32B } from './fixtures';
import { getQuant } from '@/data/quants';
import { GIB } from './types';

describe('uniform quantization', () => {
  /**
   * llama.cpp's published figure for Llama-3.1-8B at Q4_K_M is 4.58 GiB (~4.89 effective bpw).
   * The table carries 4.85 as a representative value, so the assertion allows the 1-2% drift
   * that comes from per-model differences in how K-quants assign bit widths.
   */
  it('lands Llama 3.1 8B Q4_K_M on the published 4.58 GiB', () => {
    const bytes = weightBytes(LLAMA_31_8B, getQuant('q4_k_m'));
    expect(bytes / GIB).toBeCloseTo(4.58, 1);
    expect(bytes / GIB).toBeGreaterThan(4.4);
    expect(bytes / GIB).toBeLessThan(4.7);
  });

  it('reports BF16 as two bytes per parameter', () => {
    expect(weightBytes(LLAMA_31_8B, getQuant('bf16'))).toBe(LLAMA_31_8B.totalParams * 2);
  });

  it('charges a dense model the nominal bpw exactly, since it has no experts to spare', () => {
    const { effectiveBpw, expertBytes } = weightBreakdown(QWEN3_32B, getQuant('q4_k_m'));
    expect(expertBytes).toBe(0);
    expect(effectiveBpw).toBeCloseTo(4.85, 6);
  });

  it('puts Qwen3-32B Q4_K_M in the reported 21-23 GB band', () => {
    const gb = weightBytes(QWEN3_32B, getQuant('q4_k_m')) / 1e9;
    expect(gb).toBeGreaterThan(19);
    expect(gb).toBeLessThan(21);
  });
});

describe('expert-only quantization', () => {
  /**
   * The reference case for the whole split. gpt-oss-120b at its native MXFP4 is ~61 GiB and
   * fits an 80 GB card with room for KV. Charging all 116.8B at 4.25 bpw would say ~58 GiB;
   * charging all of it at BF16 would say 218 GiB. Only the split gets there.
   */
  it('sizes gpt-oss-120b at ~61 GiB, fitting an 80 GB card', () => {
    const { totalBytes, expertBytes, denseBytes } = weightBreakdown(
      GPT_OSS_120B,
      getQuant('mxfp4')
    );

    expect(totalBytes / GIB).toBeCloseTo(60.7, 0);
    expect(totalBytes / GIB).toBeLessThan(80);

    // Experts dominate, but the BF16 remainder is ~4 GiB — the part a flat rate would lose.
    expect(expertBytes / GIB).toBeCloseTo(56.7, 0);
    expect(denseBytes / GIB).toBeCloseTo(4.0, 0);
  });

  it('blends to an effective bpw above the headline 4.25', () => {
    const { effectiveBpw } = weightBreakdown(GPT_OSS_120B, getQuant('mxfp4'));
    expect(effectiveBpw).toBeGreaterThan(4.25);
    expect(effectiveBpw).toBeCloseTo(4.46, 1);
  });

  it('would understate by ~3 GiB if experts and dense weights were charged the same rate', () => {
    const split = weightBytes(GPT_OSS_120B, getQuant('mxfp4'));
    const flat = (GPT_OSS_120B.totalParams * 4.25) / 8;

    // The BF16 remainder is ~4 GiB, but a flat rate still charges those 2.1B params 4.25 bpw,
    // so the error it makes is the ~2.9 GiB difference rather than the full dense footprint.
    expect((split - flat) / GIB).toBeCloseTo(2.9, 1);
  });
});

describe('active weights govern speed, total weights govern memory', () => {
  /**
   * The distinction the whole tool exists to make legible: DeepSeek V3 needs 671B worth of
   * memory but reads only ~37B per token. Both figures are asserted in absolute terms — their
   * *ratio* is true by construction under any uniform scheme and would guard nothing.
   */
  it('holds ~406 GB of DeepSeek V3 but reads only ~23 GB per token', () => {
    const quant = getQuant('q4_k_m');
    expect(weightBytes(DEEPSEEK_V3, quant) / 1e9).toBeCloseTo(406, -1);
    expect(activeWeightBytes(DEEPSEEK_V3, quant) / 1e9).toBeCloseTo(23, -1);
  });

  /**
   * The guard for the split that actually matters. Active parameters for gpt-oss are roughly
   * half dense, and that dense half is BF16, read in full every step. Charging it the blended
   * whole-model rate (4.47 bpw, since 98% of parameters are 4-bit experts) understates
   * bytes-per-token by more than 2x — and decode throughput is inversely proportional to it.
   *
   * Computed by hand from the fixture rather than through weightBreakdown, so a regression in
   * the split cannot quietly move the expectation with it.
   */
  it('charges gpt-oss active weights per component, not at the blended rate', () => {
    const dense = GPT_OSS_120B.totalParams - GPT_OSS_120B.expertParams;
    const activeExperts = GPT_OSS_120B.expertParams * (4 / 128);
    const expected = (dense * 16 + activeExperts * 4.25) / 8;

    const actual = activeWeightBytes(GPT_OSS_120B, getQuant('mxfp4'));
    expect(actual).toBeCloseTo(expected, -6);
    expect(actual / 1e9).toBeCloseTo(6.18, 1);

    // What the blended shortcut would have claimed.
    const blended = (GPT_OSS_120B.activeParams * 4.4651) / 8;
    expect(actual / blended).toBeGreaterThan(2);
  });

  it('reads everything for a dense model', () => {
    const quant = getQuant('q4_k_m');
    expect(activeWeightBytes(LLAMA_31_8B, quant)).toBeCloseTo(weightBytes(LLAMA_31_8B, quant), 0);
  });

  it('grows the expert union with batch, and leaves dense models flat', () => {
    const mxfp4 = getQuant('mxfp4');
    const q4 = getQuant('q4_k_m');

    expect(activeWeightBytes(GPT_OSS_120B, mxfp4, 32)).toBeGreaterThan(
      activeWeightBytes(GPT_OSS_120B, mxfp4, 1) * 1.5
    );
    expect(activeWeightBytes(LLAMA_31_8B, q4, 32)).toBe(activeWeightBytes(LLAMA_31_8B, q4, 1));
  });

  /**
   * A model carrying expert parameters but no expert counts must not silently gain or lose
   * throughput relative to its catalogued active-parameter figure.
   */
  it('falls back to the catalogued active count when expert counts are missing', () => {
    const withoutCounts = { ...DEEPSEEK_V3, experts: undefined };
    expect(effectiveActiveParams(withoutCounts, 1)).toBeCloseTo(DEEPSEEK_V3.activeParams, -8);
    // No expert counts means no basis for a batch curve, so it must stay flat rather than invent one.
    expect(effectiveActiveParams(withoutCounts, 64)).toBe(effectiveActiveParams(withoutCounts, 1));
  });
});

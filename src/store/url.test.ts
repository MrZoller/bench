import { describe, expect, it } from 'vitest';
import { configToSearch, sameScenario, searchToConfig } from './url';
import { DEFAULT_CONFIG, type Config } from './config';

/**
 * A link is the distribution mechanism for a tool like this, so the encoding has to survive a
 * stranger editing it by hand. Every test here is either "a round trip is lossless" or "nonsense
 * degrades to something usable".
 */
describe('scenario URLs', () => {
  it('writes nothing for a default scenario', () => {
    expect(configToSearch(DEFAULT_CONFIG)).toBe('');
  });

  it('writes only what differs from the default', () => {
    const search = configToSearch({ ...DEFAULT_CONFIG, deviceId: 'rtx-5090' });
    expect(search).toBe('?d=rtx-5090');
  });

  it('round-trips a fully specified scenario', () => {
    const scenario: Config = {
      modelId: 'deepseek-ai/DeepSeek-V3',
      quantId: 'q4_k_m',
      runtimeId: 'vllm',
      deviceId: 'h100-sxm',
      deviceCount: 4,
      contextTokens: 65536,
      concurrency: 8,
      promptTokens: 16384,
      kvPrecision: 'q8',
    };
    expect(searchToConfig(configToSearch(scenario))).toEqual(scenario);
  });

  it('fills absent keys from the default rather than leaving holes', () => {
    const config = searchToConfig('?d=rtx-5090');
    expect(config.deviceId).toBe('rtx-5090');
    expect(config.modelId).toBe(DEFAULT_CONFIG.modelId);
    expect(config.contextTokens).toBe(DEFAULT_CONFIG.contextTokens);
  });

  /**
   * The values themselves are validated by the store's `coerce`, which runs over whatever this
   * returns. What matters here is that an unparseable *number* reads as "unset" rather than
   * being handed on as NaN — `coerce` would clamp that, and a hand-typed `ctx=lots` should not
   * silently become the smallest legal context.
   */
  it('treats an unparseable number as unset', () => {
    expect(searchToConfig('?ctx=lots').contextTokens).toBe(DEFAULT_CONFIG.contextTokens);
    expect(searchToConfig('?n=').deviceCount).toBe(DEFAULT_CONFIG.deviceCount);
  });

  it('never throws on hostile input', () => {
    for (const search of ['', '?', '?????', '?m=', '?m=%%%', '?unknown=1', '?ctx=-1e999']) {
      expect(() => searchToConfig(search)).not.toThrow();
    }
  });

  it('compares scenarios by value, for deciding whether the URL needs rewriting', () => {
    expect(sameScenario(DEFAULT_CONFIG, { ...DEFAULT_CONFIG })).toBe(true);
    expect(sameScenario(DEFAULT_CONFIG, { ...DEFAULT_CONFIG, concurrency: 4 })).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';
import { configToSearch, configToShareSearch, sameScenario, searchToConfig } from './url';
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

  it('still writes the default scenario in full when someone asks for a link to it', () => {
    // The bare address bar is honest — it claims nothing. A copied link claims something, so the
    // default scenario is the one case where the two encoders differ, and the share side has to
    // spell it out or the highest-volume share of all drifts with the next defaults change.
    const shared = configToShareSearch(DEFAULT_CONFIG);
    const params = new URLSearchParams(shared.slice(1));

    expect([...params.keys()].sort()).toEqual(['ctx', 'd', 'kv', 'm', 'n', 'p', 'q', 'r', 'u']);
    expect(searchToConfig(shared)).toEqual(DEFAULT_CONFIG);
  });

  it('writes every field once anything differs, not just the field that differs', () => {
    // The sparse version wrote `?d=rtx-5090` and let the other eight fall back to whatever
    // DEFAULT_CONFIG says on the day the link is *opened*. A link pasted into a forum thread
    // would then change meaning the next time a default moved. Completeness is what makes the
    // querystring independent of the deployment that reads it.
    const search = configToSearch({ ...DEFAULT_CONFIG, deviceId: 'rtx-5090' });
    const params = new URLSearchParams(search.slice(1));

    expect([...params.keys()].sort()).toEqual(['ctx', 'd', 'kv', 'm', 'n', 'p', 'q', 'r', 'u']);
    expect(params.get('d')).toBe('rtx-5090');
    expect(params.get('m')).toBe(DEFAULT_CONFIG.modelId);
  });

  it('reproduces a shared scenario even when every default has since changed', () => {
    const shared: Config = {
      modelId: 'Qwen/Qwen3-32B',
      quantId: 'q5_k_m',
      runtimeId: 'vllm',
      deviceId: 'rtx-5090',
      deviceCount: 2,
      contextTokens: 32768,
      concurrency: 4,
      promptTokens: 4096,
      kvPrecision: 'q8',
    };
    const link = configToSearch(shared);

    // Stand in for a future deployment: read the link back against defaults that share not one
    // value with it. Nothing may leak through from the reader's side.
    const laterDefaults: Config = {
      modelId: 'meta-llama/Llama-3.1-8B-Instruct',
      quantId: 'q4_k_m',
      runtimeId: 'llama.cpp',
      deviceId: 'm3-ultra-512',
      deviceCount: 1,
      contextTokens: 8192,
      concurrency: 1,
      promptTokens: 1024,
      kvPrecision: 'fp16',
    };
    const params = new URLSearchParams(link.slice(1));
    const reread: Config = { ...laterDefaults };
    const fields = {
      m: 'modelId',
      q: 'quantId',
      r: 'runtimeId',
      d: 'deviceId',
      n: 'deviceCount',
      ctx: 'contextTokens',
      u: 'concurrency',
      p: 'promptTokens',
      kv: 'kvPrecision',
    } as const;

    for (const [short, full] of Object.entries(fields)) {
      const raw = params.get(short);
      if (raw === null) continue;
      if (typeof laterDefaults[full] === 'number') {
        (reread[full] as number) = Number(raw);
      } else {
        (reread[full] as string) = raw;
      }
    }

    expect(reread).toEqual(shared);
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

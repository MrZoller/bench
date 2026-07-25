import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, evaluateConfig, useConfig, type Config } from './config';
import { getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';

/**
 * The store is the boundary between untrusted input and the engine.
 *
 * Its values arrive from sliders today and from a hand-editable querystring shortly, so every
 * assertion here is about what happens when a field is nonsense — the engine below is pure
 * arithmetic and will faithfully compute a wrong answer from a wrong input.
 */

function set<K extends keyof Config>(key: K, value: Config[K]) {
  useConfig.setState(DEFAULT_CONFIG);
  useConfig.getState().set(key, value);
  return useConfig.getState();
}

describe('config coercion', () => {
  it('falls back on ids that are not in the catalog', () => {
    // A stale shared link should degrade to something usable, not a blank page.
    expect(set('modelId', 'meta-llama/Llama-4-Behemoth').modelId).toBe(DEFAULT_CONFIG.modelId);
    expect(set('deviceId', 'rtx-9090').deviceId).toBe(DEFAULT_CONFIG.deviceId);
    expect(set('quantId', 'q2_k_xxs').quantId).toBe(DEFAULT_CONFIG.quantId);
    expect(set('runtimeId', 'tensorrt').runtimeId).toBe(DEFAULT_CONFIG.runtimeId);
    expect(set('kvPrecision', 'fp8' as never).kvPrecision).toBe(DEFAULT_CONFIG.kvPrecision);
  });

  it('takes the default for values that are not numbers, rather than the minimum', () => {
    // `Number(params.get('ctx'))` on nonsense is NaN. Pinning that to the smallest legal
    // context would be its own wrong answer, quietly presented as the user's choice.
    expect(set('contextTokens', Number.NaN).contextTokens).toBe(DEFAULT_CONFIG.contextTokens);
    expect(set('concurrency', Number.NaN).concurrency).toBe(DEFAULT_CONFIG.concurrency);
    expect(set('deviceCount', Number.POSITIVE_INFINITY).deviceCount).toBe(
      DEFAULT_CONFIG.deviceCount
    );
  });

  it('clamps out-of-range numbers into the range', () => {
    expect(set('concurrency', -5).concurrency).toBe(1);
  });

  /**
   * A model cannot be asked for more context than it was trained for. The slider offers stops
   * up to 1M while every catalogued model tops out between 32K and 164K, so an uncapped value
   * would produce fit, memory and speed estimates for a window the model cannot process.
   */
  it('caps context at the selected model, not at the slider maximum', () => {
    const state = set('contextTokens', 99_000_000);
    expect(state.contextTokens).toBe(getModel(state.modelId).maxContext);
    expect(state.contextTokens).toBeLessThan(1_048_576);
  });

  it('re-caps context when the model changes beneath it', () => {
    useConfig.setState(DEFAULT_CONFIG);
    const store = useConfig.getState();

    store.set('contextTokens', 131072);
    store.set('modelId', 'Qwen/Qwen3-32B'); // 40960 max
    expect(useConfig.getState().contextTokens).toBe(getModel('Qwen/Qwen3-32B').maxContext);
  });

  /**
   * The prompt is part of the context, not a separate budget. Left independent, a 2K context
   * with a 128K prompt sized the KV cache for 2048 tokens while reporting a seven-minute TTFT —
   * a comfortable-looking budget bar for a request needing 64x the cache it drew.
   */
  it('never lets the prompt exceed the context it is part of', () => {
    const state = set('promptTokens', 131072);
    expect(state.promptTokens).toBeLessThanOrEqual(state.contextTokens);

    const shrunk = useConfig.getState();
    shrunk.set('contextTokens', 2048);
    expect(useConfig.getState().promptTokens).toBeLessThanOrEqual(2048);
  });

  /**
   * Only discrete GPUs shard a model across devices. A count above 1 elsewhere describes a rig
   * that cannot exist, and the engine would divide the weights across it without complaint.
   */
  it('refuses a multi-device rig on hardware with no transport between devices', () => {
    useConfig.setState(DEFAULT_CONFIG);
    const store = useConfig.getState();

    store.set('deviceId', 'rtx-5090');
    store.set('deviceCount', 4);
    expect(useConfig.getState().deviceCount).toBe(4);

    // Switching to a unified-memory machine has to collapse the rig, not carry the count over.
    store.set('deviceId', 'mac-studio-m3-ultra-256');
    expect(useConfig.getState().deviceCount).toBe(1);

    store.set('deviceId', 'epyc-9654');
    store.set('deviceCount', 8);
    expect(useConfig.getState().deviceCount).toBe(1);
  });
});

describe('evaluating a config', () => {
  it('produces finite numbers for the default scenario', () => {
    const { decode, prefill, placement } = evaluateConfig(DEFAULT_CONFIG);
    expect(Number.isFinite(decode.perUserTokensPerSec)).toBe(true);
    expect(Number.isFinite(prefill.ttftSeconds)).toBe(true);
    expect(placement.usedBytesPerDevice).toBeGreaterThan(0);
  });

  it('survives every coerced edge case without producing NaN', () => {
    for (const mutate of [
      () => set('contextTokens', Number.NaN),
      () => set('concurrency', -1),
      () => set('promptTokens', 99_999_999),
      () => set('deviceId', 'nope'),
    ]) {
      const { decode, placement } = evaluateConfig(mutate());
      expect(Number.isNaN(decode.perUserTokensPerSec)).toBe(false);
      expect(Number.isNaN(placement.utilization)).toBe(false);
    }
  });
});

/**
 * An expert-only quantization on a dense model is a no-op its own label denies, so the picker
 * hides it there — which means the selection has to move with it, or the control and the store
 * disagree about what is selected.
 */
describe('quantization follows the model', () => {
  it('drops an expert-only scheme when the model has no experts', () => {
    useConfig.setState(DEFAULT_CONFIG);
    const store = useConfig.getState();

    store.set('quantId', 'mxfp4');
    expect(useConfig.getState().quantId).toBe('mxfp4'); // gpt-oss is MoE, so it stands.

    store.set('modelId', 'Qwen/Qwen3-32B');
    expect(useConfig.getState().quantId).not.toBe('mxfp4');
  });

  it('leaves a uniform scheme alone on any model', () => {
    useConfig.setState(DEFAULT_CONFIG);
    const store = useConfig.getState();

    store.set('quantId', 'q4_k_m');
    store.set('modelId', 'Qwen/Qwen3-32B');
    expect(useConfig.getState().quantId).toBe('q4_k_m');
  });
});

/**
 * The store and the picker must agree about what is selectable. Both of these were cases where
 * the control offered something the store then quietly took away.
 */
describe('the store agrees with the controls', () => {
  it('keeps a multi-device count on hardware that has a link between units', () => {
    useConfig.setState(DEFAULT_CONFIG);
    const store = useConfig.getState();

    // The Spark is unified-soc *and* has ConnectX, so the picker offers a count — and the
    // store used to reset it to 1 on every change, making the linked case unevaluatable.
    store.set('deviceId', 'dgx-spark');
    store.set('deviceCount', 4);
    expect(useConfig.getState().deviceCount).toBe(4);

    store.set('deviceId', 'mac-studio-m3-ultra-256');
    expect(useConfig.getState().deviceCount).toBe(1);
  });

  /**
   * The default quant is `mxfp4`, which the picker hides for dense models — so resolving an
   * unknown id to the default *before* checking applicability landed a dense model on exactly
   * the option the rule exists to prevent.
   */
  it('never resolves an unknown quantization onto an expert-only default', () => {
    useConfig.setState({ ...DEFAULT_CONFIG, modelId: 'Qwen/Qwen3-32B' });
    useConfig.getState().set('quantId', 'not-a-real-quant');

    const { quantId, modelId } = useConfig.getState();
    expect(getModel(modelId).expertParams).toBe(0);
    expect(getQuant(quantId).denseBpw).toBeUndefined();
  });
});

describe('quantization must be able to run where it is selected', () => {
  it('drops a vendor-locked format when the hardware is another vendor', () => {
    useConfig.setState(DEFAULT_CONFIG);
    const store = useConfig.getState();

    store.set('deviceId', 'rtx-5090');
    store.set('quantId', 'nvfp4');
    expect(useConfig.getState().quantId).toBe('nvfp4');

    // NVFP4 is Blackwell-native; the MI355X's FP4 rate is for AMD's own format, and letting
    // this through would hand `peakFlops` 9.2 PFLOP/s from different silicon.
    store.set('deviceId', 'mi355x');
    expect(useConfig.getState().quantId).not.toBe('nvfp4');
  });
});

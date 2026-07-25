import { describe, expect, it } from 'vitest';
import { RUNTIMES, getRuntime, runtimesFor } from './runtimes';
import { LLAMA_CPP, MLX, VLLM } from '@/engine/fixtures';

/**
 * The app's runtime catalog and the engine's calibration fixtures must not drift.
 *
 * `speed.test.ts` anchors `LLAMA_CPP` against the DGX Spark and EPYC measurements, so those
 * fixture constants are the spec. They are duplicated here because the engine stays pure and
 * does not import from `src/data/` — which means re-calibrating a fixture to keep a reference
 * test honest would leave the shipped app on the old constants, with a fully green suite.
 *
 * This is the test that fails instead.
 */
describe('runtime catalog matches the engine fixtures', () => {
  it.each([
    ['llama.cpp', LLAMA_CPP],
    ['vllm', VLLM],
    ['mlx', MLX],
  ])('%s is identical to its calibration fixture', (id, fixture) => {
    expect(getRuntime(id)).toEqual(fixture);
  });

  it('covers exactly the runtimes the engine is calibrated for', () => {
    expect(RUNTIMES.map((r) => r.id).sort()).toEqual([LLAMA_CPP.id, MLX.id, VLLM.id].sort());
  });
});

describe('runtime support', () => {
  it('knows which runtimes can drive which hardware', () => {
    // The three cases that matter: vLLM is GPU-only, MLX is Apple-only, llama.cpp runs anywhere.
    expect(runtimesFor('discrete-gpu').map((r) => r.id)).toContain('vllm');
    expect(runtimesFor('unified-soc').map((r) => r.id)).not.toContain('vllm');
    expect(runtimesFor('unified-soc').map((r) => r.id)).toContain('mlx');
    expect(runtimesFor('cpu-ram').map((r) => r.id)).toEqual(['llama.cpp']);
  });

  it('throws on an unknown id rather than returning a default', () => {
    expect(() => getRuntime('tensorrt')).toThrow(/Unknown runtime/);
  });
});

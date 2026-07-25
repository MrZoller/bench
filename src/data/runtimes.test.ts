import { describe, expect, it } from 'vitest';
import { RUNTIMES, getRuntime, runtimesFor } from './runtimes';
import { LLAMA_CPP, MLX, VLLM } from '@/engine/fixtures';
import { getDevice } from './catalog';

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
  const ids = (deviceId: string) => runtimesFor(getDevice(deviceId)).map((r) => r.id);

  /**
   * Support is a class-and-vendor question, and every one of these fails under either axis
   * alone. `unified-soc` covers three incompatible software stacks; a runtime-wide vendor would
   * exclude vLLM from AMD's discrete accelerators, which it does drive.
   */
  it('knows which runtimes can drive which hardware', () => {
    expect(ids('rtx-5090')).toContain('vllm');
    // AMD discrete: vLLM yes, so a blanket "NVIDIA only" would be wrong.
    expect(ids('mi355x')).toContain('vllm');
    // NVIDIA unified memory: a CUDA target with an official vLLM playbook.
    expect(ids('dgx-spark')).toContain('vllm');
    // Apple and AMD unified memory: not vLLM.
    expect(ids('mac-studio-m3-ultra-256')).not.toContain('vllm');
    expect(ids('ryzen-ai-max-395')).not.toContain('vllm');

    expect(ids('mac-studio-m3-ultra-256')).toContain('mlx');
    expect(ids('dgx-spark')).not.toContain('mlx');
    expect(ids('epyc-9654')).toEqual(['llama.cpp']);
  });

  /**
   * A runtime cannot store a cache dtype it has no flag for. vLLM's `--kv-cache-dtype` takes
   * native or FP8 variants; charging 0.5 bytes per element for a 4-bit cache it cannot allocate
   * turns a long-context OOM into a reported fit.
   */
  it('states which KV cache dtypes each runtime can store', () => {
    expect(getRuntime('vllm').kvPrecisions).not.toContain('q4');
    expect(getRuntime('llama.cpp').kvPrecisions).toContain('q4');
    for (const runtime of RUNTIMES) expect(runtime.kvPrecisions).toContain('fp16');
  });

  it('throws on an unknown id rather than returning a default', () => {
    expect(() => getRuntime('tensorrt')).toThrow(/Unknown runtime/);
  });
});

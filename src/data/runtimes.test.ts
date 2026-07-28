import { describe, expect, it } from 'vitest';
import { RUNTIMES, getRuntime, kvSubstitutionFor, runtimesFor } from './runtimes';
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

/**
 * A cache charged a width nobody measured says so — the KV half of #18's rule, filed as #33.
 *
 * `kvElementBytes` falls back to the nominal figure when a runtime declares no
 * `kvBytesPerElement`, which is exact for a float format and not for an affine one. llama.cpp
 * declares 34/32 for its `q8_0` cache because the block layout is published; MLX's affine scheme
 * carries a scale and a bias per group and nobody here has measured what that costs, so its 8-bit
 * cache is charged exactly one byte. The marker is what makes that interim honest.
 */
describe('a cache charged an unmeasured width is marked', () => {
  it('marks MLX at 8-bit, which quantizes its cache affinely too', () => {
    expect(kvSubstitutionFor(getRuntime('mlx'), 'q8')).toMatch(/affine scheme/i);
  });

  /**
   * FP16 has no groups, no scales and no biases, so two bytes is exact. A marker here would be
   * crying wolf on the majority case, which is the failure the weight-axis note already records:
   * a warning on the common path trains people to ignore it where it matters.
   */
  it('stays silent on FP16, whose width is exact', () => {
    expect(kvSubstitutionFor(getRuntime('mlx'), 'fp16')).toBeUndefined();
  });

  /**
   * The two runtimes that are *not* approximating, in both directions.
   *
   * llama.cpp's `q8_0` cache is the interesting one: it also costs more than its nominal byte, and
   * it is not marked — because the catalog states the real figure instead. A marker there would
   * describe an approximation that is not being made.
   */
  it.each([
    ['llama.cpp', 'q8'],
    ['llama.cpp', 'q4'],
    ['vllm', 'q8'],
    ['vllm', 'fp16'],
  ] as const)('stays silent for %s at %s, which is measured or exact', (id, precision) => {
    expect(kvSubstitutionFor(getRuntime(id), precision)).toBeUndefined();
  });

  /**
   * The polarity, asserted rather than assumed.
   *
   * `nativeKvPrecisions` names what is *real*, so a precision added to a runtime later is marked
   * until someone declares otherwise. Naming the stand-ins instead is what let a format be added
   * to `weightFormats` and stay silently unmarked, which is the whole reason that field is
   * written the way it is.
   */
  it('marks a precision nobody has classified yet', () => {
    const mlx = getRuntime('mlx');
    const invented = { ...mlx, kvPrecisions: [...mlx.kvPrecisions, 'q4'] as const };
    expect(kvSubstitutionFor(invented, 'q4')).toBeDefined();
  });

  /**
   * A precision the runtime cannot store at all is a different refusal, made by the picker. Marking
   * it here would explain a figure that is never produced — the same boundary `substitutionFor`
   * draws for a format outside `weightFormats`.
   */
  it('says nothing about a precision the runtime cannot store', () => {
    expect(kvSubstitutionFor(getRuntime('mlx'), 'q4')).toBeUndefined();
  });

  /**
   * Both axes are declared, for every runtime that declares either. Required in the type, asserted
   * here too: the type stops a field being omitted, and this stops it being satisfied with an empty
   * list that marks everything, or a complete one that marks nothing.
   */
  it('states both axes wherever a substitution is declared', () => {
    for (const runtime of RUNTIMES.filter((r) => r.substituted)) {
      const { nativeFormats, nativeKvPrecisions, note, kvNote } = runtime.substituted!;
      expect(nativeFormats.length, `${runtime.id} marks every weight format`).toBeGreaterThan(0);
      expect(nativeKvPrecisions.length, `${runtime.id} marks every precision`).toBeGreaterThan(0);
      expect(nativeKvPrecisions.length, `${runtime.id} marks no precision`).toBeLessThan(
        runtime.kvPrecisions.length
      );
      expect(note).not.toEqual(kvNote);
    }
  });
});

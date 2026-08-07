import { describe, expect, it } from 'vitest';
import { RUNTIMES, getRuntime, kvSubstitutionFor, runtimeDrives, runtimesFor } from './runtimes';
import { LLAMA_CPP, MLX, VLLM } from '@/engine/fixtures';
import { kvElementBytes } from '@/engine/kv';
import { DEVICES, getDevice } from './catalog';

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
 * The list's own order, which is the order the Runtime picker shows (#79).
 *
 * `Bench.tsx`'s `runtimeOptions` maps `RUNTIMES` and nothing sorts it, so the file is the sequence a
 * reader gets. That was true of `devices.json` as well, where it went three rows wrong precisely because
 * nothing stated or checked it; the review of #79 found the same shape here and in `quants.ts`.
 *
 * The rule is breadth, and it is checkable without a rank field: coverage comes from `runtimeDrives`
 * over the real catalog, which answers from each runtime's `supports` rules rather than from this
 * sequence. So a runtime added with a narrow `supports` and dropped at the top of the file fails here.
 */
describe('the runtime catalog is listed in the order it states', () => {
  it('runs widest catalog coverage first', () => {
    const covered = RUNTIMES.map((runtime) => ({
      id: runtime.id,
      devices: DEVICES.filter((device) => runtimeDrives(runtime, device)).length,
    }));

    expect(
      covered.map((r) => r.devices),
      covered.map((r) => `${r.id}: ${r.devices}`).join(', ')
    ).toEqual([...covered.map((r) => r.devices)].sort((a, b) => b - a));

    // Strictly, not merely non-increasing: two runtimes covering the same number of machines would
    // leave the order between them unstated, and an unstated order is what this describe is about.
    // Today it is 43 / 26 / 11 of the 43 catalogued machines — every class, then discrete plus the
    // one NVIDIA SoC, then Apple only.
    expect(new Set(covered.map((r) => r.devices)).size).toBe(covered.length);
  });
});

/**
 * The proxy the host-resident gate used to be, and the equivalence that makes replacing it
 * figure-neutral (#209).
 *
 * `planPlacement` read `rig.device.class === 'discrete-gpu' && runtime.parallelism === 'layer'` and
 * now reads `rig.device.class === 'discrete-gpu' && runtime.hostResidentInputEmbedding`. The two
 * cannot be shown equal from the field values alone — MLX is layer-parallel and declares `false` —
 * so the claim is narrower and is exactly the one that matters: **over every pairing the catalog can
 * actually reach, the two conditions select the same runtime.** MLX is the row where they disagree,
 * and `supports` is what keeps it from a discrete GPU at all; a pairing it cannot drive is
 * `unsupported`, and no surface renders a figure from one.
 *
 * Which makes this both the proof that no answer moved and the tripwire for the next row. It is
 * deliberately a tripwire rather than an invariant: a layer-parallel runtime added for discrete GPUs
 * fails here whichever residency it declares, because the day the two conditions first differ on a
 * live pairing is the day someone has to look at what that row does with `token_embd.weight` and
 * delete this test with the reason written down — rather than inherit llama.cpp's answer from a
 * field about sharding, which is how it went unnoticed the first time.
 */
describe('the host-resident input embedding is stated, not inferred from parallelism', () => {
  it('selects the same runtime the parallelism proxy did, on every pairing the catalog reaches', () => {
    const reachable = DEVICES.flatMap((device) =>
      RUNTIMES.filter((runtime) => runtimeDrives(runtime, device)).map((runtime) => ({
        device,
        runtime,
      }))
    );
    // Discrete GPUs are the only class the gate reads at all, and a sweep that reached one runtime
    // would pass every assertion below while comparing the two conditions on one side each.
    // Asserted as a set rather than a count, because a count fails on the next GPU row for no
    // reason — what has to hold is that both answers are represented.
    const discrete = reachable.filter(({ device }) => device.class === 'discrete-gpu');
    expect(new Set(discrete.map(({ runtime }) => runtime.hostResidentInputEmbedding))).toEqual(
      new Set([true, false])
    );

    for (const { device, runtime } of discrete) {
      expect(
        runtime.hostResidentInputEmbedding,
        `${runtime.id} on ${device.id} changes answer between the two conditions`
      ).toBe(runtime.parallelism === 'layer');
    }
  });

  it('is not the same question as parallelism, and MLX is the row that proves it', () => {
    // The disagreement is real rather than hypothetical, which is what stops this pair of tests
    // being read as "the field restates `parallelism`" and simplified back into one.
    expect(getRuntime('mlx').parallelism).toBe('layer');
    expect(getRuntime('mlx').hostResidentInputEmbedding).toBe(false);
    expect(
      DEVICES.filter((d) => d.class === 'discrete-gpu' && runtimeDrives(getRuntime('mlx'), d)),
      'MLX reaches a discrete GPU, so the two conditions now differ on a live pairing'
    ).toEqual([]);

    // And the one row that does claim it is claiming llama.cpp's placement, not layer parallelism.
    expect(getRuntime('llama.cpp').hostResidentInputEmbedding).toBe(true);
    expect(getRuntime('vllm').hostResidentInputEmbedding).toBe(false);
  });
});

/**
 * Every cache width is either established or marked — the KV half of #18's rule, filed as #33 and
 * settled by #38.
 *
 * `kvElementBytes` falls back to the nominal figure when a runtime declares no
 * `kvBytesPerElement`, which is exact for a float format and not for an affine one. Each of the
 * three runtimes now has an answer: vLLM's FP8 really is one byte, llama.cpp's `q8_0` declares
 * 34/32 from its published block layout, and MLX's declares 17/16 derived from `mlx-lm`'s source.
 * The marker survives for whatever is added next without one.
 */
describe('a cache charged a width nobody established says so', () => {
  /**
   * MLX's 8-bit cache, which this suite used to assert *was* marked.
   *
   * `QuantizedKVCache` defaults to `group_size=64, bits=8` and stores an fp16 scale **and** an
   * fp16 bias per group — 8 + 16/64 + 16/64 = 8.5 bits. Derived from published source, exactly as
   * llama.cpp's was, so the marker would now be warning about a figure nobody is guessing at.
   */
  it('no longer marks MLX at 8-bit, whose width is now derived', () => {
    expect(kvSubstitutionFor(getRuntime('mlx'), 'q8')).toBeUndefined();
  });

  /**
   * And the width is actually charged, which the silence above does not establish on its own.
   *
   * These are the two halves of one claim and they fail independently: listing a precision as
   * established while leaving `kvBytesPerElement` empty silences the marker *and* keeps the wrong
   * arithmetic, which is worse than either alone.
   */
  it('charges MLX 8.5 bits per element, not the nominal 8', () => {
    expect(kvElementBytes('q8', getRuntime('mlx'))).toBeCloseTo(17 / 16, 6);
    // The nominal figure, which is what a missing declaration would fall back to.
    expect(kvElementBytes('q8', getRuntime('mlx'))).not.toBeCloseTo(1, 6);
  });

  /**
   * The derivation, restated from `mlx-lm`'s own parameters rather than as a magic 1.0625.
   *
   * This is what catches a mistyped constant or a group size that moves upstream — the figure
   * means nothing without the three numbers it comes from, and a bare `17/16` in the catalog is
   * indistinguishable from a guess that happens to look precise.
   *
   * **What deliberately has no test:** it lands on exactly llama.cpp's 34/32, and that is
   * coincidence rather than kinship — one fp16 scale per 32 elements, versus an fp16 scale *plus*
   * bias per 64, both come to half a bit. There is no assertion that can distinguish them, since
   * `17/16 === 34/32` at runtime, so the point lives here and in the catalog comment. Do not
   * "simplify" the two into a shared constant: it would tie unrelated formats together and break
   * silently when either changes its group size.
   */
  it('derives MLX’s width from the group size and the scale-plus-bias, not a bare constant', () => {
    const BITS = 8; // QuantizedKVCache(bits=8)
    const GROUP = 64; // QuantizedKVCache(group_size=64), matching --kv-group-size
    const META_BITS = 16; // scales and biases are stored at keys.dtype — fp16 on Apple silicon
    const PER_GROUP = 2; // a scale *and* a bias, which is what differs from llama.cpp

    expect(getRuntime('mlx').kvBytesPerElement?.q8).toBeCloseTo(
      (BITS + (PER_GROUP * META_BITS) / GROUP) / 8,
      6
    );
  });

  it.each([
    ['mlx', 'fp16'],
    ['llama.cpp', 'q8'],
    ['llama.cpp', 'q4'],
    ['vllm', 'q8'],
    ['vllm', 'fp16'],
  ] as const)('stays silent for %s at %s, whose width is stated', (id, precision) => {
    expect(kvSubstitutionFor(getRuntime(id), precision)).toBeUndefined();
  });

  /**
   * The polarity, asserted rather than assumed — and now the *only* thing keeping the mechanism
   * honest, since no shipped precision is marked.
   *
   * `measuredKvPrecisions` names what is established, so a precision added to a runtime later is
   * marked until someone establishes its width. Naming the stand-ins instead is what let a format
   * be added to `weightFormats` and stay silently unmarked.
   */
  it('marks a precision nobody has established a width for', () => {
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
   * The two invariants that *are* mechanical, and the one that is not.
   *
   * Held here: both axes are declared and distinct, the established list is a subset of what the
   * runtime can actually store — so an over-broad entry cannot silence a marker for a precision
   * that does not exist — and every declared width is one the engine really reads, which fails if
   * `kvElementBytes` stops consulting the catalog.
   *
   * **Not held here, deliberately: "a precision listed as established whose real width is not
   * nominal must also carry that width."** That is the #45 invariant and it stays human-enforced,
   * because it is not expressible from inside the data. vLLM's FP8 is both measured *and* nominal,
   * so "non-float precisions must declare a width" would fail it; and nothing in the catalog says
   * which precisions are group-quantized. A test that appeared to cover it would be worse than
   * this comment, since the gap it leaves — marker silent, arithmetic still nominal — is exactly
   * the state #45 was filed about. What guards MLX's case specifically is the width assertion
   * above.
   */
  it('declares both axes, and every declared width is one the engine reads', () => {
    const substituting = RUNTIMES.filter((r) => r.substituted);
    // One runtime satisfies this filter today (MLX), so without this line dropping `substituted`
    // from the catalog would empty the loop and pass every assertion in it silently.
    expect(
      substituting.length,
      'no runtime substitutes, so this loop asserts nothing'
    ).toBeGreaterThan(0);

    for (const runtime of substituting) {
      const { nativeFormats, measuredKvPrecisions, note, kvNote } = runtime.substituted!;

      expect(nativeFormats.length, `${runtime.id} marks every weight format`).toBeGreaterThan(0);
      expect(measuredKvPrecisions.length, `${runtime.id} marks every precision`).toBeGreaterThan(0);
      expect(note).not.toEqual(kvNote);

      /**
       * And both end a sentence, because both are dropped into the middle of a paragraph.
       *
       * `Bench.tsx`'s substitution panel reads "…figures below are derived from a format llama.cpp
       * cannot load. {note} They use Q4_K_M's 4.5 bpw…", so a clause here without a full stop fuses
       * two claims into one unreadable sentence — the defect #68 fixed on the Hardware picker, whose
       * fragments were composed the same way. These two are interpolated into JSX rather than joined,
       * so `sentences()` cannot cover them and the discipline has to be asserted at the source.
       */
      for (const [field, prose] of [
        ['note', note],
        ['kvNote', kvNote],
      ] as const) {
        expect(prose, `${runtime.id}'s ${field} is dropped mid-paragraph and does not end`).toMatch(
          /[.!?…]$/
        );
      }

      for (const precision of measuredKvPrecisions) {
        expect(
          runtime.kvPrecisions,
          `${runtime.id} calls ${precision} established but cannot store it`
        ).toContain(precision);
      }
    }

    // The cross-check, over every runtime rather than only the substituting ones: a width that
    // differs from nominal has to be declared wherever it is claimed to be known.
    for (const runtime of RUNTIMES) {
      for (const precision of runtime.kvPrecisions) {
        const declared = runtime.kvBytesPerElement?.[precision];
        if (declared === undefined) continue;
        expect(
          kvElementBytes(precision, runtime),
          `${runtime.id} declares ${precision} but the engine does not read it`
        ).toBeCloseTo(declared, 6);
      }
    }
  });
});

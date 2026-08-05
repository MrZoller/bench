import { describe, expect, it } from 'vitest';
import {
  CALIBRATION_BAND,
  compare,
  hasSubmittablePair,
  parseLlamaBench,
  submissionUrl,
  type Prediction,
} from './calibrate';
// The other end of the same relation: what the launch panel tells a reader to run has to be
// something this module accepts, and only a test spanning both can say so (#180).
import { decodeBenchSpan } from './launch';
import { CONTEXT_STOPS } from './stops';

/**
 * Predicted versus measured (#139).
 *
 * Two things are under test and the second is the one that matters. The parser has to read what
 * `llama-bench` actually prints, in both of its formats. And the comparison has to **refuse to
 * report a delta between two different jobs** — a measurement of the wrong scenario is noise
 * wearing a data point's chassis, and the ways that happens are invisible in the numbers.
 *
 * The fixtures below are shaped from llama-bench's own documented output rather than invented:
 * markdown columns `model | size | params | backend | ngl | test | t/s`, a `test` label that has
 * been spelled `pp 512`, `pp512` and `pp512 @ d512` across versions, and a JSON array carrying
 * `build_commit`, `n_prompt`, `n_gen`, `n_depth`, `avg_ts` and `stddev_ts`.
 */

const MARKDOWN = `
| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |  33 |        pp2048 |  7285.68 ± 100.06 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |  33 |         tg512 |     45.67 ± 0.12 |

build: 3f1ae2c0 (4123)
`;

const JSON_OUTPUT = JSON.stringify([
  {
    build_commit: '3f1ae2c0',
    build_number: 4123,
    n_prompt: 2048,
    n_gen: 0,
    n_depth: 0,
    n_gpu_layers: 33,
    avg_ts: 7285.68,
    stddev_ts: 100.06,
  },
  {
    build_commit: '3f1ae2c0',
    n_prompt: 0,
    n_gen: 512,
    // At the resident context the prediction charges every decode step against — 2,048 tokens of
    // prompt in the cache. A `tg` run from an empty cache is a weight-bound job against a KV-bound
    // prediction, which is what the depth check now catches.
    n_depth: 2048,
    n_gpu_layers: 33,
    avg_ts: 45.67,
    stddev_ts: 0.12,
  },
]);

const prediction = (over: Partial<Prediction> = {}): Prediction => ({
  prefillTokensPerSec: 7000,
  decodeTokensPerSec: 44,
  promptTokens: 2048,
  generationTokens: 512,
  concurrency: 1,
  runtimeId: 'llama.cpp',
  quantLabel: 'Q4_K_M',
  modelName: 'Llama 3.1 8B Instruct',
  totalParams: 8.03e9,
  deviceClass: 'discrete-gpu',
  deviceVendor: 'NVIDIA',
  kvType: 'f16',
  modelLayers: 32,
  gpuLayers: 33,
  // The scenario's whole window, which is what `estimateDecode` charges every step against.
  residentContextTokens: 2048,
  ...over,
});

describe('parsing what llama-bench prints', () => {
  it('reads the markdown table it prints by default', () => {
    const rows = parseLlamaBench(MARKDOWN);

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ kind: 'prefill', tokens: 2048, tokensPerSec: 7285.68 });
    expect(rows[0].stddev).toBeCloseTo(100.06, 2);
    expect(rows[1]).toMatchObject({ kind: 'decode', tokens: 512, tokensPerSec: 45.67 });
  });

  it('reads the JSON, which is the only format carrying the build', () => {
    const rows = parseLlamaBench(JSON_OUTPUT);

    expect(rows).toHaveLength(2);
    // The version-skew guard #139 names: a llama.cpp from six months ago is a different runtime for
    // calibration purposes, and markdown does not carry the commit in a parseable column.
    expect(rows[0].buildCommit).toBe('3f1ae2c0');
    expect(rows[0].gpuLayers).toBe(33);
    expect(rows.map((r) => r.kind)).toEqual(['prefill', 'decode']);
  });

  it('tolerates every spelling of the test label upstream has used', () => {
    // `pp 512`, `pp512` and `pp512 @ d512` have all shipped. A parser pinned to one of them breaks
    // on a version bump, and the reader has no idea why.
    for (const [label, expected] of [
      ['pp 512', { kind: 'prefill', tokens: 512 }],
      ['pp512', { kind: 'prefill', tokens: 512 }],
      ['tg 128', { kind: 'decode', tokens: 128 }],
      ['pp512 @ d4096', { kind: 'prefill', tokens: 512, depthTokens: 4096 }],
      ['pp512 @ d 4096', { kind: 'prefill', tokens: 512, depthTokens: 4096 }],
    ] as const) {
      const rows = parseLlamaBench(`| m | s | p | CUDA | 99 | ${label} | 123.45 ± 1.00 |`);
      expect(rows, label).toHaveLength(1);
      expect(rows[0], label).toMatchObject(expected);
    }
  });

  it('finds its columns by shape, not by position', () => {
    // The table's columns vary by build and by backend. A positional read breaks on the next
    // release; matching the `test` label and the numeric rate costs nothing and survives.
    const reordered = `| test | t/s | model | backend |\n| pp1024 | 999.9 ± 2.0 | llama | Metal |`;
    expect(parseLlamaBench(reordered)[0]).toMatchObject({ kind: 'prefill', tokensPerSec: 999.9 });
  });

  it('reports nothing rather than throwing on a paste that is not llama-bench output', () => {
    // A reader pasting the wrong thing is a mistake to say something about on the surface, not an
    // exception to handle.
    for (const junk of ['', '   ', 'hello world', '{"not": "an array"}', '| a | b |\n| c | d |']) {
      expect(parseLlamaBench(junk), junk).toEqual([]);
    }
  });

  it('ignores the build footer and the separator row', () => {
    // Both are lines with pipes or numbers in them, and both would become rows in a looser parser.
    expect(parseLlamaBench(MARKDOWN)).toHaveLength(2);
  });
});

describe('a measurement of a different job is not evidence about the model', () => {
  it('reports a delta and the band when the pair really is comparable', () => {
    const [prefill, decode] = compare(parseLlamaBench(JSON_OUTPUT), prediction());

    expect(prefill.mismatch).toBeUndefined();
    expect(prefill.error).toBeCloseTo(7285.68 / 7000 - 1, 4);
    expect(prefill.withinBand).toBe(true);
    expect(decode.mismatch).toBeUndefined();
    expect(decode.withinBand).toBe(true);
  });

  it('marks a run at the wrong prompt length rather than rescaling it', () => {
    /**
     * The temptation this refuses. Rescaling `pp512` to a 16,384-token prediction would produce a
     * plausible delta, and it would be wrong twice: prefill is quadratic so the rescaling is a model
     * rather than an observation, and the whole point of the surface is that the reader can check
     * Headroom's arithmetic instead of taking more of it on trust.
     */
    const [pair] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ promptTokens: 16384 }));

    expect(pair.mismatch).toMatch(/run at 2,048 tokens where the prediction is for 16,384/);
  });

  it('allows a nearby length, since a reader may type their own', () => {
    // The emitted command supplies the scenario's own lengths; a reader who typed theirs lands near
    // rather than on. Ten percent keeps the quadratic term still and catches pp512-against-16K.
    const [pair] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ promptTokens: 2100 }));
    expect(pair.mismatch).toBeUndefined();
  });

  it('marks a standalone run against a workload that assumes a resident prefix', () => {
    // The #139 trap. `estimatePrefill` charges an agent turn's attention against the prefix, so a
    // measurement with an empty cache is a measurement of a different workload — and the numbers
    // give no sign of it.
    const [pair] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ cachedPrefixTokens: 47616 }));

    expect(pair.mismatch).toMatch(/empty cache/);
    expect(pair.mismatch).toMatch(/charges 47,616 tokens of it/);
    expect(pair.mismatch).toMatch(/pass -d/);
  });

  it('marks a depth the prediction does not have, which is the same error inverted', () => {
    const measured = parseLlamaBench(`| m | s | p | CUDA | 99 | pp2048 @ d8192 | 5000.0 ± 1.0 |`);
    const [pair] = compare(measured, prediction());

    expect(pair.mismatch).toMatch(/depth of 8,192 where the prediction has none/);
  });

  it('marks both kinds against a multi-user prediction, for different reasons', () => {
    /**
     * The asymmetry `speed.ts` documents, surfaced where it can mislead — and **decode is marked
     * too**, which the first version got wrong on a half-true rationale.
     *
     * "Decode amortises across the batch" is true of the *weights* and false of the *cache*:
     * `estimateDecode` charges every concurrent sequence's KV read on every step, so
     * `perUserTokensPerSec` at eight users sits well below a solo `tg` wherever the cache is what
     * decode is bound by. Different arithmetic from prefill's, same conclusion — llama-bench
     * measures one sequence.
     */
    const [prefill, decode] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ concurrency: 8 }));

    expect(prefill.mismatch).toMatch(/machine-wide rate across 8 users/);
    expect(decode.mismatch).toMatch(/8 sequences' cache reads/);
  });

  it('expects a resident cache for decode and an empty one for a standalone prefill', () => {
    /**
     * **The two kinds want different depths, and the first version gave them the same one** — so it
     * flagged the correctly-reproduced `tg … -d 2048` and passed the empty-cache run that is not
     * comparable at all. Backwards, in the direction that manufactures evidence.
     */
    const [prefill, decode] = compare(parseLlamaBench(JSON_OUTPUT), prediction());
    expect(prefill.mismatch, 'a standalone prefill at depth 0 is right').toBeUndefined();
    expect(decode.mismatch, 'a tg run at the resident context is right').toBeUndefined();

    // A full model cell, so the quant check does not also fire and mask what is being tested.
    const empty = parseLlamaBench(
      `| llama 8B Q4_K - Medium | 4.58 GiB | 8.03 B | CUDA | 33 | tg512 | 45.67 ± 0.12 |`
    );
    expect(compare(empty, prediction())[0].mismatch).toMatch(
      /empty cache where the prediction charges 2,048/
    );
  });

  it('marks a paste at a different quantization, which decode is bound by', () => {
    // Q8_0 against a Q4_K_M prediction is roughly twice the bytes per token on a memory-bound
    // decode, and nothing in the two numbers says so.
    const wrongQuant = parseLlamaBench(
      `| llama 8B Q8_0 | 8.5 GiB | 8.03 B | CUDA | 33 | pp2048 | 7285.68 ± 100.06 |`
    );
    expect(compare(wrongQuant, prediction())[0].mismatch).toMatch(
      /where the figures above are for Q4_K_M/
    );

    // And llama.cpp's own spelling of the matching format does not fire it: it writes
    // "Q4_K - Medium" where the catalog writes "Q4_K_M", so a strict compare would mark every paste.
    expect(compare(parseLlamaBench(MARKDOWN), prediction())[0].mismatch).toBeUndefined();
  });

  it('marks a paste at a different cache precision', () => {
    const q8Cache = JSON.stringify([
      { n_prompt: 2048, n_gen: 0, n_depth: 0, type_k: 'q8_0', type_v: 'q8_0', avg_ts: 7285.68 },
    ]);
    expect(compare(parseLlamaBench(q8Cache), prediction())[0].mismatch).toMatch(
      /q8_0 cache where the figures above assume f16/
    );
  });

  it('marks a run that left layers on the host against a resident placement', () => {
    const spilled = JSON.stringify([
      { n_prompt: 2048, n_gen: 0, n_depth: 0, n_gpu_layers: 12, avg_ts: 900 },
    ]);
    expect(compare(parseLlamaBench(spilled), prediction({ gpuLayers: 32 }))[0].mismatch).toMatch(
      /12 layers on the GPU where the placement above puts 32 of 32/
    );

    // And says nothing when the prediction makes no claim.
    expect(
      compare(parseLlamaBench(spilled), prediction({ gpuLayers: undefined }))[0].mismatch
    ).toBeUndefined();
  });

  it('accepts every spelling of "all the layers"', () => {
    /**
     * Where the two halves of this project disagreed with each other. llama.cpp counts the output
     * tensor a position past the repeating blocks, so #136's emitter passes `layers + 1` for a
     * fully-resident placement, and readers type `-ngl 99` for the same thing. Comparing against
     * the layer count alone would have marked a run that followed Headroom's own command.
     */
    for (const ngl of [32, 33, 99]) {
      const run = JSON.stringify([
        { n_prompt: 2048, n_gen: 0, n_depth: 0, n_gpu_layers: ngl, avg_ts: 7285.68 },
      ]);
      expect(
        compare(parseLlamaBench(run), prediction({ modelLayers: 32, gpuLayers: 32 }))[0].mismatch,
        `-ngl ${ngl}`
      ).toBeUndefined();
    }
  });

  it('accepts the emitted flag for a partial placement, and the older one it replaced', () => {
    /**
     * **The same disagreement one layer down** (#204). `prediction.gpuLayers` counts *repeating
     * layers*; `n_gpu_layers` in a paste is the flag, which counts the output tensor a slot past
     * them. Since #204 the Launch panel emits `-ngl N + 1` for a spilling placement of `N` layers,
     * so comparing exactly against `N` would mark a run that followed Headroom's own command — the
     * failure the fully-resident branch above was written for, arriving on the branch it left out.
     *
     * `-ngl N` is accepted alongside it, and **not because the two are equivalent** — it loads
     * `N - 1` repeating layers, a measurably different placement. It is accepted because every
     * spilling command this panel emitted between #169 and #204 was a bare `-ngl N`, so those
     * pastes exist. A backwards-compatibility tolerance with a shelf life, narrowed in #208
     * together with the fully-resident arm, which has the same hole and far more of it.
     */
    for (const ngl of [12, 13]) {
      const run = JSON.stringify([
        { n_prompt: 2048, n_gen: 0, n_depth: 0, n_gpu_layers: ngl, avg_ts: 900 },
      ]);
      expect(
        compare(parseLlamaBench(run), prediction({ modelLayers: 32, gpuLayers: 12 }))[0].mismatch,
        `-ngl ${ngl}`
      ).toBeUndefined();
    }

    // And the tolerance is exactly one slot wide, not a band: a genuinely different split still
    // fails, on both sides.
    for (const ngl of [11, 14]) {
      const run = JSON.stringify([
        { n_prompt: 2048, n_gen: 0, n_depth: 0, n_gpu_layers: ngl, avg_ts: 900 },
      ]);
      expect(
        compare(parseLlamaBench(run), prediction({ modelLayers: 32, gpuLayers: 12 }))[0].mismatch,
        `-ngl ${ngl}`
      ).toMatch(/layers on the GPU where the placement above puts 12 of 32/);
    }
  });

  it('rejects a GPU run against a CPU prediction, which a one-sided check let through', () => {
    // `cpu-ram` predicts zero GPU layers, and only rejecting *fewer* than predicted let every
    // positive count pass — so the EPYC-shaped measurements this feature exists for could be
    // satisfied by a GPU run.
    const onGpu = JSON.stringify([
      { n_prompt: 2048, n_gen: 0, n_depth: 0, n_gpu_layers: 32, avg_ts: 7285.68 },
    ]);
    expect(
      compare(parseLlamaBench(onGpu), prediction({ gpuLayers: 0, modelLayers: 32 }))[0].mismatch
    ).toMatch(/32 layers on the GPU where the placement above puts 0 of 32/);

    /**
     * **And `-ngl 1` is not a second spelling of zero**, which is where the #204 widening above had
     * to stop. A prediction of no GPU layers is a machine with no GPU or a card with no room for a
     * layer, and the emitter passes `-ngl 0` for both; `-ngl 1` puts the whole output table on a
     * GPU, so accepting it would re-open the hole the two-sided check closed.
     */
    const oneSlot = JSON.stringify([
      { n_prompt: 2048, n_gen: 0, n_depth: 0, n_gpu_layers: 1, avg_ts: 7285.68 },
    ]);
    expect(
      compare(parseLlamaBench(oneSlot), prediction({ gpuLayers: 0, modelLayers: 32 }))[0].mismatch
    ).toMatch(/1 layers on the GPU where the placement above puts 0 of 32/);
  });

  it('will not read a markdown paste as confirming a non-default cache', () => {
    // A paste carrying no cache precision must not sail past a Q8 prediction — unverifiable is not
    // the same as matching, and markdown is the default output. `MARKDOWN` is the shape that has to
    // stay unverifiable *after* #181 as well: llama-bench prints the cache columns only when they
    // are not at their default, so a table whose header does not list them is a table that says
    // nothing about the cache, and reading the header is what tells the two apart.
    expect(compare(parseLlamaBench(MARKDOWN), prediction({ kvType: 'q8_0' }))[0].mismatch).toMatch(
      /without a stated cache precision/
    );
    // And an f16 prediction is unaffected, since nothing about it is unverifiable.
    expect(compare(parseLlamaBench(MARKDOWN), prediction())[0].mismatch).toBeUndefined();
  });

  it('reads the cache columns a markdown paste prints, rather than calling it unverifiable', () => {
    // **This assertion is the inversion of the one #175 pinned here**, and the inversion is the
    // point of having pinned it. Until #181 the limitation was the *parser* rather than the format:
    // `parseMarkdown` had no branch for `type_k`/`type_v` at all, so it never yielded `kvTypes`
    // whether or not llama-bench printed them — and the panel's own measure command passes
    // `-ctk q8_0 -ctv q8_0 -o md`, which makes it print them. The reader who followed the panel
    // exactly was told their correct run looked like f16.
    //
    // **And `ngl` went with them, which this fixture is what found** (Codex, on #175).
    // `parseMarkdown` located `ngl` by *position* — the cell before `test` — because a bare integer
    // has no distinctive shape. The cache columns sit between the two in llama-bench's own layout,
    // so the cell before `test` was `type_v` and the layer count was lost on exactly the output the
    // panel's command produces. Both are one fix: the parser now reads the header row and indexes
    // by column name, so the order of the middle columns costs nothing.
    const withCacheColumns = `
| model                  |     params | backend | ngl | type_k | type_v |          test |          t/s |
| ---------------------- | ---------: | ------- | --: | ------ | ------ | ------------: | -----------: |
| llama 8B Q4_K - Medium |     8.03 B | CUDA    |  33 | q8_0   | q8_0   | tg512 @ d2048 | 45.67 ± 0.12 |
`;
    const parsed = parseLlamaBench(withCacheColumns);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].kvTypes).toEqual({ k: 'q8_0', v: 'q8_0' });
    expect(parsed[0].gpuLayers).toBe(33);
    // The run this fixture describes *is* the one the q8_0 figures assume, and it now reads that
    // way: no cache reason, and a layer check that ran rather than being skipped.
    expect(compare(parsed, prediction({ kvType: 'q8_0' }))[0].mismatch).toBeUndefined();
  });

  it('does not require a decode run to be any particular length', () => {
    // `perUserTokensPerSec` is a steady-state per-token rate and does not depend on how many
    // tokens are asked for, so requiring `n_gen` to match the window's remainder rejected every
    // ordinary tg128 against a scenario that merely happened to leave 2,192 tokens spare. What
    // matters for decode is the cache it reads, which is the depth check.
    const short = JSON.stringify([{ n_prompt: 0, n_gen: 128, n_depth: 2048, avg_ts: 45.67 }]);
    expect(compare(parseLlamaBench(short), prediction())[0].mismatch).toBeUndefined();
  });

  it('marks a llama-bench paste against a runtime llama-bench cannot measure', () => {
    // #139's own stated scope: llama-bench loads GGUF and speaks for llama.cpp placements alone.
    const [pair] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ runtimeId: 'vllm' }));
    expect(pair.mismatch).toMatch(/speaks for llama\.cpp placements only/);
  });

  it('drops a combined pg row rather than reading its blended rate as prefill', () => {
    /**
     * `llama-bench` computes a row's rate as `(n_prompt + n_gen) / time`, so a `-pg` row's `avg_ts`
     * is a blend dominated by the slow half — 7,000 t/s of prefill and 100 of decode come out near
     * 473. Read as prefill that is a 93% miss with nothing to mark it, and there is no way to
     * recover either rate from one number.
     */
    const pg = JSON.stringify([
      { n_prompt: 512, n_gen: 128, n_depth: 0, avg_ts: 473.1, build_commit: 'abc' },
    ]);
    expect(parseLlamaBench(pg)).toEqual([]);
    // The markdown form was already safe, by failing the anchored label pattern. Now deliberately.
    expect(parseLlamaBench(`| m | s | p | CUDA | 33 | pp512+tg128 | 473.1 ± 1.0 |`)).toEqual([]);
  });

  it('lists every reason rather than the first one', () => {
    const [pair] = compare(
      parseLlamaBench(JSON_OUTPUT),
      prediction({ promptTokens: 16384, cachedPrefixTokens: 8192, concurrency: 4 })
    );

    expect(pair.mismatch).toMatch(/16,384/);
    expect(pair.mismatch).toMatch(/empty cache/);
    expect(pair.mismatch).toMatch(/4 users/);
  });

  it('judges the band at the figure the engine’s own tests assert', () => {
    const rows = parseLlamaBench(JSON_OUTPUT);
    // Just inside and just outside, from the same measurement, so the boundary is the thing tested
    // rather than the arithmetic around it.
    const inside = compare(
      rows,
      prediction({ prefillTokensPerSec: 7285.68 / (1 + CALIBRATION_BAND) })
    );
    const outside = compare(
      rows,
      prediction({ prefillTokensPerSec: 7285.68 / (1 + CALIBRATION_BAND + 0.01) })
    );

    expect(inside[0].withinBand).toBe(true);
    expect(outside[0].withinBand).toBe(false);
  });
});

/**
 * The other side of every check above: **the one command Headroom itself prints has to survive
 * them** (#180).
 *
 * It did not. `llamaBench()` emitted its decode run at `prompt + prefix` — the cache resident when
 * generation begins — while this module expects `contextTokens`, which is what `estimateDecode`
 * charges every step's cache read at. On the default 8K-prompt/32K-context scenario that is 8,192
 * against 32,768, so the likeliest path a reader takes through this feature — copy the measure
 * command, run it, paste the result — produced a row marked unusable.
 *
 * The relation is a *ratio* rather than a shared constant, which is why this sweeps the stops
 * instead of asserting one pair of numbers. `decodeBenchSpan` leaves room for the generation inside
 * the window, so the emitted depth is `contextTokens - gen` and the depth the run actually averages
 * over is `contextTokens - gen / 2`; neither is the figure the engine charges, and both have to stay
 * inside the tolerance at every context a reader can select — including the 512 floor `coerce`
 * clamps at, where a flat 128 tokens of generation would be a quarter of the window and rejected.
 */
describe('the command the launch panel prints is one this comparison accepts', () => {
  /** Every context a reader can reach: the fixed stops, plus the floor below the smallest of them. */
  const contexts = [512, ...CONTEXT_STOPS];

  it('accepts the emitted depth at every context stop', () => {
    for (const contextTokens of contexts) {
      const { depth, gen } = decodeBenchSpan({
        contextTokens,
        concurrency: 1,
        kvPrecision: 'fp16',
      });
      const [pair] = compare(
        [{ kind: 'decode', tokens: gen, depthTokens: depth, tokensPerSec: 44 }],
        prediction({ residentContextTokens: contextTokens, promptTokens: contextTokens })
      );

      expect(pair.mismatch, `${contextTokens}: -d ${depth} -n ${gen}`).toBeUndefined();
      // And the command is one llama-bench can run: it sizes n_ctx as n_prompt + n_gen + n_depth,
      // so the two flags together have to fit the window the panel priced.
      expect(depth + gen, `${contextTokens}`).toBe(contextTokens);
    }
  });

  it('accepts the depth the run averages over, which is the shallower of the two', () => {
    // The cache grows as it generates, so a reader's measurement is not taken at `-d` throughout.
    // Half a generation shallower is still the same claim, and it is the one the numbers embody.
    for (const contextTokens of contexts) {
      const { depth, gen } = decodeBenchSpan({
        contextTokens,
        concurrency: 1,
        kvPrecision: 'fp16',
      });
      const meanDepth = Math.round(depth + gen / 2);
      const [pair] = compare(
        [{ kind: 'decode', tokens: gen, depthTokens: meanDepth, tokensPerSec: 44 }],
        prediction({ residentContextTokens: contextTokens, promptTokens: contextTokens })
      );

      expect(pair.mismatch, `${contextTokens}: mean depth ${meanDepth}`).toBeUndefined();
    }
  });

  it('would have rejected the depth it emitted before, on the default scenario', () => {
    // The regression this closes, stated as the numbers #180 was filed with rather than as a rule.
    const [pair] = compare(
      [{ kind: 'decode', tokens: 24576, depthTokens: 8192, tokensPerSec: 44 }],
      prediction({ residentContextTokens: 32768, promptTokens: 8192 })
    );

    expect(pair.mismatch).toMatch(/depth of 8,192 where the prediction charges 32,768/);
  });
});

describe('what the second review round found', () => {
  it('marks a different model at the same format', () => {
    // A Llama Q4_K_M measurement against a DeepSeek Q4_K_M prediction passed on the format alone,
    // and the generated issue then labelled it as the DeepSeek run — a wrong data point entering
    // the record under a name nobody will question.
    const [pair] = compare(
      parseLlamaBench(MARKDOWN),
      prediction({ modelName: 'DeepSeek V3', quantLabel: 'Q4_K_M' })
    );
    expect(pair.mismatch).toMatch(/where the figures above are for DeepSeek V3/);
  });

  it('charges decode against the whole window, not the prompt', () => {
    // `estimateDecode` prices every step at `usage.contextTokens`. At the default 8K-prompt,
    // 32K-context scenario the first version accepted a run near 8K depth and marked the run at
    // the modelled 32K — grading a measurement against a rate it does not describe.
    const at32k = prediction({ promptTokens: 2048, residentContextTokens: 32768 });
    const [, decode] = compare(parseLlamaBench(JSON_OUTPUT), at32k);
    expect(decode.mismatch).toMatch(/depth of 2,048 where the prediction charges 32,768/);
  });

  it('reads the layer count out of the markdown table too', () => {
    // The default output carries an `ngl` column, so a run with half the model on the host was
    // accepted against a fully-resident prediction on any non-JSON paste.
    const partial = parseLlamaBench(
      `| llama 8B Q4_K - Medium | 4.58 GiB | 8.03 B | CUDA | 12 | pp2048 | 900.0 ± 1.0 |`
    );
    expect(partial[0].gpuLayers).toBe(12);
    expect(compare(partial, prediction())[0].mismatch).toMatch(/12 layers on the GPU/);
  });

  it('requires both halves of the cache to match, not either', () => {
    // They are charged separately and they are separate flags; a run matching on K alone is not a
    // run at this precision.
    const mixed = JSON.stringify([
      { n_prompt: 2048, n_gen: 0, n_depth: 0, type_k: 'f16', type_v: 'q4_0', avg_ts: 7285.68 },
    ]);
    expect(compare(parseLlamaBench(mixed), prediction())[0].mismatch).toMatch(
      /f16\/q4_0 cache where the figures above assume f16/
    );
  });
});

describe('what the third review round found', () => {
  it('identifies the model by its parameter count, not by its name', () => {
    /**
     * The name check catches a cross-family paste and misses Qwen3 8B against Qwen3 32B — llama.cpp
     * writes an architecture where the catalog writes a product, so the two never agree past the
     * first word. Both formats print a parameter count, which is the same quantity on both sides.
     */
    const bigger = parseLlamaBench(
      `| qwen3 32B Q4_K - Medium | 18.5 GiB | 32.76 B | CUDA | 65 | pp2048 | 3000.0 ± 1.0 |`
    );
    expect(bigger[0].params).toBeCloseTo(32.76e9, -8);
    expect(compare(bigger, prediction())[0].mismatch).toMatch(
      /32.8B model where the figures above are for 8.0B/
    );
  });

  it('marks a Metal run against a device that is not Apple', () => {
    // Checked only where the backend *contradicts* the device: a vendor-to-backend table would be
    // inventing data, and llama.cpp's names vary by build. Metal is Apple's alone.
    const metal = parseLlamaBench(
      `| llama 8B Q4_K - Medium | 4.58 GiB | 8.03 B | Metal | 33 | pp2048 | 900.0 ± 1.0 |`
    );
    expect(compare(metal, prediction())[0].mismatch).toMatch(/run on Metal/);
  });

  it('marks a CPU run against a graphics card', () => {
    const cpu = parseLlamaBench(
      `| llama 8B Q4_K - Medium | 4.58 GiB | 8.03 B | CPU | 0 | pp2048 | 40.0 ± 1.0 |`
    );
    expect(compare(cpu, prediction())[0].mismatch).toMatch(
      /run on the CPU where the figures above are for a graphics card/
    );
  });

  it('refuses to compare against a configuration that cannot run', () => {
    // `impossible` means the cache and activations alone are over the ceiling, so the rates beside
    // it describe a machine that cannot load the model — any measurement pasted against them was
    // necessarily taken on something else, and the panel was producing a percentage anyway.
    const [pair] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ impossible: true }));
    expect(pair.mismatch).toMatch(/cannot run at all/);
    expect(hasSubmittablePair([pair])).toBe(false);
  });

  it('makes no claim about the length when the window leaves no room to generate', () => {
    // The first version floored the expectation at one token, so a prompt filling the window
    // rejected every normal decode row against a length nothing can satisfy.
    const [, decode] = compare(parseLlamaBench(JSON_OUTPUT), prediction({ generationTokens: 0 }));
    expect(decode.mismatch).toBeUndefined();
  });
});

/**
 * Reading the header row (#181), and the two defects that were one defect.
 *
 * `parseMarkdown` did not read the header at all. `type_k`/`type_v` had no branch, so no markdown
 * paste ever carried a cache precision; and `ngl` was found by *position* — the cell before `test` —
 * which those same cache columns displace. llama-bench prints a column for every setting that is not
 * at its default, in its own field order, between `backend` and `test`, so the middle of the table
 * is exactly the part a position cannot describe.
 *
 * The fixtures below are that field order rather than an invented one: `model`, `size`, `params`,
 * `backend`, then `ngl` (omitted entirely for a CPU backend), `threads` (added for one), `type_k`,
 * `type_v`, `ts` — and `test`, `t/s` last, always.
 */
describe('what reading the header row settled', () => {
  /**
   * What the panel's own two commands print. `-ngl 33 -ctk q8_0 -ctv q8_0 -o md`, twice, so
   * llama-bench prints `ngl`, `type_k` and `type_v` — and two tables with a header each, because
   * `llamaBench()` emits two invocations and a reader pastes what both of them printed.
   */
  const PANEL_MARKDOWN = `
| model                  |       size |     params | backend | ngl | type_k | type_v |   test |              t/s |
| ---------------------- | ---------: | ---------: | ------- | --: | -----: | -----: | -----: | ---------------: |
| llama 8B Q4_K - Medium |   4.58 GiB |     8.03 B | CUDA    |  33 |   q8_0 |   q8_0 | pp2048 | 7285.68 ± 100.06 |

| model                  |       size |     params | backend | ngl | type_k | type_v |          test |          t/s |
| ---------------------- | ---------: | ---------: | ------- | --: | -----: | -----: | ------------: | -----------: |
| llama 8B Q4_K - Medium |   4.58 GiB |     8.03 B | CUDA    |  33 |   q8_0 |   q8_0 | tg512 @ d2048 | 45.67 ± 0.12 |
`;

  it('carries the cache precision and the layer count through a markdown paste', () => {
    const rows = parseLlamaBench(PANEL_MARKDOWN);

    expect(rows).toHaveLength(2);
    for (const row of rows) {
      expect(row.kvTypes).toEqual({ k: 'q8_0', v: 'q8_0' });
      expect(row.gpuLayers).toBe(33);
    }
    // Which is the whole complaint in #181: the reader who followed the panel exactly was told
    // their correctly reproduced run "looks like an f16 run". Both rows now compare clean.
    const [prefill, decode] = compare(rows, prediction({ kvType: 'q8_0' }));
    expect(prefill.mismatch).toBeUndefined();
    expect(decode.mismatch).toBeUndefined();
  });

  it('does not let an offloaded run compare clean against a resident placement', () => {
    /**
     * **The sharper half, as a whole number rather than as a rule.** A reader who ran 12 of 32
     * layers on the GPU with a q8_0 cache, against a fully-resident f16 prediction: two independent
     * differences, both invisible in the numbers, and the panel reported neither. `ngl` was
     * displaced by the cache columns so the placement check was skipped, and `kvTypes` was never
     * read so the cache check had nothing to fire on — an 87% miss, unmarked and submittable.
     */
    const offloaded = `
| model                  |       size |     params | backend | ngl | type_k | type_v |   test |            t/s |
| ---------------------- | ---------: | ---------: | ------- | --: | -----: | -----: | -----: | -------------: |
| llama 8B Q4_K - Medium |   4.58 GiB |     8.03 B | CUDA    |  12 |   q8_0 |   q8_0 | pp2048 | 900.00 ± 10.00 |
`;
    const [pair] = compare(parseLlamaBench(offloaded), prediction());

    expect(pair.error).toBeCloseTo(900 / 7000 - 1, 4);
    expect(pair.mismatch).toMatch(/12 layers on the GPU where the placement above puts 33 of 32/);
    expect(pair.mismatch).toMatch(/q8_0 cache where the figures above assume f16/);
    expect(hasSubmittablePair([pair])).toBe(false);
  });

  it('finds ngl past a column it has never seen before', () => {
    // `-ts` is on the panel's own multi-GPU command, and llama-bench prints a `ts` column for it —
    // between `ngl` and `test`, same as the cache pair. So the displacement does not need a cache
    // column at all, and at f16 nothing else would have marked the row: an offloaded multi-GPU run
    // compared clean. Indexing by name is what makes the middle of the table free to change.
    const withSplit = `
| model                  |     params | backend | ngl |                  ts |   test |            t/s |
| ---------------------- | ---------: | ------- | --: | ------------------: | -----: | -------------: |
| llama 8B Q4_K - Medium |     8.03 B | CUDA    |  12 | 0.27/0.27/0.23/0.23 | pp2048 | 900.00 ± 10.00 |
`;
    const [row] = parseLlamaBench(withSplit);

    expect(row.gpuLayers).toBe(12);
    expect(compare([row], prediction())[0].mismatch).toMatch(/12 layers on the GPU/);
  });

  it('reads the columns in whatever order the header puts them', () => {
    // The order is llama-bench's own and it has changed before. Nothing here depends on it —
    // including which half of the cache pair is which, which a positional read would have had to
    // guess and a transposed one would report as a mixed-precision run that never happened.
    const reordered = `
| test   |            t/s | type_v | ngl | type_k | model                  |
| pp2048 | 900.00 ± 10.00 |   q4_0 |  20 |   q8_0 | llama 8B Q4_K - Medium |
`;
    expect(parseLlamaBench(reordered)[0]).toMatchObject({
      gpuLayers: 20,
      kvTypes: { k: 'q8_0', v: 'q4_0' },
    });
  });

  it('does not read a thread count as a layer count', () => {
    /**
     * **The defect inverted, and the one that would have cost the most.** llama-bench omits `ngl`
     * for a CPU backend and prints `threads` instead, so the cell before `test` was the thread
     * count — a 96-thread EPYC run read as 96 layers on a GPU it does not have, and marked as a
     * different job against the `cpu-ram` placement that predicts zero. That is a *false* rejection
     * of exactly the measurements the second calibration anchor is made of.
     *
     * A header with no `ngl` column means the run stated no layer count, and unstated skips the
     * check — which is this module's rule for every optional field the paste may omit.
     */
    const onHost = `
| model                  |       size |     params | backend | threads |          test |          t/s |
| ---------------------- | ---------: | ---------: | ------- | ------: | ------------: | -----------: |
| llama 8B Q4_K - Medium |   4.58 GiB |     8.03 B | CPU     |      96 | tg512 @ d2048 |  5.20 ± 0.03 |
`;
    const [row] = parseLlamaBench(onHost);

    expect(row.gpuLayers).toBeUndefined();
    expect(
      compare([row], prediction({ deviceClass: 'cpu-ram', deviceVendor: 'AMD', gpuLayers: 0 }))[0]
        .mismatch
    ).toBeUndefined();
  });

  it('will not complete a half-stated cache pair', () => {
    // `-ctk q8_0` alone prints `type_k` and leaves `type_v` at a default llama-bench does not
    // print. Filling in `f16` for the missing half would invent the exact field the cache check
    // exists to compare — a mixed `q8_0`/`f16` run is a run at neither precision — so one column is
    // a paste that has not stated its cache precision, and it is told so.
    const halfStated = `
| model                  |     params | backend | ngl | type_k |   test |              t/s |
| ---------------------- | ---------: | ------- | --: | -----: | -----: | ---------------: |
| llama 8B Q4_K - Medium |     8.03 B | CUDA    |  33 |   q8_0 | pp2048 | 7285.68 ± 100.06 |
`;
    const [row] = parseLlamaBench(halfStated);

    expect(row.kvTypes).toBeUndefined();
    // And the rest of the header still reads, which is the point of indexing rather than bailing.
    expect(row.gpuLayers).toBe(33);
    expect(compare([row], prediction({ kvType: 'q8_0' }))[0].mismatch).toMatch(
      /without a stated cache precision/
    );
  });

  it('still reads a row pasted without its header', () => {
    /**
     * A reader pastes one row out of a table. That is not a parse failure: `ngl` falls back to the
     * position it used before, which is what the table's own layout puts there when nothing
     * displaced it — and the common bare paste is a GPU row where the position is right.
     *
     * When something has displaced it — here the cache columns, with no header to name them — the
     * fallback returns nothing rather than a neighbour, because `q8_0` is not an integer. The
     * fallback is as good as it was and no better, which is the argument for the header rather than
     * an argument against the fallback.
     */
    const bare = `| llama 8B Q4_K - Medium | 4.58 GiB | 8.03 B | CUDA | 12 | pp2048 | 900.0 ± 1.0 |`;
    expect(parseLlamaBench(bare)[0]).toMatchObject({ gpuLayers: 12 });
    expect(parseLlamaBench(bare)[0].kvTypes).toBeUndefined();

    const bareWithCache = `| llama 8B Q4_K - Medium | 4.58 GiB | 8.03 B | CUDA | 12 | q8_0 | q8_0 | pp2048 | 900.0 ± 1.0 |`;
    const [row] = parseLlamaBench(bareWithCache);
    expect(row.gpuLayers).toBeUndefined();
    expect(row.kvTypes).toBeUndefined();
  });

  it('keeps an empty cell, because the header counts cells', () => {
    // Dropping empties was harmless while every lookup was by shape and is not once one is by
    // index: a blank cell shifts every column after it out from under the header, and this row
    // would be read against a header one column wider than itself — losing both fields again.
    const blankSize = `
| model                  | size |     params | backend | ngl | type_k | type_v |   test |              t/s |
| ---------------------- | ---- | ---------: | ------- | --: | -----: | -----: | -----: | ---------------: |
| llama 8B Q4_K - Medium |      |     8.03 B | CUDA    |  33 |   q8_0 |   q8_0 | pp2048 | 7285.68 ± 100.06 |
`;
    expect(parseLlamaBench(blankSize)[0]).toMatchObject({
      gpuLayers: 33,
      kvTypes: { k: 'q8_0', v: 'q8_0' },
    });
  });

  it('will not name a row the header does not fit', () => {
    // A header describes a row of its own width and nothing else — llama-bench writes one cell per
    // field for both, so a row of a different width came from somewhere else or arrived truncated.
    // Here the `type_v` cell is missing, which would slide the test label into the cache precision
    // and report a run at a `q8_0`/`pp2048` cache: a fabricated answer to the question the panel
    // was asked, which is worse than the absent one. Such a row is read as though it had no header.
    const ragged = `
| model                  |     params | backend | ngl | type_k | type_v |   test |            t/s |
| ---------------------- | ---------: | ------- | --: | -----: | -----: | -----: | -------------: |
| llama 8B Q4_K - Medium |     8.03 B | CUDA    |  12 |   q8_0 | pp2048 | 900.00 ± 10.00 |
`;
    const [row] = parseLlamaBench(ragged);

    expect(row.kvTypes).toBeUndefined();
    expect(row.gpuLayers).toBeUndefined();
    expect(row).toMatchObject({ kind: 'prefill', tokens: 2048, tokensPerSec: 900 });
  });

  it('reads the same measurement out of both formats', () => {
    /**
     * **The guard against these two drifting apart again.** `type_k`/`type_v` were read on the JSON
     * path and not on the markdown one, and nothing compared the two — which is how the gap
     * survived a release and two reviews. A paste of one run in either format is the same run, and
     * this asserts it field by field rather than by inspection.
     */
    const asJson = JSON.stringify([
      {
        model_type: 'llama 8B Q4_K - Medium',
        model_n_params: 8.03e9,
        backend: 'CUDA',
        n_gpu_layers: 33,
        type_k: 'q8_0',
        type_v: 'q8_0',
        n_prompt: 2048,
        n_gen: 0,
        n_depth: 0,
        avg_ts: 7285.68,
        stddev_ts: 100.06,
      },
    ]);
    const [fromJson] = parseLlamaBench(asJson);
    const [fromMarkdown] = parseLlamaBench(PANEL_MARKDOWN);

    expect(fromMarkdown).toEqual({
      ...fromJson,
      /**
       * The one field the two formats cannot agree on exactly, and it is the table's arithmetic
       * rather than the parser's: markdown prints `8.03 B` where JSON carries the count, so the
       * two are the same quantity to the three figures the table has. Well inside the 10% the
       * model check compares at, which is what that tolerance is for.
       */
      params: expect.closeTo(8.03e9, 0),
    });
  });
});

describe('the submission carries the scenario, not a description of it', () => {
  const url = (over: Parameters<typeof submissionUrl>[0] | undefined = undefined) =>
    submissionUrl(
      over ?? {
        repoUrl: 'https://github.com/MrZoller/headroom',
        scenarioUrl: 'https://mrzoller.github.io/headroom/?m=x&d=rtx-5090',
        deviceName: 'GeForce RTX 5090',
        deviceCount: 1,
        modelName: 'Llama 3.1 8B Instruct',
        comparisons: compare(parseLlamaBench(JSON_OUTPUT), prediction()),
      }
    );

  const bodyOf = (href: string) => new URL(href).searchParams.get('body') ?? '';

  it('puts the scenario link in the body, since it is the reproducible half', () => {
    // `llama-bench` names the model file and the backend but not the host reliably, so the URL is
    // what ties a measurement to a device row.
    expect(bodyOf(url())).toContain('https://mrzoller.github.io/headroom/?m=x&d=rtx-5090');
  });

  it('carries both figures and the error, as a table a maintainer can read', () => {
    const body = bodyOf(url());
    expect(body).toContain('| predicted t/s | measured t/s | error |');
    expect(body).toContain('7285.7');
    expect(body).toContain('7000.0');
  });

  it('names a missing build rather than omitting the field', () => {
    // A build nobody recorded and a build nobody asked for are different states, and the record has
    // to be able to tell them apart.
    const body = bodyOf(
      url({
        repoUrl: 'https://github.com/MrZoller/headroom',
        scenarioUrl: 'https://example.test/?d=rtx-5090',
        deviceName: 'GeForce RTX 5090',
        deviceCount: 1,
        modelName: 'Llama 3.1 8B Instruct',
        // Markdown output, which carries no commit.
        comparisons: compare(parseLlamaBench(MARKDOWN), prediction()),
      })
    );

    expect(body).toMatch(/llama\.cpp build:.*not in the pasted output/);
    expect(body).toMatch(/-o json/);
  });

  it('names the rig rather than one of its cards', () => {
    // The scenario link keeps the count, but the title and the Machine field are what a maintainer
    // groups by — an eight-card run filed as "RTX 5090" is grouped with the single-card ones.
    const href = url({
      repoUrl: 'https://github.com/MrZoller/headroom',
      scenarioUrl: 'https://example.test/?d=rtx-5090&n=8',
      deviceName: 'GeForce RTX 5090',
      deviceCount: 8,
      modelName: 'Llama 3.1 8B Instruct',
      comparisons: compare(parseLlamaBench(JSON_OUTPUT), prediction()),
    });

    expect(bodyOf(href)).toContain('8x GeForce RTX 5090');
    expect(new URL(href).searchParams.get('title')).toContain('8x GeForce RTX 5090');
  });

  it('writes only the comparable pairs into the table', () => {
    /**
     * A row the panel has just called "not comparable" carries a percentage that is a difference
     * between two *jobs*. Writing it into the issue strips the explanation and leaves a number that
     * reads as evidence — which is how a bad data point enters the record and is never questioned.
     */
    const mixed = compare(parseLlamaBench(JSON_OUTPUT), prediction({ promptTokens: 16384 }));
    expect(mixed.some((c) => c.mismatch !== undefined)).toBe(true);
    expect(hasSubmittablePair(mixed.filter((c) => c.mismatch !== undefined))).toBe(false);

    const body = bodyOf(
      url({
        repoUrl: 'https://github.com/MrZoller/headroom',
        scenarioUrl: 'https://example.test/?d=rtx-5090',
        deviceName: 'GeForce RTX 5090',
        deviceCount: 1,
        modelName: 'Llama 3.1 8B Instruct',
        comparisons: mixed,
      })
    );
    // One data row per comparable pair, and none for the marked one.
    const dataRows = body.split('\n').filter((line) => /^\| (prefill|decode) \|/.test(line));
    expect(dataRows).toHaveLength(mixed.filter((c) => c.mismatch === undefined).length);
  });

  it('is a plain issues/new link with nothing else in it', () => {
    // No backend and no telemetry: the reader opens a GitHub form and sees exactly what they are
    // about to post. Same shape the weekly catalog refresh already proved out.
    const href = url();
    expect(href.startsWith('https://github.com/MrZoller/headroom/issues/new?')).toBe(true);
    expect(new URL(href).searchParams.get('labels')).toBe('calibration');
    expect(new URL(href).searchParams.get('title')).toContain('Llama 3.1 8B Instruct');
  });
});

import { describe, expect, it } from 'vitest';
import {
  CALIBRATION_BAND,
  compare,
  hasSubmittablePair,
  parseLlamaBench,
  submissionUrl,
  type Prediction,
} from './calibrate';

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
  });

  it('will not read a markdown paste as confirming a non-default cache', () => {
    // A paste carrying no cache precision must not sail past a Q8 prediction — unverifiable is not
    // the same as matching, and markdown is the default output.
    expect(compare(parseLlamaBench(MARKDOWN), prediction({ kvType: 'q8_0' }))[0].mismatch).toMatch(
      /without a stated cache precision/
    );
    // And an f16 prediction is unaffected, since nothing about it is unverifiable.
    expect(compare(parseLlamaBench(MARKDOWN), prediction())[0].mismatch).toBeUndefined();
  });

  it('treats a markdown paste as unverifiable even when it prints the cache columns', () => {
    // The limitation is the *parser*, not the format, and the comment here claimed the opposite
    // until Codex caught it on #175: `parseMarkdown` has no branch for `type_k`/`type_v` at all, so
    // it never yields `kvTypes` whether or not llama-bench printed them. That matters because the
    // panel's own measure command passes `-ctk q8_0 -ctv q8_0 -o md`, which makes it print them —
    // so the reader who follows the panel exactly is told their correct run looks like f16.
    //
    // Reading those columns is #181. When it lands this assertion inverts, which is the point of
    // pinning it: the current behaviour is a known gap rather than an unexamined one.
    //
    // **And `ngl` goes with them, which this fixture is what found** (Codex, on #175).
    // `parseMarkdown` locates `ngl` by *position* — the cell before `test` — because a bare integer
    // has no distinctive shape. The cache columns sit between the two in llama-bench's own layout,
    // so the cell before `test` is `type_v` and the layer count is lost on exactly the output the
    // panel's command produces. `describeMismatch` then skips the layer check, and an offloaded run
    // compares clean against a resident prediction. Same root as #181 — the parser does not read the
    // header row — and filed there.
    const withCacheColumns = `
| model                  |     params | backend | ngl | type_k | type_v |   test |          t/s |
| ---------------------- | ---------: | ------- | --: | ------ | ------ | -----: | -----------: |
| llama 8B Q4_K - Medium |     8.03 B | CUDA    |  33 | q8_0   | q8_0   |  tg512 | 45.67 ± 0.12 |
`;
    const parsed = parseLlamaBench(withCacheColumns);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].kvTypes).toBeUndefined();
    expect(parsed[0].gpuLayers).toBeUndefined();
    expect(compare(parsed, prediction({ kvType: 'q8_0' }))[0].mismatch).toMatch(
      /without a stated cache precision/
    );
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

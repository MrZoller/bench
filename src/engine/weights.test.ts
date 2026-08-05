import { describe, expect, it } from 'vitest';
import {
  activeWeightBytes,
  effectiveActiveParams,
  fixedParams,
  outputProjectionParams,
  prefillComputeParams,
  weightBreakdown,
  weightBytes,
} from './weights';
import {
  DEEPSEEK_V3,
  GEMMA_3_12B,
  GPT_OSS_120B,
  GPT_OSS_20B,
  LLAMA_31_8B,
  LLAMA_32_3B,
  QWEN3_32B,
} from './fixtures';
import { getQuant } from '@/data/quants';
import { GIB } from './types';
import type { ModelSpec } from './types';

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
   * bytes-per-token by ~1.8x — and decode throughput is inversely proportional to it.
   *
   * The dense term is `activeDenseParams`, not `totalParams - expertParams`: the latter is the
   * residency figure and re-adds a 0.58B embedding table that an untied model reads one row of.
   *
   * Computed by hand from the fixture rather than through weightBreakdown, so a regression in
   * the split cannot quietly move the expectation with it.
   */
  it('charges gpt-oss active weights per component, not at the blended rate', () => {
    const activeExperts = GPT_OSS_120B.expertParams * (4 / 128);
    const expected = (GPT_OSS_120B.activeDenseParams * 16 + activeExperts * 4.25) / 8;

    const actual = activeWeightBytes(GPT_OSS_120B, getQuant('mxfp4'));
    expect(actual).toBeCloseTo(expected, -6);
    expect(actual / 1e9).toBeCloseTo(5.02, 1);

    // What the blended shortcut would have claimed.
    const blended = (GPT_OSS_120B.activeParams * 4.4651) / 8;
    expect(actual / blended).toBeGreaterThan(1.7);
  });

  /**
   * A dense model does *not* read every weight it stores. Llama 3.1 keeps `lm_head` as its own
   * tensor, so the 128256 x 4096 input embedding is a row lookup at decode — 0.32 GB of the
   * 4.87 GB file that never crosses the bus per step.
   */
  it('skips the untied embedding table for a dense model', () => {
    const quant = getQuant('q4_k_m');
    const embeddingBytes = (LLAMA_31_8B.vocabSize * LLAMA_31_8B.hiddenSize * quant.bpw) / 8;

    expect(activeWeightBytes(LLAMA_31_8B, quant)).toBeCloseTo(
      weightBytes(LLAMA_31_8B, quant) - embeddingBytes,
      0
    );
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

/**
 * The tied-embedding and vision-tower corrections, exercised through the engine rather than
 * asserted as catalog data.
 *
 * These are the guards that fail if `activeDenseParams` is ever rederived from
 * `tie_word_embeddings` — a key Gemma 3 does not set — or if a vision tower is folded back
 * into the per-token count.
 */
describe('per-token basis excludes what a text token does not read', () => {
  const quant = getQuant('q4_k_m');

  it('keeps a tied embedding table in the decode read', () => {
    // Tied means the table *is* the output projection: a full vocab matmul every step.
    const embedding = GEMMA_3_12B.vocabSize * GEMMA_3_12B.hiddenSize;
    const withoutTable = activeWeightBytes(GEMMA_3_12B, quant) - (embedding * quant.bpw) / 8;

    expect(activeWeightBytes(GEMMA_3_12B, quant)).toBeCloseTo(
      ((GEMMA_3_12B.totalParams - GEMMA_3_12B.nonLanguageParams!) * quant.bpw) / 8,
      0
    );
    // Subtracting it, as the untied case correctly does, would drop ~0.6 GB of real traffic.
    expect(activeWeightBytes(GEMMA_3_12B, quant) - withoutTable).toBeGreaterThan(0.5e9);
  });

  it('excludes the vision tower from the decode read but not from memory', () => {
    const towerBytes = (GEMMA_3_12B.nonLanguageParams! * quant.bpw) / 8;

    expect(weightBytes(GEMMA_3_12B, quant) - activeWeightBytes(GEMMA_3_12B, quant)).toBeCloseTo(
      towerBytes,
      0
    );
  });

  /**
   * The distinction stated directly, rather than through a comparison with `activeParams`.
   *
   * That comparison used to show an inversion — a tied model reading *above* its published
   * figure — and stopped meaning anything once dense models began reporting `activeParams`
   * equal to `totalParams`. Asserting against the published convention made the test hostage
   * to a convention; asserting the physical rule does not.
   */
  it('keeps the embedding in the per-token basis exactly when it is tied', () => {
    const embedding = (m: typeof LLAMA_31_8B) => m.vocabSize * m.hiddenSize;

    // Tied: the table is the output projection, so it stays — only the vision tower comes out.
    expect(GEMMA_3_12B.activeDenseParams).toBe(
      GEMMA_3_12B.totalParams - GEMMA_3_12B.nonLanguageParams!
    );
    expect(GEMMA_3_12B.activeDenseParams).toBeGreaterThan(embedding(GEMMA_3_12B));

    // Untied: `lm_head` is its own tensor, so the input table is a row lookup and comes out.
    expect(LLAMA_31_8B.activeDenseParams).toBe(LLAMA_31_8B.totalParams - embedding(LLAMA_31_8B));
  });

  /**
   * The fallback for an MoE model missing its expert counts has to reconstruct the per-token
   * fraction from the same dense basis `activeParams` was built on. Subtracting the full dense
   * half instead implied 9.5% of gpt-oss-20b's experts per token where its config says 4 of 32.
   *
   * DeepSeek cannot show this — its round-trip is exact either way — so the assertion is on
   * gpt-oss, where the embedding is a third of the dense half.
   */
  it('recovers the true per-token expert fraction without expert counts', () => {
    const withoutCounts = { ...GPT_OSS_20B, experts: undefined };
    const routed = effectiveActiveParams(withoutCounts, 1) - GPT_OSS_20B.activeDenseParams;

    expect(routed / GPT_OSS_20B.expertParams).toBeCloseTo(4 / 32, 3);
  });
});

/**
 * What repeats and what does not (#165).
 *
 * A per-layer figure divides `layerBytes`, never `totalBytes`. The two are the same question only
 * for a model whose vocabulary is small against its depth, which is not the shape of anything
 * people run at the small end — and the consumer of the difference is a layer *count*, which ends
 * up in a shell command.
 */
describe('the fixed tensors are separated from the repeating stack', () => {
  const quant = getQuant('q4_k_m');

  it('adds back up, whatever the model', () => {
    for (const model of [LLAMA_31_8B, LLAMA_32_3B, QWEN3_32B, GEMMA_3_12B, GPT_OSS_120B]) {
      const { fixedBytes, layerBytes, totalBytes, denseBytes } = weightBreakdown(model, quant);

      expect(fixedBytes + layerBytes, model.name).toBeCloseTo(totalBytes, 0);
      expect(fixedBytes, model.name).toBeGreaterThan(0);
      // Fixed tensors are never routed experts, so they cannot exceed the dense half.
      expect(fixedBytes, model.name).toBeLessThanOrEqual(denseBytes);
    }
  });

  it('counts one vocabulary table when it is tied and two when it is not', () => {
    // The same correction `activeDenseParams` makes, asked for a different reason: there the
    // question is what a token reads, here it is what a device holds whole. A tied table is read
    // every step *and* occupies exactly one device, so the two answers differ on the tie.
    const table = (m: typeof LLAMA_31_8B) => m.vocabSize * m.hiddenSize;

    expect(fixedParams(LLAMA_32_3B)).toBe(table(LLAMA_32_3B));
    expect(fixedParams(LLAMA_31_8B)).toBe(2 * table(LLAMA_31_8B));
    // And a tower is resident without being in any layer, so it is fixed too.
    expect(fixedParams(GEMMA_3_12B)).toBe(table(GEMMA_3_12B) + GEMMA_3_12B.nonLanguageParams!);
  });

  it('is 12% of Llama 3.2 3B, which is a layer or two of a 28-layer count', () => {
    // The figure #165 was filed on: 128,256 x 3,072 tied against 3.21B total. Charging it evenly
    // across the layers makes each "layer" 14% heavier than a layer, so a byte budget that holds
    // 28 of the real thing is read as holding 24.
    const { fixedBytes, layerBytes, totalBytes } = weightBreakdown(LLAMA_32_3B, quant);

    expect(fixedBytes / totalBytes).toBeCloseTo(0.123, 3);
    expect(totalBytes / LLAMA_32_3B.layers / (layerBytes / LLAMA_32_3B.layers)).toBeCloseTo(
      1.14,
      2
    );
  });

  it('leaves every byte total exactly where it was', () => {
    // The split is a partition, not a re-sizing. Nothing that reads a weight total may move.
    expect(weightBytes(LLAMA_31_8B, getQuant('bf16'))).toBe(LLAMA_31_8B.totalParams * 2);
    for (const model of [LLAMA_32_3B, GEMMA_3_12B, GPT_OSS_120B]) {
      const b = weightBreakdown(model, quant);
      expect(b.totalBytes, model.name).toBe(b.expertBytes + b.denseBytes);
    }
  });

  /**
   * The three placements, and the identity a placement seeds bins from (#182).
   *
   * `fixedBytes` is a file figure and stays one; what upstream disproves is the premise it used to
   * carry, that the tensors in it go to *one* device. They go to three: the input embedding to the
   * host, the output projection to the last `-ts` device, a tower to the first GPU. The sum has to
   * survive that, because `layerBytes` is still `totalBytes - fixedBytes` and a placement that seeds
   * bins from components which do not add up would lose or double-count bytes silently.
   */
  describe('and the fixed block is three placements rather than one', () => {
    it('adds back up to the fixed block, on every model and every format', () => {
      for (const model of [LLAMA_31_8B, LLAMA_32_3B, QWEN3_32B, GEMMA_3_12B, GPT_OSS_120B]) {
        for (const format of ['bf16', 'q8_0', 'q4_k_m', 'q3_k_m', 'mxfp4']) {
          const b = weightBreakdown(model, getQuant(format));
          const where = `${model.name} ${format}`;
          expect(b.hostResidentBytes + b.outputBytes + b.towerBytes, where).toBeCloseTo(
            b.fixedBytes,
            0
          );
          for (const part of [b.hostResidentBytes, b.outputBytes, b.towerBytes]) {
            expect(part, where).toBeGreaterThanOrEqual(0);
          }
        }
      }
    });

    it('keeps the identity when the defensive clamp fires', () => {
      /**
       * `fixedBytes` is clamped to `denseBytes` so `layerBytes` cannot go negative, and the clamp
       * is documented as defensive rather than reachable — no catalog row's vocabulary and tower
       * outweigh its own dense half. Unreachable is not the same as untested: the three components
       * are scaled by whatever the clamp took, and a clamp that quietly broke their sum would be
       * worse than the negative `layerBytes` it exists to prevent.
       *
       * Synthesised rather than hunted for (#197), since the whole point is a shape the catalog
       * does not have: a 4,096-hidden model with a two-million-token vocabulary, which is two
       * tables larger than the model.
       */
      const absurd: ModelSpec = {
        ...LLAMA_31_8B,
        id: 'test/clamped',
        vocabSize: 2_000_000,
        hiddenSize: 4_096,
        totalParams: 8_030_261_248,
        tiedEmbeddings: false,
      };
      const b = weightBreakdown(absurd, quant);

      expect(b.fixedBytes).toBe(b.denseBytes);
      expect(b.fixedBytes).toBeLessThan((fixedParams(absurd) * quant.bpw) / 8);
      expect(b.hostResidentBytes + b.outputBytes + b.towerBytes).toBeCloseTo(b.fixedBytes, 0);
      expect(b.layerBytes).toBe(0);
    });

    it('charges a tied model no host-resident table, because the card holds one too', () => {
      // llama.cpp materialises a tied table twice — the host's copy and a `TENSOR_DUPLICATED`
      // output on the last GPU — and adds the duplicate's bytes to `size_data` itself. The file
      // carries one table and the card holds one, so nothing comes off the card's budget. An
      // untied model has two in the file and the card holds one, which is the whole over-charge.
      const table = (m: typeof LLAMA_31_8B) => (m.vocabSize * m.hiddenSize * quant.bpw) / 8;

      const tied = weightBreakdown(LLAMA_32_3B, quant);
      expect(tied.hostResidentBytes).toBe(0);
      expect(tied.outputBytes).toBeCloseTo(table(LLAMA_32_3B), 0);

      const untied = weightBreakdown(LLAMA_31_8B, quant);
      expect(untied.hostResidentBytes).toBeCloseTo(table(LLAMA_31_8B), 0);
      expect(untied.outputBytes).toBeCloseTo(table(LLAMA_31_8B), 0);

      // 7.6% of Qwen3 8B's file and 6.5% of Llama 3.1 8B's is what a discrete GPU stops being
      // charged for — one table, not the whole fixed block.
      expect(untied.hostResidentBytes / untied.totalBytes).toBeCloseTo(0.065, 2);
    });

    it('separates a tower from the table, since they sit at opposite ends of the rig', () => {
      // Gemma 3 12B is tied *and* multimodal, so its fixed block is a table and a tower with
      // nothing host-resident — and the two go to the last card and the first respectively.
      const b = weightBreakdown(GEMMA_3_12B, quant);

      expect(b.hostResidentBytes).toBe(0);
      expect(b.towerBytes).toBeCloseTo((GEMMA_3_12B.nonLanguageParams! * quant.bpw) / 8, 0);
      expect(b.outputBytes).toBeCloseTo(
        (GEMMA_3_12B.vocabSize * GEMMA_3_12B.hiddenSize * quant.bpw) / 8,
        0
      );
      expect(b.towerBytes).toBeGreaterThan(0);
      expect(b.outputBytes).toBeGreaterThan(b.towerBytes);
    });

    it('gives a text-only model no tower to place', () => {
      for (const model of [LLAMA_31_8B, LLAMA_32_3B, QWEN3_32B]) {
        expect(weightBreakdown(model, quant).towerBytes, model.name).toBe(0);
      }
    });
  });
});

/**
 * Prefill computes logits for the positions that need them — one, for generation — not for
 * every prompt token. The output projection therefore sits outside the prefill FLOPs basis
 * while staying inside the decode one, which is the only place the two bases diverge.
 */
describe('prefill excludes the output projection', () => {
  it.each([
    ['untied', GPT_OSS_20B],
    ['tied', GEMMA_3_12B],
  ])('drops exactly one vocab x hidden matmul for a %s model', (_label, model) => {
    const gap = effectiveActiveParams(model, 1) - prefillComputeParams(model);
    expect(gap).toBeCloseTo(model.vocabSize * model.hiddenSize, -6);
  });

  it('is a material share of a wide-vocabulary model', () => {
    // gpt-oss carries a 201K vocabulary against 3.6B active: 16% of the linear term.
    const share =
      (effectiveActiveParams(GPT_OSS_20B, 1) - prefillComputeParams(GPT_OSS_20B)) /
      effectiveActiveParams(GPT_OSS_20B, 1);
    expect(share).toBeGreaterThan(0.15);
  });
});

/**
 * The output projection is per-request work, not per-token work, and the difference only shows
 * up at the ends: negligible on a long prompt, most of the pass on a one-token one.
 */
describe('output projection is charged once per prefill', () => {
  const flops = (model: typeof GPT_OSS_20B, promptTokens: number) =>
    2 * (prefillComputeParams(model) * promptTokens + outputProjectionParams(model));

  it('is a sixth of a single-token prompt rather than vanishing from it', () => {
    const withProjection = flops(GPT_OSS_20B, 1);
    const withoutIt = 2 * prefillComputeParams(GPT_OSS_20B) * 1;

    // 0.58B projection within a 3.60B one-token pass: omitting it loses 16% of the work. The
    // expert term is most of the remainder, which is why this is a sixth and not a half.
    expect(1 - withoutIt / withProjection).toBeCloseTo(0.16, 2);
  });

  it('is a rounding error on a long prompt', () => {
    const perTokenShare =
      outputProjectionParams(GPT_OSS_20B) / (prefillComputeParams(GPT_OSS_20B) * 4096);
    expect(perTokenShare).toBeLessThan(0.001);
  });

  it('never charges it per token, at any prompt length', () => {
    // Doubling the prompt must add exactly the per-token term twice over, not the projection too.
    const delta = flops(GPT_OSS_20B, 2048) - flops(GPT_OSS_20B, 1024);
    expect(delta).toBe(2 * prefillComputeParams(GPT_OSS_20B) * 1024);
  });
});

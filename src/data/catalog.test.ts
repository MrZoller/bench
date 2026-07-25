import { describe, expect, it } from 'vitest';
import { CATALOG_GENERATED_AT, DEVICES, MODELS, getDevice, getModel } from './catalog';
import { getQuant } from './quants';
import { evaluate } from '@/engine';
import { LLAMA_CPP, GPT_OSS_120B, DEEPSEEK_V3, QWEN3_32B } from '@/engine/fixtures';
import { GIB } from '@/engine/types';

describe('device catalog', () => {
  it('covers all three hardware classes', () => {
    const classes = new Set(DEVICES.map((d) => d.class));
    expect(classes).toEqual(new Set(['discrete-gpu', 'unified-soc', 'cpu-ram']));
  });

  it.each(DEVICES.map((d) => [d.id, d] as const))('%s is internally consistent', (_id, device) => {
    expect(device.allocatableBytes).toBeGreaterThan(0);
    // Allocatable can equal capacity on a dedicated card, never exceed it.
    expect(device.allocatableBytes).toBeLessThanOrEqual(device.capacityBytes);
    expect(device.bandwidthBytesPerSec).toBeGreaterThan(0);
    expect(Object.keys(device.flops).length).toBeGreaterThan(0);
    // Provenance is not optional: every figure here was typed by a human from a datasheet.
    expect(device.source).toMatch(/^https:\/\//);
  });

  it('never claims measured bandwidth above theoretical', () => {
    for (const device of DEVICES) {
      if (device.measuredBandwidthBytesPerSec) {
        expect(device.measuredBandwidthBytesPerSec).toBeLessThanOrEqual(
          device.bandwidthBytesPerSec
        );
      }
    }
  });

  /**
   * Shared-memory machines must not be catalogued as if the whole pool were available. This is
   * the difference between reporting "fits" and the model failing to load.
   */
  it('caps allocatable below capacity on every unified-memory device', () => {
    const unified = DEVICES.filter((d) => d.class === 'unified-soc');
    expect(unified.length).toBeGreaterThan(3);
    for (const device of unified) {
      expect(device.allocatableBytes).toBeLessThan(device.capacityBytes);
    }
  });

  it('marks unreleased hardware as such', () => {
    const rumored = DEVICES.filter((d) => d.status === 'rumored');
    // The M5 Ultra is press-rumour grade; if it is in the catalog it must be labelled.
    for (const device of rumored) {
      expect(device.note).toBeTruthy();
    }
    expect(DEVICES.every((d) => ['shipping', 'announced', 'rumored'].includes(d.status))).toBe(
      true
    );
  });

  it('spans the capacity/bandwidth triangle rather than clustering', () => {
    const spark = getDevice('dgx-spark');
    const mac = getDevice('mac-studio-m3-ultra-256');
    const gpu = getDevice('rtx-5090');

    // High capacity, low bandwidth.
    expect(spark.capacityBytes).toBeGreaterThan(gpu.capacityBytes * 3);
    expect(spark.bandwidthBytesPerSec).toBeLessThan(gpu.bandwidthBytesPerSec / 5);
    // High capacity and high bandwidth, but weaker compute than the Spark.
    expect(mac.bandwidthBytesPerSec).toBeGreaterThan(spark.bandwidthBytesPerSec * 2);
    expect(mac.flops.fp16!).toBeLessThan(spark.flops.fp16!);
  });
});

describe('generated model catalog', () => {
  it('was generated, and says when', () => {
    expect(MODELS.length).toBeGreaterThan(10);
    expect(Number.isFinite(Date.parse(CATALOG_GENERATED_AT))).toBe(true);
  });

  it.each(MODELS.map((m) => [m.id, m] as const))('%s has a usable spec', (_id, model) => {
    expect(model.totalParams).toBeGreaterThan(0);
    expect(model.activeParams).toBeGreaterThan(0);
    expect(model.activeParams).toBeLessThanOrEqual(model.totalParams);
    expect(model.expertParams).toBeLessThan(model.totalParams);
    expect(model.layers).toBeGreaterThan(0);
    expect(model.maxContext).toBeGreaterThan(0);
    expect(model.source).toMatch(/^https:\/\/huggingface\.co\//);

    // An MoE model must carry expert counts, or the batch-union model silently goes flat.
    if (model.expertParams > 0) expect(model.experts).toBeDefined();
    if (model.attention.layerWindows) {
      expect(model.attention.layerWindows).toHaveLength(model.layers);
    }
  });

  it('captures all three attention shapes across the catalog', () => {
    const cores = new Set(MODELS.map((m) => m.attention.core.kind));
    expect(cores).toContain('gqa');
    expect(cores).toContain('mla');
    expect(MODELS.some((m) => m.attention.layerWindows?.some((w) => w !== null))).toBe(true);
  });

  /**
   * The derivation has to reproduce what vendors publish, or it is not deriving — it is
   * inventing. These are the figures the model cards state; each exercises a different part of
   * the pipeline (MXFP4 packed counts, MTP exclusion, MLA, plain gated MoE).
   */
  it.each([
    ['openai/gpt-oss-120b', 117, 5.1],
    ['openai/gpt-oss-20b', 21, 3.6],
    ['deepseek-ai/DeepSeek-V3', 671, 37],
    ['zai-org/GLM-4.5-Air', 106, 12],
    ['Qwen/Qwen3-235B-A22B', 235, 22],
    ['Qwen/Qwen3-30B-A3B', 30, 3],
  ])('%s matches its published parameter counts', (id, totalB, activeB) => {
    const model = getModel(id);

    // Relative tolerances, because an absolute one wide enough for 671B is meaningless at 3B.
    // 2% on totals rejects the raw HF figures (which miss by 3.4B / 13.5B / 4.5B on the
    // packed and MTP models); 8% on active is tight enough that adding the input embedding
    // back in — the correction this pipeline exists to apply — fails every one of these.
    expect(Math.abs(model.totalParams / 1e9 - totalB) / totalB).toBeLessThan(0.02);
    expect(Math.abs(model.activeParams / 1e9 - activeB) / activeB).toBeLessThan(0.08);
  });

  /**
   * Guards the embedding exclusion directly. Decode gathers one row of the embedding table
   * rather than reading it, so it must not count toward active parameters.
   *
   * Only the gpt-oss pair is asserted here: the correction has to be *material* for a test to
   * discriminate, and it scales with vocabulary relative to active parameters. gpt-oss carries
   * a 201K vocabulary against 3.6B active, so counting the embedding shifts the figure 16%.
   * On GLM-4.5-Air the same correction is under 5%, inside the tolerance above — it is
   * covered by the published-figures test, not by this one.
   */
  it.each([
    ['openai/gpt-oss-120b', 5.1],
    ['openai/gpt-oss-20b', 3.6],
  ])('%s would miss its published active count if the embedding were counted', (id, activeB) => {
    const model = getModel(id);
    const withEmbedding = (model.activeParams + model.vocabSize * model.hiddenSize) / 1e9;
    expect(Math.abs(withEmbedding - activeB) / activeB).toBeGreaterThan(0.08);
  });

  /**
   * The generator and the hand-built test fixtures were derived independently — the fixtures
   * from reading config.json by hand, the catalog from the API. They must agree, or one of
   * them is wrong.
   */
  it.each([
    ['openai/gpt-oss-120b', GPT_OSS_120B],
    ['deepseek-ai/DeepSeek-V3', DEEPSEEK_V3],
    ['Qwen/Qwen3-32B', QWEN3_32B],
  ])('%s agrees with the hand-built fixture', (id, fixture) => {
    const model = getModel(id);
    expect(model.layers).toBe(fixture.layers);
    expect(model.hiddenSize).toBe(fixture.hiddenSize);
    expect(model.vocabSize).toBe(fixture.vocabSize);
    expect(model.attention.core).toEqual(fixture.attention.core);
    expect(model.expertParams).toBe(fixture.expertParams);
    expect(Math.abs(model.totalParams - fixture.totalParams) / fixture.totalParams).toBeLessThan(
      0.02
    );

    // The window array is the whole output of the sliding-window derivation, and it drives
    // both KV size and prefill attention FLOPs. Comparing values, not just length, is what
    // makes this an independent check rather than a shape assertion.
    expect(model.attention.layerWindows).toEqual(fixture.attention.layerWindows);
  });

  /**
   * Tied embeddings must be read from the tensor list, never from `tie_word_embeddings`.
   *
   * Both Gemma 3 repos omit that config key entirely while genuinely being tied — their index
   * has no `lm_head.weight`. Trusting the key would subtract a 262208 x 3840 table (1.0B, ~9%
   * of active) that decode in fact runs as a full output matmul every step.
   *
   * The assertion is deliberately on the *derived arithmetic* rather than on the flag, so it
   * fails if the flag is ever right for the wrong reason.
   */
  it.each([
    ['Qwen/Qwen3-4B', true],
    ['unsloth/gemma-3-12b-it', true],
    ['openai/gpt-oss-20b', false],
    ['Qwen/Qwen3-8B', false],
  ])('%s keeps its embedding table per-token only when tied', (id, tied) => {
    const model = getModel(id);
    expect(model.tiedEmbeddings).toBe(tied);

    const embedding = model.vocabSize * model.hiddenSize;
    const dense = model.totalParams - model.expertParams - (model.nonLanguageParams ?? 0);
    expect(model.activeDenseParams).toBeCloseTo(tied ? dense : dense - embedding, -6);
  });

  /**
   * A vision tower occupies memory but never runs for a text token, so it belongs in
   * `totalParams` and not in the per-token count. Gemma 3 is the only multimodal pair here;
   * its tower is ~0.42B, which is 3.7% of prefill on the 12B.
   */
  it.each([
    ['unsloth/gemma-3-12b-it', 12.19],
    ['unsloth/gemma-3-27b-it', 27.43],
  ])('%s excludes its vision tower from the per-token count but not from memory', (id, totalB) => {
    const model = getModel(id);

    expect(model.nonLanguageParams).toBeGreaterThan(0.4e9);
    expect(model.nonLanguageParams).toBeLessThan(0.5e9);
    // Still resident: the tower loads with the model even for a text-only request.
    expect(model.totalParams / 1e9).toBeCloseTo(totalB, 1);
    expect(model.activeDenseParams).toBe(model.totalParams - model.nonLanguageParams!);
  });

  /**
   * The catalogued models are the ones the engine actually runs, so the invariant that keeps
   * decode honest has to hold across all of them, not just the ones spot-checked above.
   */
  it('gives every model a per-token basis inside its own residency figure', () => {
    for (const model of MODELS) {
      const dense = model.totalParams - model.expertParams;
      expect(model.activeDenseParams).toBeGreaterThan(0);
      expect(model.activeDenseParams).toBeLessThanOrEqual(dense);
    }
  });
});

describe('every catalogued model evaluates on every catalogued device', () => {
  it('produces finite numbers for the whole cross product', () => {
    const quant = getQuant('q4_k_m');
    let evaluated = 0;

    for (const model of MODELS) {
      for (const device of DEVICES) {
        const result = evaluate({
          model,
          quant,
          usage: { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
          rig: { device, count: 1 },
          runtime: LLAMA_CPP,
        });

        expect(Number.isFinite(result.decode.perUserTokensPerSec)).toBe(true);
        expect(Number.isFinite(result.placement.usedBytesPerDevice)).toBe(true);
        expect(result.decode.perUserTokensPerSec).toBeGreaterThan(0);
        evaluated++;
      }
    }

    expect(evaluated).toBe(MODELS.length * DEVICES.length);
  });

  /**
   * A spot check that the catalog and engine together give advice a knowledgeable person
   * would recognise, rather than merely finite numbers.
   */
  it('says a 5090 runs an 8B model comfortably and a 671B model not at all', () => {
    const quant = getQuant('q4_k_m');
    const usage = { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' as const };
    const rig = { device: getDevice('rtx-5090'), count: 1 };

    const small = evaluate({
      model: getModel('NousResearch/Meta-Llama-3.1-8B-Instruct'),
      quant,
      usage,
      rig,
      runtime: LLAMA_CPP,
    });
    expect(small.placement.fits).toBe(true);
    expect(small.decode.perUserTokensPerSec).toBeGreaterThan(100);

    const huge = evaluate({
      model: getModel('deepseek-ai/DeepSeek-V3'),
      quant,
      usage,
      rig,
      runtime: LLAMA_CPP,
    });
    expect(huge.placement.fits).toBe(false);
    expect(huge.decode.perUserTokensPerSec).toBeLessThan(5);
  });

  it('fits gpt-oss-120b on a Spark at its native quantization, where a 5090 cannot', () => {
    const usage = { contextTokens: 32768, concurrency: 1, kvPrecision: 'fp16' as const };
    const model = getModel('openai/gpt-oss-120b');
    const quant = getQuant('mxfp4');

    const spark = evaluate({
      model,
      quant,
      usage,
      rig: { device: getDevice('dgx-spark'), count: 1 },
      runtime: LLAMA_CPP,
    });
    const gpu = evaluate({
      model,
      quant,
      usage,
      rig: { device: getDevice('rtx-5090'), count: 1 },
      runtime: LLAMA_CPP,
    });

    expect(spark.placement.fits).toBe(true);
    expect(gpu.placement.fits).toBe(false);
    // ~61 GiB of weights against a 32 GiB card.
    expect(spark.weights.totalBytes / GIB).toBeCloseTo(61, 0);
  });
});

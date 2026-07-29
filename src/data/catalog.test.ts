import { describe, expect, it } from 'vitest';
import {
  CATALOG_GENERATED_AT,
  DEVICES,
  DEVICE_ID_ALIASES,
  MODELS,
  canonicalDeviceId,
  getDevice,
  getModel,
  modelsByPopularity,
  toDevice,
  type DeviceRow,
} from './catalog';
import devicesJson from './devices.json';
import { getQuant } from './quants';
import { evaluate } from '@/engine';
import { LLAMA_CPP, GPT_OSS_120B, DEEPSEEK_V3, QWEN3_32B } from '@/engine/fixtures';
import { GIB } from '@/engine/types';
import { maxAllocatablePerDevice, raisingCeilingWouldHelp } from '@/engine/placement';

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

  /**
   * The convention, not the ordering.
   *
   * The check here used to be that a measured figure never exceeded the theoretical one, which
   * passed just as happily with a measured figure present as absent — and one row had one. The
   * engine then applied `bandwidthEfficiency` and `CLASS_BANDWIDTH_UTILIZATION` on top of it, so
   * Strix Halo was charged the sticker-to-real gap twice and every one of its throughput figures
   * read 16.8% under the treatment the other 24 devices get. On a grid whose purpose is
   * ranking hardware against hardware, one row was being ranked on a different basis.
   *
   * Both calibration anchors are pinned against theoretical peaks — the DGX Spark at 273 GB/s and
   * the EPYC 9654 at 460.8 — so the constants *are* that gap. A measured figure in the catalog is
   * not a second effect to charge.
   */
  it('carries a theoretical peak for every device and a measured figure for none', () => {
    for (const device of DEVICES) {
      expect(device.bandwidthBytesPerSec).toBeGreaterThan(0);
      expect(device).not.toHaveProperty('measuredBandwidthBytesPerSec');
    }
  });

  /**
   * Asserted against the raw rows, not only the loaded devices.
   *
   * `toDevice` no longer maps `measuredBandwidthGBs`, and the `as DeviceRow[]` cast tolerates
   * excess JSON properties — so a curator re-adding the key by hand passes both `tsc` and the
   * check above, and leaves a figure sitting in the catalog implying it is used. That is the
   * misleading provenance this convention exists to prevent, one step short of the arithmetic.
   */
  it('states no measured figure anywhere in the raw catalog either', () => {
    for (const row of devicesJson.devices as Record<string, unknown>[]) {
      const measured = Object.keys(row).filter((k) => /measured/i.test(k));
      expect(measured, `${String(row.id)} states ${measured.join(', ')}`).toEqual([]);
    }
  });

  /**
   * Named rather than swept, because it is the row that had the override and the one a future
   * curator is most likely to re-add it to: AMD rates the part at 256 GB/s and real workloads land
   * near 213, which is a tempting 17% to fold in. The note carries the provenance instead.
   */
  it('rates Strix Halo at the sticker, and says where the real figure went', () => {
    const strix = getDevice('ryzen-ai-max-395');

    expect(strix.bandwidthBytesPerSec).toBe(256 * 1e9);
    expect(strix.note).toMatch(/256 GB\/s/);
    expect(strix.note).toMatch(/213/);
    expect(strix.note).toMatch(/bandwidthEfficiency|CLASS_BANDWIDTH_UTILIZATION/);
  });

  /**
   * The other convention `devices.json` states in prose: every rate in a row is *dense*, so the
   * ladder doubles at each halving of element width — int8 and fp8 at 2x fp16, fp4 at 4x.
   *
   * Worth pinning because the derivation is per-vendor and per-generation prose (a sparse FP4
   * headline is an eighth of dense fp16 on Blackwell, a sparse FP8 one a quarter on Ada, and the
   * datacenter parts are transcribed rather than derived) and a curator applying the wrong step to
   * one dtype produces a row where only the ratios disagree. That is the shape of the #51 failure:
   * a stated convention and the catalog coming apart without either looking wrong on its own. What
   * this cannot see is a row halved consistently but the wrong number of times, which is why the
   * worked examples in `$comment-compute` name the headline figure they start from.
   */
  it('states every rate in a row on the same dense basis', () => {
    for (const device of DEVICES) {
      const { fp16, bf16, fp8, int8, fp4 } = device.flops;
      if (fp16 === undefined) continue;

      for (const [dtype, value, multiple] of [
        ['bf16', bf16, 1],
        ['fp8', fp8, 2],
        ['int8', int8, 2],
        ['fp4', fp4, 4],
      ] as const) {
        if (value === undefined) continue;
        expect(
          value / fp16,
          `${device.id} states ${dtype} at ${(value / fp16).toFixed(2)}x its fp16 rate`
        ).toBeCloseTo(multiple, 1);
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

/**
 * Coverage, which is a different property from accuracy and fails in a way accuracy checks cannot
 * see: a machine that is absent is not wrong about anything.
 *
 * Every assertion here is a measurement from #78 taken against the 25-row catalog. Three vendors'
 * consumer lines and the entire sub-$1000 tier were missing, so the answer for most of the audience
 * for a "will it run" tool was not an incomplete comparison but no row to select at all. Written as
 * properties of the catalog rather than as a list of expected ids, because the point is the shape of
 * the coverage and not which particular card satisfies it.
 */
describe('the catalog covers the hardware the audience owns', () => {
  const gpus = DEVICES.filter((d) => d.class === 'discrete-gpu');
  const priced = gpus.filter((d) => d.msrpUsd !== undefined);

  /**
   * The headline number from the issue: the cheapest catalogued GPU was the 5080 at $999, so every
   * question from below that price — which is most of them — had no hardware to ask it about.
   */
  it('prices a GPU below the $999 floor the catalog used to start at', () => {
    const cheapest = Math.min(...priced.map((d) => d.msrpUsd!));
    expect(cheapest).toBeLessThan(350);

    // A tier, not a token row. Six of the nine consumer NVIDIA/AMD/Intel rows sit under $999.
    expect(priced.filter((d) => d.msrpUsd! < 999).length).toBeGreaterThanOrEqual(6);
  });

  /**
   * A person running llama.cpp on ROCm or Vulkan against a Radeon, or on SYCL against an Arc, had
   * nothing to select: AMD appeared only as datacenter Instinct parts and Intel not at all.
   */
  it('offers a consumer card from all three GPU vendors', () => {
    // Containment rather than set equality: the claim is that each of the three is *present* in
    // this tier, and a fourth vendor arriving is not a reason for this to fail.
    const consumer = new Set(priced.filter((d) => d.msrpUsd! <= 1000).map((d) => d.vendor));
    for (const vendor of ['NVIDIA', 'AMD', 'Intel']) {
      expect(consumer, `no ${vendor} card at or under $1000`).toContain(vendor);
    }
  });

  it('catalogues Intel in both classes it competes in', () => {
    // `DEVICE_CLASSES` and the runtimes' `supports` checks never had an Intel problem — the rows
    // simply were not written. Xeon 6 with MRDIMM is the other half of the CPU inference story,
    // and without it `cpu-ram` read as an AMD-only technique.
    const classes = new Set(DEVICES.filter((d) => d.vendor === 'Intel').map((d) => d.class));
    expect(classes).toContain('discrete-gpu');
    expect(classes).toContain('cpu-ram');
  });

  it('does not make CPU inference look like an AMD technique', () => {
    const intel = DEVICES.filter((d) => d.class === 'cpu-ram' && d.vendor === 'Intel');
    const fastest = Math.max(...intel.map((d) => d.bandwidthBytesPerSec));

    // 12 channels of MRDIMM-8800 is 844.8 GB/s to the digit — exact arithmetic, like the EPYC rows.
    expect(fastest).toBe(844.8 * 1e9);
    // And well above the fastest EPYC row in the catalog (the 9755 at 576), which is the point: the
    // class had four rows and three vendors' worth of nothing, so high-bandwidth CPU inference read
    // as an AMD technique. Stated as a floor on the Intel figure rather than as "Intel is the
    // fastest cpu-ram row", because a 12-channel EPYC arriving later is not a failure of a claim
    // about Intel being here — the same containment reasoning as the vendor check above.
    expect(fastest).toBeGreaterThan(576 * 1e9);
  });

  /**
   * The 3060 12GB argument from the issue, checked against the engine rather than asserted: the
   * tier matters because it is where the *verdict* boundary sits, and a row that could not actually
   * run the model would be coverage in name only.
   *
   * One row has to clear it, not all of them. This tier is where most of the will-it-run questions
   * come from, so the next cards added to it are things like a 10 GiB Arc B570 at $219 or an 8 GiB
   * 5060 at $299 — correct rows that cannot hold a 12B, which needs 8.3 GiB of weights before any
   * cache. Swept with `every`, this guard would fail on exactly the coverage work it exists to
   * protect, and the only ways back to green would be deleting the row or lowering the bar.
   */
  it('has a machine under $400 that really runs a 12B at Q4_K_M', () => {
    const budget = gpus.filter((d) => (d.msrpUsd ?? Number.POSITIVE_INFINITY) < 400);
    expect(budget.length).toBeGreaterThan(0);

    const verdicts = budget.map((device) => {
      const result = evaluate({
        model: getModel('unsloth/gemma-3-12b-it'),
        quant: getQuant('q4_k_m'),
        usage: { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
        rig: { device, count: 1 },
        runtime: LLAMA_CPP,
      });

      return {
        id: device.id,
        fits: result.placement.fits,
        // Fast enough to be worth doing, not merely resident.
        tokensPerSec: Math.round(result.decode.perUserTokensPerSec),
      };
    });

    const capable = verdicts.filter((v) => v.fits && v.tokensPerSec > 15);
    expect(
      capable.length,
      `no sub-$400 row holds a 12B at 8K above 15 tok/s: ${JSON.stringify(verdicts)}`
    ).toBeGreaterThan(0);
  });

  /**
   * Every Apple row was a maxed configuration, so the machine most Mac owners have — and the
   * generation they are deciding whether to replace — were both missing.
   */
  it('covers Apple below 64 GiB, and back to the M1', () => {
    const apple = DEVICES.filter((d) => d.vendor === 'Apple');

    expect(apple.some((d) => d.capacityBytes <= 16 * GIB)).toBe(true);
    expect(apple.some((d) => /M1 /.test(d.name))).toBe(true);
    expect(apple.some((d) => /M2 Ultra/.test(d.name))).toBe(true);
    // Four memory tiers below the old 64 GiB floor would be one machine repeated; the spread is
    // what makes the class answerable.
    expect(new Set(apple.map((d) => d.bandwidthBytesPerSec)).size).toBeGreaterThanOrEqual(6);
  });

  /**
   * The Studio ships with an M4 Max and the only M4 Max row was a MacBook Pro — which is a
   * different machine, because the Studio's base part is the binned bandwidth bin.
   */
  it('separates the Studio M4 Max from the MacBook Pro one', () => {
    const studio = getDevice('mac-studio-m4-max-36');
    const laptop = getDevice('macbook-pro-m4-max-128');

    expect(studio.bandwidthBytesPerSec).toBe(410 * 1e9);
    expect(laptop.bandwidthBytesPerSec).toBe(546 * 1e9);
  });

  /**
   * **An Apple row's compute must be reproducible from a GPU core count the row itself states.**
   *
   * This is a convention test rather than a value test, and it exists because the same defect was
   * filed three times in one review round: a row identified by memory capacity while carrying the
   * compute of a GPU bin that capacity does not imply. The base MacBook Air at $999 has 8 GPU cores
   * and carried the 10-core rate; the 64 GiB M1 Max was offered as 24- and 32-core and carried the
   * 32-core rate; the 192 GiB M2 Ultra was offered as 60- and 76-core and carried the 76-core rate.
   * Each was a 25-33% overstatement of prefill on a machine somebody actually owns, and each was
   * invisible because no test connected the name to the number.
   *
   * Apple publishes no per-core figure, but within a generation it is constant — which is what makes
   * this checkable. `devices.json`'s own `$comment-compute` already states the rule ("Apple: FP32 at
   * 2x, from the per-GPU-core rate of that generation"); this asserts it.
   *
   * So every Apple row has to name its core count somewhere a reader will see it, and the arithmetic
   * has to come out. Rows whose capacity genuinely pins the bin (128 GiB only ever shipped with the
   * 40-core M4 Max) still state it, because "capacity implies it" is exactly the reasoning that was
   * wrong three times.
   */
  it('derives every Apple row from a GPU core count the row states', () => {
    // fp16 TFLOPS per GPU core, by generation. Constant within a generation; the M4 family runs
    // 0.85 across Air, Pro and Max, which is the property that makes the check meaningful.
    const perCore: Record<string, number> = { M1: 0.65, M2: 0.716, M3: 0.675, M4: 0.85, M5: 0.9 };
    const apple = DEVICES.filter((d) => d.vendor === 'Apple');
    expect(apple.length).toBeGreaterThan(8);

    for (const device of apple) {
      const stated = /(\d+)-core GPU/.exec(device.name) ?? /(\d+)-core GPU/.exec(device.note ?? '');
      expect(
        stated,
        `${device.id} states no GPU core count, so nothing connects "${device.name}" to its ${device.flops.fp16} TFLOPS`
      ).not.toBeNull();

      const generation = /\bM(\d)\b/.exec(device.name)?.[0];
      expect(generation, `${device.id} does not name an Apple generation`).toBeDefined();
      const rate = perCore[generation!];
      expect(rate, `no per-core rate recorded for ${generation}`).toBeDefined();

      const cores = Number(stated![1]);
      const expected = cores * rate;
      // `toDevice` converts the row's TFLOPS to FLOPS; the per-core rates above are TFLOPS.
      const actual = device.flops.fp16! / 1e12;
      expect(
        Math.abs(actual - expected) / expected,
        `${device.id} states ${cores} GPU cores, which is ${expected.toFixed(1)} TFLOPS at ${generation}'s ${rate}/core, but carries ${actual}`
      ).toBeLessThan(0.06);
    }
  });
});

/**
 * Coverage of the *model* catalog, which is the same property #78 established for the device one and
 * fails the same way: a model that is absent is not wrong about anything, and nothing in the repo
 * noticed. The weekly refresh re-derives every figure on every row and cannot see a row that was
 * never seeded, so the seed list sat unchanged from the day the catalog was built while every number
 * in it stayed seven days old.
 *
 * Every assertion here is a measurement from #77 against the 17-row catalog, written as a property of
 * the catalog rather than as a list of expected ids — the shape of the coverage is the point, and
 * which particular repo satisfies it will change again. What they have in common is that they were
 * all false before this list was re-probed.
 */
describe('the catalog covers the models people are choosing between', () => {
  const byId = new Map(MODELS.map((m) => [m.id, m]));
  const paramsB = (m: (typeof MODELS)[number]) => m.totalParams / 1e9;

  /**
   * The headline gap: the largest catalogued model was DeepSeek's 671B, so the top of the open-weight
   * range had no row — and it is exactly where the MLA-versus-naive gap the project exists to
   * demonstrate is widest, since a 1T MoE caches 61 layers of compressed latent rather than 61 layers
   * of 64 KV heads.
   */
  it('reaches the top of the open-weight range', () => {
    expect(Math.max(...MODELS.map(paramsB))).toBeGreaterThan(1000);
  });

  /**
   * And the 480B class between them, which is what a 512 GB Mac is actually bought for. Nothing sat
   * between 235B and 671B.
   */
  it('carries a model in the 400-600B class', () => {
    const between = MODELS.filter((m) => paramsB(m) > 400 && paramsB(m) < 600);
    expect(between.length).toBeGreaterThan(0);
    // MoE, specifically: a dense model at that size answers a question no catalogued machine can ask.
    expect(between.some((m) => m.expertParams > 0)).toBe(true);
  });

  /**
   * The other end, and most of the audience: the smallest row was Qwen3-4B, so anyone deciding what
   * an 8 GB card runs had one model to select and no way to compare it against anything.
   */
  it('answers the 8 GB question with more than one row', () => {
    expect(Math.min(...MODELS.map(paramsB))).toBeLessThan(3.5);
    // A tier rather than a token row: four models under 4.5B, from three publishers.
    const small = MODELS.filter((m) => paramsB(m) < 4.5);
    expect(small.length).toBeGreaterThanOrEqual(4);
    expect(new Set(small.map((m) => m.org)).size).toBeGreaterThanOrEqual(3);
  });

  /**
   * Five organisations shipping models people run locally had no row at all. Asserted as membership
   * rather than as a count, since the reason each one matters is different — Phi-4 is high traffic in
   * a size class the catalog barely covered, Granite is IBM's current generation, Command A+ is a
   * 219B sparse MoE, Seed-OSS is a 36B dense.
   *
   * NVIDIA is deliberately not in this list and is the interesting absence: every current Nemotron is
   * either a Mamba-2 hybrid or a per-block NAS export, and both are refused by the generator rather
   * than catalogued at 10x and 13x their real cache. See `NOT_SEEDED` in `scripts/build-catalog.ts`.
   */
  it('represents the publishers whose models people actually run', () => {
    const orgs = new Set(MODELS.map((m) => m.org));
    for (const org of ['Microsoft', 'IBM', 'Cohere', 'ByteDance', 'Moonshot AI', 'MiniMax']) {
      expect(orgs, `no model from ${org}`).toContain(org);
    }
  });

  /**
   * MLA at a scale someone owns hardware for.
   *
   * The family was represented only by 671B models, which is an argument about a cache nobody
   * reading the page can run. A 30B MLA MoE puts the same comparison on a laptop: GLM 4.7 Flash
   * caches 4.5 KiB/token against Qwen3-30B-A3B's 24.0 at the same active parameter count.
   */
  it('carries MLA at a size that fits on one consumer machine', () => {
    const smallMla = MODELS.filter((m) => m.attention.core.kind === 'mla' && paramsB(m) < 150);
    expect(smallMla.length).toBeGreaterThan(0);
    // Two sizes, not one, or "MLA is what the enormous models do" survives with an extra row.
    expect(new Set(smallMla.map((m) => Math.round(paramsB(m)))).size).toBeGreaterThan(1);
  });

  /**
   * The stale-sibling problem, which is the one a user cannot see: the catalog carried the older
   * member of six families and not the newer one, so picking the row that is there means picking the
   * one that has been superseded.
   *
   * The successor is checked against the *predecessor's own* release date rather than against a
   * hard-coded one, so this keeps meaning what it says as both rows move.
   */
  it.each([
    ['NousResearch/Meta-Llama-3.1-70B-Instruct', 'unsloth/Llama-3.3-70B-Instruct'],
    ['Qwen/Qwen3-4B', 'Qwen/Qwen3-4B-Instruct-2507'],
    ['Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-30B-A3B-Instruct-2507'],
    ['Qwen/Qwen3-235B-A22B', 'Qwen/Qwen3-235B-A22B-Instruct-2507'],
    ['deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-V3.1'],
    ['zai-org/GLM-4.5-Air', 'zai-org/GLM-4.7'],
    ['mistralai/Mistral-Small-24B-Instruct-2501', 'mistralai/Mistral-Small-4-119B-2603'],
  ])('carries %s and the model that has replaced it', (older, newer) => {
    const before = byId.get(older);
    const after = byId.get(newer);
    expect(before, `${older} left the catalog`).toBeDefined();
    expect(
      after,
      `${newer} is missing, so the stale sibling is the only one a user can pick`
    ).toBeDefined();

    // Both rows carry HF's own repo creation date, so this is a check that the "successor" really is
    // one rather than a name that reads newer.
    expect(Date.parse(after!.releasedAt!)).toBeGreaterThan(Date.parse(before!.releasedAt!));
  });

  /**
   * Deliberately *not* asserted here: that the newest row is within some number of months of
   * `CATALOG_GENERATED_AT`.
   *
   * It is the obvious way to state "the catalog has not aged", and it is a test that passes today and
   * fails on a date — on somebody else's unrelated pull request, for a reason their diff has nothing
   * to do with. It also barely discriminated: against the 17-row catalog the gap was 11.7 months,
   * because gpt-oss shipped two weeks after the newest seed and dragged the figure under any
   * threshold loose enough to be safe.
   *
   * Absence needs a mechanism rather than an assertion, and the mechanism is
   * `reportSeedCandidates()` in `scripts/build-catalog.ts`: every refresh ends by asking the hub what
   * the field is downloading and printing whatever this list neither carries nor has written down a
   * reason for. That puts the evidence in front of a person weekly without failing anyone's build.
   */
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
   *
   * GLM 4.7 is here because it was the one row that stopped reconciling and nothing failed: it
   * shipped 35.06B active against Z.ai's stated 32B — 9.6% out, on a row whose own note quotes
   * "355B-A32B" into the control that renders the derived figure. The cause was its published total
   * standing in for a measured one: 355B is 2.2B above the sum of the architecture's own tensors, and
   * `denseParams` is `totalParams - expertParams` with the experts exact, so the whole 2.2B landed in
   * a 16.8B residual and the decode basis with it. The seed now carries the measured 352.8B and the
   * generator checks the active count against the published one on every refresh
   * (`reconcileActiveParams`); this is the same claim at the other end of the pipeline.
   */
  it.each([
    ['openai/gpt-oss-120b', 117, 5.1],
    ['openai/gpt-oss-20b', 21, 3.6],
    ['deepseek-ai/DeepSeek-V3', 671, 37],
    ['zai-org/GLM-4.5-Air', 106, 12],
    ['zai-org/GLM-4.7', 355, 32],
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
    // Derived independently on both sides — by hand from config.json for the fixture, from the
    // API for the catalog — so this is the check that catches a wrong width in either.
    expect(model.attention.projectionWidth).toBe(fixture.attention.projectionWidth);
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
    ['ibm-granite/granite-4.1-8b', true],
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
   * A tied model that ships the output table anyway, which is the mirror of the Gemma 3 case above
   * and was unguarded: `granite-4.1-8b` states `tie_word_embeddings: true` *and* carries
   * `lm_head.weight` at [100352, 4096] beside an identically-shaped `model.embed_tokens.weight`.
   *
   * `from_pretrained` lists that tensor in `_tied_weights_keys` and overwrites it with the embedding
   * at load, and llama.cpp's converter drops it for the same reason — so the resident model holds one
   * table where the index counts two. Read as an untied projection it was wrong twice: 8.79B against
   * 8.38B of weights, and an embedding subtracted from a per-token count that reads it every step.
   *
   * Asserted as the exact difference rather than as a flag, because the two halves are one claim: if
   * the row is tied, the total is 0.41B lighter, and a fix that changed only one of them would be a
   * new inconsistency rather than a correction.
   */
  it('counts one output table on a tied model that ships two', () => {
    const granite = getModel('ibm-granite/granite-4.1-8b');
    const table = granite.vocabSize * granite.hiddenSize;

    expect(granite.tiedEmbeddings).toBe(true);
    expect(table).toBe(100352 * 4096);
    // 8,791,592,960 elements are in the safetensors index; one 411,041,792-element table of them is
    // the duplicate the loader discards.
    expect(granite.totalParams).toBe(8_791_592_960 - table);
    // And decode reads that table, so the per-token basis is the whole thing.
    expect(granite.activeDenseParams).toBe(granite.totalParams);
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
   * Attention scales by the query projection width, which is not the hidden size for most
   * current models. Asserted against widths read from each repo's own config, and deliberately
   * including a model that projects *narrower* — a one-directional test would pass under a
   * "multiply hidden size by a constant" regression.
   */
  it.each([
    ['zai-org/GLM-4.5-Air', 12288, 3.0],
    ['deepseek-ai/DeepSeek-V3', 20480, 2.857],
    ['Qwen/Qwen3-30B-A3B', 4096, 2.0],
    ['openai/gpt-oss-20b', 4096, 1.422],
    ['unsloth/gemma-3-27b-it', 4096, 0.762],
    ['NousResearch/Meta-Llama-3.1-8B-Instruct', 4096, 1.0],
  ])('%s projects attention to its own width, not the hidden size', (id, width, ratio) => {
    const model = getModel(id);
    expect(model.attention.projectionWidth).toBe(width);
    expect(model.attention.projectionWidth / model.hiddenSize).toBeCloseTo(ratio, 2);
  });

  it('has most of the catalog projecting to something other than its hidden size', () => {
    const differing = MODELS.filter((m) => m.attention.projectionWidth !== m.hiddenSize);
    // 25 of 35 today. If this ever drops to zero the field has silently become hiddenSize again.
    expect(differing.length).toBeGreaterThan(8);
    // Both directions are represented, so the correction cannot be a one-way fudge.
    expect(differing.some((m) => m.attention.projectionWidth > m.hiddenSize)).toBe(true);
    expect(differing.some((m) => m.attention.projectionWidth < m.hiddenSize)).toBe(true);
  });

  /**
   * Mirrored seeds carry the canonical repo's traffic, not the mirror's. NousResearch's Llama
   * 3.1 70B has ~4.8K downloads against Meta's ~1.24M, which sorted the best-known model in the
   * catalog to last place.
   */
  it.each([
    ['NousResearch/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Llama-3.1-70B-Instruct'],
    ['NousResearch/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'],
    ['unsloth/gemma-3-12b-it', 'google/gemma-3-12b-it'],
    ['unsloth/gemma-3-27b-it', 'google/gemma-3-27b-it'],
  ])('%s takes its popularity from the canonical repo', (id, canonical) => {
    const model = getModel(id);
    expect(model.popularity?.measuredOn).toBe(canonical);
    expect(model.popularity?.downloads ?? 0).toBeGreaterThan(500_000);
  });

  it('does not rank the best-known model last', () => {
    const ranked = modelsByPopularity();
    const llama70 = ranked.findIndex((m) => m.id.endsWith('Meta-Llama-3.1-70B-Instruct'));
    expect(llama70).toBeGreaterThanOrEqual(0);
    expect(llama70).toBeLessThan(ranked.length - 1);
  });

  /**
   * Sliding-window metadata has to be complete or absent, never partial. A short `layer_types`
   * array and a missing array are indistinguishable to `layerWindows?.[i]`, so a partial one
   * would silently read as full attention — overstating KV and prefill for precisely the models
   * the hybrid handling exists to get right.
   */
  it('gives every sliding-window model a window for every layer', () => {
    for (const model of MODELS) {
      const windows = model.attention.layerWindows;
      if (!windows) continue;

      expect(windows).toHaveLength(model.layers);
      // An array that is all-null is the same thing as no array, and should have been omitted.
      expect(windows.some((w) => w !== null)).toBe(true);
      for (const w of windows) {
        if (w !== null) expect(w).toBeGreaterThan(0);
      }
    }
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

/**
 * `devices.json` is the hand-edited catalog of the two, and the JSON import types every string
 * field as `string` — so the casts narrowing them to the engine's unions were the only thing
 * between a typo and the engine, and a cast checks nothing. `toModel` makes this argument for the
 * generated catalog. These are the same argument for the file that needs it more.
 *
 * Each case below is what the typo actually produced before the guard, not a hypothetical.
 */
describe('a hand-typed device row is validated, not trusted', () => {
  // The RTX 5090's row, copied from `devices.json` rather than retyped, so a reader diffing the
  // two finds them identical.
  const ROW: DeviceRow = {
    id: 'rtx-5090',
    name: 'GeForce RTX 5090',
    vendor: 'NVIDIA',
    class: 'discrete-gpu',
    status: 'shipping',
    capacityGiB: 32,
    allocatableGiB: 31,
    bandwidthGBs: 1792,
    tflops: { fp16: 419, fp8: 838, fp4: 1676 },
    interconnect: 'PCIe 5.0 x16',
    hostLinkGBs: 63.0,
    tdpWatts: 575,
    msrpUsd: 1999,
    releasedAt: '2025-01-30',
    source: 'https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216',
  };

  it('is the row the catalog actually ships', () => {
    // Pins the fixture to the file. A guard demonstrated on an invented row proves less than one
    // demonstrated on a real one, and this is the assertion that keeps the two from drifting.
    expect(toDevice(ROW)).toEqual(getDevice('rtx-5090'));
  });

  it('refuses a misspelled class rather than reporting hardware nothing can drive', () => {
    // Every runtime's `supports` check misses, and `CLASS_BANDWIDTH_UTILIZATION[class]` is
    // undefined underneath, which takes decode to zero.
    expect(() => toDevice({ ...ROW, class: 'discrete_gpu' })).toThrow(/unsupported class/i);
    expect(() => toDevice({ ...ROW, class: 'discrete_gpu' })).toThrow(/rtx-5090/);
  });

  it('refuses a misspelled status rather than dropping the device out of the Matrix', () => {
    // The Matrix filters to `shipping`, so this one is silent: the device simply stops appearing.
    expect(() => toDevice({ ...ROW, status: 'shiping' })).toThrow(/unsupported status/i);
  });

  it('refuses a misspelled compute dtype rather than dividing by a zero rate', () => {
    // The loudest of the three: `peakFlops` has no fp16 to fall back to, so prefill reports a
    // time to first token of Infinity for a device whose datasheet is otherwise intact.
    expect(() => toDevice({ ...ROW, tflops: { fp61: 209.5, fp8: 419 } })).toThrow(
      /unsupported compute dtype/i
    );
  });

  it('names what it expected, since a human is the one fixing the row', () => {
    expect(() => toDevice({ ...ROW, class: 'gpu' })).toThrow(/discrete-gpu, unified-soc, cpu-ram/);
  });

  it('still accepts every member of each union', () => {
    // Guards the check against being satisfied by a list narrower than the type — `announced` has
    // no row in the catalog today, so nothing else would notice it being rejected.
    for (const cls of ['discrete-gpu', 'unified-soc', 'cpu-ram'] as const) {
      expect(toDevice({ ...ROW, class: cls }).class).toBe(cls);
    }
    for (const status of ['shipping', 'announced', 'rumored'] as const) {
      expect(toDevice({ ...ROW, status }).status).toBe(status);
    }
    for (const dtype of ['fp16', 'bf16', 'fp8', 'fp4', 'int8'] as const) {
      expect(toDevice({ ...ROW, tflops: { [dtype]: 100 } }).flops[dtype]).toBeGreaterThan(0);
    }
  });
});

/**
 * A ceiling that can be raised has to say how far, and the answer is never all of physical memory.
 *
 * Every Apple row was `allocatableTunable` with no stated maximum, so `maxAllocatablePerDevice`
 * fell back to capacity and all six resolved to 100% of RAM. The app then told the owner of a
 * 96 GiB Mac Studio that a 95.5 GiB configuration would fit once they raised the ceiling — a
 * machine with nothing left for the OS, the window server, or the inference process's own unwired
 * allocations. `iogpu.wired_limit_mb` *accepts* that value; the distance between what the sysctl
 * parses and what the machine survives is the whole subject of the field.
 */
describe('a raiseable allocation ceiling states how far it raises', () => {
  const tunable = DEVICES.filter((d) => d.allocatableTunable);

  it('covers every machine with a setting to raise', () => {
    // Nine Apple rows and the Ryzen. A sweep that silently matched nothing would prove nothing.
    expect(tunable.length).toBeGreaterThanOrEqual(10);
  });

  it.each(tunable.map((d) => [d.id, d] as const))('%s reserves room for the OS', (_id, device) => {
    const max = maxAllocatablePerDevice(device);

    expect(device.maxAllocatableBytes).toBeDefined();
    // Strictly below capacity: this is the assertion the whole issue turns on.
    expect(max).toBeLessThan(device.capacityBytes);
    // And never below the default it is supposed to be a ceiling for.
    expect(max).toBeGreaterThanOrEqual(device.allocatableBytes);
    // The reason is written down, since the reserve is a judgement rather than a datasheet figure.
    expect(device.note).toBeTruthy();
  });

  /**
   * The distinction must survive the fix. A ceiling that reserved so much that nothing sat between
   * the default and the maximum would satisfy every assertion above while quietly turning
   * "raiseable" into "will not run" — the failure in the other direction, which the Envelope,
   * Telemetry and Matrix legends all describe to the user.
   */
  it('leaves a band between the default and the ceiling on every Apple machine', () => {
    const apple = tunable.filter((d) => d.vendor === 'Apple');
    expect(apple).toHaveLength(9);

    for (const device of apple) {
      expect(maxAllocatablePerDevice(device)).toBeGreaterThan(device.allocatableBytes);
      expect(raisingCeilingWouldHelp(device, device.allocatableBytes + 1)).toBe(true);
    }
  });

  it('refuses to promise the last byte of RAM on any of them', () => {
    for (const device of tunable) {
      expect(raisingCeilingWouldHelp(device, device.capacityBytes)).toBe(false);
      expect(raisingCeilingWouldHelp(device, device.capacityBytes - 1)).toBe(false);
    }
  });

  /**
   * The case from the issue, named rather than swept: the 96 GiB Mac Studio and a 95.5 GiB
   * configuration, which the app offered to make fit.
   */
  it('no longer offers a 96 GiB Mac Studio a 95.5 GiB configuration', () => {
    const studio = getDevice('mac-studio-m3-ultra-96');
    expect(raisingCeilingWouldHelp(studio, 95.5 * GIB)).toBe(false);
    // But the configurations the setting really does rescue still say so.
    expect(raisingCeilingWouldHelp(studio, 80 * GIB)).toBe(true);
  });
});

/**
 * Two rules, and the interesting part is where they cross.
 *
 * The default is what Metal reports as its recommended working set — two thirds of RAM at 32 GiB
 * and below, three quarters above — because that is the figure llama.cpp warns against exceeding and
 * the one a user has before touching a sysctl. The ceiling is capacity minus max(8 GiB, 1/16 of
 * RAM), the reserve #53 established.
 *
 * Below 32 GiB those two meet and then invert: a 24 GiB machine recommends 16 GiB and reserves down
 * to 16 GiB, and a 16 GiB machine would have a ceiling *under* its own default. Six maxed
 * configurations all sat far above the crossover, so the rule looked universal; the machines most
 * Mac owners have sit at or below it. Those rows state no raiseable ceiling rather than one that
 * promises less than the default, which is `maxAllocatablePerDevice`'s under-promising direction
 * applied one step earlier — at curation time.
 */
describe('the Apple rows derive both of their allocation figures', () => {
  const apple = DEVICES.filter((d) => d.vendor === 'Apple');

  it('covers machines on both sides of the crossover', () => {
    expect(apple.length).toBeGreaterThanOrEqual(11);
    expect(apple.some((d) => d.capacityBytes <= 32 * GIB)).toBe(true);
    expect(apple.some((d) => d.capacityBytes > 32 * GIB)).toBe(true);
  });

  it.each(apple.map((d) => [d.id, d] as const))(
    '%s defaults to what Metal recommends as a working set',
    (_id, device) => {
      const ram = device.capacityBytes / GIB;
      const recommended = ram <= 32 ? Math.floor((ram * 2) / 3) : ram * 0.75;
      expect(device.allocatableBytes / GIB).toBe(recommended);
    }
  );

  it.each(apple.map((d) => [d.id, d] as const))(
    '%s states a ceiling only where the OS reserve leaves one',
    (_id, device) => {
      const reserve = Math.max(8 * GIB, device.capacityBytes / 16);
      const ceiling = device.capacityBytes - reserve;

      if (ceiling > device.allocatableBytes) {
        expect(device.allocatableTunable).toBe(true);
        expect(device.maxAllocatableBytes).toBe(ceiling);
      } else {
        // Nothing to raise to. Stating the flag here would put "raiseable to 16 GiB" beside a
        // default of 16 GiB in the picker, or — with the reserve applied literally — a ceiling
        // below the memory the machine already offers.
        expect(device.allocatableTunable).toBeUndefined();
        expect(device.maxAllocatableBytes).toBeUndefined();
        expect(maxAllocatablePerDevice(device)).toBe(device.allocatableBytes);
        expect(raisingCeilingWouldHelp(device, device.allocatableBytes + 1)).toBe(false);
      }
    }
  );
});

/**
 * The pairing is enforced at load, not left to the curator: the failure is silent on every surface
 * and reads as generosity, which is why it survived a 25-row manual verification pass.
 */
describe('the catalog refuses a ceiling it cannot justify', () => {
  const TUNABLE_ROW: DeviceRow = {
    id: 'mac-studio-m3-ultra-96',
    name: 'Mac Studio M3 Ultra (96 GB)',
    vendor: 'Apple',
    class: 'unified-soc',
    status: 'shipping',
    capacityGiB: 96,
    allocatableGiB: 72,
    allocatableTunable: true,
    maxAllocatableGiB: 88,
    bandwidthGBs: 819,
    tflops: { fp16: 54 },
    source: 'https://www.apple.com/mac-studio/specs/',
  };

  it('accepts the row the catalog ships', () => {
    expect(toDevice(TUNABLE_ROW).maxAllocatableBytes).toBe(88 * GIB);
  });

  /** The shape every Apple row had: tunable, with nothing saying how far. */
  const withoutMax = (): DeviceRow => {
    const row = { ...TUNABLE_ROW };
    delete row.maxAllocatableGiB;
    return row;
  };

  it('refuses a tunable row that states no maximum', () => {
    expect(() => toDevice(withoutMax())).toThrow(/no maxAllocatableGiB/i);
  });

  it('refuses a ceiling at or above physical memory', () => {
    expect(() => toDevice({ ...TUNABLE_ROW, maxAllocatableGiB: 96 })).toThrow(/reserve room/i);
    expect(() => toDevice({ ...TUNABLE_ROW, maxAllocatableGiB: 128 })).toThrow(/reserve room/i);
  });

  it('refuses a ceiling below the default it is meant to raise', () => {
    expect(() => toDevice({ ...TUNABLE_ROW, maxAllocatableGiB: 64 })).toThrow(/below its own/i);
  });

  it('refuses a maximum on a row whose ceiling does not move', () => {
    // The pairing from the other side. `maxAllocatablePerDevice` ignores a maximum without the
    // flag, so the stated ceiling would vanish from every surface without a word.
    const untunable = { ...TUNABLE_ROW };
    delete untunable.allocatableTunable;

    expect(() => toDevice(untunable)).toThrow(/without allocatableTunable/i);
    expect(() => toDevice({ ...TUNABLE_ROW, allocatableTunable: false })).toThrow(
      /without allocatableTunable/i
    );
  });

  it('leaves a fixed-ceiling row alone', () => {
    // A card whose ceiling cannot be raised states no maximum, and must not be asked for one —
    // otherwise the guard would reject most of the committed rows.
    const fixed = withoutMax();
    delete fixed.allocatableTunable;

    expect(() => toDevice(fixed)).not.toThrow();
    expect(toDevice(fixed).maxAllocatableBytes).toBeUndefined();
  });
});

/**
 * A device id is in other people's links, so renaming a row is a compatibility change.
 *
 * `rtx-a6000-ada` named a product that does not exist: the Ampere card is the RTX A6000 and the Ada
 * one is the RTX 6000 Ada Generation, and the id fused them while every spec on the row was the Ada
 * card's. `url.ts` writes `deviceId` into every shared scenario as `d`, and the failure without an
 * alias is not a broken link — it is `coerce` falling back to the default device and showing a
 * stranger a different machine's numbers under the sender's URL.
 */
describe('a renamed device keeps the links that already name it', () => {
  it('no longer ships the id that fused two real products', () => {
    expect(DEVICES.map((d) => d.id)).not.toContain('rtx-a6000-ada');
    expect(getDevice('rtx-6000-ada').name).toBe('RTX 6000 Ada');
  });

  it('resolves the old id to the row it always described', () => {
    // Same object, not merely the same specs: an alias that produced a copy would compare unequal
    // wherever the app holds a device by reference.
    expect(getDevice('rtx-a6000-ada')).toBe(getDevice('rtx-6000-ada'));
  });

  it('still refuses an id that never existed', () => {
    expect(() => getDevice('rtx-a6000')).toThrow(/unknown device/i);
  });

  /**
   * The alias table is a plain object and the ids arrive from a querystring, so `?d=toString` reads
   * a *function* off the prototype chain. With a `??` lookup that is not nullish, so the fallback
   * does not fire and a function is returned from a signature promising a string. The degradation
   * is the same either way — `getDevice` misses and `coerce` falls back to the default — but the
   * type is a lie, and `narrow` in this same file uses `Object.hasOwn` for exactly this shape.
   */
  it.each(['toString', 'constructor', 'hasOwnProperty', '__proto__'])(
    'does not read %s off the prototype of the alias table',
    (id) => {
      expect(canonicalDeviceId(id)).toBe(id);
      expect(() => getDevice(id)).toThrow(/unknown device/i);
    }
  );

  /**
   * The invariant rather than the one entry: an alias pointing at a row that has since been renamed
   * again resolves to nothing, and an alias that shadows a live id makes that row unreachable. Both
   * are silent, and both are one careless edit away.
   */
  it('points every alias at a row that exists, and shadows none', () => {
    const ids = new Set(DEVICES.map((d) => d.id));
    for (const [from, to] of Object.entries(DEVICE_ID_ALIASES)) {
      expect(ids.has(to), `${from} aliases missing row ${to}`).toBe(true);
      expect(ids.has(from), `${from} is both an alias and a row`).toBe(false);
    }
  });
});

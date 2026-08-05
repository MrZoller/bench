import { describe, expect, it } from 'vitest';
import {
  DEFAULT_HOST_BANDWIDTH,
  allocatablePerDevice,
  clampUsageToContext,
  effectivePromptTokens,
  maxAllocatablePerDevice,
  raisingCeilingWouldHelp,
  canShard,
  maxContextThatFits,
  offloadBandwidth,
  planPlacement,
} from './placement';
import {
  DEEPSEEK_V3,
  DGX_SPARK,
  GEMMA_3_12B,
  GPT_OSS_120B,
  LLAMA_31_8B,
  LLAMA_32_3B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_256,
  MLX,
  QWEN3_32B,
  RTX_4090,
  RTX_5090,
  STRIX_HALO_395,
  VLLM,
} from './fixtures';
import { achievedBandwidth } from './speed';
import { isSlidingLayer } from './kv';
import { weightBreakdown } from './weights';
import { getQuant } from '@/data/quants';
import { GIB } from './types';
import type { DeviceSpec, ModelSpec, UsageSpec } from './types';

const usage = (contextTokens: number, concurrency = 1): UsageSpec => ({
  contextTokens,
  concurrency,
  kvPrecision: 'fp16',
});

describe('allocatable is not capacity', () => {
  /**
   * The plan's headline accuracy case. A 256 GB Mac caps GPU-wired memory near 75% by
   * default, so a configuration needing more than ~192 GB must be refused even though the
   * box says 256 GB — and must be accepted once the user raises the wired limit.
   */
  it('refuses a config above the default macOS wired-memory ceiling, and accepts it once raised', () => {
    const quant = getQuant('bf16'); // ~217 GiB of weights: over the default cap, under capacity
    const rig = { device: MAC_STUDIO_M3_ULTRA_256, count: 1 };

    const atDefault = planPlacement(GPT_OSS_120B, quant, usage(4096), rig, MLX);
    expect(atDefault.fits).toBe(false);
    expect(atDefault.usedBytesPerDevice).toBeLessThan(MAC_STUDIO_M3_ULTRA_256.capacityBytes);

    // Same machine, same model, wired limit lifted toward capacity.
    const tuned: DeviceSpec = {
      ...MAC_STUDIO_M3_ULTRA_256,
      allocatableBytes: Math.floor(0.95 * MAC_STUDIO_M3_ULTRA_256.capacityBytes),
    };
    const raised = planPlacement(
      GPT_OSS_120B,
      quant,
      usage(4096),
      { device: tuned, count: 1 },
      MLX
    );
    expect(raised.fits).toBe(true);
  });

  it('honours the Strix Halo Variable Graphics Memory ceiling rather than its 128 GB sticker', () => {
    // Variable Graphics Memory exposes 96 of 128 GB; sizing against the sticker would
    // overstate what the model can have by a third.
    const rig = { device: STRIX_HALO_395, count: 1 };
    expect(allocatablePerDevice(rig, LLAMA_CPP)).toBe(96 * GIB);
    expect(allocatablePerDevice(rig, LLAMA_CPP)).toBeLessThan(STRIX_HALO_395.capacityBytes);
  });

  it('applies vLLM prealloc as a ceiling that llama.cpp does not have', () => {
    const rig = { device: RTX_5090, count: 1 };
    const withVllm = allocatablePerDevice(rig, VLLM);
    const withLlamaCpp = allocatablePerDevice(rig, LLAMA_CPP);

    expect(withVllm).toBeLessThan(withLlamaCpp);
    expect(withVllm).toBeCloseTo(RTX_5090.capacityBytes * 0.9, -8);
  });
});

describe('fit', () => {
  it('fits an 8B model at Q4 on a single 5090 with room to spare', () => {
    const plan = planPlacement(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      usage(8192),
      {
        device: RTX_5090,
        count: 1,
      },
      LLAMA_CPP
    );

    expect(plan.fits).toBe(true);
    expect(plan.utilization).toBeLessThan(0.5);
  });

  it('fits gpt-oss-120b on a Spark where it will not fit one 5090', () => {
    const quant = getQuant('mxfp4');

    const onSpark = planPlacement(
      GPT_OSS_120B,
      quant,
      usage(8192),
      {
        device: DGX_SPARK,
        count: 1,
      },
      LLAMA_CPP
    );
    const on5090 = planPlacement(
      GPT_OSS_120B,
      quant,
      usage(8192),
      {
        device: RTX_5090,
        count: 1,
      },
      LLAMA_CPP
    );

    expect(onSpark.fits).toBe(true);
    expect(on5090.fits).toBe(false);
    expect(on5090.offloadFraction).toBeGreaterThan(0);
  });

  it('shards across a multi-GPU rig', () => {
    const quant = getQuant('q4_k_m');
    const one = planPlacement(
      QWEN3_32B,
      quant,
      usage(8192),
      { device: RTX_5090, count: 1 },
      LLAMA_CPP
    );
    const two = planPlacement(
      QWEN3_32B,
      quant,
      usage(8192),
      { device: RTX_5090, count: 2 },
      LLAMA_CPP
    );

    // The *repeating stack* is what divides, and since #182 that is visible in the figures rather
    // than hidden by a smeared fixed block. Qwen3 32B is untied, so a single card's budget is the
    // file less the host-resident input table, and the output projection seeds the last bin — which
    // pushes one repeating layer onto the other card, 33/31 rather than 32/32.
    const { layerBytes } = weightBreakdown(QWEN3_32B, quant);
    expect(two.weightBytesPerDevice).toBeCloseTo((layerBytes / QWEN3_32B.layers) * 33, -6);
    // Still comfortably under what one card was holding, which is what "shards" has to mean.
    expect(two.weightBytesPerDevice).toBeLessThan(one.weightBytesPerDevice * 0.55);
    expect(two.fits).toBe(true);
  });

  it('reports offload as impossible on unified memory, which has nowhere slower to spill', () => {
    // No quantization makes this fit; on a discrete GPU it would offload, here it simply can't.
    const plan = planPlacement(
      GPT_OSS_120B,
      getQuant('bf16'),
      usage(4096),
      {
        device: DGX_SPARK,
        count: 1,
      },
      LLAMA_CPP
    );

    expect(plan.fits).toBe(false);
    expect(plan.offloadFraction).toBe(0);
    expect(plan.impossible).toBe(true);
  });

  it('flags a runtime that cannot drive the device class at all', () => {
    const plan = planPlacement(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      usage(4096),
      {
        device: MAC_STUDIO_M3_ULTRA_256,
        count: 1,
      },
      VLLM
    );
    expect(plan.unsupported).toMatch(/vLLM/);
  });

  /**
   * `quantApplies` enforced this for the picker and the store, so the app never showed one of
   * these pairings — but every caller reaching the engine directly walked straight past it and
   * got capacity and throughput for a checkpoint the runtime cannot open. A rule the UI applies
   * on the engine's behalf is a rule the engine does not have.
   */
  it('flags a weight format the runtime cannot load', () => {
    const rig = { device: RTX_5090, count: 1 };

    // AWQ is a vLLM format; llama.cpp reads GGUF. Nothing about the hardware rejects it, which is
    // why the vendor/dtype check alone let it through.
    const awqOnLlamaCpp = planPlacement(
      LLAMA_31_8B,
      getQuant('awq_4bit'),
      usage(4096),
      rig,
      LLAMA_CPP
    );
    expect(awqOnLlamaCpp.unsupported).toMatch(/llama\.cpp/);

    // And the reverse, so the check cannot be satisfied by a list that happens to be one-sided.
    const ggufOnVllm = planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), rig, VLLM);
    expect(ggufOnVllm.unsupported).toMatch(/vLLM/);

    // The pairing each runtime *can* load stays clean, so the check is not simply refusing work.
    expect(
      planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), rig, LLAMA_CPP).unsupported
    ).toBeUndefined();
    expect(
      planPlacement(LLAMA_31_8B, getQuant('awq_4bit'), usage(4096), rig, VLLM).unsupported
    ).toBeUndefined();
  });
});

describe('maximum context', () => {
  it('finds a context that fits and reports the next step up as over budget', () => {
    const quant = getQuant('q4_k_m');
    const rig = { device: RTX_5090, count: 1 };
    const base = usage(4096);

    const max = maxContextThatFits(LLAMA_31_8B, quant, base, rig, LLAMA_CPP);
    expect(max).toBeGreaterThan(0);

    expect(
      planPlacement(LLAMA_31_8B, quant, { ...base, contextTokens: max }, rig, LLAMA_CPP).fits
    ).toBe(true);
    if (max < LLAMA_31_8B.maxContext) {
      expect(
        planPlacement(LLAMA_31_8B, quant, { ...base, contextTokens: max + 1 }, rig, LLAMA_CPP).fits
      ).toBe(false);
    }
  });

  it('shrinks as concurrency rises, since every sequence carries its own cache', () => {
    const quant = getQuant('q4_k_m');
    const rig = { device: RTX_5090, count: 1 };

    const alone = maxContextThatFits(LLAMA_31_8B, quant, usage(4096, 1), rig, LLAMA_CPP);
    const crowded = maxContextThatFits(LLAMA_31_8B, quant, usage(4096, 8), rig, LLAMA_CPP);
    expect(crowded).toBeLessThan(alone);
  });

  it('returns zero when even one token cannot fit', () => {
    const plan = maxContextThatFits(
      GPT_OSS_120B,
      getQuant('bf16'),
      usage(4096),
      { device: RTX_5090, count: 1 },
      LLAMA_CPP
    );
    expect(plan).toBe(0);
  });
});

describe('memory breakdown', () => {
  it('accounts for every byte it reports as used', () => {
    const plan = planPlacement(
      QWEN3_32B,
      getQuant('q4_k_m'),
      usage(32768),
      {
        device: RTX_5090,
        count: 1,
      },
      LLAMA_CPP
    );

    expect(plan.usedBytesPerDevice).toBeCloseTo(
      plan.weightBytesPerDevice + plan.kvBytesPerDevice + plan.activationBytesPerDevice,
      -3
    );
  });

  it('grows KV, and only KV, when context grows', () => {
    const quant = getQuant('q4_k_m');
    const rig = { device: RTX_5090, count: 1 };

    const short = planPlacement(QWEN3_32B, quant, usage(4096), rig, LLAMA_CPP);
    const long = planPlacement(QWEN3_32B, quant, usage(16384), rig, LLAMA_CPP);

    expect(long.weightBytesPerDevice).toBe(short.weightBytesPerDevice);
    expect(long.kvBytesPerDevice).toBeCloseTo(short.kvBytesPerDevice * 4, -6);
    expect(long.kvBytesPerDevice / GIB).toBeGreaterThan(3);
  });
});

/**
 * KV does not shard the way weights do, and assuming it does is optimistic in the one direction
 * that matters — it reports a rig fitting when the layout it would really produce does not.
 */
describe('the KV cache shards only as far as the model allows', () => {
  const plan = (model: typeof QWEN3_32B, count: number) =>
    planPlacement(
      model,
      getQuant('q4_k_m'),
      { contextTokens: 32768, concurrency: 16, kvPrecision: 'fp16' },
      { device: RTX_5090, count },
      VLLM
    );

  it('stops dividing once every rank holds a whole KV head', () => {
    // Qwen3-32B has 8 KV heads. Up to 8 cards each rank gets at least one head and the cache
    // divides; past that the heads are replicated and per-card KV stops falling.
    const at8 = plan(QWEN3_32B, 8);
    const at16 = plan(QWEN3_32B, 16);

    expect(at8.kvBytesPerDevice).toBeCloseTo(at8.totalKvBytes / 8, -3);
    expect(at16.kvBytesPerDevice).toBeCloseTo(at8.kvBytesPerDevice, -3);
    // Weights keep sharding — it is only KV that has a floor.
    expect(at16.weightBytesPerDevice).toBeCloseTo(at8.weightBytesPerDevice / 2, -3);
  });

  it('never divides an MLA latent cache at all', () => {
    // One latent per token per layer, with no head axis to split along, so vLLM replicates it
    // on every rank. The old code divided by the full device count — off by 8x on 8 cards.
    for (const count of [1, 2, 4, 8]) {
      const p = plan(DEEPSEEK_V3, count);
      expect(p.kvBytesPerDevice).toBeCloseTo(p.totalKvBytes, -3);
    }
  });
});

/**
 * Spilled weights read at the slower of host RAM and the bus to it. Modelling only host RAM
 * made every offloaded configuration 2.5x too fast on a PCIe 4.0 card.
 */
describe('offload crosses a real bus', () => {
  it('takes the device host link when it is slower than host RAM', () => {
    // 80 GB/s of DDR5 behind a 31.5 GB/s PCIe 4.0 link.
    expect(offloadBandwidth({ device: RTX_4090, count: 1 }, DEFAULT_HOST_BANDWIDTH)).toBeCloseTo(
      31.5e9,
      -6
    );
    // And behind a 63 GB/s PCIe 5.0 link, still the link.
    expect(offloadBandwidth({ device: RTX_5090, count: 1 }, DEFAULT_HOST_BANDWIDTH)).toBeCloseTo(
      63e9,
      -6
    );
  });

  it('adds up the links on a multi-card rig, then stops at host memory', () => {
    // Each card streams its own shard over its own link, so two PCIe 4.0 cards move 63 GB/s
    // between them — charging one card's 31.5 to the whole rig doubled the transfer time.
    expect(offloadBandwidth({ device: RTX_4090, count: 2 }, DEFAULT_HOST_BANDWIDTH)).toBeCloseTo(
      63e9,
      -6
    );
    // Four of them would exceed host memory itself, which is then the binding constraint.
    expect(offloadBandwidth({ device: RTX_4090, count: 4 }, DEFAULT_HOST_BANDWIDTH)).toBe(
      DEFAULT_HOST_BANDWIDTH
    );
  });

  it('falls back to host RAM where there is no host to cross to', () => {
    // Unified memory has no separate host: the pool in question already is system memory.
    expect(offloadBandwidth({ device: DGX_SPARK, count: 1 }, DEFAULT_HOST_BANDWIDTH)).toBe(
      DEFAULT_HOST_BANDWIDTH
    );
  });
});

/**
 * "You could raise this" is advice, and advice that cannot be taken is worse than none.
 */
describe('a ceiling is only raiseable as far as the platform allows', () => {
  const ryzen: DeviceSpec = { ...STRIX_HALO_395, allocatableTunable: true };

  /**
   * This test used to assert the opposite, and was the reason the claim survived: `iogpu.wired_
   * limit_mb` is indeed a default rather than a hardware limit, so "raiseable up to physical
   * memory" reads as correct. It confuses what the sysctl *accepts* with what the machine
   * survives. macOS, the window server and the inference process's own unwired allocations still
   * need room, so the ceiling raises to 240 of 256 GiB, not to 256.
   */
  it('treats a Mac default as raiseable, but never to the last byte of RAM', () => {
    const max = maxAllocatablePerDevice(MAC_STUDIO_M3_ULTRA_256);

    expect(max).toBeGreaterThan(MAC_STUDIO_M3_ULTRA_256.allocatableBytes);
    expect(max).toBeLessThan(MAC_STUDIO_M3_ULTRA_256.capacityBytes);

    // The band between the default and the real ceiling is still rescued by raising it — that is
    // the distinction this whole function exists to draw, and it must not be lost to the fix.
    const between = MAC_STUDIO_M3_ULTRA_256.allocatableBytes + 1;
    expect(raisingCeilingWouldHelp(MAC_STUDIO_M3_ULTRA_256, between)).toBe(true);

    // Above the ceiling it is not, however much physical memory the box has left.
    const wired = MAC_STUDIO_M3_ULTRA_256.capacityBytes - 1;
    expect(raisingCeilingWouldHelp(MAC_STUDIO_M3_ULTRA_256, wired)).toBe(false);
  });

  /**
   * The defensive floor behind the catalog guard. A tunable row that states no maximum is read as
   * not raiseable, rather than as raiseable to physical capacity — the assumption that made every
   * Apple row claim 100% of RAM. Wrong in the safe direction now: it under-promises.
   */
  it('promises nothing for a tunable ceiling that states no maximum', () => {
    const unstated: DeviceSpec = { ...MAC_STUDIO_M3_ULTRA_256 };
    delete (unstated as { maxAllocatableBytes?: number }).maxAllocatableBytes;

    expect(maxAllocatablePerDevice(unstated)).toBe(unstated.allocatableBytes);
    expect(raisingCeilingWouldHelp(unstated, unstated.allocatableBytes + 1)).toBe(false);
  });

  it('refuses to promise more than Variable Graphics Memory exposes', () => {
    // 96 of 128 GB is the AMD maximum, and it is already the catalogued default — so there is
    // nothing to raise, and a 117 GiB configuration cannot be rescued by a setting.
    const stated = { ...ryzen, maxAllocatableBytes: 96 * GIB };
    expect(maxAllocatablePerDevice(stated)).toBe(96 * GIB);
    expect(raisingCeilingWouldHelp(stated, 117 * GIB)).toBe(false);
    expect(raisingCeilingWouldHelp(stated, 90 * GIB)).toBe(false);
  });

  it('never claims a fixed ceiling can move', () => {
    expect(raisingCeilingWouldHelp(RTX_5090, 1)).toBe(false);
    expect(maxAllocatablePerDevice(RTX_5090)).toBe(RTX_5090.allocatableBytes);
  });
});

/**
 * Under a layer split, weights and cache travel together: a card that owns a layer owns both.
 * Rounding only one of them up describes a machine that does not exist.
 */
describe('an indivisible layer count rounds weights up too', () => {
  it('charges the busiest card, not the average one', () => {
    // DeepSeek V3 has 61 layers. Over two cards that is 31 and 30, so the busy one holds 31/61
    // of the model — not half of it.
    const plan = (count: number) =>
      planPlacement(
        DEEPSEEK_V3,
        getQuant('iq4_xs'),
        { contextTokens: 16384, concurrency: 1, kvPrecision: 'fp16' },
        { device: RTX_5090, count },
        LLAMA_CPP
      );

    const two = plan(2);
    const one = plan(1);
    // 31 layers of the repeating stack, and the output projection is on the *other* card — it
    // seeds the last bin, so the busiest bin by combined load is the one holding a layer more
    // rather than the one holding the table (#182). The whole file over 61 was the divisor before,
    // and it charged this card for a share of tensors it does not hold.
    const perLayer = weightBreakdown(DEEPSEEK_V3, getQuant('iq4_xs')).layerBytes / 61;
    expect(two.weightBytesPerDevice).toBeCloseTo(31 * perLayer, -3);
    // And the same divisor as the cache, which is the property that was broken.
    expect(two.kvBytesPerDevice).toBeCloseTo((one.kvBytesPerDevice * 31) / 61, -3);
  });

  it('leaves tensor-parallel rigs dividing evenly, because they do', () => {
    const plan = (count: number) =>
      planPlacement(
        DEEPSEEK_V3,
        getQuant('iq4_xs'),
        { contextTokens: 16384, concurrency: 1, kvPrecision: 'fp16' },
        { device: RTX_5090, count },
        VLLM
      );
    expect(plan(2).weightBytesPerDevice).toBeCloseTo(plan(1).weightBytesPerDevice / 2, -3);
  });
});

/**
 * A layer count is not a KV divisor on a hybrid model, and a layer split is not a speedup.
 * Both were being assumed, and both flatter multi-card rigs in the direction that reports a fit.
 */
describe('layer splits are sized, not divided', () => {
  const gemma = (count: number, runtime = LLAMA_CPP) =>
    planPlacement(
      GEMMA_3_12B,
      getQuant('q4_k_m'),
      { contextTokens: 131072, concurrency: 8, kvPrecision: 'fp16' },
      { device: RTX_5090, count },
      runtime
    );

  it('charges the busiest card for the full-attention layers it lands', () => {
    // Gemma 3 12B has 8 full-attention layers among 48, and at 128K each caches far more than a
    // sliding one. Over five cards the full layers cannot be spread evenly — someone holds two —
    // so the busiest card holds more than a fifth of the cache. Five is reachable: the store
    // accepts any device count from a URL, and `DEVICE_COUNT_STOPS` is only what the slider offers.
    const five = gemma(5);
    const evenShare = gemma(1).kvBytesPerDevice / 5;

    expect(five.kvBytesPerDevice).toBeGreaterThan(evenShare * 1.15);

    // And a count that *does* divide the full layers evenly gets essentially the even share, so
    // the model is charging for real imbalance rather than adding a blanket penalty. Not *exactly*
    // even since #182: the output projection seeds the last bin and the vision tower the first, so
    // the walk balances 5,7,7,7,7,6,6,3 layers rather than six apiece. Each card still lands
    // exactly one full-attention layer — those are the eight heaviest and go out first — so what
    // is left over is a difference in sliding layers, which cache ~1/128 as much: under 1% against
    // the 20% the five-card split carries.
    const eight = gemma(8);
    const evenEighth = gemma(1).kvBytesPerDevice / 8;
    expect(eight.kvBytesPerDevice).toBeGreaterThan(evenEighth);
    expect(eight.kvBytesPerDevice).toBeLessThan(evenEighth * 1.01);
  });

  it('takes both figures from the same device', () => {
    // The card carrying the big full-attention caches is the one a balanced scheduler gives
    // fewer layers, so the heaviest-KV card and the heaviest-weight card are different cards.
    // Adding those two maxima describes a device that does not exist — and reported spill for
    // a rig that fits.
    const p = gemma(4);
    const { layerBytes, towerBytes, outputBytes } = weightBreakdown(
      GEMMA_3_12B,
      getQuant('q4_k_m')
    );
    const perLayerWeight = layerBytes / GEMMA_3_12B.layers;

    // Whatever share of the weights a card holds, it must be a whole number of layers plus
    // whichever fixed tensor was *placed* on it — the vision tower on the first card, the output
    // projection on the last, nothing at all in between (#182). The divisor is `layerBytes`, not
    // the file: charging a layer a share of tensors no layer holds is the #165 defect, and now
    // that the blocks sit where upstream puts them the whole-number property is exact rather than
    // a coincidence of the smearing.
    const last = p.assignment.shares.length - 1;
    for (const [i, s] of p.assignment.shares.entries()) {
      const placed = (i === 0 ? towerBytes : 0) + (i === last ? outputBytes : 0);
      expect((s.weightBytes - placed) / perLayerWeight, `bin ${i}`).toBeCloseTo(s.layers, 6);
    }

    // And the two together must never exceed what one device could hold of each.
    expect(p.weightBytesPerDevice).toBeLessThanOrEqual(gemma(1).weightBytesPerDevice + 1);
    expect(p.kvBytesPerDevice).toBeLessThanOrEqual(gemma(1).kvBytesPerDevice + 1);
  });

  it('never claims a card holds more than the whole cache', () => {
    for (const count of [1, 2, 4, 8]) {
      const p = gemma(count);
      expect(p.kvBytesPerDevice).toBeLessThanOrEqual(p.totalKvBytes + 1);
    }
  });

  /**
   * `offloadFraction` is charged against the *whole model's* active weights by both speed
   * estimators, so it has to be a rig-wide quantity. Derived from the busiest device it was a
   * per-device fraction, and a layer split is exactly where those two differ: the cards hold
   * different amounts, so one can be over its ceiling while the remaining serial stages stay
   * resident — and every one of them was billed host-bus time all the same.
   */
  describe('the spill fraction is the rig’s, not the busiest device’s', () => {
    // What the old code computed: the busiest device's overflow over the busiest device's weights.
    const perDeviceFraction = (p: ReturnType<typeof planPlacement>) =>
      Math.min(1, Math.max(0, -p.headroomBytes) / Math.max(p.weightBytesPerDevice, 1));

    it('charges only the devices that are over, not every serial stage', () => {
      // Gemma 3 12B at 128K over 8 users is cache-dominated, and its 8 full-attention layers cache
      // ~128x what a sliding one does. Eight of them over five cards is three cards holding two and
      // two holding one, so the split is lopsided by construction: three cards sit at the top load
      // and two sit well under it. Lower the ceiling just below that top load and exactly three of
      // the five spill — which is the shape the issue describes, and which a fraction taken from
      // one device cannot express.
      const scenario = { contextTokens: 131072, concurrency: 8, kvPrecision: 'fp16' as const };
      const plan = (device: DeviceSpec) =>
        planPlacement(GEMMA_3_12B, getQuant('q4_k_m'), scenario, { device, count: 5 }, LLAMA_CPP);

      const roomy = plan(RTX_5090);
      expect(roomy.fits).toBe(true);

      const deficit = 0.25 * GIB;
      const tight: DeviceSpec = {
        ...RTX_5090,
        allocatableBytes: roomy.usedBytesPerDevice - deficit,
        capacityBytes: 200 * GIB,
      };
      const spilling = plan(tight);

      expect(spilling.fits).toBe(false);
      // Offload genuinely rescues this: the cache still fits, it is the weights on top that do not.
      expect(spilling.impossible).toBe(false);

      // Three cards over by `deficit` each, two resident — and every spilled byte is a weight,
      // since none of the three is over by more than the ~0.29 GiB of weights it holds.
      expect(spilling.offloadFraction * spilling.totalWeightBytes).toBeCloseTo(3 * deficit, -6);

      // What the two readings say about the same rig. The busiest card holds 4% of the model's
      // weights, so expressing its overflow as a fraction of *those* put almost the whole model on
      // the host bus — eight times the streamed volume, straight onto decode and TTFT together.
      expect(perDeviceFraction(spilling)).toBeGreaterThan(0.85);
      expect(spilling.offloadFraction).toBeLessThan(0.12);
    });

    it('holds on real hardware, where every card spills but by different amounts', () => {
      // No synthetic ceiling: gpt-oss-120b at Q4_K_M over four 4090s at 128K and 16 users. Every
      // card is over, so this is the milder half of the defect — the busiest card's ratio is still
      // not the rig's, and it overstates the streamed volume by 14%.
      const p = planPlacement(
        GPT_OSS_120B,
        getQuant('q4_k_m'),
        { contextTokens: 131072, concurrency: 16, kvPrecision: 'fp16' },
        { device: RTX_4090, count: 4 },
        LLAMA_CPP
      );

      expect(p.fits).toBe(false);
      expect(p.impossible).toBe(false);
      expect(p.offloadFraction).toBeGreaterThan(0);
      expect(p.offloadFraction).toBeLessThan(perDeviceFraction(p));
    });

    it('never claims more spills than the rig is over by, or than the model has', () => {
      for (const count of [1, 2, 3, 4, 5, 8]) {
        for (const context of [32768, 131072]) {
          const p = planPlacement(
            GEMMA_3_12B,
            getQuant('q4_k_m'),
            { contextTokens: context, concurrency: 8, kvPrecision: 'fp16' },
            { device: RTX_4090, count },
            LLAMA_CPP
          );
          if (p.unsupported) continue;

          expect(p.offloadFraction).toBeGreaterThanOrEqual(0);
          expect(p.offloadFraction).toBeLessThanOrEqual(1);
          // A resident rig spills nothing, whatever the packing looks like.
          if (p.fits) expect(p.offloadFraction).toBe(0);

          // And the bytes it claims are on the host bus cannot exceed what the rig is over by —
          // the sum of every card's deficit, which the busiest card's deficit times the card count
          // bounds from above.
          const spilled = p.offloadFraction * p.totalWeightBytes;
          expect(spilled).toBeLessThanOrEqual(Math.max(0, -p.headroomBytes) * count + 1);
        }
      }
    });

    it('leaves the uniform case exactly where it was', () => {
      // Tensor parallelism hands every rank the same load, so summing n identical overflows over n
      // identical shards has to give back the per-device ratio. This is the regression guard on the
      // path that was never wrong.
      //
      // FP8, not IQ4_XS: `VLLM.weightFormats` does not include the GGUF K-quants, so every
      // iteration of the first attempt returned `unsupported` and skipped, and the guard on this
      // change's load-bearing claim asserted nothing while reporting green. Hence the counter
      // below — the same `continue` will hollow this out again the next time a fixture moves.
      let asserted = 0;
      for (const count of [1, 2, 4, 8]) {
        const p = planPlacement(
          DEEPSEEK_V3,
          getQuant('fp8'),
          { contextTokens: 32768, concurrency: 4, kvPrecision: 'fp16' },
          { device: RTX_5090, count },
          VLLM
        );
        if (p.unsupported || p.impossible) continue;
        expect(p.offloadFraction).toBeGreaterThan(0);
        expect(p.offloadFraction).toBeCloseTo(perDeviceFraction(p), 6);
        asserted += 1;
      }
      expect(asserted).toBe(4);
    });

    /**
     * The predicate and its sentence are one claim, which is this repo's most-repeated lesson.
     * Widening `impossible` to every device broke an implication the panels were relying on: it
     * used to be true by construction that an impossible offloadable rig had *the busiest device's*
     * cache over the ceiling, so BudgetBar could rebuild the figure from `kvBytesPerDevice`.
     */
    it('names the device that made it impossible, not the one the readout describes', () => {
      // Gemma 3 12B over three 4090s at 128K and 8 users: the packing gives two cards seven and
      // eight layers against the third's thirty-three, so the busiest card by combined load is the
      // one with the *least* cache.
      //
      // **At Q8_0 rather than Q4_K_M, and #182 is why.** The two readings were within 0.2% of each
      // other on combined load at Q4_K_M, and seeding the vision tower onto the first bin was
      // enough to swap which one `busiest` returns — so the scenario stopped exhibiting the
      // property rather than the property stopping being true. That is `floorBytesPerDevice`'s own
      // docblock arriving in its test: `weightBytesPerDevice` is an argmax readout and a test
      // resting on which bin wins needs a margin. Q8_0 has one, at 24.95 GiB of floor against a
      // 18.57 GiB readout and a 23 GiB ceiling.
      const p = planPlacement(
        GEMMA_3_12B,
        getQuant('q8_0'),
        { contextTokens: 131072, concurrency: 8, kvPrecision: 'fp16' },
        { device: RTX_4090, count: 3 },
        LLAMA_CPP
      );

      expect(p.impossible).toBe(true);
      // The floor that actually refused it is over the ceiling...
      expect(p.floorBytesPerDevice).toBeGreaterThan(p.allocatableBytesPerDevice);
      // ...and it is not the floor of the device this readout describes, which is comfortably
      // under it. Rebuilding the sentence from these two fields printed a figure that disproved
      // the refusal beside it.
      expect(p.kvBytesPerDevice + p.activationBytesPerDevice).toBeLessThan(
        p.allocatableBytesPerDevice
      );
    });

    it('keeps the floor and the busiest device together whenever every device holds the same', () => {
      // Tensor parallelism, and any single-device rig: there is only one load, so the two figures
      // are the same number and the distinction above cannot arise.
      for (const runtime of [VLLM, LLAMA_CPP]) {
        for (const count of runtime === VLLM ? [1, 2, 4] : [1]) {
          const p = planPlacement(
            GEMMA_3_12B,
            getQuant(runtime === VLLM ? 'fp8' : 'q4_k_m'),
            { contextTokens: 131072, concurrency: 8, kvPrecision: 'fp16' },
            { device: RTX_4090, count },
            runtime
          );
          if (p.unsupported) continue;
          expect(p.floorBytesPerDevice).toBeCloseTo(
            p.kvBytesPerDevice + p.activationBytesPerDevice,
            6
          );
        }
      }
    });
  });

  it('does not grant a serial split aggregate bandwidth', () => {
    // Whole layers run in sequence for one token, so a single stream sees one card's bandwidth
    // however many cards there are. Tensor parallelism really does add channels.
    const perDevice = achievedBandwidth({ device: RTX_5090, count: 1 }, LLAMA_CPP);
    expect(achievedBandwidth({ device: RTX_5090, count: 8 }, LLAMA_CPP)).toBeCloseTo(perDevice, -6);

    const tp1 = achievedBandwidth({ device: RTX_5090, count: 1 }, VLLM);
    expect(achievedBandwidth({ device: RTX_5090, count: 8 }, VLLM)).toBeGreaterThan(tp1 * 4);
  });
});

/**
 * Sharding needs a transport, and the split ran regardless of whether one exists — eight Mac
 * Studios came back as a supported placement holding an eighth of the model each, over an
 * interconnect the catalog says they do not have. The Bench hides its device-count control for
 * these rows, but that is one surface's store protecting itself; the Matrix, the Envelope and any
 * direct `evaluate` caller reach `planPlacement` without passing through it.
 */
describe('a rig is only sharded when its devices can talk to each other', () => {
  const macRig = (count: number) => ({ device: MAC_STUDIO_M3_ULTRA_256, count });

  it('refuses a multi-device rig with no interconnect', () => {
    expect(canShard(MAC_STUDIO_M3_ULTRA_256)).toBe(false);

    const plan = planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), macRig(8), MLX);
    expect(plan.unsupported).toMatch(/no interconnect/i);
  });

  it('does not divide the model across devices it has refused', () => {
    const one = planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), macRig(1), MLX);
    const eight = planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), macRig(8), MLX);

    // The figure that gave it away: an eighth of the weights per machine, for a rig that cannot
    // move a tensor between them.
    expect(eight.weightBytesPerDevice).toBe(one.weightBytesPerDevice);
  });

  it('still shards hardware that has a link', () => {
    // The Spark is `unified-soc` like the Mac and has ConnectX-7, which is the whole reason
    // `canShard` keys on the transport rather than the device class.
    const rig = { device: DGX_SPARK, count: 2 };
    const plan = planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), rig, LLAMA_CPP);

    expect(plan.unsupported).toBeUndefined();
  });

  it('leaves a single device of unshardable hardware alone', () => {
    const plan = planPlacement(LLAMA_31_8B, getQuant('q4_k_m'), usage(4096), macRig(1), MLX);
    expect(plan.unsupported).toBeUndefined();
  });
});

/**
 * The grids evaluate a scenario chosen at one point against a whole row of contexts, so every
 * field that describes the working set has to be narrowed to the cell being drawn. The prompt was;
 * the cached prefix was not, for as long as both existed.
 */
describe('a scenario narrowed to one cell holds its whole working set', () => {
  const scenario = (over: Partial<UsageSpec> = {}): UsageSpec => ({
    contextTokens: 65536,
    concurrency: 1,
    kvPrecision: 'fp16',
    ...over,
  });

  it('caps the prompt at the context it is being asked about', () => {
    const narrowed = clampUsageToContext(scenario({ promptTokens: 32768 }), 2048);

    expect(narrowed.contextTokens).toBe(2048);
    expect(narrowed.promptTokens).toBe(2048);
  });

  it('caps the cached prefix at the room the prompt leaves, not at the context', () => {
    // Both fit the window alone. Together they are three times it, which is a session that
    // cannot be resident while that prompt is being read.
    const narrowed = clampUsageToContext(
      scenario({ promptTokens: 2048, cachedPrefixTokens: 4096 }),
      2048
    );

    expect(narrowed.promptTokens).toBe(2048);
    expect(narrowed.cachedPrefixTokens).toBe(0);
  });

  it('never lets the prompt and the prefix exceed the window between them', () => {
    for (const contextTokens of [512, 2048, 8192, 65536]) {
      for (const promptTokens of [1, 1024, 40000, 200000]) {
        for (const cachedPrefixTokens of [0, 1024, 50000, 1_000_000]) {
          const narrowed = clampUsageToContext(
            scenario({ promptTokens, cachedPrefixTokens }),
            contextTokens
          );

          expect(
            effectivePromptTokens(narrowed) + (narrowed.cachedPrefixTokens ?? 0)
          ).toBeLessThanOrEqual(contextTokens);
        }
      }
    }
  });

  it('holds an unstated prompt to the same bound as a stated one', () => {
    // The Envelope leaves `promptTokens` unset, so the prefix has to be measured against the
    // default prefill uses — 90% of the window — rather than against the whole window.
    const narrowed = clampUsageToContext(scenario({ cachedPrefixTokens: 1_000_000 }), 8192);

    expect(narrowed.promptTokens).toBeUndefined();
    expect(narrowed.cachedPrefixTokens).toBe(8192 - Math.floor(8192 * 0.9));
  });

  it('leaves a working set that already fits entirely alone', () => {
    const fits = scenario({ contextTokens: 8192, promptTokens: 2048, cachedPrefixTokens: 4096 });

    expect(clampUsageToContext(fits, 8192)).toEqual(fits);
  });

  it('does not invent fields the scenario did not state', () => {
    const narrowed = clampUsageToContext(scenario(), 4096);

    expect(narrowed.promptTokens).toBeUndefined();
    expect(narrowed.cachedPrefixTokens).toBeUndefined();
  });
});

/**
 * The assignment `planPlacement` packs, now kept rather than discarded (#136).
 *
 * Nothing here is new modelling: every figure comes from the same bins the byte totals already
 * came from. What the tests are for is the *reading* of those figures, because the one number the
 * launch emitter takes from this — `residentLayers`, which becomes llama.cpp's `-ngl` — is a layer
 * count derived from bytes, and this engine has already shipped one defect (#14) that was a ratio
 * from one scope applied to a quantity from another. The difference now is that the wrong answer
 * would be pasted into a shell rather than printed on a panel.
 */
describe('the layer assignment survives the packing', () => {
  const gemma = (count: number, runtime = LLAMA_CPP) =>
    planPlacement(
      GEMMA_3_12B,
      getQuant('q4_k_m'),
      { contextTokens: 131072, concurrency: 8, kvPrecision: 'fp16' },
      { device: RTX_5090, count },
      runtime
    );

  it('hands out every layer exactly once under a layer split', () => {
    for (const count of [1, 2, 3, 4, 5, 8]) {
      const { shares } = gemma(count).assignment;
      const handed = shares.reduce((sum, s) => sum + s.deviceCount * s.layers, 0);

      expect(handed, `${count} cards`).toBe(GEMMA_3_12B.layers);
    }
  });

  /**
   * **Which layers, not only how many** (#166).
   *
   * The count was the whole of what survived the packing, so the object described a *family* of
   * assignments rather than the one it had sized — and the members of that family have radically
   * different loads. This is the partition property that makes the new list a real answer rather
   * than a plausible one: every layer is somewhere, no layer is in two places, and each list agrees
   * with the count beside it.
   */
  it('says which layers each card holds, and hands every index out exactly once', () => {
    for (const count of [1, 2, 3, 4, 5, 8]) {
      const { shares } = gemma(count).assignment;
      const seen: number[] = [];

      for (const share of shares) {
        // The count and the list are two spellings of one fact, and a reader may use either.
        expect(share.layerIndices, `${count} cards`).toHaveLength(share.layers);
        // In the model's own order, which is what "layers 5, 11 and 17" means — the packing walks
        // them heaviest-first, so the list would otherwise arrive in load order.
        expect(
          [...share.layerIndices].sort((a, b) => a - b),
          `${count} cards`
        ).toEqual([...share.layerIndices]);
        seen.push(...share.layerIndices);
      }

      expect(
        seen.sort((a, b) => a - b),
        `${count} cards`
      ).toEqual(Array.from({ length: GEMMA_3_12B.layers }, (_, i) => i));
    }
  });

  /**
   * **The finding itself, now provable from the export.** #166's argument is that many assignments
   * share one set of counts and have radically different loads, so the counts alone do not
   * reproduce the layout the bytes beside them describe. On this rig the packing inverts them: the
   * cards holding the *most* layers hold the *fewest* full-attention ones, because one of those
   * caches ~128x what a sliding layer does at 128K over 8 users. Nothing in `layers` says that, and
   * the indices say it outright.
   */
  it('gives the cards with the most layers the fewest full-attention ones', () => {
    const shares = [...gemma(5).assignment.shares].sort((a, b) => a.layers - b.layers);
    const unbounded = shares.map(
      (s) => s.layerIndices.filter((i) => !isSlidingLayer(GEMMA_3_12B, i)).length
    );

    // The premise, asserted rather than assumed: the counts really are lopsided here.
    expect(shares[shares.length - 1].layers - shares[0].layers).toBeGreaterThan(1);

    // The inversion. The lightest-by-count card carries more of the expensive layers than the
    // heaviest-by-count one, which is the whole reason a count is not a description.
    expect(unbounded[0]).toBeGreaterThan(unbounded[unbounded.length - 1]);
    // And between them the cards hold every full-attention layer the model has, so the composition
    // is a partition of the same stack the counts partition.
    expect(unbounded.reduce((a, b) => a + b, 0)).toBe(
      Array.from({ length: GEMMA_3_12B.layers }, (_, i) => i).filter(
        (i) => !isSlidingLayer(GEMMA_3_12B, i)
      ).length
    );
  });

  it('gives a tensor-parallel rank every layer, because it holds a slice of each', () => {
    // The distinction `Assignment.parallelism` exists for. Four vLLM ranks each hold a quarter of
    // every tensor, so a rank's layer count is the model's and its *bytes* are the quarter — where
    // four llama.cpp cards each hold a quarter of the layers whole.
    const tp = planPlacement(
      GEMMA_3_12B,
      getQuant('bf16'),
      { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
      { device: RTX_5090, count: 4 },
      VLLM
    ).assignment;

    expect(tp.parallelism).toBe('tensor');
    expect(tp.shares).toHaveLength(1);
    expect(tp.shares[0].deviceCount).toBe(4);
    expect(tp.shares[0].layers).toBe(GEMMA_3_12B.layers);
    // And *which* layers is every layer, for the same reason — the list is not a layer-split
    // artefact that goes empty on the other branch. `parallelism` is still what says a rank holds
    // a slice of each of these rather than all of them whole.
    expect(tp.shares[0].layerIndices).toEqual(
      Array.from({ length: GEMMA_3_12B.layers }, (_, i) => i)
    );
  });

  /**
   * **The counts are wildly uneven on a hybrid model, and that is the finding rather than a
   * tolerance to widen.** My first version of this test asserted they land within one of each
   * other and failed by 19.
   *
   * The packing balances the *combined* per-layer cost, and at 128K over 8 users one of Gemma's
   * full-attention layers caches ~128x what a sliding one does — so a card that lands one full
   * layer is full, and a card holding only sliding layers takes twenty of them to reach the same
   * load. That is the whole reason `DeviceShare.layers` is counted during the walk rather than
   * recovered as `layers / deviceCount` afterwards: the even-split figure is not merely imprecise
   * here, it is wrong by a factor of three on the same rig.
   */
  it('hands out layers by weight of cache, not in equal counts', () => {
    const counts = gemma(5)
      .assignment.shares.map((s) => s.layers)
      .sort((a, b) => a - b);

    expect(counts.reduce((a, b) => a + b, 0)).toBe(GEMMA_3_12B.layers);
    // An even split would put every card within one of 48/5. The real assignment is not close.
    expect(counts[counts.length - 1] - counts[0]).toBeGreaterThan(1);

    // And the loads it balanced *are* close, which is what makes the count spread the right answer
    // rather than a bad packing.
    const loads = gemma(5)
      .assignment.shares.map((s) => s.weightBytes + s.kvBytes)
      .sort((a, b) => a - b);
    expect(loads[loads.length - 1] / loads[0]).toBeLessThan(1.5);
  });

  describe('resident layers are what `-ngl` is', () => {
    it('puts every layer on the device when nothing spills', () => {
      const resident = planPlacement(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        usage(4096),
        { device: RTX_5090, count: 1 },
        LLAMA_CPP
      );

      expect(resident.offloadFraction).toBe(0);
      expect(resident.assignment.residentLayers).toBe(LLAMA_31_8B.layers);
    });

    it('drops below the layer count exactly when weights spill', () => {
      // A 70B at Q4 is ~40 GiB against a 4090's 23 — comfortably a spill, and not an impossible
      // one, so there is a real layer count to report rather than a refusal.
      const spilled = planPlacement(
        QWEN3_32B,
        getQuant('bf16'),
        usage(4096),
        { device: RTX_4090, count: 1 },
        LLAMA_CPP
      );

      expect(spilled.offloadFraction).toBeGreaterThan(0);
      expect(spilled.assignment.residentLayers).toBeGreaterThan(0);
      expect(spilled.assignment.residentLayers).toBeLessThan(QWEN3_32B.layers);
    });

    /**
     * The property that makes the number safe to paste, and the one a rig-wide ratio breaks.
     *
     * Under an uneven layer split some cards are over their ceiling and some are not, so a device
     * holding ten resident layers sits beside one holding nine of ten. Deriving each card's count
     * from `offloadFraction` would charge the resident cards for the busy ones' overflow and
     * under-report `-ngl` across the whole rig — the #14 shape, in a shell command.
     */
    it('never claims a card holds more resident layers than it was given', () => {
      for (const count of [1, 2, 3, 4, 5, 8]) {
        for (const share of gemma(count).assignment.shares) {
          expect(share.residentLayers, `${count} cards`).toBeLessThanOrEqual(share.layers);
          expect(share.residentLayers, `${count} cards`).toBeGreaterThanOrEqual(0);
          expect(Number.isInteger(share.residentLayers), `${count} cards`).toBe(true);
        }
      }
    });

    it('leaves a card that is not over its ceiling holding all of its layers', () => {
      // The direct statement of the above: with the ceiling set just under the heaviest bin, the
      // cards below it must be untouched. A rig-wide fraction cannot express that.
      const ceiling = gemma(5);
      const heaviest = Math.max(...ceiling.assignment.shares.map((s) => s.weightBytes + s.kvBytes));
      const lightest = Math.min(...ceiling.assignment.shares.map((s) => s.weightBytes + s.kvBytes));
      expect(lightest, 'the split is even, so this proves nothing').toBeLessThan(heaviest);

      const squeezed = planPlacement(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        { contextTokens: 131072, concurrency: 8, kvPrecision: 'fp16' },
        {
          device: {
            ...RTX_5090,
            allocatableBytes: Math.floor(
              (heaviest + lightest) / 2 + ceiling.activationBytesPerDevice
            ),
          },
          count: 5,
        },
        LLAMA_CPP
      );

      const untouched = squeezed.assignment.shares.filter((s) => s.residentLayers === s.layers);
      const spilling = squeezed.assignment.shares.filter((s) => s.residentLayers < s.layers);
      expect(untouched.length, 'no card kept all its layers').toBeGreaterThan(0);
      expect(spilling.length, 'no card spilled, so the squeeze did nothing').toBeGreaterThan(0);
    });

    it('sums across a layer split and does not multiply a tensor-parallel rig by its rank count', () => {
      // The two arms of the one expression, and the reason they cannot be the same arm. Four cards
      // under a layer split hold four different quarters, so the rig's resident count is the sum;
      // four vLLM ranks hold slices of the same layers, so summing would report a rig running four
      // times the model it has.
      const split = gemma(4).assignment;
      expect(split.residentLayers).toBe(
        split.shares.reduce((sum, s) => sum + s.deviceCount * s.residentLayers, 0)
      );
      expect(split.residentLayers).toBeLessThanOrEqual(GEMMA_3_12B.layers);

      const tp = planPlacement(
        GEMMA_3_12B,
        getQuant('bf16'),
        { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
        { device: RTX_5090, count: 4 },
        VLLM
      ).assignment;
      expect(tp.residentLayers).toBeLessThanOrEqual(GEMMA_3_12B.layers);
    });

    it('reports every layer resident on hardware with nowhere to spill from', () => {
      // Unified memory has no faster tier, so `canOffload` is false and nothing is ever charged as
      // spilled — which means a placement that does not fit still reports every layer resident.
      // `impossible` is the field that says it will not run; this one describes a layout.
      const mac = planPlacement(
        GPT_OSS_120B,
        getQuant('bf16'),
        usage(4096),
        { device: MAC_STUDIO_M3_ULTRA_256, count: 1 },
        MLX
      );

      expect(mac.impossible).toBe(true);
      expect(mac.offloadFraction).toBe(0);
      expect(mac.assignment.residentLayers).toBe(GPT_OSS_120B.layers);
    });
  });

  /**
   * The count is taken against the layers, not against the file (#165).
   *
   * `layerSplitBins` used to charge every layer `totalWeightBytes / layers`, which is a layer plus
   * a share of tensors no layer holds — the embedding table, the output projection where it is
   * untied, any vision tower. As a byte total that was right in aggregate; as a *layer count* it is
   * a different quantity, and #163 exported these bins as a layer count while #136 printed it as
   * `-ngl`. So the count divides `layerWeightBytes`.
   *
   * **#165 left the bytes alone and #182 has now moved them**, which is why the scope test below
   * reads the opposite way round from the one it replaces. The count's basis is unchanged; what
   * changed underneath it is that `weightBytes` and `layerWeightBytes` now differ by a *placed*
   * block rather than a smeared share of one.
   */
  describe('a layer count is taken against the layers', () => {
    const quant = getQuant('q4_k_m');

    /**
     * Llama 3.2 3B is the row #165 was filed against: 12.3% of the file is a tied 128,256 x 3,072
     * table, spread across 28 layers that do not hold it. Not the catalog's largest such share —
     * Gemma 3 4B is 25.4% with its tower — so the gap this shows is a floor rather than a headline.
     * The ceiling is stated rather than found, because a 3B model does not spill on anything anyone
     * sells, and the spilled branch is the only one where the two bases differ at all.
     */
    it('reports a count the device can actually load, where the whole-file basis did not', () => {
      const { layerBytes, fixedBytes } = weightBreakdown(LLAMA_32_3B, quant);
      const perLayer = layerBytes / LLAMA_32_3B.layers;

      const p = planPlacement(
        LLAMA_32_3B,
        quant,
        usage(8192),
        { device: { ...RTX_4090, allocatableBytes: 2 * GIB }, count: 1 },
        LLAMA_CPP
      );
      const share = p.assignment.shares[0];

      expect(p.fits).toBe(false);
      expect(p.impossible).toBe(false);

      // What the card has left for weights once the cache and the overhead are on it, and what the
      // reported count asks of that: this many real layers, beside the table it also holds.
      const budget = p.allocatableBytesPerDevice - share.kvBytes - p.activationBytesPerDevice;
      expect(share.residentLayers * perLayer + fixedBytes).toBeLessThanOrEqual(budget);

      // The same overflow read against the whole share, which is what this used to do. It asks the
      // card for 0.62 GiB where the card has 0.50, and `-ngl` is where that lands.
      // `deviceWeightBytes`, since #182: the fraction is of what the cards were asked to hold.
      const spilled = p.offloadFraction * p.deviceWeightBytes;
      const wholeFileBasis = Math.floor(
        ((share.weightBytes - spilled) / share.weightBytes) * share.layers
      );
      expect(wholeFileBasis).toBeGreaterThan(share.residentLayers);
      expect(wholeFileBasis * perLayer + fixedBytes).toBeGreaterThan(budget);
    });

    it('places the fixed tensors where the rig holds them, and accounts for every byte', () => {
      // #165's scope line, inverted by #182. It used to read "leaves every byte figure exactly
      // where it was", and the reason was that assigning the fixed tensors to one bin is a change
      // to what the product answers rather than to a layer count — true, and now made on purpose
      // with upstream read rather than reasoned about.
      //
      // What survives is the accounting: nothing may be lost or double-counted by the placing. The
      // bins sum to `deviceWeightBytes`, and each bin is a whole number of repeating layers plus
      // exactly the block that was placed on it.
      const { layerBytes, towerBytes, outputBytes } = weightBreakdown(GEMMA_3_12B, quant);
      const perLayer = layerBytes / GEMMA_3_12B.layers;

      for (const count of [1, 2, 3, 4, 5, 8]) {
        const p = gemma(count);
        const held = p.assignment.shares.reduce((sum, s) => sum + s.deviceCount * s.weightBytes, 0);
        expect(held, `${count} cards`).toBeCloseTo(p.deviceWeightBytes, 0);

        const last = p.assignment.shares.length - 1;
        for (const [i, s] of p.assignment.shares.entries()) {
          const placed = (i === 0 ? towerBytes : 0) + (i === last ? outputBytes : 0);
          expect((s.weightBytes - placed) / perLayer, `${count} cards, bin ${i}`).toBeCloseTo(
            s.layers,
            6
          );
        }
      }
    });

    /**
     * The ordering invariant, which is a new one and is not a preference (#182).
     *
     * llama.cpp indexes the output slot through the same `upper_bound` over `tensor_split` that the
     * repeating layers go through, so it puts the output projection on the **last** device with a
     * share of the `-ngl` window — whichever bin Headroom nominated. A packing that seeds any other
     * bin is asking for a layout llama.cpp will not execute, and `launch.ts` would then print a
     * `-ts` whose extra slot lands on a card that was never sized for the table.
     *
     * Synthesised rather than pinned to a catalog row (#197): what makes the property visible is an
     * untied table large enough to see beside a layer, and that is a shape rather than a model.
     */
    it('emits the bin holding the output projection last, whatever the packing', () => {
      const model: ModelSpec = {
        ...LLAMA_31_8B,
        id: 'test/ordering',
        // A deliberately fat vocabulary against a shallow stack, so the table is worth several
        // layers and the walk has a real imbalance to work around.
        vocabSize: 200_000,
        hiddenSize: 4_096,
        layers: 12,
        tiedEmbeddings: false,
      };
      const { layerBytes, outputBytes } = weightBreakdown(model, quant);
      const perLayer = layerBytes / model.layers;
      expect(outputBytes).toBeGreaterThan(perLayer);

      for (const count of [2, 3, 4, 5, 8]) {
        const p = planPlacement(
          model,
          quant,
          { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
          { device: RTX_5090, count },
          LLAMA_CPP
        );
        const shares = p.assignment.shares;
        const extra = shares.map((s) => s.weightBytes - s.layers * perLayer);

        // The last bin carries the table, and no other bin carries anything.
        expect(extra[extra.length - 1], `${count} cards`).toBeCloseTo(outputBytes, 0);
        for (const [i, e] of extra.entries()) {
          if (i < extra.length - 1) expect(e, `${count} cards, bin ${i}`).toBeCloseTo(0, 0);
        }
        // And no bin is emitted holding nothing but the table — a card the model cannot use is
        // dropped rather than described (#182, decision 4).
        for (const [i, s] of shares.entries())
          expect(s.layers, `${count} cards, bin ${i}`).toBeGreaterThan(0);
      }
    });

    it('suppresses a bin the model cannot put a layer on, rather than emitting an empty card', () => {
      // A table worth more than three layers over eight cards: the walk would leave the seeded bin
      // with the table and nothing else, which is legal, reproducible and reads as a bug in a
      // command. The rig is told it has more cards than the model can use instead.
      const model: ModelSpec = {
        ...LLAMA_31_8B,
        id: 'test/suppression',
        vocabSize: 262_144,
        hiddenSize: 4_096,
        layers: 8,
        tiedEmbeddings: false,
      };
      const p = planPlacement(
        model,
        quant,
        { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16' },
        { device: RTX_5090, count: 8 },
        LLAMA_CPP
      );

      expect(p.assignment.shares.length).toBeLessThan(8);
      for (const s of p.assignment.shares) expect(s.layers).toBeGreaterThan(0);
      // Every layer is still placed, and the rig still holds every byte it is charged for.
      expect(p.assignment.shares.reduce((n, s) => n + s.layers, 0)).toBe(model.layers);
      expect(
        p.assignment.shares.reduce((sum, s) => sum + s.deviceCount * s.weightBytes, 0)
      ).toBeCloseTo(p.deviceWeightBytes, 0);
    });

    it('leaves a fully resident rig reporting every layer, on either parallelism', () => {
      // The two bases agree exactly whenever nothing spills, which is most of the catalog most of
      // the time — so this fix must be invisible there, and the tensor-parallel branch untouched.
      for (const runtime of [LLAMA_CPP, VLLM]) {
        for (const count of [1, 2, 4]) {
          const p = planPlacement(
            LLAMA_32_3B,
            getQuant(runtime === VLLM ? 'fp8' : 'q4_k_m'),
            usage(8192),
            { device: RTX_5090, count },
            runtime
          );
          if (p.unsupported) continue;

          expect(p.fits, `${runtime.id} x${count}`).toBe(true);
          expect(p.assignment.residentLayers, `${runtime.id} x${count}`).toBe(LLAMA_32_3B.layers);
        }
      }
    });
  });

  it('keeps the shares agreeing with the per-device figures the panels read', () => {
    // The busiest bin is what `weightBytesPerDevice` and `kvBytesPerDevice` come from, so a share
    // list that disagreed with them would be a second derivation of one quantity — which is the
    // failure this file's own history is mostly made of.
    for (const count of [1, 2, 5]) {
      const p = gemma(count);
      const busiest = p.assignment.shares.reduce((a, b) =>
        b.weightBytes + b.kvBytes > a.weightBytes + a.kvBytes ? b : a
      );

      expect(busiest.weightBytes, `${count} cards`).toBeCloseTo(p.weightBytesPerDevice, 6);
      expect(busiest.kvBytes, `${count} cards`).toBeCloseTo(p.kvBytesPerDevice, 6);
    }
  });
});

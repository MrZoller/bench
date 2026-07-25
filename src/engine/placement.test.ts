import { describe, expect, it } from 'vitest';
import {
  DEFAULT_HOST_BANDWIDTH,
  allocatablePerDevice,
  maxAllocatablePerDevice,
  raisingCeilingWouldHelp,
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
import { getQuant } from '@/data/quants';
import { GIB } from './types';
import type { DeviceSpec, UsageSpec } from './types';

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

    expect(two.weightBytesPerDevice).toBeCloseTo(one.weightBytesPerDevice / 2, -6);
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

  it('treats a Mac default as raiseable up to physical memory', () => {
    // `iogpu.wired_limit_mb` is a default at 75%, not a hardware limit.
    expect(maxAllocatablePerDevice(MAC_STUDIO_M3_ULTRA_256)).toBe(
      MAC_STUDIO_M3_ULTRA_256.capacityBytes
    );
    const between = MAC_STUDIO_M3_ULTRA_256.allocatableBytes + 1;
    expect(raisingCeilingWouldHelp(MAC_STUDIO_M3_ULTRA_256, between)).toBe(true);
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
    expect(two.weightBytesPerDevice).toBeCloseTo((one.weightBytesPerDevice * 31) / 61, -3);
    // And the same divisor as the cache, which is the property that was broken.
    expect(two.weightBytesPerDevice / one.weightBytesPerDevice).toBeCloseTo(
      two.kvBytesPerDevice / one.kvBytesPerDevice,
      6
    );
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

    expect(five.kvBytesPerDevice).toBeGreaterThan(evenShare);

    // And a count that *does* divide the full layers evenly gets the even share, so the model is
    // charging for real imbalance rather than adding a blanket penalty.
    const eight = gemma(8);
    expect(eight.kvBytesPerDevice).toBeCloseTo(gemma(1).kvBytesPerDevice / 8, -3);
  });

  it('never claims a card holds more than the whole cache', () => {
    for (const count of [1, 2, 4, 8]) {
      const p = gemma(count);
      expect(p.kvBytesPerDevice).toBeLessThanOrEqual(p.totalKvBytes + 1);
    }
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

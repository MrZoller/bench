import { describe, expect, it } from 'vitest';
import { allocatablePerDevice, maxContextThatFits, planPlacement } from './placement';
import {
  DGX_SPARK,
  GPT_OSS_120B,
  LLAMA_31_8B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_256,
  MLX,
  QWEN3_32B,
  RTX_5090,
  STRIX_HALO_395,
  VLLM,
} from './fixtures';
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

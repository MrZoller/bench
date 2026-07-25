import { describe, expect, it } from 'vitest';
import { achievedBandwidth, estimateDecode, estimatePrefill } from './speed';
import { effectiveActiveParams } from './weights';
import { planPlacement } from './placement';
import {
  DEEPSEEK_V3,
  DGX_SPARK,
  EPYC_9654,
  GPT_OSS_20B,
  LLAMA_31_8B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_256,
  QWEN3_32B,
  RTX_5090,
  VLLM,
} from './fixtures';
import { getQuant } from '@/data/quants';
import type { DeviceSpec, ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { denseParams, TFLOP } from './types';

function decode(
  model: ModelSpec,
  quant: QuantSpec,
  usage: UsageSpec,
  rig: Rig,
  runtime: RuntimeSpec = LLAMA_CPP
) {
  const placement = planPlacement(model, quant, usage, rig, runtime);
  return estimateDecode(model, quant, usage, rig, runtime, placement);
}

const single = (contextTokens: number): UsageSpec => ({
  contextTokens,
  concurrency: 1,
  kvPrecision: 'fp16',
});

/**
 * Two independent published measurements, at opposite ends of the hardware range. If the
 * roofline is only calibrated for discrete GPUs, one of these fails — which is the point of
 * anchoring both rather than one.
 *
 * The band is +/-30%: tight enough to catch a structural error, honest about what a roofline
 * can claim when it cannot see the scheduler or the kernels.
 */
describe('calibration against published benchmarks', () => {
  it('reproduces DGX Spark decoding gpt-oss-20b at ~49.7 tok/s', () => {
    const result = decode(GPT_OSS_20B, getQuant('mxfp4'), single(4096), {
      device: DGX_SPARK,
      count: 1,
    });

    expect(result.perUserTokensPerSec).toBeGreaterThan(49.7 * 0.7);
    expect(result.perUserTokensPerSec).toBeLessThan(49.7 * 1.3);
  });

  it('reproduces EPYC 9654 decoding DeepSeek-671B Q8 at ~6 tok/s', () => {
    const result = decode(DEEPSEEK_V3, getQuant('q8_0'), single(4096), {
      device: EPYC_9654,
      count: 1,
    });

    expect(result.perUserTokensPerSec).toBeGreaterThan(6 * 0.7);
    expect(result.perUserTokensPerSec).toBeLessThan(6 * 1.3);
  });

  it('reproduces DGX Spark prefilling gpt-oss-20b at ~2053 tok/s', () => {
    const result = estimatePrefill(
      GPT_OSS_20B,
      getQuant('mxfp4'),
      { ...single(4096), promptTokens: 1024 },
      { device: DGX_SPARK, count: 1 },
      LLAMA_CPP
    );

    // Two-sided, like the decode anchors. A one-sided floor here previously hid an 8x error:
    // reading the tensor-core rate off storage width handed llama.cpp the FP4 peak it cannot
    // reach, and the test passed anyway because 33,000 is greater than 500.
    expect(result.prefillTokensPerSec).toBeGreaterThan(2053 * 0.7);
    expect(result.prefillTokensPerSec).toBeLessThan(2053 * 1.3);
  });

  it('does not hand llama.cpp a tensor-core rate it cannot reach', () => {
    const rig = { device: DGX_SPARK, count: 1 };
    const usage = { ...single(4096), promptTokens: 1024 };

    // IQ4_XS stores at 4.46 bits but dequantizes to fp16 to compute. Keying the rate off
    // storage width would have picked the Spark's 1000 TFLOP FP4 number.
    const iq4 = estimatePrefill(GPT_OSS_20B, getQuant('iq4_xs'), usage, rig, LLAMA_CPP);
    const bf16 = estimatePrefill(GPT_OSS_20B, getQuant('bf16'), usage, rig, LLAMA_CPP);
    expect(iq4.prefillTokensPerSec).toBeCloseTo(bf16.prefillTokensPerSec, 0);
  });

  /**
   * The other half of that rule: a runtime that *does* dispatch natively must get the faster
   * rate. Without this the low-precision branch is dead code — llama.cpp returns fp16
   * unconditionally, so nothing else in the suite reaches it.
   */
  it('gives a native runtime the low-precision rate, and falls back when the device lacks it', () => {
    const usage = { ...single(4096), promptTokens: 1024 };
    const rig = { device: RTX_5090, count: 1 };

    const bf16 = estimatePrefill(QWEN3_32B, getQuant('bf16'), usage, rig, VLLM);
    const fp8 = estimatePrefill(QWEN3_32B, getQuant('fp8'), usage, rig, VLLM);
    // The 5090 lists 419 TFLOP fp16 and 838 TFLOP fp8.
    expect(fp8.prefillTokensPerSec / bf16.prefillTokensPerSec).toBeCloseTo(2, 1);

    // The 5090 has real FP4 tensor cores at 4x fp16, so NVFP4 — a uniform format — gets them.
    // The no-fallback rule is asserted separately, against cards that genuinely lack the units.
    const nvfp4 = estimatePrefill(QWEN3_32B, getQuant('nvfp4'), usage, rig, VLLM);
    expect(nvfp4.prefillTokensPerSec / bf16.prefillTokensPerSec).toBeCloseTo(4, 0);
    expect(nvfp4.prefillTokensPerSec).toBeGreaterThan(fp8.prefillTokensPerSec);
  });

  it('charges sliding-window layers linear rather than quadratic prefill attention', () => {
    const usage = { ...single(32768), promptTokens: 16384 };
    const rig = { device: DGX_SPARK, count: 1 };
    const result = estimatePrefill(GPT_OSS_20B, getQuant('mxfp4'), usage, rig, LLAMA_CPP);

    // 12 full layers over the whole prompt, 12 sliding layers capped at 128 keys.
    const span = 12 * 16384 + 12 * 128;
    expect(result.attentionFlops).toBe(4 * 16384 * span * GPT_OSS_20B.attention.projectionWidth);

    // Treating every layer as full attention — the mistake kv.ts already avoids on the
    // memory side — would overstate the attention term by nearly 2x.
    const uniform = 4 * 16384 * (24 * 16384) * GPT_OSS_20B.attention.projectionWidth;
    expect(uniform / result.attentionFlops).toBeGreaterThan(1.9);
  });

  it('puts an 8B dense model on a 5090 in the expected few-hundred tok/s range', () => {
    const result = decode(LLAMA_31_8B, getQuant('q4_k_m'), single(4096), {
      device: RTX_5090,
      count: 1,
    });
    expect(result.perUserTokensPerSec).toBeGreaterThan(200);
    expect(result.perUserTokensPerSec).toBeLessThan(400);
  });
});

describe('the capacity/bandwidth/compute triangle', () => {
  /**
   * The comparison the whole tool exists to make: same model, two machines, opposite
   * strengths. Spark fits it comfortably and decodes slowly; the Mac decodes it much faster.
   */
  it('has the Mac out-decoding the Spark on the same model, by roughly their bandwidth ratio', () => {
    const usage = single(8192);
    const quant = getQuant('mxfp4');

    const spark = decode(GPT_OSS_20B, quant, usage, { device: DGX_SPARK, count: 1 });
    const mac = decode(GPT_OSS_20B, quant, usage, {
      device: MAC_STUDIO_M3_ULTRA_256,
      count: 1,
    });

    expect(mac.perUserTokensPerSec).toBeGreaterThan(spark.perUserTokensPerSec);
    // 819 GB/s against 273 GB/s.
    expect(mac.perUserTokensPerSec / spark.perUserTokensPerSec).toBeCloseTo(3, 0);
  });

  it('has the Spark out-prefilling the Mac despite decoding slower', () => {
    const usage = { ...single(32768), promptTokens: 16384 };
    const quant = getQuant('mxfp4');

    const spark = estimatePrefill(
      GPT_OSS_20B,
      quant,
      usage,
      { device: DGX_SPARK, count: 1 },
      LLAMA_CPP
    );
    const mac = estimatePrefill(
      GPT_OSS_20B,
      quant,
      usage,
      { device: MAC_STUDIO_M3_ULTRA_256, count: 1 },
      LLAMA_CPP
    );

    expect(spark.ttftSeconds).toBeLessThan(mac.ttftSeconds);
  });
});

describe('long context shifts the bottleneck to KV', () => {
  it('is weight-bound at short context and KV-bound at long context', () => {
    const quant = getQuant('q4_k_m');
    const rig = { device: RTX_5090, count: 1 };

    expect(decode(LLAMA_31_8B, quant, single(1024), rig).kvBound).toBe(false);
    // 128 KiB/token against ~4.9 GB of weights: KV overtakes somewhere past ~37K tokens.
    expect(decode(LLAMA_31_8B, quant, single(131072), rig).kvBound).toBe(true);
  });

  it('slows decode materially as context grows, even though the weights never change', () => {
    const quant = getQuant('q4_k_m');
    const rig = { device: RTX_5090, count: 1 };

    const short = decode(LLAMA_31_8B, quant, single(1024), rig).perUserTokensPerSec;
    const long = decode(LLAMA_31_8B, quant, single(131072), rig).perUserTokensPerSec;
    expect(long).toBeLessThan(short / 2);
  });
});

describe('MoE experts accumulate with batch size', () => {
  /**
   * DeepSeek publishes 37B active. The decode basis sits just under it, and the gap is exactly
   * the input embedding: 129280 x 7168 = 0.93B, which an untied model reads a single row of per
   * token rather than in full. Counting the whole dense half instead gives 37.5B — the residency
   * figure, not the per-token one.
   */
  it('lands just under the published active count, short by the embedding table', () => {
    const basis = effectiveActiveParams(DEEPSEEK_V3, 1);
    expect(basis / 1e9).toBeCloseTo(36.6, 0);

    const embedding = DEEPSEEK_V3.vocabSize * DEEPSEEK_V3.hiddenSize;
    const residencyBasis = denseParams(DEEPSEEK_V3) + DEEPSEEK_V3.expertParams * (8 / 256);
    expect(residencyBasis - basis).toBeCloseTo(embedding, -6);
    expect(basis).toBeLessThan(DEEPSEEK_V3.activeParams);
  });

  it('approaches total parameters as batch grows', () => {
    const big = effectiveActiveParams(DEEPSEEK_V3, 512);
    expect(big).toBeGreaterThan(DEEPSEEK_V3.totalParams * 0.9);
    expect(big).toBeLessThanOrEqual(DEEPSEEK_V3.totalParams);
  });

  it('leaves a dense model unchanged at any batch size', () => {
    expect(effectiveActiveParams(LLAMA_31_8B, 64)).toBe(LLAMA_31_8B.activeDenseParams);
    expect(effectiveActiveParams(LLAMA_31_8B, 64)).toBe(effectiveActiveParams(LLAMA_31_8B, 1));
  });

  it('gains less per-user throughput from batching than a dense model does', () => {
    // Aggregate throughput still rises with batch — batching helps — but the expert union
    // grows too, so MoE gains less than the dense case. This is the effect that surprises
    // people putting an MoE model behind a multi-user endpoint.
    const quant = getQuant('q4_k_m');
    const rig = { device: RTX_5090, count: 8 };

    const gain = (model: ModelSpec) => {
      const at = (concurrency: number) =>
        decode(model, quant, { contextTokens: 2048, concurrency, kvPrecision: 'fp16' }, rig)
          .aggregateTokensPerSec;
      return at(32) / at(1);
    };

    expect(gain(DEEPSEEK_V3)).toBeLessThan(gain(LLAMA_31_8B));
  });
});

describe('offload is a cliff', () => {
  it('collapses throughput once weights spill to host RAM', () => {
    const quant = getQuant('q4_k_m');
    const usage = single(4096);

    // 406 GB of weights against a 32 GB card: almost all of it has to live in host RAM.
    const rig = { device: RTX_5090, count: 1 };
    const placement = planPlacement(DEEPSEEK_V3, quant, usage, rig, LLAMA_CPP);
    expect(placement.offloadFraction).toBeGreaterThan(0);

    const result = estimateDecode(DEEPSEEK_V3, quant, usage, rig, LLAMA_CPP, placement);
    expect(result.offloadPenalty).toBeDefined();
    expect(result.perUserTokensPerSec).toBeLessThan(
      result.offloadPenalty!.withoutOffloadTokensPerSec / 2
    );
  });
});

/**
 * Compute precision is looked up per dtype, and the lookup has to be able to *miss*.
 *
 * INT8 and FP8 run at the same rate on hardware that has both, which makes aliasing them
 * tempting. Ampere is the counterexample: 2x fp16 on INT8 tensor cores, no FP8 units at all.
 * Aliasing in either direction invents a rate one of those cards cannot reach.
 */
describe('compute precision lookup', () => {
  const ampereLike: DeviceSpec = {
    ...RTX_5090,
    id: 'int8-only',
    // The Ampere shape: an INT8 rate at 2x fp16, and no FP8 entry whatsoever.
    flops: { fp16: 142 * TFLOP, int8: 284 * TFLOP },
  };
  const prefill = (quantId: string, device: DeviceSpec = ampereLike) =>
    estimatePrefill(
      QWEN3_32B,
      getQuant(quantId),
      { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16', promptTokens: 8192 },
      { device, count: 1 },
      VLLM
    ).prefillTokensPerSec;

  it('reaches the catalogued INT8 rate on a card that publishes one', () => {
    expect(prefill('int8') / prefill('bf16')).toBeCloseTo(2, 1);
  });

  it('refuses to lend the INT8 rate to an FP8 quant', () => {
    // No FP8 tensor cores means FP8 falls back to fp16, not to the INT8 headline.
    expect(prefill('fp8')).toBeCloseTo(prefill('bf16'), 1);
  });

  it('still serves INT8 from the FP8 rate when only that is published', () => {
    const fp8Only: DeviceSpec = { ...RTX_5090, flops: { fp16: 419 * TFLOP, fp8: 838 * TFLOP } };
    expect(prefill('int8', fp8Only) / prefill('bf16', fp8Only)).toBeCloseTo(2, 1);
  });
});

/**
 * Interconnect tiers. Matching only /nvlink/ collapsed Infinity Fabric and Ethernet into the
 * PCIe default, which at eight devices compounds over three doublings — and the two err in
 * opposite directions, so a single default cannot serve both.
 */
describe('tensor-parallel scaling by interconnect', () => {
  const withLink = (interconnect: string): DeviceSpec => ({ ...RTX_5090, interconnect });
  const aggregate = (interconnect: string, count: number) =>
    achievedBandwidth({ device: withLink(interconnect), count }, VLLM);

  it('treats Infinity Fabric as fabric-class, not PCIe', () => {
    expect(aggregate('Infinity Fabric', 8)).toBeCloseTo(aggregate('NVLink 4.0', 8), -9);
    expect(aggregate('Infinity Fabric', 8)).toBeGreaterThan(aggregate('PCIe 5.0 x16', 8) * 1.3);
  });

  it('puts an Ethernet link below PCIe rather than equal to it', () => {
    expect(aggregate('ConnectX-7 200GbE', 8)).toBeLessThan(aggregate('PCIe 5.0 x16', 8));
  });

  it('applies the tier only once there is more than one device', () => {
    const links = ['NVLink 5.0', 'PCIe 4.0 x16', 'ConnectX-7 200GbE'];
    // Identical at count 1 — no link is crossed, so its quality cannot matter...
    for (const link of links) {
      expect(aggregate(link, 1)).toBeCloseTo(aggregate('NVLink 5.0', 1), -9);
    }
    // ...and all three separate at count 2, which is what makes the agreement above meaningful
    // rather than an artifact of the early return.
    const [fabric, pcie, network] = links.map((l) => aggregate(l, 2));
    expect(fabric).toBeGreaterThan(pcie);
    expect(pcie).toBeGreaterThan(network);
  });
});

/**
 * The precision lookup must be able to miss in *both* directions, and the two rules are
 * symmetric: a card is only ever given a rate for silicon it actually has.
 */
describe('precision fallbacks never invent silicon', () => {
  const prefill = (quantId: string, flops: DeviceSpec['flops']) =>
    estimatePrefill(
      QWEN3_32B,
      getQuant(quantId),
      { contextTokens: 8192, concurrency: 1, kvPrecision: 'fp16', promptTokens: 8192 },
      { device: { ...RTX_5090, flops }, count: 1 },
      VLLM
    ).prefillTokensPerSec;

  it('does not lend an FP8 rate to FP4 on a card with no FP4 units', () => {
    // The H100 shape: strong FP8, no FP4 at all.
    const hopper = { fp16: 989 * TFLOP, fp8: 1979 * TFLOP };
    expect(prefill('nvfp4', hopper)).toBeCloseTo(prefill('bf16', hopper), 0);
  });

  it('does not lend an INT8 rate to FP8 on a card with no FP8 units', () => {
    // The Ampere shape: INT8 tensor cores, no FP8.
    const ampere = { fp16: 142 * TFLOP, int8: 284 * TFLOP };
    expect(prefill('fp8', ampere)).toBeCloseTo(prefill('bf16', ampere), 0);
  });

  it('still uses a real FP4 rate when the card publishes one', () => {
    const blackwell = { fp16: 419 * TFLOP, fp8: 838 * TFLOP, fp4: 1676 * TFLOP };
    expect(prefill('nvfp4', blackwell) / prefill('bf16', blackwell)).toBeCloseTo(4, 0);
  });
});

/**
 * An expert-only quantization computes at two rates, and crediting the whole prefill pass at
 * the faster one overstates every model with a substantial dense half.
 */
describe('expert-only formats are timed at two rates', () => {
  // A real Blackwell shape: FP4 units present at 4x fp16.
  // The 5090 fixture already carries real FP4 rates; named for what the test is about.
  const blackwell = RTX_5090;
  const rate = (model: typeof QWEN3_32B, quantId: string) =>
    estimatePrefill(
      model,
      getQuant(quantId),
      { contextTokens: 4096, concurrency: 1, kvPrecision: 'fp16', promptTokens: 1024 },
      { device: blackwell, count: 1 },
      VLLM
    ).prefillTokensPerSec;

  it('gives a dense model no speedup at all from an expert-only format', () => {
    // MXFP4 sets denseBpw: 16, so with no routed experts every tensor stays BF16. Claiming the
    // FP4 peak here was a 4x on work that does not exist.
    expect(rate(QWEN3_32B, 'mxfp4')).toBeCloseTo(rate(QWEN3_32B, 'bf16'), 0);
  });

  it('gives an MoE model a partial speedup, between the two rates', () => {
    const speedup = rate(GPT_OSS_20B, 'mxfp4') / rate(GPT_OSS_20B, 'bf16');
    expect(speedup).toBeGreaterThan(1.5);
    expect(speedup).toBeLessThan(4);
  });

  it('still gives a uniform 4-bit format the full rate', () => {
    // NVFP4 has no denseBpw, so nothing is spared and the whole pass runs at the FP4 peak.
    for (const model of [QWEN3_32B, GPT_OSS_20B]) {
      expect(rate(model, 'nvfp4') / rate(model, 'bf16')).toBeCloseTo(4, 0);
    }
  });
});

/**
 * Once the expert half computes at a different rate from everything else, FLOP counts stop
 * being comparable and the attention/linear bound has to be judged on time.
 */
describe('the prefill bound is judged on time, not FLOPs', () => {
  // The 5090 fixture already carries real FP4 rates; named for what the test is about.
  const blackwell = RTX_5090;

  it('reports attention-bound when attention costs more time than the linear layers', () => {
    const result = estimatePrefill(
      GPT_OSS_20B,
      getQuant('mxfp4'),
      { contextTokens: 32768, concurrency: 1, kvPrecision: 'fp16', promptTokens: 16384 },
      { device: blackwell, count: 1 },
      VLLM
    );

    // Attention is the smaller FLOP count and still the larger share of the wall clock, because
    // most of the linear work runs at the 4x FP4 rate and attention does not.
    expect(result.attentionFlops).toBeLessThan(result.linearFlops);
    expect(result.attentionBound).toBe(true);
  });

  it('agrees with the FLOP comparison when both halves run at one rate', () => {
    // bf16 is uniform, so time is proportional to FLOPs and the two tests coincide.
    const result = estimatePrefill(
      GPT_OSS_20B,
      getQuant('bf16'),
      { contextTokens: 32768, concurrency: 1, kvPrecision: 'fp16', promptTokens: 16384 },
      { device: blackwell, count: 1 },
      VLLM
    );
    expect(result.attentionBound).toBe(result.attentionFlops > result.linearFlops);
  });
});

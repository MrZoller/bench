import type {
  DeviceClass,
  DeviceSpec,
  ModelSpec,
  QuantSpec,
  Rig,
  RuntimeSpec,
  UsageSpec,
} from './types';
import { effectiveBandwidth } from './types';
import { attentionSpanPerToken, kvReadBytesPerToken } from './kv';
import { activeWeightBytes, outputProjectionParams, prefillComputeParams } from './weights';
import type { Placement } from './placement';
import { DEFAULT_HOST_BANDWIDTH } from './placement';

/**
 * Throughput and latency, as a roofline.
 *
 * Decode is memory-bound: every token re-reads the active weights and the whole KV cache for
 * its sequence, doing about one FLOP per byte. Prefill is compute-bound: the entire prompt
 * goes through the network in parallel. A device can be strong at one and weak at the other —
 * a DGX Spark prefills fast and decodes slowly, a Mac Studio does the reverse — which is
 * precisely what a single "speed" number cannot express.
 *
 * **On accuracy.** This is a roofline, not a simulator. Against the three published anchors in
 * speed.test.ts it reads ~19% over on DGX Spark decode, ~10% over on Spark prefill, and within
 * 1% on EPYC decode. It cannot model scheduler behaviour, per-model kernel quality, or thermal
 * throttling. The app must present these as estimates with a band, never as promises.
 *
 * Those first two used to read ~10% and ~6% *under*, and none of the constants below were
 * touched to move them. What changed is that the per-token parameter basis stopped counting
 * work the hardware does not do — the input embedding table decode never reads, and the output
 * projection prefill computes for one position rather than every prompt token. The old
 * calibration was partly absorbing both, which is why correcting them moved gpt-oss-20b decode
 * by 31% while barely touching DeepSeek on EPYC, whose embedding is 2.5% of its active
 * parameters.
 *
 * The knobs were deliberately left alone rather than re-centred on the Spark points: re-tuning
 * a fudge factor immediately after removing the error it was masking is how the next error gets
 * hidden. The residual is now honest and sits inside the +/-30% band the tests assert.
 *
 * **On the calibration constants.** `bandwidthEfficiency` and `CLASS_BANDWIDTH_UTILIZATION`
 * are two free multiplicative knobs fitted to two data points, and only their *product* is
 * observable. The split between "what the runtime achieves" and "what the memory subsystem
 * allows" is not identifiable from this data — it is a defensible physical story, not a
 * measured decomposition. It becomes a testable claim the first time a second CPU-capable
 * runtime or a second CPU device is added, and should be re-derived then rather than assumed.
 */

/**
 * Fraction of nominal memory bandwidth a device class actually delivers on GEMV, on top of
 * whatever the runtime achieves.
 *
 * GPUs and unified-memory SoCs have memory controllers built to be saturated by the compute
 * units in front of them. CPU cores cannot do the same to a 12-channel DDR5 subsystem: the
 * EPYC anchor implies 52% of nominal where the Spark anchor implies 90%, and that gap is
 * architectural rather than a property of any model or runtime.
 *
 * Only `cpu-ram` departs from 1.0, so in practice this is a CPU-specific correction rather
 * than a general per-class model. See the header note on identifiability before tuning it.
 */
const CLASS_BANDWIDTH_UTILIZATION: Record<DeviceClass, number> = {
  'discrete-gpu': 1.0,
  'unified-soc': 1.0,
  'cpu-ram': 0.62,
};

/**
 * Per-doubling tensor-parallel efficiency, by interconnect tier.
 *
 * Three tiers rather than "NVLink or everything else". Matching only `/nvlink/` put AMD's
 * Infinity Fabric and the Spark's Ethernet link in the same bucket despite them sitting on
 * opposite sides of PCIe — and at eight devices the constant compounds over three doublings,
 * so the two cases were wrong by ~40% in opposite directions.
 *
 *   - `fabric`  — on-package/on-node switched links. NVLink, and AMD's Infinity Fabric, whose
 *     ~896 GB/s of peer bandwidth per GPU in an 8-OAM node is NVLink-class.
 *   - `pcie`    — the commodity case: cards in slots, sharing a root complex.
 *   - `network` — Ethernet or InfiniBand between chassis. A Spark's 200GbE is ~25 GB/s per
 *     direction, well *below* PCIe 5.0 x16, so the old default flattered it.
 *
 * These carry the same identifiability caveat as the bandwidth constants in the header: they
 * are a defensible ordering, not a measured decomposition. No published multi-device benchmark
 * currently pins them, and nothing in the app reaches `count > 1` yet.
 */
const TP_SCALING = { fabric: 0.95, pcie: 0.85, network: 0.7 } as const;

function tpEfficiency(rig: Rig): number {
  const count = Math.max(1, rig.count);
  if (count <= 1) return 1;

  const link = rig.device.interconnect ?? '';
  const base = /nvlink|infinity fabric|xgmi/i.test(link)
    ? TP_SCALING.fabric
    : /ethernet|gbe|infiniband|connectx/i.test(link)
      ? TP_SCALING.network
      : TP_SCALING.pcie;

  return base ** Math.log2(count);
}

/** Aggregate memory bandwidth the rig actually delivers. */
export function achievedBandwidth(rig: Rig, runtime: RuntimeSpec): number {
  return (
    effectiveBandwidth(rig.device) *
    runtime.bandwidthEfficiency *
    CLASS_BANDWIDTH_UTILIZATION[rig.device.class] *
    Math.max(1, rig.count) *
    tpEfficiency(rig)
  );
}

export interface DecodeEstimate {
  /** Tokens per second seen by one user. */
  perUserTokensPerSec: number;
  /** Tokens per second across all concurrent sequences. */
  aggregateTokensPerSec: number;
  /** Bytes moved per decode step, split so the UI can show what dominates. */
  weightReadBytes: number;
  kvReadBytes: number;
  /** True when KV traffic outweighs weight traffic — the long-context regime. */
  kvBound: boolean;
  /** Set when weights spill to host RAM, which is usually the whole explanation. */
  offloadPenalty?: { fraction: number; withoutOffloadTokensPerSec: number };
}

export function estimateDecode(
  model: ModelSpec,
  quant: QuantSpec,
  usage: UsageSpec,
  rig: Rig,
  runtime: RuntimeSpec,
  placement: Placement,
  hostBandwidth = DEFAULT_HOST_BANDWIDTH
): DecodeEstimate {
  const batch = Math.max(1, usage.concurrency);
  const contextTokens = Math.max(1, usage.contextTokens);

  const weightReadBytes = activeWeightBytes(model, quant, batch);
  // Each sequence re-reads its own cache every step.
  const kvReadBytes = kvReadBytesPerToken(model, contextTokens, usage.kvPrecision) * batch;

  const deviceBandwidth = achievedBandwidth(rig, runtime);

  const offload = placement.offloadFraction;
  const onDeviceBytes = weightReadBytes * (1 - offload) + kvReadBytes;
  const offloadedBytes = weightReadBytes * offload;

  const secondsPerStep = onDeviceBytes / deviceBandwidth + offloadedBytes / hostBandwidth;
  const aggregateTokensPerSec = secondsPerStep > 0 ? batch / secondsPerStep : 0;

  const estimate: DecodeEstimate = {
    perUserTokensPerSec: aggregateTokensPerSec / batch,
    aggregateTokensPerSec,
    weightReadBytes,
    kvReadBytes,
    kvBound: kvReadBytes > weightReadBytes,
  };

  if (offload > 0) {
    const resident = (weightReadBytes + kvReadBytes) / deviceBandwidth;
    estimate.offloadPenalty = {
      fraction: offload,
      withoutOffloadTokensPerSec: resident > 0 ? 1 / resident : 0,
    };
  }

  return estimate;
}

/**
 * Peak FLOPS at the precision this configuration will actually compute in.
 *
 * Gated on the runtime, not just the format: llama.cpp dequantizes every GGUF to fp16 before
 * the matmul, so a Blackwell card's FP4 headline number is unreachable from it. Reading the
 * rate off storage width alone overstates prefill by up to 8x.
 */
function peakFlops(device: DeviceSpec, quant: QuantSpec, runtime: RuntimeSpec): number {
  const f = device.flops;
  const fp16 = f.fp16 ?? f.bf16 ?? 0;
  if (!runtime.nativeLowPrecision) return fp16;

  switch (quant.computeDtype) {
    case 'fp4':
      // Deliberately does not fall back to fp8, for the same reason fp8 does not fall back to
      // int8: an H100 has FP8 tensor cores and no FP4 ones, so an NVFP4 quant there runs
      // dequantized, not at twice fp16. Lending it the FP8 rate reported a Blackwell-native
      // format as usable on hardware that cannot dispatch it.
      return f.fp4 ?? fp16;
    case 'fp8':
      // Deliberately does not fall back to int8. A card with INT8 tensor cores and no FP8 ones
      // cannot run an FP8 kernel at the INT8 rate; it runs it at fp16.
      return f.fp8 ?? fp16;
    case 'int8':
      // The reverse fallback is safe: hardware with FP8 units runs INT8 at the same rate.
      return f.int8 ?? f.fp8 ?? fp16;
    case 'fp16':
      return fp16;
  }
}

export interface PrefillEstimate {
  /** Seconds until the first token appears. */
  ttftSeconds: number;
  /** Prompt tokens processed per second. */
  prefillTokensPerSec: number;
  /** FLOPs split, so the UI can show when quadratic attention takes over. */
  linearFlops: number;
  attentionFlops: number;
  /** True when attention outweighs the linear layers — the long-prompt regime. */
  attentionBound: boolean;
  /** Set when offloaded weights have to be streamed in before the prompt can be processed. */
  offloadPenalty?: { fraction: number };
}

export function estimatePrefill(
  model: ModelSpec,
  quant: QuantSpec,
  usage: UsageSpec,
  rig: Rig,
  runtime: RuntimeSpec,
  placement?: Placement,
  hostBandwidth = DEFAULT_HOST_BANDWIDTH
): PrefillEstimate {
  const contextTokens = Math.max(1, usage.contextTokens);
  const promptTokens = Math.max(1, usage.promptTokens ?? Math.floor(contextTokens * 0.9));

  // Two FLOPs per parameter per token for the linear layers. MoE routes each token through
  // only its selected experts, so this uses active rather than total parameters — FLOPs scale
  // with per-token active params, while bytes scale with the batch-wide expert union.
  //
  // Not `model.activeParams`: that is the published figure, which subtracts the embedding table
  // even when it is tied and therefore run as a full output matmul, and which counts a vision
  // tower that a text-only prompt never touches.
  //
  // The output projection is charged once, not per token: logits are produced only for the
  // position that needs them. Per-token it would be 16% of a gpt-oss-20b prompt pass; dropped
  // entirely it understates a *short* prompt by the same margin, since at one token the
  // projection is most of the work.
  const linearFlops =
    2 * (prefillComputeParams(model) * promptTokens + outputProjectionParams(model));
  // QK^T and AV. Quadratic on full-attention layers, but only linear on sliding-window ones,
  // which attend over their window however long the prompt gets. Overtakes the linear term on
  // long prompts — why time-to-first-token degrades faster than people expect at big contexts.
  // Scaled by the attention projection width, not the hidden size: a model is free to project
  // into a wider or narrower query space than its residual stream, and most current ones do.
  const attentionFlops =
    4 * promptTokens * attentionSpanPerToken(model, promptTokens) * model.attention.projectionWidth;

  const peak = peakFlops(rig.device, quant, runtime);
  const achieved = peak * runtime.computeEfficiency * Math.max(1, rig.count) * tpEfficiency(rig);

  let ttftSeconds = achieved > 0 ? (linearFlops + attentionFlops) / achieved : Infinity;

  // Offloaded weights must cross the host bus once per prefill pass before compute can start.
  // Without this the offload cliff is invisible on the number users watch first.
  //
  // Sized at the *prompt-batch* expert union, not the single-token one: a pass over hundreds
  // of tokens routes through essentially every expert, so the streamed volume is the whole
  // offloaded weight set. Charging the batch-1 union understated MoE TTFT by up to 5x, and
  // dense models could never reveal it because for them the two are identical.
  const offload = placement?.offloadFraction ?? 0;
  if (offload > 0) {
    const streamedBytes = activeWeightBytes(model, quant, promptTokens) * offload;
    ttftSeconds += streamedBytes / hostBandwidth;
  }

  return {
    ttftSeconds,
    prefillTokensPerSec:
      ttftSeconds > 0 && Number.isFinite(ttftSeconds) ? promptTokens / ttftSeconds : 0,
    linearFlops,
    attentionFlops,
    attentionBound: attentionFlops > linearFlops,
    ...(offload > 0 ? { offloadPenalty: { fraction: offload } } : {}),
  };
}

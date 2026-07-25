/**
 * The vocabulary the whole engine speaks.
 *
 * Two unit conventions, kept explicit because mixing them is the classic source of
 * quiet 7% errors in this domain:
 *   - **Memory** is binary. A "32GB" card holds 32 GiB = 32 * 2^30 bytes. Stored as bytes.
 *   - **Bandwidth** is decimal. "1792 GB/s" means 1792e9 bytes/sec. Stored as bytes/sec.
 */

/** 2^30 — memory capacities are binary. */
export const GIB = 1024 ** 3;
/** 1e9 — bandwidth and FLOPS figures are decimal. */
export const GB = 1e9;
export const TFLOP = 1e12;

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/**
 * How a model caches keys and values. This is the single most consequential thing
 * to get right: the naive `2 * layers * kv_heads * head_dim` formula that most
 * calculators apply to everything is wrong for two whole families of model, by
 * multiples, in the direction that tells people to buy hardware they don't need.
 */
export type AttentionCore =
  /** Multi-head and grouped-query attention. MHA is simply the case where kvHeads == queryHeads. */
  | { kind: 'gqa'; kvHeads: number; headDim: number }
  /**
   * Multi-head latent attention (DeepSeek V3/V4 family). Caches one compressed latent
   * per token per layer rather than per-head keys and values — so there is no factor
   * of two and no head multiplier. Roughly 3-5x smaller than GQA at the same scale.
   */
  | { kind: 'mla'; kvLoraRank: number; qkRopeHeadDim: number };

export interface AttentionSpec {
  core: AttentionCore;
  /**
   * Per-layer attention window in tokens; `null` means that layer attends over the full
   * context. Absent entirely means every layer is full attention.
   *
   * Models like gpt-oss (alternating sliding/full, 128-token window) and Gemma (5:1)
   * cap most of their layers' KV at the window size, so their cache stops growing with
   * context on those layers. At 128K that is a ~2x difference on total KV.
   */
  layerWindows?: readonly (number | null)[];
}

export interface ModelSpec {
  /** Hugging Face repo id, e.g. `Qwen/Qwen3-32B`. */
  id: string;
  name: string;
  org: string;

  /** Exact total parameter count, summed from HF's safetensors index. */
  totalParams: number;
  /**
   * Parameters read per forward pass. Equals totalParams for dense models. For MoE this is
   * dense params + shared experts + the routed experts actually selected per token — it
   * governs decode speed, while totalParams governs memory. That split is the single most
   * misunderstood thing about running these models.
   */
  activeParams: number;
  /**
   * Parameters living in routed expert FFNs. Needed separately because native quantization
   * schemes quantize experts far harder than the rest of the network (gpt-oss ships MXFP4
   * experts with BF16 attention), so a flat params * bpw is wrong for exactly the models
   * people most want to run.
   */
  expertParams: number;

  /**
   * Routed expert counts. Present only for MoE. Needed beyond `activeParams` because the set
   * of experts read grows with batch size — a single token touches `perToken` of them, but a
   * batch of 32 collectively touches most of them, so MoE throughput scales with concurrency
   * very differently from dense.
   */
  experts?: { total: number; perToken: number };

  layers: number;
  hiddenSize: number;
  vocabSize: number;
  attention: AttentionSpec;

  /** Quantization the weights ship in from the vendor, if not bf16. */
  nativeQuant?: string;
  maxContext: number;

  popularity?: { downloads: number; likes: number };
  releasedAt?: string;
  /** Provenance for every derived figure — this catalog is generated, never typed from memory. */
  source: string;
}

/** Parameters that are not routed experts, and so are always read and usually less quantized. */
export function denseParams(model: ModelSpec): number {
  return model.totalParams - model.expertParams;
}

export function isMoE(model: ModelSpec): boolean {
  return model.expertParams > 0;
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

export interface QuantSpec {
  id: string;
  label: string;
  /**
   * Effective bits per weight, including block metadata. Always higher than the name
   * suggests: GGUF K-quants carry a scale and a min per 256-weight block, so Q4_K_M is
   * ~4.85 bits per weight in practice, not 4.
   */
  bpw: number;
  /**
   * Bits per weight for everything that isn't a routed expert, when the scheme deliberately
   * spares those tensors. gpt-oss ships MXFP4 experts with BF16 attention and embeddings
   * (`quantization_config.modules_to_not_convert` spells this out), so charging the whole
   * model 4.25 bpw understates it by ~4 GB. Absent means the scheme is uniform.
   */
  denseBpw?: number;
  /**
   * Precision the tensor cores would compute in, if the runtime dispatches to them natively.
   *
   * Deliberately explicit rather than inferred from `bpw`: storage width and compute width
   * are different things. IQ4_XS and AWQ store at ~4.3 bits but dequantize and accumulate in
   * fp16, so keying off bit width would hand them Blackwell's FP4 rate and overstate prefill
   * by ~8x. Only formats with real low-precision kernels (MXFP4, NVFP4, FP8) claim otherwise.
   */
  computeDtype: 'fp16' | 'fp8' | 'fp4';
  /** Rough quality cost vs bf16, for UI guidance only — never fed into the math. */
  qualityNote?: string;
  source: string;
}

/** Bytes per element for KV cache storage. */
export type KvPrecision = 'fp16' | 'q8' | 'q4';

export const KV_BYTES: Record<KvPrecision, number> = {
  fp16: 2,
  q8: 1,
  q4: 0.5,
};

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/**
 * Hardware does not sort by VRAM. These three classes sit at genuinely different corners
 * of the capacity/bandwidth/compute triangle, and collapsing them to one number is what
 * makes existing calculators give bad advice about the machines people actually compare.
 */
export type DeviceClass =
  /** Dedicated VRAM, high bandwidth, PCIe or NVLink between cards. */
  | 'discrete-gpu'
  /** One pool shared with the CPU, an allocation ceiling below nominal RAM, capacity and bandwidth decoupled. */
  | 'unified-soc'
  /** System RAM as model memory. Enormous capacity, bandwidth set by channel count. */
  | 'cpu-ram';

/** Whether a spec is something you can buy today. Rumoured hardware must be labelled as such in the UI. */
export type DeviceStatus = 'shipping' | 'announced' | 'rumored';

export interface DeviceSpec {
  id: string;
  name: string;
  vendor: string;
  class: DeviceClass;
  status: DeviceStatus;

  /** Nominal memory, in bytes. */
  capacityBytes: number;
  /**
   * Bytes actually allocatable to model weights and KV. Below capacity on shared-memory
   * machines: macOS caps GPU-wired memory near 75% by default, Strix Halo exposes 96 of
   * its 128 GB, CPU inference must leave the OS room. Getting this wrong is the difference
   * between "fits" and "OOM on load".
   */
  allocatableBytes: number;
  /** Whether that ceiling can be raised by the user (macOS iogpu.wired_limit_mb, AMD VGM). */
  allocatableTunable?: boolean;

  /** Theoretical peak memory bandwidth, bytes/sec. */
  bandwidthBytesPerSec: number;
  /**
   * Measured bandwidth where a credible benchmark exists. Preferred over theoretical when
   * present — Strix Halo's 256 GB/s sticker versus ~213 GB/s real is 17% of the answer.
   */
  measuredBandwidthBytesPerSec?: number;

  /** Dense FLOPS at the precision used for prefill, by dtype id. */
  flops: Partial<Record<'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8', number>>;

  interconnect?: string;
  tdpWatts?: number;
  msrpUsd?: number;
  releasedAt?: string;
  source: string;
}

/** The bandwidth figure the roofline should actually use. */
export function effectiveBandwidth(device: DeviceSpec): number {
  return device.measuredBandwidthBytesPerSec ?? device.bandwidthBytesPerSec;
}

// ---------------------------------------------------------------------------
// Runtimes
// ---------------------------------------------------------------------------

export interface RuntimeSpec {
  id: string;
  label: string;
  /** Fixed per-device allocation for context, kernels and framework state, in bytes. */
  overheadBytes: number;
  /**
   * Fraction of theoretical bandwidth actually reached during decode. Real engines land
   * around 0.6-0.85 depending on kernel quality and how well the model maps to the backend.
   */
  bandwidthEfficiency: number;
  /** Fraction of peak FLOPS reached during prefill. */
  computeEfficiency: number;
  /**
   * Whether the runtime dispatches quantized weights to low-precision tensor cores, or
   * dequantizes to fp16 first. llama.cpp does the latter for every GGUF format, so it sees
   * fp16 rates on hardware whose headline number is FP4 — the gap between a card's marketing
   * TFLOPS and what a local setup actually gets.
   */
  nativeLowPrecision: boolean;
  /**
   * Whether the runtime reserves a fixed fraction of the device up front regardless of need
   * (vLLM's gpu_memory_utilization) rather than allocating as it goes (llama.cpp).
   */
  preallocFraction?: number;
  /** Device classes this runtime can drive at all. */
  supports: readonly DeviceClass[];
  source: string;
}

// ---------------------------------------------------------------------------
// The question being asked
// ---------------------------------------------------------------------------

export interface UsageSpec {
  /** Tokens of context per sequence — prompt plus generation. */
  contextTokens: number;
  /** Sequences held in the KV cache at once. */
  concurrency: number;
  /** Prompt length used for time-to-first-token. Defaults to most of the context. */
  promptTokens?: number;
  kvPrecision: KvPrecision;
}

export interface Rig {
  device: DeviceSpec;
  count: number;
}

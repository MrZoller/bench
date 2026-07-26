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
   * Width of the query/value projection — `num_attention_heads * head_dim` — which QK^T and AV
   * scale by.
   *
   * Deliberately not `hiddenSize`. A model may project into a wider or narrower attention space
   * than its residual stream, and most current ones do: GLM-4.5-Air is 3x its hidden size and
   * DeepSeek 2.9x, while Gemma 3 27B and Mistral Small are *narrower*. Substituting hidden size
   * understated GLM's attention term by 67% and overstated Gemma 3 27B's by 31% — errors in
   * opposite directions, so no single correction factor could have absorbed them.
   *
   * For MLA this is the mean of the query space (`heads * (qk_nope + qk_rope)`) and the value
   * space (`heads * v_head_dim`), which differ; the engine charges QK and AV at one rate.
   */
  projectionWidth: number;
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
   * Non-expert parameters a single token actually reads. This, not `activeParams`, is what
   * decode bytes and prefill FLOPs are built from.
   *
   * It differs from `totalParams - expertParams` by two subtractions that pull in opposite
   * directions between models, which is why neither published figure can stand in for it:
   *   - the input embedding, when the model does *not* tie it to the output projection. An
   *     untied table is a row lookup and read once per token; a tied one is a full vocab
   *     matmul on every step and must stay.
   *   - non-text towers. Gemma 3's vision encoder occupies memory but never runs for a text
   *     token, so it belongs in `totalParams` and nowhere near a per-token count.
   *
   * Charging the whole dense half instead overstated gpt-oss-20b's decode traffic by 31%.
   *
   * Prefill subtracts one further term that decode does not — see `prefillComputeParams` — so
   * this is the decode basis, not a shared one.
   */
  activeDenseParams: number;
  /** Whether the output projection reuses the input embedding table. */
  tiedEmbeddings?: boolean;
  /** Parameters in non-text towers — resident, but not run for a text token. */
  nonLanguageParams?: number;
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

  popularity?: {
    downloads: number;
    likes: number;
    /**
     * Repo the figures were read from, when it differs from `id`. Gated originals are seeded
     * via open mirrors, but a mirror's traffic is not the model's — Meta's Llama 3.1 70B has
     * ~255x the downloads of the NousResearch copy the weights come from.
     */
    measuredOn?: string;
  };
  releasedAt?: string;
  /** Commit every figure on this row was derived from, so a suspicious number is reproducible. */
  revision?: string;
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
   * by ~8x. Only formats with real low-precision kernels (MXFP4, NVFP4, FP8, INT8) claim
   * otherwise.
   *
   * `int8` is tracked apart from `fp8` even though the two run at the same rate on hardware
   * that has both, because plenty of hardware doesn't: Ampere has INT8 tensor cores and no
   * FP8 at all. Collapsing them would hand an FP8 quant a rate that card cannot reach.
   */
  computeDtype: 'fp16' | 'fp8' | 'fp4' | 'int8';
  /**
   * Hardware this format needs to run at all, when it is not an open standard.
   *
   * NVFP4 needs both halves and neither alone is sufficient. Vendor, because AMD's MI355X
   * publishes a 9.2 PFLOP/s FP4 rate for its *own* format and handing that to NVFP4 is a
   * plausible impossibility. Dtype, because "NVIDIA" also covers the 3090, the 4090 and the
   * H100, none of which have FP4 tensor cores — a vendor-only rule accepted every pre-Blackwell
   * card in the catalog.
   *
   * MXFP4 carries neither: it is the OCP microscaling standard, both vendors implement it, and
   * a runtime without native support simply dequantizes it.
   */
  requires?: {
    vendor?: string;
    /** A rate the device must actually publish for this dtype. */
    dtype?: 'fp4' | 'fp8' | 'int8';
  };
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
  /**
   * Highest the allocation ceiling can actually be raised to, in bytes.
   *
   * Absent means "as far as physical memory allows" — true of Apple's `iogpu.wired_limit_mb`,
   * where the catalogued figure is a 75% default rather than a limit. Present where the platform
   * states a real maximum: AMD's Variable Graphics Memory exposes 96 of the Ryzen AI Max+'s
   * 128 GB, so its default *is* its maximum and raising the setting buys nothing.
   *
   * Without this the UI told a Ryzen owner that a 117 GiB configuration would fit once they
   * raised the ceiling, which the platform does not permit.
   */
  maxAllocatableBytes?: number;

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
  /**
   * Bandwidth of the link to host memory, bytes/sec — the rate offloaded weights actually
   * stream at.
   *
   * Distinct from `interconnect`, which is the *device-to-device* transport `tpEfficiency`
   * models: an H100 SXM talks to its neighbours over NVLink 4.0 and to the host over PCIe 5.0,
   * and only the second one governs spill. Absent on unified-memory and CPU machines, where
   * there is no separate host to cross to.
   */
  hostLinkBytesPerSec?: number;
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
  /**
   * Hardware this runtime can drive, as class-and-optionally-vendor pairs.
   *
   * Neither axis alone is enough. `unified-soc` covers Apple silicon, NVIDIA's GB10 and AMD's
   * Strix Halo — one memory topology, three incompatible software stacks — so MLX needs the
   * vendor. But a single runtime-wide vendor is equally wrong: vLLM drives AMD's discrete
   * accelerators *and* NVIDIA's unified-memory Spark, while driving no Apple hardware at all.
   * Per-entry vendors are the smallest thing that expresses both.
   */
  supports: readonly { class: DeviceClass; vendor?: string }[];
  /**
   * How this runtime spreads a model over several devices, which decides how the cache divides.
   *
   * `tensor` shards every layer across every card, so KV divides by attention head and stops
   * when each rank holds one — and an MLA latent, having no head axis, is replicated whole.
   * `layer` gives each card entire layers and their entire KV buffers, so everything divides by
   * the device count with no exception. vLLM defaults to the first, llama.cpp and Ollama to the
   * second, and applying one layout to both rejects rigs that work.
   */
  parallelism: 'tensor' | 'layer';
  /**
   * Weight formats this runtime can load, by `QuantSpec.id`.
   *
   * The same guarantee `kvPrecisions` already gave the cache, which is what makes its absence
   * here an omission rather than a design: llama.cpp loads GGUF and cannot read an AWQ
   * checkpoint, MLX reads its own formats and neither GGUF K-quants nor AWQ. Without it the app
   * would report capacity and throughput for a pairing that cannot be loaded at all.
   */
  weightFormats: readonly string[];
  /**
   * Set when some of `weightFormats` are stand-ins *by width* rather than formats this runtime
   * really loads.
   *
   * The engine cannot tell the difference — a roofline consumes bits per weight, and a stand-in of
   * the right width produces plausible arithmetic either way. That is exactly what makes it worth
   * recording: MLX quantizes with its own affine scheme at 4 and 8 bits, the catalog has no MLX
   * entries for those, and other catalogued formats fill in. So every memory and throughput
   * figure for an
   * Apple-silicon configuration derives from a format MLX does not read, while the vLLM and
   * llama.cpp figures do not — and nothing on screen said which was which.
   *
   * Kept in `weightFormats` rather than removed from it: dropping the substitution restricts Apple
   * silicon to BF16 alone, which makes a headline case largely unusable. A documented
   * approximation beats an honest refusal here; an *undocumented* one beats neither.
   *
   * **The list is of the formats that are real, not of the stand-ins**, and the polarity is the
   * whole point. Naming the stand-ins meant adding a width to `weightFormats` and forgetting the
   * second list left it offered, scored, and unmarked — silently. Naming the natives makes a
   * newly-added format marked until someone declares it real, so the failure mode is a warning
   * nobody needed rather than a figure nobody questioned.
   *
   * One object rather than two optional fields, so a runtime cannot declare a substitution and omit
   * its explanation — which returned `undefined` from the lookup, read as "not a substitution" at
   * both call sites, and reached exactly the invisible state this exists to abolish.
   */
  substituted?: {
    /** Formats here that the runtime genuinely loads. Everything else in `weightFormats` stands in. */
    nativeFormats: readonly string[];
    /** What the substitution is, in one clause, for the marker shown beside those figures. */
    note: string;
  };
  /** KV cache dtypes the runtime can actually store. */
  kvPrecisions: readonly KvPrecision[];
  /**
   * Bytes per cached element where this runtime's format costs more than its nominal width.
   *
   * `KV_BYTES` is the nominal figure and it is exact for a float format — vLLM's FP8 really is
   * one byte. It is not exact for llama.cpp, whose `q8_0` and `q4_0` KV store 32-element blocks
   * with a 2-byte scale attached: 34/32 and 18/32 bytes per element, the same block-metadata
   * overhead `quants.ts` already documents on the weight side.
   *
   * 6% and 12% sound ignorable and are not, because the cache is what pushes a configuration
   * over: Qwen3 4B at Q8_0 on a 4090 at 32K by eight users read 22.6 GiB against a 23 GiB
   * ceiling and "fits", where the real layout needs 23.7 and has to offload.
   */
  kvBytesPerElement?: Partial<Record<KvPrecision, number>>;
  /**
   * What this runtime *calls* a precision, where its own name differs from the generic one.
   *
   * `KvPrecision` is a width, and the widths are shared: one byte per element is one byte
   * whether the runtime spells it `q8_0` or `fp8_e4m3`. The names are not shared, and labelling
   * vLLM's cache "Q8" named a `--kv-cache-dtype` value that does not exist — it takes
   * `auto`/`fp8`/`fp8_e5m2`/`fp8_e4m3` and has no integer option.
   *
   * A label rather than a fourth `KvPrecision` member, because nothing in the arithmetic differs
   * and a wider type would have to be threaded through `KV_BYTES`, placement, store coercion and
   * the URL codec to express a distinction the engine never uses. If precision *identity* ever
   * becomes something the engine reasons about — llama.cpp's `q8_0` KV really does carry block
   * scales and run nearer 8.5 effective bits — that is the point to split the type, and this
   * field is what would be replaced.
   */
  kvLabels?: Partial<Record<KvPrecision, string>>;
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
  /**
   * Tokens already in the cache that `promptTokens` attends against, rather than re-reads.
   *
   * Absent — the default, and every archetype but one — means a standalone prompt: `promptTokens`
   * is the whole working set, which is what a single-shot request sends and what every published
   * anchor measures. Present, it splits the two apart: a coding agent's turn sends ~16K *new*
   * tokens into a session that is already resident, and prefix caching (vLLM's APC, llama.cpp's
   * cache reuse) means it does not pay to read the session again.
   *
   * Opt-in rather than derived from `contextTokens - promptTokens`, because deriving it would
   * change every archetype at once — including the single-prompt scenarios the calibration is
   * pinned to. An archetype that re-reads its history says nothing here and gets the old answer.
   *
   * Note the direction: declaring a prefix makes prefill *slower*, not faster. It buys not
   * re-reading the prefix, and it costs attending against it.
   */
  cachedPrefixTokens?: number;
  kvPrecision: KvPrecision;
}

export interface Rig {
  device: DeviceSpec;
  count: number;
}

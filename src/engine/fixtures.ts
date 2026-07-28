import type { DeviceSpec, ModelSpec, RuntimeSpec } from './types';
import { GB, GIB, TFLOP } from './types';

/**
 * Hand-verified specs used to pin the engine to reality.
 *
 * Every architecture figure here was read from the model's own `config.json` on Hugging Face
 * (URLs in each `source`), not recalled. The generated catalog will supersede these for the
 * app; they stay because tests need fixtures that don't move when the catalog refreshes.
 */

/** Alternating sliding/full attention, starting with sliding — gpt-oss's pattern. */
function alternatingWindows(layers: number, window: number): (number | null)[] {
  return Array.from({ length: layers }, (_, i) => (i % 2 === 0 ? window : null));
}

/** Dense, plain GQA, full attention. The baseline case every calculator gets right. */
export const LLAMA_31_8B: ModelSpec = {
  id: 'meta-llama/Llama-3.1-8B-Instruct',
  name: 'Llama 3.1 8B Instruct',
  org: 'Meta',
  totalParams: 8.03e9,
  activeParams: 8.03e9,
  // Untied lm_head (the index carries its own tensor), so decode reads a single embedding row
  // rather than the whole 128256 x 4096 table.
  activeDenseParams: 8.03e9 - 128256 * 4096,
  tiedEmbeddings: false,
  expertParams: 0,
  layers: 32,
  hiddenSize: 4096,
  vocabSize: 128256,
  // 32 query heads x 128 = 4096, equal to hidden size. The case where the old shortcut
  // happened to be right, which is why it kept passing.
  attention: { core: { kind: 'gqa', kvHeads: 8, headDim: 128 }, projectionWidth: 32 * 128 },
  maxContext: 131072,
  source: 'https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/raw/main/config.json',
};

/** Dense GQA at a size where the 32 GB consumer ceiling starts to bite. */
export const QWEN3_32B: ModelSpec = {
  id: 'Qwen/Qwen3-32B',
  name: 'Qwen3 32B',
  org: 'Alibaba',
  totalParams: 32.8e9,
  activeParams: 32.8e9,
  activeDenseParams: 32.8e9 - 151936 * 5120,
  tiedEmbeddings: false,
  expertParams: 0,
  layers: 64,
  hiddenSize: 5120,
  vocabSize: 151936,
  // 64 query heads x 128 = 8192 against a 5120 hidden size: 1.6x.
  attention: { core: { kind: 'gqa', kvHeads: 8, headDim: 128 }, projectionWidth: 64 * 128 },
  maxContext: 40960,
  source: 'https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json',
};

/**
 * MLA. 61 layers x (kv_lora_rank 512 + qk_rope_head_dim 64).
 *
 * `totalParams` is the published 671B rather than the 684.53B that HF's safetensors index
 * reports: the difference is the Multi-Token Prediction module, which ships in the repo but
 * is not loaded for ordinary inference. Counting it would overstate weights by ~13B.
 *
 * The expert split below reproduces the published 37B active figure exactly, which is the
 * check that the MoE derivation is structurally right:
 *   dense 17.1B + (8/256 routed) 20.4B = 37.5B.
 */
export const DEEPSEEK_V3: ModelSpec = {
  id: 'deepseek-ai/DeepSeek-V3',
  name: 'DeepSeek V3',
  org: 'DeepSeek',
  totalParams: 671e9,
  activeParams: 37e9,
  activeDenseParams: 671e9 - 58 * 256 * 3 * 7168 * 2048 - 129280 * 7168,
  tiedEmbeddings: false,
  // 58 MoE layers (61 - first_k_dense_replace 3) x 256 experts x 3 matrices x 7168 x 2048.
  expertParams: 58 * 256 * 3 * 7168 * 2048,
  experts: { total: 256, perToken: 8 },
  layers: 61,
  hiddenSize: 7168,
  vocabSize: 129280,
  // MLA projects wider than it caches: 128 heads x (128 nope + 64 rope) = 24576 for queries
  // against 128 x 128 = 16384 for values. The engine charges QK and AV at one rate, so this is
  // their mean — 2.9x the 7168 hidden size.
  attention: {
    core: { kind: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
    projectionWidth: (128 * (128 + 64) + 128 * 128) / 2,
  },
  nativeQuant: 'fp8',
  maxContext: 163840,
  source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json',
};

/**
 * Tied embeddings *and* a vision tower — the two corrections that pull the per-token parameter
 * count in opposite directions, in one model.
 *
 * Its `config.json` omits `tie_word_embeddings` entirely, yet the safetensors index has no
 * `lm_head.weight`: the table is shared, so decode runs it as a full 262208 x 3840 output
 * matmul every step and it must *not* be subtracted. The 0.42B vision tower is the reverse —
 * resident in memory, never run for a text token.
 *
 * Hence `activeDenseParams` (11.77B) sitting *above* the published active figure (11.18B),
 * which is the one shape in the catalog where those two numbers invert.
 */
export const GEMMA_3_12B: ModelSpec = {
  id: 'unsloth/gemma-3-12b-it',
  name: 'Gemma 3 12B',
  org: 'Google',
  totalParams: 12_187_325_040,
  // A dense model's active count is its total — the same contract the generator now applies.
  // The embedding subtraction belongs only to the MoE published-figure convention.
  activeParams: 12_187_325_040,
  // The physical one subtracts the vision tower, and keeps the tied table.
  activeDenseParams: 12_187_325_040 - 421_290_864,
  tiedEmbeddings: true,
  nonLanguageParams: 421_290_864,
  expertParams: 0,
  layers: 48,
  hiddenSize: 3840,
  vocabSize: 262208,
  attention: {
    core: { kind: 'gqa', kvHeads: 8, headDim: 256 },
    // 16 query heads x 256 = 4096 against a 3840 hidden size: 1.07x, the narrowest gap here.
    projectionWidth: 16 * 256,
    // sliding_window_pattern 6: every 6th layer attends over the full context.
    layerWindows: Array.from({ length: 48 }, (_, i) => ((i + 1) % 6 === 0 ? null : 1024)),
  },
  maxContext: 131072,
  source: 'https://huggingface.co/unsloth/gemma-3-12b-it/raw/main/config.json',
};

/**
 * Hybrid sliding-window MoE with expert-only MXFP4 quantization — the model that breaks
 * both of the simplifying assumptions other calculators make, which is why it's a fixture.
 */
export const GPT_OSS_120B: ModelSpec = {
  id: 'openai/gpt-oss-120b',
  name: 'gpt-oss 120B',
  org: 'OpenAI',
  totalParams: 116.8e9,
  activeParams: 5.1e9,
  activeDenseParams: 116.8e9 - 36 * 128 * 3 * 2880 * 2880 - 201088 * 2880,
  tiedEmbeddings: false,
  // 36 layers x 128 experts x 3 matrices x 2880 x 2880.
  expertParams: 36 * 128 * 3 * 2880 * 2880,
  experts: { total: 128, perToken: 4 },
  layers: 36,
  hiddenSize: 2880,
  vocabSize: 201088,
  attention: {
    core: { kind: 'gqa', kvHeads: 8, headDim: 64 },
    // 64 query heads x 64 = 4096 against a 2880 hidden size: 1.42x.
    projectionWidth: 64 * 64,
    layerWindows: alternatingWindows(36, 128),
  },
  nativeQuant: 'mxfp4',
  maxContext: 131072,
  source: 'https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json',
};

/**
 * The small sibling, and the engine's main speed calibration anchor: LMSYS measured it on a
 * DGX Spark at 2,053 tok/s prefill and 49.7 tok/s decode under Ollama.
 */
export const GPT_OSS_20B: ModelSpec = {
  id: 'openai/gpt-oss-20b',
  name: 'gpt-oss 20B',
  org: 'OpenAI',
  totalParams: 20.9e9,
  activeParams: 3.6e9,
  activeDenseParams: 20.9e9 - 24 * 32 * 3 * 2880 * 2880 - 201088 * 2880,
  tiedEmbeddings: false,
  // 24 layers x 32 experts x 3 matrices x 2880 x 2880.
  expertParams: 24 * 32 * 3 * 2880 * 2880,
  experts: { total: 32, perToken: 4 },
  layers: 24,
  hiddenSize: 2880,
  vocabSize: 201088,
  attention: {
    core: { kind: 'gqa', kvHeads: 8, headDim: 64 },
    projectionWidth: 64 * 64,
    layerWindows: alternatingWindows(24, 128),
  },
  nativeQuant: 'mxfp4',
  maxContext: 131072,
  source: 'https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json',
};

// ---------------------------------------------------------------------------
// Devices — one from each corner of the capacity/bandwidth/compute triangle
// ---------------------------------------------------------------------------

/** Fast at everything, in only 32 GB. */
export const RTX_5090: DeviceSpec = {
  id: 'rtx-5090',
  name: 'GeForce RTX 5090',
  vendor: 'NVIDIA',
  class: 'discrete-gpu',
  status: 'shipping',
  capacityBytes: 32 * GIB,
  allocatableBytes: 31 * GIB, // display and desktop compositor take a slice
  bandwidthBytesPerSec: 1792 * GB,
  // Blackwell: real FP4 tensor cores, as devices.json records. Omitting them here made every
  // fixture-based NVFP4 estimate 4x too low and left the card's actual FP4 path untested.
  flops: { fp16: 419 * TFLOP, fp8: 838 * TFLOP, fp4: 1676 * TFLOP },
  interconnect: 'PCIe 5.0 x16',
  // The link to the *host*, not the one to a neighbouring card. Offloaded weights cross this.
  hostLinkBytesPerSec: 63 * GB,
  tdpWatts: 575,
  msrpUsd: 1999,
  source: 'https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216',
};

/**
 * A 4090 exists in the fixtures for one reason: it is PCIe 4.0, half the 5090's host link, so
 * it is the case that shows offload bandwidth is a device property rather than a constant.
 */
export const RTX_4090: DeviceSpec = {
  id: 'rtx-4090',
  name: 'GeForce RTX 4090',
  vendor: 'NVIDIA',
  class: 'discrete-gpu',
  status: 'shipping',
  capacityBytes: 24 * GIB,
  allocatableBytes: 23 * GIB,
  bandwidthBytesPerSec: 1008 * GB,
  flops: { fp16: 165.2 * TFLOP, fp8: 330.3 * TFLOP },
  interconnect: 'PCIe 4.0 x16',
  hostLinkBytesPerSec: 31.5 * GB,
  tdpWatts: 450,
  msrpUsd: 1599,
  source: 'https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889',
};

/**
 * 128 GB at only 273 GB/s, but real Blackwell compute behind it. Fits models a 5090 can't
 * touch, decodes them slowly, and processes prompts fast — the clearest single example of
 * why one VRAM number can't answer the question.
 */
export const DGX_SPARK: DeviceSpec = {
  id: 'dgx-spark',
  name: 'DGX Spark (GB10)',
  vendor: 'NVIDIA',
  class: 'unified-soc',
  status: 'shipping',
  capacityBytes: 128 * GIB,
  allocatableBytes: 120 * GIB, // coherent pool; OS still needs room
  bandwidthBytesPerSec: 273 * GB,
  // Dense, matching devices.json. NVIDIA's headline 1 PetaFLOP for GB10 is the sparse figure.
  flops: { fp16: 125 * TFLOP, fp8: 250 * TFLOP, fp4: 500 * TFLOP },
  interconnect: 'ConnectX-7 200GbE',
  tdpWatts: 240,
  msrpUsd: 3999,
  source: 'https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/',
};

/**
 * The inverse of Spark: bandwidth-rich, compute-poor. macOS caps GPU-wired memory near 75%
 * of RAM by default (`iogpu.wired_limit_mb`), which is why allocatable is well under capacity.
 */
export const MAC_STUDIO_M3_ULTRA_256: DeviceSpec = {
  id: 'mac-studio-m3-ultra-256',
  name: 'Mac Studio M3 Ultra (256 GB)',
  vendor: 'Apple',
  class: 'unified-soc',
  status: 'shipping',
  capacityBytes: 256 * GIB,
  allocatableBytes: Math.floor(0.75 * 256 * GIB),
  allocatableTunable: true,
  bandwidthBytesPerSec: 819 * GB,
  flops: { fp16: 54 * TFLOP },
  tdpWatts: 270,
  msrpUsd: 5599,
  source: 'https://www.apple.com/mac-studio/specs/',
};

/**
 * The big one, and the clearest case of a ceiling that is not a limit: 512 GiB of memory with
 * 384 GiB handed out by default. A configuration between those two figures does not run today
 * and runs after one `sysctl`, which is a different answer from "will not run".
 */
export const MAC_STUDIO_M3_ULTRA_512: DeviceSpec = {
  id: 'mac-studio-m3-ultra-512',
  name: 'Mac Studio M3 Ultra (512 GB)',
  vendor: 'Apple',
  class: 'unified-soc',
  status: 'shipping',
  capacityBytes: 512 * GIB,
  allocatableBytes: Math.floor(0.75 * 512 * GIB),
  allocatableTunable: true,
  bandwidthBytesPerSec: 819 * GB,
  flops: { fp16: 54 * TFLOP },
  tdpWatts: 270,
  msrpUsd: 9499,
  source: 'https://www.apple.com/mac-studio/specs/',
};

/**
 * Cheap capacity, modest everything else. Sticker bandwidth is 256 GB/s; measured lands
 * around 213 GB/s, and that 17% is the difference between a right and a wrong tok/s figure.
 */
export const STRIX_HALO_395: DeviceSpec = {
  id: 'ryzen-ai-max-395',
  name: 'Ryzen AI Max+ 395 (128 GB)',
  vendor: 'AMD',
  class: 'unified-soc',
  status: 'shipping',
  capacityBytes: 128 * GIB,
  allocatableBytes: 96 * GIB, // Variable Graphics Memory ceiling
  allocatableTunable: true,
  // ...and that ceiling is also the maximum: VGM exposes 96 of the 128 GB, so the setting is
  // already as high as it goes. The comment above said this and the data did not.
  maxAllocatableBytes: 96 * GIB,
  bandwidthBytesPerSec: 256 * GB,
  flops: { fp16: 59 * TFLOP },
  tdpWatts: 120,
  msrpUsd: 1999,
  source: 'https://www.amd.com/en/products/processors/laptop/ryzen/ai-max.html',
};

/** The floor of the roofline: enormous capacity, channel-limited bandwidth. */
export const EPYC_9654: DeviceSpec = {
  id: 'epyc-9654',
  name: 'EPYC 9654 (12-ch DDR5-4800)',
  vendor: 'AMD',
  class: 'cpu-ram',
  status: 'shipping',
  capacityBytes: 768 * GIB,
  allocatableBytes: 720 * GIB,
  bandwidthBytesPerSec: 460.8 * GB,
  // CPU inference realises a small fraction of nominal vector throughput on GEMM.
  flops: { fp16: 6 * TFLOP },
  tdpWatts: 360,
  source: 'https://www.amd.com/en/products/processors/server/epyc/9004-series/amd-epyc-9654.html',
};

// ---------------------------------------------------------------------------
// Runtimes
// ---------------------------------------------------------------------------

/**
 * Bandwidth efficiency is anchored on DGX Spark decode (49.7 tok/s implies 90% of the 273
 * GB/s pool; 0.82 is held deliberately conservative and reads ~9% under). Compute efficiency
 * is anchored on the matching prefill measurement (2,053 tok/s implies ~15 TFLOPS against a
 * 125 TFLOP fp16 peak). It dequantizes GGUF to fp16 before the matmul, so it never reaches a
 * card's FP4 or FP8 headline rate — see `nativeLowPrecision`.
 */
export const LLAMA_CPP: RuntimeSpec = {
  id: 'llama.cpp',
  label: 'llama.cpp / Ollama',
  overheadBytes: 0.6 * GIB,
  bandwidthEfficiency: 0.82,
  computeEfficiency: 0.12,
  nativeLowPrecision: false,
  supports: [{ class: 'discrete-gpu' }, { class: 'unified-soc' }, { class: 'cpu-ram' }],
  parallelism: 'layer',
  weightFormats: ['bf16', 'q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'iq4_xs', 'q3_k_m', 'mxfp4'],
  kvPrecisions: ['fp16', 'q8', 'q4'],
  // q8_0/q4_0 KV blocks carry a 2-byte scale per 32 elements.
  kvBytesPerElement: { q8: 34 / 32, q4: 18 / 32 },
  source: 'https://github.com/ggml-org/llama.cpp',
};

export const VLLM: RuntimeSpec = {
  id: 'vllm',
  label: 'vLLM',
  overheadBytes: 1.5 * GIB,
  bandwidthEfficiency: 0.85,
  computeEfficiency: 0.3,
  // Dispatches FP8 and FP4 weights to the tensor cores rather than dequantizing first.
  nativeLowPrecision: true,
  // Reserves a fixed fraction of the device up front regardless of what the model needs.
  preallocFraction: 0.9,
  supports: [{ class: 'discrete-gpu' }, { class: 'unified-soc', vendor: 'NVIDIA' }],
  parallelism: 'tensor',
  weightFormats: ['bf16', 'fp8', 'int8', 'nvfp4', 'mxfp4', 'awq_4bit'],
  kvPrecisions: ['fp16', 'q8'],
  // One byte per element, but vLLM spells it fp8_e4m3 and has no integer option.
  kvLabels: { q8: 'FP8' },
  source: 'https://docs.vllm.ai/',
};

export const MLX: RuntimeSpec = {
  id: 'mlx',
  label: 'MLX (Apple)',
  overheadBytes: 0.5 * GIB,
  bandwidthEfficiency: 0.8,
  computeEfficiency: 0.15,
  nativeLowPrecision: false,
  // Class is too coarse here: `unified-soc` also covers the DGX Spark and Strix Halo.
  supports: [{ class: 'unified-soc', vendor: 'Apple' }],
  parallelism: 'layer',
  weightFormats: ['bf16', 'int8', 'q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'iq4_xs', 'q3_k_m'],
  // BF16 is the one real MLX format here; every other entry above is a width standing in — `int8`
  // included, since MLX's 8-bit is affine and the catalogued row is LLM.int8() at a flat 8.0.
  substituted: {
    nativeFormats: ['bf16'],
    note: 'MLX quantizes with its own affine scheme and the catalog has no measured entry for it, so another catalogued format of the same nominal width stands in.',
    // Both widths are established, so neither is marked — FP16 is a plain float, and the 8-bit
    // cache is derived from `mlx-lm`'s own source below. The *weight* formats are still stand-ins.
    measuredKvPrecisions: ['fp16', 'q8'],
    kvNote:
      'MLX quantizes the cache with the same affine scheme its weights use, which carries a scale and a bias per group, and the catalog has no established width for this precision — so it is charged its nominal figure, which understates the cache rather than overstating it.',
  },
  kvPrecisions: ['fp16', 'q8'],
  // `QuantizedKVCache(group_size=64, bits=8)`, with an fp16 scale *and* an fp16 bias per group:
  // 8 + 16/64 + 16/64 = 8.5 bits. Lands on llama.cpp's 34/32 by coincidence, not kinship — see
  // the derivation in `runtimes.ts`.
  kvBytesPerElement: { q8: 17 / 16 },
  source:
    'https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py (QuantizedKVCache: group_size=64, bits=8; scales and biases at keys.dtype), https://github.com/ml-explore/mlx',
};

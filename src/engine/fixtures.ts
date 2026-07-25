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
  expertParams: 0,
  layers: 32,
  hiddenSize: 4096,
  vocabSize: 128256,
  attention: { core: { kind: 'gqa', kvHeads: 8, headDim: 128 } },
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
  expertParams: 0,
  layers: 64,
  hiddenSize: 5120,
  vocabSize: 151936,
  attention: { core: { kind: 'gqa', kvHeads: 8, headDim: 128 } },
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
  // 58 MoE layers (61 - first_k_dense_replace 3) x 256 experts x 3 matrices x 7168 x 2048.
  expertParams: 58 * 256 * 3 * 7168 * 2048,
  experts: { total: 256, perToken: 8 },
  layers: 61,
  hiddenSize: 7168,
  vocabSize: 129280,
  attention: { core: { kind: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 } },
  nativeQuant: 'fp8',
  maxContext: 163840,
  source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json',
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
  // 36 layers x 128 experts x 3 matrices x 2880 x 2880.
  expertParams: 36 * 128 * 3 * 2880 * 2880,
  experts: { total: 128, perToken: 4 },
  layers: 36,
  hiddenSize: 2880,
  vocabSize: 201088,
  attention: {
    core: { kind: 'gqa', kvHeads: 8, headDim: 64 },
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
  // 24 layers x 32 experts x 3 matrices x 2880 x 2880.
  expertParams: 24 * 32 * 3 * 2880 * 2880,
  experts: { total: 32, perToken: 4 },
  layers: 24,
  hiddenSize: 2880,
  vocabSize: 201088,
  attention: {
    core: { kind: 'gqa', kvHeads: 8, headDim: 64 },
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
  flops: { fp16: 419 * TFLOP, fp8: 838 * TFLOP },
  interconnect: 'PCIe 5.0 x16',
  tdpWatts: 575,
  msrpUsd: 1999,
  source: 'https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216',
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
  flops: { fp16: 125 * TFLOP, fp4: 1000 * TFLOP },
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
  bandwidthBytesPerSec: 256 * GB,
  measuredBandwidthBytesPerSec: 213 * GB,
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
  supports: ['discrete-gpu', 'unified-soc', 'cpu-ram'],
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
  supports: ['discrete-gpu'],
  source: 'https://docs.vllm.ai/',
};

export const MLX: RuntimeSpec = {
  id: 'mlx',
  label: 'MLX (Apple)',
  overheadBytes: 0.5 * GIB,
  bandwidthEfficiency: 0.8,
  computeEfficiency: 0.15,
  nativeLowPrecision: false,
  supports: ['unified-soc'],
  source: 'https://github.com/ml-explore/mlx',
};

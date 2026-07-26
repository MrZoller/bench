import type { DeviceSpec, RuntimeSpec } from '@/engine/types';
import { GIB } from '@/engine/types';

/**
 * Inference runtimes, as the app offers them.
 *
 * The runtime is not a cosmetic choice, which is why it is a first-class picker rather than an
 * assumption. Three of its properties change the answer materially:
 *
 *   - **What hardware it can drive at all.** vLLM does not run on a Mac's unified memory; MLX
 *     runs on nothing else. Picking a runtime a device cannot use should say so, not quietly
 *     produce a number.
 *   - **Whether it reserves memory up front.** vLLM takes a fixed fraction of the device
 *     regardless of model size; llama.cpp allocates as it goes. On a 24 GB card that is several
 *     gigabytes of difference before a single weight is loaded.
 *   - **Whether it reaches the tensor cores.** llama.cpp dequantizes every GGUF to fp16 before
 *     the matmul, so a Blackwell card's FP4 headline is unreachable from it. This is the single
 *     biggest lever on time-to-first-token, and no VRAM calculator models it.
 *
 * The efficiency constants carry the caveat documented at the top of `speed.ts`: they are fitted
 * to a small number of published measurements, and only their product with the device-class
 * utilisation is observable.
 */
export const RUNTIMES: readonly RuntimeSpec[] = [
  {
    id: 'llama.cpp',
    label: 'llama.cpp / Ollama',
    overheadBytes: 0.6 * GIB,
    bandwidthEfficiency: 0.82,
    // Low because every GGUF is dequantized to fp16 before the matmul — the tensor cores'
    // low-precision rates are simply not reachable from here.
    computeEfficiency: 0.12,
    nativeLowPrecision: false,
    supports: [{ class: 'discrete-gpu' }, { class: 'unified-soc' }, { class: 'cpu-ram' }],
    // GGUF K-quants for the cache as well as the weights: `--cache-type-k q8_0 q4_0`.
    // Splits by whole layers across cards by default, so the cache divides evenly.
    parallelism: 'layer',
    // GGUF: the K-quants and I-quants, plus the plain float baselines it converts to. Not AWQ,
    // which is a different checkpoint format it cannot read, and not the vendor 4-bit schemes.
    weightFormats: ['bf16', 'q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'iq4_xs', 'q3_k_m', 'mxfp4'],
    kvPrecisions: ['fp16', 'q8', 'q4'],
    // `q8_0` and `q4_0` KV store 32-element blocks with a 2-byte scale: 34/32 and 18/32 bytes
    // per element. The same block-metadata overhead quants.ts documents on the weight side.
    kvBytesPerElement: { q8: 34 / 32, q4: 18 / 32 },
    source: 'https://github.com/ggml-org/llama.cpp',
  },
  {
    id: 'vllm',
    label: 'vLLM',
    overheadBytes: 1.5 * GIB,
    bandwidthEfficiency: 0.85,
    computeEfficiency: 0.3,
    // Dispatches FP8 and FP4 weights to the tensor cores rather than dequantizing first.
    nativeLowPrecision: true,
    // Reserves a fixed fraction of the device up front regardless of what the model needs.
    preallocFraction: 0.9,
    // Discrete accelerators from either vendor, plus NVIDIA's unified-memory Spark, which is a
    // CUDA target with an official playbook. Not Apple or AMD unified memory.
    supports: [{ class: 'discrete-gpu' }, { class: 'unified-soc', vendor: 'NVIDIA' }],
    // Tensor-parallel by default: every layer sharded across every rank.
    parallelism: 'tensor',
    // Safetensors checkpoints: float baselines, the vendor low-precision formats, and AWQ.
    // Not GGUF, which is llama.cpp's container.
    weightFormats: ['bf16', 'fp8', 'int8', 'nvfp4', 'mxfp4', 'awq_4bit'],
    // `--kv-cache-dtype` takes auto/native or an FP8 variant. There is no 4-bit KV cache, and
    // offering one lets a long-context OOM be reported as a comfortable fit.
    kvPrecisions: ['fp16', 'q8'],
    // vLLM's one-byte cache is FP8, not integer Q8 — `fp8_e4m3`/`fp8_e5m2`, with no int8 option
    // at all. One byte per element either way, so no figure on screen changes, but "Q8" named a
    // flag a user cannot type.
    kvLabels: { q8: 'FP8' },
    source: 'https://docs.vllm.ai/',
  },
  {
    id: 'mlx',
    label: 'MLX (Apple)',
    overheadBytes: 0.5 * GIB,
    bandwidthEfficiency: 0.8,
    computeEfficiency: 0.15,
    nativeLowPrecision: false,
    // Class is too coarse here: `unified-soc` also covers the DGX Spark and Strix Halo.
    supports: [{ class: 'unified-soc', vendor: 'Apple' }],
    // Single-machine only in the catalogue, so this never divides anything today — declared
    // because the field is required, and a layer split is what a multi-device MLX would do.
    parallelism: 'layer',
    // MLX quantizes with its own affine scheme at 4 and 8 bits, and the catalog has no
    // MLX-native entries for those — so other catalogued formats stand in *by width*, which is
    // what a roofline over bits-per-weight actually consumes. Not exact: MLX 4-bit is nearer 4.5 bpw
    // than Q4_K_M's 4.85. Recorded as a modelling choice rather than a claim that MLX reads
    // GGUF, which it does not. What it genuinely cannot do is AWQ or the vendor formats.
    weightFormats: ['bf16', 'int8', 'q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'iq4_xs', 'q3_k_m'],
    substituted: {
      // The one MLX genuinely loads. Everything else above is a stand-in, and stays that way by
      // default if a format is added and nobody says otherwise.
      //
      // `int8` is not on this list, and the comment above is why: MLX's 8-bit is affine at 8 bits
      // too, and the catalogued `int8` row is LLM.int8() — per-channel, no group metadata, 8.0 bpw
      // exactly, cited to arXiv 2208.07339 and offered to vLLM. It is a stand-in for MLX exactly as
      // Q8_0 is, and leaving it native inverted the two: on a 235B, `int8` reported 13.7 GiB
      // lighter than the marked `q8_0` and carried no mark at all. BF16 has no groups, no scales
      // and no biases, so 16 bpw is exact and it stays. Raised by Codex on PR #32.
      nativeFormats: ['bf16'],
      note: 'MLX quantizes with its own affine scheme and the catalog has no measured entry for it, so another catalogued format of the same nominal width stands in.',
    },
    kvPrecisions: ['fp16', 'q8'],
    source: 'https://github.com/ml-explore/mlx',
  },
] as const;

const BY_ID = new Map(RUNTIMES.map((r) => [r.id, r]));

export function getRuntime(id: string): RuntimeSpec {
  const runtime = BY_ID.get(id);
  if (!runtime) throw new Error(`Unknown runtime: ${id}`);
  return runtime;
}

/** Whether a runtime can drive a particular device, by class and vendor. */
export function runtimeDrives(runtime: RuntimeSpec, device: DeviceSpec): boolean {
  return runtime.supports.some(
    (s) => s.class === device.class && (s.vendor === undefined || s.vendor === device.vendor)
  );
}

/** Runtimes that can drive a given device. */
export function runtimesFor(device: DeviceSpec): readonly RuntimeSpec[] {
  return RUNTIMES.filter((r) => runtimeDrives(r, device));
}

/**
 * The note explaining that this pairing's figures rest on a stand-in format, or `undefined` when
 * they do not.
 *
 * One function, because the marker has to appear on every surface that renders a figure and three
 * hand-written copies of "is this a substitution" is how one of them comes to disagree — the
 * failure this codebase has hit with the host-RAM caveat, the ceiling check and the KV label.
 */
export function substitutionFor(runtime: RuntimeSpec, quantId: string): string | undefined {
  const substituted = runtime.substituted;
  if (!substituted) return undefined;
  // A format this runtime cannot load at all is a different refusal, made by `planPlacement`, and
  // marking it here would explain a figure that is never produced.
  if (!runtime.weightFormats.includes(quantId)) return undefined;
  if (substituted.nativeFormats.includes(quantId)) return undefined;
  return substituted.note;
}

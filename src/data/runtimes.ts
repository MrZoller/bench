import type { RuntimeSpec } from '@/engine/types';
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
    supports: ['discrete-gpu', 'unified-soc', 'cpu-ram'],
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
    supports: ['discrete-gpu'],
    source: 'https://docs.vllm.ai/',
  },
  {
    id: 'mlx',
    label: 'MLX (Apple)',
    overheadBytes: 0.5 * GIB,
    bandwidthEfficiency: 0.8,
    computeEfficiency: 0.15,
    nativeLowPrecision: false,
    supports: ['unified-soc'],
    source: 'https://github.com/ml-explore/mlx',
  },
] as const;

const BY_ID = new Map(RUNTIMES.map((r) => [r.id, r]));

export function getRuntime(id: string): RuntimeSpec {
  const runtime = BY_ID.get(id);
  if (!runtime) throw new Error(`Unknown runtime: ${id}`);
  return runtime;
}

/** Runtimes that can actually drive a device class — the rest are offered but marked. */
export function runtimesFor(deviceClass: RuntimeSpec['supports'][number]): readonly RuntimeSpec[] {
  return RUNTIMES.filter((r) => r.supports.includes(deviceClass));
}

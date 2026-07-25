import type { QuantSpec } from '@/engine/types';

/**
 * Effective bits per weight for every quantization scheme the app offers.
 *
 * These are *effective* figures, not nominal ones. GGUF K-quants store a scale and a min per
 * 256-weight block, so the real cost always exceeds the number in the name — Q4_K_M is ~4.85
 * bpw, not 4. Using the nominal value understates a 70B model by roughly 7 GB, which is
 * exactly the kind of error that turns "fits" into "doesn't".
 *
 * `computeDtype` is tracked separately from `bpw` because storage width and compute width are
 * different things. Every GGUF format dequantizes to fp16 before the matmul regardless of how
 * few bits it stores, so inferring the tensor-core rate from bit width would hand IQ4_XS
 * Blackwell's FP4 throughput and overstate prefill by ~8x.
 *
 * A caveat worth stating plainly: effective bpw drifts by 1-2% between models, because
 * K-quants assign different bit widths to different tensor types and the embedding share of
 * total parameters varies. llama.cpp's own published figure for Llama-3.1-8B Q4_K_M is 4.89
 * bpw; a small-vocab model lands slightly lower. The table carries a representative value and
 * the tests assert within a tolerance band rather than pretending to more precision than exists.
 */
export const QUANTS: readonly QuantSpec[] = [
  {
    id: 'bf16',
    label: 'BF16 / FP16',
    bpw: 16,
    computeDtype: 'fp16',
    qualityNote: 'Reference quality — the weights as trained.',
    source: 'https://huggingface.co/docs/transformers/main/en/main_classes/model',
  },
  {
    id: 'fp8',
    label: 'FP8 (E4M3)',
    bpw: 8,
    computeDtype: 'fp8',
    qualityNote: 'Near-lossless. How DeepSeek and several frontier open models ship natively.',
    source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json',
  },
  {
    id: 'int8',
    label: 'INT8',
    bpw: 8,
    // Its own dtype, not an alias for fp8: Ampere reaches 2x fp16 on INT8 while having no FP8
    // tensor cores whatsoever, so the two rates have to be looked up separately.
    computeDtype: 'int8',
    qualityNote: 'Near-lossless with good calibration.',
    source: 'https://arxiv.org/abs/2208.07339',
  },
  {
    id: 'nvfp4',
    label: 'NVFP4',
    // 4-bit mantissa plus an FP8 scale per 16-element block: 4 + 8/16.
    bpw: 4.5,
    computeDtype: 'fp4',
    qualityNote: 'Blackwell-native 4-bit. Better quality than MXFP4 at slightly more memory.',
    source:
      'https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/',
  },
  {
    id: 'mxfp4',
    label: 'MXFP4 (expert-only)',
    // 4-bit mantissa plus a shared 8-bit exponent per 32-element block: 4 + 8/32.
    bpw: 4.25,
    // Attention, router, embeddings and lm_head stay BF16 — see modules_to_not_convert.
    denseBpw: 16,
    computeDtype: 'fp4',
    qualityNote: 'How gpt-oss ships. Experts at 4-bit, attention untouched.',
    source: 'https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json',
  },
  {
    id: 'q8_0',
    label: 'Q8_0',
    bpw: 8.5,
    computeDtype: 'fp16',
    qualityNote: 'Indistinguishable from BF16 in practice.',
    source: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md',
  },
  {
    id: 'q6_k',
    label: 'Q6_K',
    bpw: 6.57,
    computeDtype: 'fp16',
    qualityNote: 'Negligible loss. The safe choice when memory allows.',
    source: 'https://github.com/ggml-org/llama.cpp/discussions/5063',
  },
  {
    id: 'q5_k_m',
    label: 'Q5_K_M',
    bpw: 5.67,
    computeDtype: 'fp16',
    qualityNote: 'Very small loss.',
    source: 'https://github.com/ggml-org/llama.cpp/discussions/5063',
  },
  {
    id: 'q4_k_m',
    label: 'Q4_K_M',
    bpw: 4.85,
    computeDtype: 'fp16',
    qualityNote: 'The default local trade. Small but measurable loss; the usual sweet spot.',
    source: 'https://github.com/ggml-org/llama.cpp/discussions/5063',
  },
  {
    id: 'iq4_xs',
    label: 'IQ4_XS',
    bpw: 4.46,
    computeDtype: 'fp16',
    qualityNote: 'Slightly smaller than Q4_K_M at similar quality; slower prompt processing.',
    source: 'https://github.com/ggml-org/llama.cpp/discussions/5063',
  },
  {
    id: 'q3_k_m',
    label: 'Q3_K_M',
    bpw: 3.91,
    computeDtype: 'fp16',
    qualityNote: 'Noticeable degradation. Reach for a smaller model at Q4 first.',
    source: 'https://github.com/ggml-org/llama.cpp/discussions/5063',
  },
  {
    id: 'awq_4bit',
    label: 'AWQ 4-bit',
    // 4 bits plus FP16 scale and zero-point per 128-element group.
    bpw: 4.25,
    // Marlin and comparable kernels store int4 but accumulate in fp16.
    computeDtype: 'fp16',
    qualityNote: 'Activation-aware 4-bit for GPU serving stacks.',
    source: 'https://arxiv.org/abs/2306.00978',
  },
] as const;

const BY_ID = new Map(QUANTS.map((q) => [q.id, q]));

export function getQuant(id: string): QuantSpec {
  const quant = BY_ID.get(id);
  if (!quant) throw new Error(`Unknown quantization: ${id}`);
  return quant;
}

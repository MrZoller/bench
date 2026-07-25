import type { ModelSpec, RuntimeSpec, UsageSpec } from './types';

/**
 * Runtime overhead and activation workspace.
 *
 * The smallest and least precise of the three memory terms — a gigabyte or two rather than
 * tens — but it is what makes a model that "just fits" on paper fail to load in practice, so
 * it is worth carrying explicitly rather than hiding in a fudge factor.
 *
 * The runtime matters more than the model here. llama.cpp allocates roughly what it needs;
 * vLLM reserves a fixed fraction of the card up front (`gpu_memory_utilization`, 0.9 by
 * default) whether or not the model wants it. Someone comparing the two on the same hardware
 * sees a real difference in what's left for KV, and no other calculator surfaces it.
 */

/** Tokens processed per prefill chunk. Sets the size of the transient activation buffers. */
const PREFILL_CHUNK_TOKENS = 512;

/**
 * Live intermediate buffers per token in flight — residual stream, attention projections,
 * and the FFN intermediate. An approximation of a per-architecture detail; the term is small
 * enough that refining it further would be false precision.
 */
const LIVE_BUFFERS = 8;

export interface ActivationBreakdown {
  /** Fixed per-device cost: driver context, kernels, framework state. */
  runtimeOverheadBytes: number;
  /** Transient buffers proportional to tokens in flight. */
  workspaceBytes: number;
  /** Output logits, which scale with vocabulary and concurrency. */
  logitsBytes: number;
  totalBytes: number;
}

export function activationBreakdown(
  model: ModelSpec,
  usage: UsageSpec,
  runtime: RuntimeSpec
): ActivationBreakdown {
  const tokensInFlight = Math.max(usage.concurrency, PREFILL_CHUNK_TOKENS);
  const workspaceBytes = tokensInFlight * model.hiddenSize * 2 * LIVE_BUFFERS;

  // Logits are materialised in fp32, one row of the vocabulary per sequence. Small for a 32K
  // vocabulary, not small for the 200K+ vocabularies that recent models ship.
  const logitsBytes = usage.concurrency * model.vocabSize * 4;

  return {
    runtimeOverheadBytes: runtime.overheadBytes,
    workspaceBytes,
    logitsBytes,
    totalBytes: runtime.overheadBytes + workspaceBytes + logitsBytes,
  };
}

export function activationBytes(model: ModelSpec, usage: UsageSpec, runtime: RuntimeSpec): number {
  return activationBreakdown(model, usage, runtime).totalBytes;
}

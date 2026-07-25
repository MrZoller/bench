import type { ModelSpec, QuantSpec } from './types';
import { denseParams } from './types';

/**
 * Weight sizing.
 *
 * `params * bits / 8` is right only when a scheme quantizes the whole network evenly, and the
 * models people most want to run are exactly the ones where it doesn't. gpt-oss ships MXFP4
 * routed experts alongside BF16 attention and embeddings; charging the entire 116.8B at 4.25
 * bpw understates it by about 4 GB.
 *
 * The same split has to be applied to the *active* parameters, and that is easier to get
 * wrong: active params are roughly half dense for gpt-oss, and charging that dense half the
 * blended whole-model rate understates bytes-read-per-token by ~1.9x — which lands directly
 * on decode throughput.
 */

export interface WeightBreakdown {
  /** Routed expert FFN weights — the bulk of any MoE model. */
  expertBytes: number;
  /** Everything else: attention, embeddings, norms, router, shared experts, dense FFN layers. */
  denseBytes: number;
  totalBytes: number;
  /** Blended bits per weight across the whole model. For display; never use it to size a subset. */
  effectiveBpw: number;
}

export function weightBreakdown(model: ModelSpec, quant: QuantSpec): WeightBreakdown {
  const dense = denseParams(model);
  // A uniform scheme spares nothing; `denseBpw` is set only when the scheme deliberately does.
  const denseBpw = quant.denseBpw ?? quant.bpw;

  const expertBytes = (model.expertParams * quant.bpw) / 8;
  const denseBytes = (dense * denseBpw) / 8;
  const totalBytes = expertBytes + denseBytes;

  return {
    expertBytes,
    denseBytes,
    totalBytes,
    effectiveBpw: (totalBytes * 8) / model.totalParams,
  };
}

export function weightBytes(model: ModelSpec, quant: QuantSpec): number {
  return weightBreakdown(model, quant).totalBytes;
}

/**
 * Fraction of routed experts touched by a batch of `batch` tokens.
 *
 * One token selects `perToken` of them; a batch collectively selects more, approaching all of
 * them as the batch grows. This is why MoE throughput improves with concurrency far less than
 * dense throughput does — an effect that surprises people putting these models behind a server.
 */
export function expertFraction(model: ModelSpec, batch: number): number {
  if (model.expertParams === 0) return 0;
  const n = Math.max(1, batch);

  if (model.experts) {
    const { total, perToken } = model.experts;
    return 1 - (1 - perToken / total) ** n;
  }

  // Without expert counts the union can't be modelled, so fall back to the catalog's own
  // active-parameter figure and hold it flat across batch. Better a known-conservative
  // estimate than a fabricated curve.
  const implied = (model.activeParams - denseParams(model)) / model.expertParams;
  return Math.min(1, Math.max(0, implied));
}

/**
 * Parameters actually read for a decode step at a given batch size. Recovers a model's
 * published active-parameter count at batch 1 to within the rounding the vendor applied.
 */
export function effectiveActiveParams(model: ModelSpec, batch: number): number {
  if (model.expertParams === 0) return model.activeParams;
  return denseParams(model) + model.expertParams * expertFraction(model, batch);
}

/**
 * Bytes of weight read for a decode step.
 *
 * Charges the dense and expert halves at their own rates rather than a blended average. For a
 * uniform scheme the two are identical; for expert-only schemes like MXFP4 the difference is
 * about a factor of two on the number that sets decode speed.
 */
export function activeWeightBytes(model: ModelSpec, quant: QuantSpec, batch = 1): number {
  const denseBpw = quant.denseBpw ?? quant.bpw;
  const activeExpertParams = model.expertParams * expertFraction(model, batch);
  return (denseParams(model) * denseBpw + activeExpertParams * quant.bpw) / 8;
}

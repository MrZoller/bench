import type { ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { activationBreakdown, type ActivationBreakdown } from './activations';
import { hasSlidingLayers, kvBytesPerToken, marginalKvBytesPerToken } from './kv';
import {
  DEFAULT_HOST_BANDWIDTH,
  maxContextThatFits,
  normalizeRig,
  normalizeUsage,
  planPlacement,
  type Placement,
} from './placement';
import {
  estimateDecode,
  estimatePrefill,
  type DecodeEstimate,
  type PrefillEstimate,
} from './speed';
import { weightBreakdown, type WeightBreakdown } from './weights';

export * from './types';
export * from './kv';
export * from './weights';
export * from './activations';
export * from './placement';
export * from './speed';
export * from './verdict';

/** Everything the UI needs to answer "can I run this, and how comfortably". */
export interface Scenario {
  model: ModelSpec;
  quant: QuantSpec;
  usage: UsageSpec;
  rig: Rig;
  runtime: RuntimeSpec;
  /** Bandwidth of host RAM, for the offload path. */
  hostBandwidth?: number;
}

export interface Evaluation {
  weights: WeightBreakdown;
  activations: ActivationBreakdown;
  placement: Placement;
  decode: DecodeEstimate;
  prefill: PrefillEstimate;
  /** KV per token with every layer caching — the figure that compares architectures. */
  kvBytesPerToken: number;
  /** KV cost of one more token at the current context. Lower for hybrid models past their window. */
  marginalKvBytesPerToken: number;
  /** Largest context this rig can hold at the current concurrency. */
  maxContextTokens: number;
  hasSlidingLayers: boolean;
}

/**
 * The single entry point. Everything beneath it is pure, so this is cheap enough to call on
 * every slider frame — which is what makes the Bench feel like direct manipulation rather
 * than a form you submit.
 *
 * Inputs are normalized once here so that placement, decode and prefill cannot disagree
 * about what a degenerate scenario (zero context, zero devices) means.
 */
export function evaluate(scenario: Scenario): Evaluation {
  const { model, quant, runtime, hostBandwidth = DEFAULT_HOST_BANDWIDTH } = scenario;
  const usage = normalizeUsage(scenario.usage);
  const rig = normalizeRig(scenario.rig);

  const placement = planPlacement(model, quant, usage, rig, runtime);

  return {
    weights: weightBreakdown(model, quant),
    activations: activationBreakdown(model, usage, runtime),
    placement,
    decode: estimateDecode(model, quant, usage, rig, runtime, placement, hostBandwidth),
    prefill: estimatePrefill(model, quant, usage, rig, runtime, placement, hostBandwidth),
    kvBytesPerToken: kvBytesPerToken(model, usage.kvPrecision),
    marginalKvBytesPerToken: marginalKvBytesPerToken(model, usage.contextTokens, usage.kvPrecision),
    maxContextTokens: maxContextThatFits(model, quant, usage, rig, runtime),
    hasSlidingLayers: hasSlidingLayers(model),
  };
}

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
export * from './envelope';
export * from './matrix';

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
  /** Largest context this rig can hold with every weight resident. */
  maxContextTokens: number;
  /**
   * Largest context that can actually be run, allowing weights to spill to host RAM. Equal to
   * `maxContextTokens` whenever nothing is offloaded, and far larger when something is.
   */
  runnableContextTokens: number;
  /**
   * The context actually selected, after normalization.
   *
   * Carried so a caller can tell "the hardware would hold more" from "the hardware would hold
   * more and the model would not accept it" — `maxContextThatFits` already caps at the model's
   * own limit, so at that limit the two figures coincide and headroom stops meaning growth.
   */
  contextTokens: number;
  hasSlidingLayers: boolean;
}

/** Where the bytes land and how fast it runs — the three answers, without the context limits. */
export interface ScenarioEstimate {
  placement: Placement;
  decode: DecodeEstimate;
  prefill: PrefillEstimate;
}

/**
 * The cheap half of `evaluate`: one placement and the two speed estimates over it.
 *
 * Separated because `maxContextTokens` and `runnableContextTokens` are each a binary search over
 * the model's whole context range — roughly twenty `planPlacement` calls apiece — and a caller
 * that wants only the three answers above was paying about forty times what it used. The verdict
 * layer is that caller: it re-evaluates every archetype at that archetype's own scenario, discards
 * both limits every time, and does it on each render of the strip.
 *
 * A narrower function rather than a `skipLimits` flag on `evaluate`, so the discarded work is
 * unreachable rather than merely opt-out. `evaluate` is defined as this plus the extras, which is
 * what stops the two drifting about how a scenario is normalized.
 */
export function estimateScenario(scenario: Scenario): ScenarioEstimate {
  const { model, quant, runtime, hostBandwidth = DEFAULT_HOST_BANDWIDTH } = scenario;
  const usage = normalizeUsage(scenario.usage);
  const rig = normalizeRig(scenario.rig);

  const placement = planPlacement(model, quant, usage, rig, runtime);

  return {
    placement,
    decode: estimateDecode(model, quant, usage, rig, runtime, placement, hostBandwidth),
    prefill: estimatePrefill(model, quant, usage, rig, runtime, placement, hostBandwidth),
  };
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
  const { model, quant, runtime } = scenario;
  const usage = normalizeUsage(scenario.usage);
  const rig = normalizeRig(scenario.rig);

  const { placement, decode, prefill } = estimateScenario(scenario);

  return {
    weights: weightBreakdown(model, quant),
    activations: activationBreakdown(model, usage, runtime),
    placement,
    decode,
    prefill,
    // The runtime matters here for the same reason it does in placement: llama.cpp's q8_0 cache
    // costs more than its nominal byte. Omitting it made the headline bytes-per-token figure
    // disagree with the total placement charges for the same cache — 6% at q8, 12% at q4.
    kvBytesPerToken: kvBytesPerToken(model, usage.kvPrecision, runtime),
    marginalKvBytesPerToken: marginalKvBytesPerToken(
      model,
      usage.contextTokens,
      usage.kvPrecision,
      runtime
    ),
    maxContextTokens: maxContextThatFits(model, quant, usage, rig, runtime),
    runnableContextTokens: maxContextThatFits(model, quant, usage, rig, runtime, {
      allowOffload: true,
    }),
    contextTokens: usage.contextTokens,
    hasSlidingLayers: hasSlidingLayers(model),
  };
}

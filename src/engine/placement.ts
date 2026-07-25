import type { DeviceClass, ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { activationBytes } from './activations';
import { kvBytesTotal } from './kv';
import { weightBytes } from './weights';

/**
 * Where the bytes actually land.
 *
 * Two things here are routinely wrong elsewhere:
 *
 *   1. **Allocatable is not capacity.** A 128 GB Mac cannot hand 128 GB to a model — macOS
 *      caps GPU-wired memory near 75% of RAM by default. Strix Halo exposes 96 of its 128 GB.
 *      vLLM reserves a fixed fraction up front. Sizing against the number on the box is how
 *      you get a configuration that reports "fits" and then OOMs on load.
 *   2. **Offload is a cliff, not a slope.** When weights spill to host RAM over PCIe, the
 *      spilled fraction reads at a small fraction of device bandwidth. It is the most common
 *      reason a working setup is inexplicably slow, and it should be visible in the output
 *      rather than buried in an efficiency constant.
 */

/** Typical dual-channel DDR5 desktop bandwidth — the speed offloaded weights read at. */
export const DEFAULT_HOST_BANDWIDTH = 80e9;

/** Device classes as prose. The enum reads fine in code and badly in a sentence. */
const DEVICE_CLASS_PROSE: Record<DeviceClass, string> = {
  'discrete-gpu': 'discrete GPUs',
  'unified-soc': 'unified-memory hardware',
  'cpu-ram': 'CPU and system RAM',
};

export interface Placement {
  fits: boolean;

  /** Per-device figures, after tensor-parallel sharding across the rig. */
  weightBytesPerDevice: number;
  kvBytesPerDevice: number;
  activationBytesPerDevice: number;
  usedBytesPerDevice: number;
  allocatableBytesPerDevice: number;

  /** Totals across the whole rig, for the headline readout. */
  totalWeightBytes: number;
  totalKvBytes: number;

  /** Spare room per device; negative when over. */
  headroomBytes: number;
  utilization: number;

  /**
   * Fraction of weights that must live in host RAM. Zero when everything fits. Only
   * meaningful for discrete GPUs — a unified-memory machine has no faster tier to fall
   * back from, so an over-budget config there simply does not run.
   */
  offloadFraction: number;
  /** True when the configuration is over budget and offload cannot rescue it. */
  impossible: boolean;

  /** Set when the runtime cannot drive this class of device at all. */
  unsupported?: string;
}

/** Memory a single device can actually give the model, after the runtime takes its cut. */
export function allocatablePerDevice(rig: Rig, runtime: RuntimeSpec): number {
  const { device } = rig;
  const ceiling =
    runtime.preallocFraction === undefined
      ? device.allocatableBytes
      : Math.min(device.allocatableBytes, device.capacityBytes * runtime.preallocFraction);
  return Math.max(0, ceiling);
}

/**
 * Clamp a scenario to values every module agrees on.
 *
 * Without this the modules disagree at the boundaries: `kvBytesTotal` would drop the entire
 * KV term at concurrency 0 and report that anything fits, while `estimateDecode` clamps to a
 * single sequence and reports a real throughput for it. `maxContextThatFits` inherits the
 * former and would claim full context on hardware that cannot hold it.
 */
/**
 * `Math.max(1, Math.floor(NaN))` is NaN, so a plain clamp would pass garbage straight through
 * and paint NaN across every field of the result. Scenarios arrive from sliders and from
 * hand-editable querystrings, where `Number(params.get('ctx'))` on nonsense yields NaN.
 */
function positiveInt(value: number, fallback = 1): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : fallback;
}

export function normalizeUsage(usage: UsageSpec): UsageSpec {
  return {
    ...usage,
    contextTokens: positiveInt(usage.contextTokens),
    concurrency: positiveInt(usage.concurrency),
    ...(usage.promptTokens === undefined ? {} : { promptTokens: positiveInt(usage.promptTokens) }),
  };
}

export function normalizeRig(rig: Rig): Rig {
  return { ...rig, count: positiveInt(rig.count) };
}

export function planPlacement(
  model: ModelSpec,
  quant: QuantSpec,
  rawUsage: UsageSpec,
  rawRig: Rig,
  runtime: RuntimeSpec
): Placement {
  const usage = normalizeUsage(rawUsage);
  const rig = normalizeRig(rawRig);

  const totalWeightBytes = weightBytes(model, quant);
  const totalKvBytes = kvBytesTotal(
    model,
    usage.contextTokens,
    usage.concurrency,
    usage.kvPrecision
  );
  const activations = activationBytes(model, usage, runtime);

  // Tensor parallelism shards weights and KV evenly; activations are per-device, not shared.
  const shards = rig.count;
  const weightBytesPerDevice = totalWeightBytes / shards;
  const kvBytesPerDevice = totalKvBytes / shards;
  const usedBytesPerDevice = weightBytesPerDevice + kvBytesPerDevice + activations;

  const allocatableBytesPerDevice = allocatablePerDevice(rig, runtime);
  const headroomBytes = allocatableBytesPerDevice - usedBytesPerDevice;
  const fits = headroomBytes >= 0;

  // Only a discrete GPU has somewhere slower to spill to. On unified memory or CPU RAM the
  // pool in question *is* system memory, so over budget means it does not run.
  const canOffload = rig.device.class === 'discrete-gpu';
  const deficit = Math.max(0, -headroomBytes);
  const offloadFraction =
    fits || !canOffload ? 0 : Math.min(1, deficit / Math.max(weightBytesPerDevice, 1));

  // Even offloading every weight leaves KV and activations, which must sit on the device.
  const impossible =
    !fits && (!canOffload || kvBytesPerDevice + activations > allocatableBytesPerDevice);

  const unsupported = !runtime.supports.includes(rig.device.class)
    ? `${runtime.label} does not run on ${DEVICE_CLASS_PROSE[rig.device.class]}.`
    : runtime.requiresVendor && rig.device.vendor !== runtime.requiresVendor
      ? `${runtime.label} runs only on ${runtime.requiresVendor} hardware.`
      : undefined;

  return {
    fits,
    weightBytesPerDevice,
    kvBytesPerDevice,
    activationBytesPerDevice: activations,
    usedBytesPerDevice,
    allocatableBytesPerDevice,
    totalWeightBytes,
    totalKvBytes,
    headroomBytes,
    utilization: usedBytesPerDevice / allocatableBytesPerDevice,
    offloadFraction,
    impossible,
    unsupported,
  };
}

/**
 * Largest context that still fits, at the given concurrency. Binary search rather than an
 * inverted formula, because hybrid sliding-window models make KV a non-linear function of
 * context — the closed form would be wrong for exactly the models it matters most for.
 */
export interface ContextLimitOptions {
  /**
   * Count a context as reachable when the weights spill to host RAM, rather than requiring a
   * fully resident placement.
   *
   * The distinction is not cosmetic. `fits` is false for *any* offloaded configuration, even at
   * a one-token context, so the resident limit collapses to zero the moment a model is too big
   * for the card — reporting "caps out at 0" for a rig that would happily hold 128K of KV once
   * the weights are offloaded. Ask for the resident figure when saying what fits comfortably,
   * and the offload-aware one when saying what can actually be run.
   */
  allowOffload?: boolean;
}

export function maxContextThatFits(
  model: ModelSpec,
  quant: QuantSpec,
  usage: UsageSpec,
  rig: Rig,
  runtime: RuntimeSpec,
  { allowOffload = false }: ContextLimitOptions = {}
): number {
  const attempt = (contextTokens: number) => {
    const placement = planPlacement(model, quant, { ...usage, contextTokens }, rig, runtime);
    if (placement.unsupported) return false;
    return allowOffload ? !placement.impossible : placement.fits;
  };

  if (!attempt(1)) return 0;

  let low = 1;
  let high = model.maxContext;
  if (attempt(high)) return high;

  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2);
    if (attempt(mid)) low = mid;
    else high = mid;
  }
  return low;
}

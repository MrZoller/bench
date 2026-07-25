import type { ModelSpec, DeviceSpec, QuantSpec, RuntimeSpec, UsageSpec } from './types';
import { planPlacement } from './placement';
import { estimateDecode, estimatePrefill } from './speed';

/**
 * Every model against every device, at one usage setting.
 *
 * The Bench and the Envelope both answer questions about a configuration you have already
 * chosen. This is the surface for the question that comes before that — "what are my options" —
 * and it is the one that makes the capacity/bandwidth/compute triangle visible, because the
 * three measures disagree about which hardware is best and the disagreement is the point.
 *
 * Deliberately three separate readings rather than a composite score. A device that holds a
 * model nobody can wait for and a device that runs a smaller one instantly are both "good" on
 * some axis, and averaging them produces a number that recommends neither.
 */

export type MatrixMeasure = 'fit' | 'decode' | 'ttft';

export interface MatrixCell {
  modelId: string;
  deviceId: string;
  /** Which format this cell was actually evaluated at — not always the one selected. */
  quantId: string;
  /** False when the pair cannot run at all — no measure is meaningful then. */
  runs: boolean;
  /** Why not, when it does not. */
  blockedBy?: string;
  /** Used bytes over allocatable. Above 1 means it did not fit resident. */
  utilization: number;
  offloadFraction: number;
  tokensPerSec: number;
  ttftSeconds: number;
}

export interface MatrixRequest {
  models: readonly ModelSpec[];
  devices: readonly DeviceSpec[];
  runtime: RuntimeSpec;
  usage: UsageSpec;
  deviceCount: number;
  /**
   * The quantization to use for a given pair.
   *
   * A function rather than one value, because a single format cannot serve the whole grid: an
   * expert-only scheme like MXFP4 is a no-op on a dense model, and forcing it across the
   * catalog blanked more than half the rows for a reason that has nothing to do with the
   * hardware being compared. Callers substitute something applicable and say so.
   */
  quantFor: (model: ModelSpec, device: DeviceSpec) => QuantSpec;
}

export function computeMatrix(request: MatrixRequest): MatrixCell[][] {
  const { models, devices, quantFor, runtime, usage, deviceCount } = request;

  return models.map((model) =>
    devices.map((device) => {
      const quant = quantFor(model, device);
      const base = { modelId: model.id, deviceId: device.id, quantId: quant.id };
      const rig = { device, count: deviceCount };
      const placement = planPlacement(model, quant, usage, rig, runtime);

      if (placement.unsupported || placement.impossible) {
        return {
          ...base,
          runs: false,
          blockedBy: placement.unsupported ?? 'Does not fit',
          utilization: placement.utilization,
          offloadFraction: 0,
          tokensPerSec: 0,
          ttftSeconds: 0,
        };
      }

      const decode = estimateDecode(model, quant, usage, rig, runtime, placement);
      const prefill = estimatePrefill(model, quant, usage, rig, runtime, placement);

      return {
        ...base,
        runs: true,
        utilization: placement.utilization,
        offloadFraction: placement.offloadFraction,
        tokensPerSec: decode.perUserTokensPerSec,
        ttftSeconds: prefill.ttftSeconds,
      };
    })
  );
}

/**
 * The value a measure reads off a cell, normalised so 1 is best and 0 is worst.
 *
 * Normalised against the grid rather than against an absolute scale, because the useful
 * comparison is between the options in front of you: a heatmap where every cell is pale because
 * nothing reaches some theoretical maximum tells you nothing about which to buy.
 */
export function measureValue(cell: MatrixCell, measure: MatrixMeasure): number | undefined {
  if (!cell.runs) return undefined;
  switch (measure) {
    case 'fit':
      // Headroom, so more is better and an offloaded fit scores below any resident one.
      return cell.offloadFraction > 0 ? 0 : Math.max(0, 1 - cell.utilization);
    case 'decode':
      return cell.tokensPerSec;
    case 'ttft':
      // Inverted: less time is better.
      return cell.ttftSeconds > 0 ? 1 / cell.ttftSeconds : 0;
  }
}

/** Largest value across the grid for a measure, for scaling the ramp. */
export function measureMax(cells: MatrixCell[][], measure: MatrixMeasure): number {
  let max = 0;
  for (const row of cells) {
    for (const cell of row) {
      const value = measureValue(cell, measure);
      if (value !== undefined && value > max) max = value;
    }
  }
  return max;
}

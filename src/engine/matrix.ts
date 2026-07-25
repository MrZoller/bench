import type { ModelSpec, DeviceSpec, QuantSpec, RuntimeSpec, UsageSpec } from './types';
import { planPlacement, raisingCeilingWouldHelp } from './placement';
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
  /** The context it was evaluated at, capped at this model's own limit. */
  contextTokens: number;
  /** False when the pair cannot run at all — no measure is meaningful then. */
  runs: boolean;
  /** Why not, when it does not. */
  blockedBy?: string;
  /**
   * Set when the cell is over the *default* allocation on a machine that lets you raise it.
   *
   * A setting and a hardware limit are not the same answer, and this grid is read as a shortlist:
   * DeepSeek V3 at Q5_K_M needs about 445 GiB, which is past the 512 GB Mac Studio's 384 GiB
   * default and inside the 512 it can be tuned to. Collapsing that into the same "will not run"
   * as a configuration beyond physical capacity strikes a machine off the list over a checkbox.
   * The Envelope and Telemetry both preserved the distinction; this surface dropped it.
   */
  raiseCeilingWouldHelp?: boolean;
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
      /**
       * Clamped per row, because `maxContext` differs across the grid — 32K on some models,
       * 164K on others — and neither `planPlacement` nor `estimateDecode` knows about it. Left
       * unclamped, a 40K model was scored for a 128K request it cannot accept, and clicking
       * that cell then produced different numbers in the Bench, where `coerce` does clamp.
       */
      const contextTokens = Math.min(usage.contextTokens, model.maxContext);
      const cellUsage: UsageSpec = {
        ...usage,
        contextTokens,
        promptTokens: Math.min(usage.promptTokens ?? contextTokens, contextTokens),
      };

      const base = {
        modelId: model.id,
        deviceId: device.id,
        quantId: quant.id,
        contextTokens,
      };
      const rig = { device, count: deviceCount };
      const placement = planPlacement(model, quant, cellUsage, rig, runtime);

      if (placement.unsupported || placement.impossible) {
        // Same call `raisingCeilingWouldHelp` serves in the Envelope and Telemetry, rather than a
        // third re-derivation of "is this a setting or a wall" — the two that already existed
        // disagreed once, which is why it lives in `placement.ts`.
        const raiseable =
          placement.unsupported === undefined &&
          raisingCeilingWouldHelp(device, placement.usedBytesPerDevice);

        return {
          ...base,
          runs: false,
          blockedBy:
            placement.unsupported ?? (raiseable ? 'Past the default allocation' : 'Does not fit'),
          ...(raiseable ? { raiseCeilingWouldHelp: true } : {}),
          utilization: placement.utilization,
          offloadFraction: 0,
          tokensPerSec: 0,
          ttftSeconds: 0,
        };
      }

      const decode = estimateDecode(model, quant, cellUsage, rig, runtime, placement);
      const prefill = estimatePrefill(model, quant, cellUsage, rig, runtime, placement);

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

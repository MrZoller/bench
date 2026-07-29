import type { ModelSpec, DeviceSpec, QuantSpec, RuntimeSpec, UsageSpec } from './types';
import {
  clampUsageToContext,
  planPlacement,
  raisingCeilingWouldHelp,
  wasEvaluated,
} from './placement';
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
  /**
   * Whether the pair was judged on its numbers, as against turned away on a categorical ground.
   *
   * `runs` collapses two unlike failures — a runtime that cannot load this model on this device
   * at all, and one whose bytes were counted and did not fit — and `blockedBy` carries the
   * difference only as prose, so a caller cannot read it. One caller has to: the stand-in warning
   * asks whether any figure on this grid came from a format the runtime cannot really load, and a
   * capacity failure *is* such a figure. Its verdict, its tooltip and its raise-the-ceiling
   * recommendation all rest on the stand-in's bit width. Only a cell refused on a ground that
   * never consulted the arithmetic rests on nothing at all. Raised by Codex on PR #32.
   */
  evaluated: boolean;
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
      // Through `clampUsageToContext` so `cachedPrefixTokens` is held to the row's context too —
      // it is part of the working set in the same way the prompt is, and left unclamped a prefix
      // past the model's own limit took one cell from 16 s to 273 s. The prompt still defaults to
      // the whole context here, which is this grid's reading and not the Envelope's: a row is
      // scored for the largest request it can accept.
      const cellUsage: UsageSpec = clampUsageToContext(
        { ...usage, promptTokens: usage.promptTokens ?? contextTokens },
        contextTokens
      );

      const base = {
        modelId: model.id,
        deviceId: device.id,
        quantId: quant.id,
        contextTokens,
      };
      const rig = { device, count: deviceCount };
      const placement = planPlacement(model, quant, cellUsage, rig, runtime);

      if (placement.unsupported || placement.impossible) {
        // Shared with the Bench's banner, which asks the same question of a single placement — see
        // `wasEvaluated`. Absent an `unsupported`, the bytes were counted and came up short, so the
        // cell's verdict did come from whatever format the row was scored at.
        const evaluated = wasEvaluated(placement);
        // Same call `raisingCeilingWouldHelp` serves in the Envelope and Telemetry, rather than a
        // third re-derivation of "is this a setting or a wall" — the two that already existed
        // disagreed once, which is why it lives in `placement.ts`.
        const raiseable =
          evaluated && raisingCeilingWouldHelp(device, placement.usedBytesPerDevice);

        return {
          ...base,
          runs: false,
          evaluated,
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
        evaluated: true,
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

/**
 * The two ends of what a measure actually spans on this grid, and the top value for scaling.
 *
 * The cells rather than only their numbers, because the number a reader needs at the end of a ramp
 * is the one the cell itself reports — `tokensPerSec`, `ttftSeconds` — and `measureValue` is not
 * always that. It inverts TTFT so that larger is better, so the ramp's *low* end is the *longest*
 * wait, and recovering the seconds from `1 / value` is both a second derivation of a figure the cell
 * already holds and a floating-point round trip. Handing back the cell lets a label read the field.
 *
 * **There is deliberately no `min`.** Nothing scales against the bottom: `fill` anchors its log
 * curve at zero rather than at the lowest cell, so a minimum would be a number no mark is derived
 * from — and the legend's low label comes off `low` for the reason above. `max` is here because it
 * *is* what the ramp divides by, and deriving it twice is how a scale and its legend come to
 * disagree about the same grid.
 */
export interface MeasureRange {
  /** The cell at the worst end of the ramp. */
  low: MatrixCell;
  /** The cell at the best end. */
  high: MatrixCell;
  /** `high`'s value — what the ramp is scaled against. */
  max: number;
}

/**
 * The span a measure covers across the grid, or `undefined` when nothing ran.
 *
 * Ordered by `measureValue`, which is the ramp's own ordering rather than any cell field: for TTFT
 * that puts the slowest cell at `low`, which is the point. A grid where no pair runs has no span —
 * it also has no ink, since `fill` returns the empty fill for every cell — so the absence is a
 * value the caller can render rather than a zero it has to interpret.
 */
export function measureRange(
  cells: MatrixCell[][],
  measure: MatrixMeasure
): MeasureRange | undefined {
  let range: MeasureRange | undefined;
  let min = Number.POSITIVE_INFINITY;

  for (const row of cells) {
    for (const cell of row) {
      const value = measureValue(cell, measure);
      if (value === undefined) continue;
      if (range === undefined) {
        range = { low: cell, high: cell, max: value };
        min = value;
        continue;
      }
      if (value < min) {
        range.low = cell;
        min = value;
      }
      if (value > range.max) {
        range.high = cell;
        range.max = value;
      }
    }
  }

  return range;
}

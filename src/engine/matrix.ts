import type { ModelSpec, DeviceSpec, QuantSpec, RuntimeSpec, UsageSpec } from './types';
import {
  clampUsageToContext,
  planPlacement,
  raisingCeilingWouldHelp,
  wasEvaluated,
} from './placement';
import { estimateDecode, estimatePrefill } from './speed';
import { MEASURE_DIRECTION, measureOf, type Measure } from './measure';

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

/**
 * The measure vocabulary, under the name this grid's callers already use.
 *
 * Shared with the Envelope since #65 — the same three questions over different axes — and the type
 * lives in `measure.ts` beside the direction each one runs in. An alias rather than a second union,
 * because two spellings of one type is how a surface comes to accept a measure the other cannot
 * read.
 */
export type MatrixMeasure = Measure;

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
 * The value a measure reads off a cell, in the measure's own units — or nothing at all.
 *
 * **Not oriented**, since #97: the reading is `measureOf` and the direction is `MEASURE_DIRECTION`,
 * both shared with the Envelope so each is stated once. What stays here is the set of questions only
 * this grid can answer — whether the cell ran, whether a spill puts it off the scale rather than
 * merely low on it, and whether a reading exists at all.
 *
 * Read against the grid rather than against an absolute scale, because the useful comparison is
 * between the options in front of you: a heatmap where every cell is pale because nothing reaches
 * some theoretical maximum tells you nothing about which to buy.
 */
export function measureValue(cell: MatrixCell, measure: MatrixMeasure): number | undefined {
  if (!cell.runs) return undefined;
  // An offloaded fit scores below any resident one — a categorical answer rather than a degree,
  // since the weights are crossing the bus whatever the headroom arithmetic says.
  if (measure === 'fit' && cell.offloadFraction > 0) return 0;
  /**
   * A cell that ran and was not timed — at either end — is off this scale rather than on it.
   *
   * **Both ends, and each was wrong in the opposite direction under the old orientation.** A zero
   * means never timed, and in seconds zero is instantaneous: it would take the brightest step and
   * drag the domain's floor to it, where under the reciprocal it was `1 / 0` guarded to the worst
   * reading. An `Infinity` is `estimatePrefill`'s deliberate answer for a device whose compute rate
   * the catalog does not state (`index.test.ts` pins it), and under the reciprocal it became `0` —
   * the worst reading, by luck rather than by design. In seconds it poisons the domain: `max` is
   * `Infinity`, the span is `Infinity`, and every placement on the grid comes out `NaN`, which
   * indexes the ramp with `undefined` and leaves *every* cell unpainted. So the fix that made the
   * zero explicit has to make the infinity explicit too, or one guard replaces two.
   *
   * `undefined` rather than the darkest step, because that is what this function already means by
   * "no reading": the caller paints a hole, and a hole is what a cell nobody could time is.
   */
  if (measure === 'ttft' && !(cell.ttftSeconds > 0 && Number.isFinite(cell.ttftSeconds)))
    return undefined;
  return measureOf(cell, measure);
}

/**
 * The two ends of what a measure actually spans on this grid, and the domain its ramp is placed in.
 *
 * The cells rather than only their numbers, because the number a reader needs at the end of a ramp
 * is the one the cell itself reports — `tokensPerSec`, `ttftSeconds` — and handing back the cell
 * lets a label read the field rather than re-deriving it.
 *
 * **`low` is a tie in the ordinary case, and a caller reading anything but the ramp value off it is
 * reading an arbitrary cell.** `measureValue('fit')` returns exactly 0 for every offloaded cell by
 * design, so on any grid where something spills — most of them — the low end is a whole population
 * and `low` is whichever member comes first in row-major order. That is safe only because every one
 * of them yields the same *value*, which is what the legend prints; see `rampEnd`, which argues why
 * the fit label is the ramp's figure rather than the worst spiller's sentence.
 *
 * **`domain` replaced a bare `max`, and the floor is the half that changed**
 * ([#97](https://github.com/MrZoller/headroom/issues/97)). The old field was `high`'s value and the
 * caller floored the ramp at zero, on the argument that this grid runs from a desktop CPU to a B200
 * so its bottom really is near nothing. That argument holds for headroom and for tokens per second
 * and does not survive being pointed at a latency. Zero seconds is not a reading any cell can have,
 * and anchoring a log curve at an unreachable point is what took `placed` down to roughly
 * `t_fastest / t`: a cell ten times slower than the grid's fastest landed on step 0, on a grid whose
 * span is far more than tenfold. **A zero floor anchors the ramp's worst end, and for a
 * lower-is-better measure the worst end is unbounded** — there is nothing to anchor to, so the
 * domain is the grid's own span. Stated here rather than at the call site because the ramp and the
 * legend are two readings of one grid, and deriving the domain twice is how they come to disagree.
 */
export interface MeasureRange {
  /** The cell at the worst end of the ramp. */
  low: MatrixCell;
  /** The cell at the best end. */
  high: MatrixCell;
  /**
   * What the ramp is placed inside, in the measure's own units and always stated low-to-high.
   *
   * Which end is *good* is `MEASURE_DIRECTION`, not this — see `magnitudeFill`.
   */
  domain: { min: number; max: number };
}

/**
 * The span a measure covers across the grid, or `undefined` when nothing ran.
 *
 * Ordered by `measureValue` **and the measure's direction**, so `low` is the worst cell for a
 * lower-is-better measure too: on TTFT that is the largest number rather than the smallest, which is
 * the sign flip the old inverted value was hiding. A grid where no pair runs has no span — it also
 * has no ink, since `fill` returns the empty fill for every cell — so the absence is a value the
 * caller can render rather than a zero it has to interpret.
 */
export function measureRange(
  cells: MatrixCell[][],
  measure: MatrixMeasure
): MeasureRange | undefined {
  const higherIsBetter = MEASURE_DIRECTION[measure] === 'higher';

  let smallest: { cell: MatrixCell; value: number } | undefined;
  let largest: { cell: MatrixCell; value: number } | undefined;

  for (const row of cells) {
    for (const cell of row) {
      const value = measureValue(cell, measure);
      if (value === undefined) continue;
      if (smallest === undefined || value < smallest.value) smallest = { cell, value };
      if (largest === undefined || value > largest.value) largest = { cell, value };
    }
  }

  if (smallest === undefined || largest === undefined) return undefined;

  return {
    low: (higherIsBetter ? smallest : largest).cell,
    high: (higherIsBetter ? largest : smallest).cell,
    /**
     * Floored at zero only where zero is a reading — see the interface above. For headroom it is
     * one: `measureValue` returns exactly 0 for every offloaded cell. For tokens per second it is
     * the physical bottom of the scale, and keeping it is what lets this grid say "worse" rather
     * than "worst here".
     */
    domain: higherIsBetter
      ? { min: 0, max: largest.value }
      : { min: smallest.value, max: largest.value },
  };
}

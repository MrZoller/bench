import type { ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { planPlacement } from './placement';
import { estimateDecode } from './speed';

/**
 * The feasibility field: how a configuration behaves across the whole usage plane.
 *
 * The Bench answers "does this work at the settings I picked". This answers the question people
 * actually have, which is "how much room do I have before it stops working" — a region rather
 * than a point. Context on one axis, concurrent users on the other, because those are the two
 * things a deployment grows in and the two that multiply into the KV cache.
 *
 * Pure, like the rest of the engine, and cheap enough to sweep a whole grid on every render:
 * each cell is a placement plus a decode estimate over a handful of numbers.
 */

/** Why a cell is not comfortable, in the order a reader would want to hear it. */
export type CellState =
  /** Fits with room, and fast enough to use interactively. */
  | 'comfortable'
  /** Fits, but either close to the ceiling or slow enough to notice. */
  | 'tight'
  /** Runs only because weights spill to host RAM. */
  | 'offloaded'
  /** Does not run at all. */
  | 'over';

export interface EnvelopeCell {
  contextTokens: number;
  concurrency: number;
  state: CellState;
  /**
   * Why a cell is tight, since it means two unrelated things — nearly full, or too slow — and
   * a reader looking at one amber square cannot tell which without being told.
   */
  tightBecause?: 'capacity' | 'speed';
  /** Per-user decode, so the table can say what "tight" costs. */
  tokensPerSec: number;
  utilization: number;
}

export interface EnvelopeGrid {
  contexts: readonly number[];
  concurrencies: readonly number[];
  /** Row-major: `cells[concurrencyIndex][contextIndex]`. */
  cells: EnvelopeCell[][];
}

export interface EnvelopeRequest {
  model: ModelSpec;
  quant: QuantSpec;
  runtime: RuntimeSpec;
  rig: Rig;
  usage: UsageSpec;
  contexts: readonly number[];
  concurrencies: readonly number[];
  /** Below this, per-user decode is too slow to call comfortable. */
  usableTokensPerSec: number;
  /** Above this share of the ceiling, a cell is tight even when it is fast. */
  tightUtilization: number;
  /**
   * The rate as the UI will *print* it.
   *
   * Injected rather than assumed, in the same shape as `prefillAt` in the verdict layer. The
   * classification has to be made on the figure a reader sees: printing a rounded "15 tok/s"
   * beside a state decided on 14.7 is a label contradicting the number next to it, which this
   * project has now shipped three times in other places.
   */
  displayedRate?: (tokensPerSec: number) => number;
}

export function computeEnvelope(request: EnvelopeRequest): EnvelopeGrid {
  const {
    model,
    quant,
    runtime,
    rig,
    usage,
    contexts,
    concurrencies,
    usableTokensPerSec,
    tightUtilization,
    displayedRate = (n) => n,
  } = request;

  const cells = concurrencies.map((concurrency) =>
    contexts.map((contextTokens) => {
      const cellUsage: UsageSpec = { ...usage, contextTokens, concurrency };
      const placement = planPlacement(model, quant, cellUsage, rig, runtime);

      // A runtime that cannot drive this hardware makes every cell impossible, not merely
      // over budget — there is no configuration of context and concurrency that rescues it.
      if (placement.unsupported || placement.impossible) {
        return {
          contextTokens,
          concurrency,
          state: 'over' as const,
          tokensPerSec: 0,
          utilization: placement.utilization,
        };
      }

      const decode = estimateDecode(model, quant, cellUsage, rig, runtime, placement);
      const tokensPerSec = decode.perUserTokensPerSec;
      const slow = displayedRate(tokensPerSec) < usableTokensPerSec;
      const full = placement.utilization > tightUtilization;

      // Offload is called out separately rather than folded into "tight": it runs, but for a
      // structural reason a user can act on, and it is the single most common explanation for
      // a setup being mysteriously slow.
      const state: CellState =
        placement.offloadFraction > 0 ? 'offloaded' : slow || full ? 'tight' : 'comfortable';

      return {
        contextTokens,
        concurrency,
        state,
        // Capacity named first when both apply: running out of memory is the harder wall, and
        // the one a user cannot trade away by accepting a slower answer.
        ...(state === 'tight'
          ? { tightBecause: full ? ('capacity' as const) : ('speed' as const) }
          : {}),
        tokensPerSec,
        utilization: placement.utilization,
      };
    })
  );

  return { contexts, concurrencies, cells };
}

/**
 * The largest context that stays comfortable at each concurrency — the frontier of the region.
 *
 * Returned separately from the grid because it is what someone actually reads off the picture:
 * not "which cells are green" but "how far can I push this before it stops being pleasant".
 */
export function comfortableFrontier(grid: EnvelopeGrid): (number | undefined)[] {
  return grid.cells.map((row) => {
    let best: number | undefined;
    for (const cell of row) {
      if (cell.state === 'comfortable') best = cell.contextTokens;
    }
    return best;
  });
}

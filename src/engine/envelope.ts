import type { ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from './types';
import { planPlacement, raisingCeilingWouldHelp } from './placement';
import { estimateDecode, estimatePrefill } from './speed';

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
  /** Fits with room, fast enough to use interactively, and starts answering promptly. */
  | 'comfortable'
  /** Fits, but close to the ceiling, slow to type, or slow to start. */
  | 'tight'
  /** Runs only because weights spill to host RAM. */
  | 'offloaded'
  /** Over what the hardware can hold. */
  | 'over'
  /**
   * The runtime cannot drive this hardware at all.
   *
   * Its own state rather than folded into `over`, because the two need opposite advice: `over`
   * says buy more memory or quantize harder, `unsupported` says pick a different runtime. The
   * grid is uniformly one or the other, so a legend that called both "past what this hardware
   * can hold" was telling an MLX-on-RTX-5090 user to go shopping.
   */
  | 'unsupported';

export interface EnvelopeCell {
  contextTokens: number;
  concurrency: number;
  state: CellState;
  /**
   * Why a cell is tight, since it means three unrelated things — nearly full, slow to type, or
   * slow to start — and a reader looking at one amber square cannot tell which without being
   * told.
   */
  tightBecause?: 'capacity' | 'speed' | 'latency';
  /**
   * Why a cell does not run: past the hardware, or merely past a ceiling that can be raised.
   *
   * macOS caps wired GPU memory near 75% of RAM and AMD exposes a Variable Graphics Memory
   * setting, so on those machines the catalog figure is an untuned default rather than a limit.
   * The Telemetry tile already says so; the grid painted the same cells "past what this hardware
   * can hold", which contradicts the tile and hides the one change that would fix it.
   */
  overBecause?: 'capacity' | 'allocation';
  /** Per-user decode, so the table can say what "tight" costs. */
  tokensPerSec: number;
  /** Time to first token, for the same reason: it is half of what "usable" means. */
  ttftSeconds: number;
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
   * Above this first-token latency, a cell is tight however fast it then types.
   *
   * Decode speed alone was the whole test, so a resident configuration with a long prompt could
   * be painted green here while the Telemetry tile beside it read "Slow start" in red — the two
   * surfaces contradicting each other about one scenario. A minute of waiting is not comfortable
   * at any tokens per second.
   */
  usableTtftSeconds: number;
  /**
   * First-token latency as the UI will print it, for the same reason `displayedRate` exists.
   */
  displayedTtft?: (ttftSeconds: number) => number;
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
    usableTtftSeconds,
    displayedRate = (n) => n,
    displayedTtft = (n) => n,
  } = request;

  const cells = concurrencies.map((concurrency) =>
    contexts.map((contextTokens) => {
      /**
       * The prompt is clamped to the cell's own context, not carried across from the slider.
       *
       * `coerce` already enforces this for the selected scenario, and for the same reason: the
       * prompt is *part* of the context, so a 32K prompt in a 2K column describes a request that
       * cannot be made. Decode never read `promptTokens`, so carrying it through was harmless
       * until prefill was added here — at which point every column was timed for a prompt six of
       * seven of them cannot hold, and the whole region went amber at 41 s the moment the prompt
       * slider passed 16K.
       */
      const cellUsage: UsageSpec = {
        ...usage,
        contextTokens,
        concurrency,
        ...(usage.promptTokens === undefined
          ? {}
          : { promptTokens: Math.min(usage.promptTokens, contextTokens) }),
      };
      const placement = planPlacement(model, quant, cellUsage, rig, runtime);

      // A runtime that cannot drive this hardware is a different failure from running out of
      // room, and the fix is different too — neither is rescued by any context or concurrency,
      // but only one of them is about memory.
      if (placement.unsupported) {
        return {
          contextTokens,
          concurrency,
          state: 'unsupported' as const,
          tokensPerSec: 0,
          ttftSeconds: 0,
          utilization: placement.utilization,
        };
      }

      if (placement.impossible) {
        // Within what the platform will actually hand out is a different sentence from past the
        // hardware — and a different action. Physical capacity is the wrong bound: AMD's
        // Variable Graphics Memory stops at 96 of the Ryzen's 128 GiB, so a cell between those
        // two is past the machine as far as any setting is concerned. `raisingCeilingWouldHelp`
        // already knew that; this branch was re-deriving a weaker version of it.
        const raiseable = raisingCeilingWouldHelp(rig.device, placement.usedBytesPerDevice);

        return {
          contextTokens,
          concurrency,
          state: 'over' as const,
          overBecause: raiseable ? ('allocation' as const) : ('capacity' as const),
          tokensPerSec: 0,
          ttftSeconds: 0,
          utilization: placement.utilization,
        };
      }

      const decode = estimateDecode(model, quant, cellUsage, rig, runtime, placement);
      const prefill = estimatePrefill(model, quant, cellUsage, rig, runtime, placement);
      const tokensPerSec = decode.perUserTokensPerSec;
      const ttftSeconds = prefill.ttftSeconds;
      const slow = displayedRate(tokensPerSec) < usableTokensPerSec;
      const full = placement.utilization > tightUtilization;
      // Latency is half of usable. A cell that decodes at 40 tok/s after a 90-second wait is
      // not comfortable, and the tile next to this panel already says so.
      const slowStart = displayedTtft(ttftSeconds) > usableTtftSeconds;

      // Offload is called out separately rather than folded into "tight": it runs, but for a
      // structural reason a user can act on, and it is the single most common explanation for
      // a setup being mysteriously slow.
      const state: CellState =
        placement.offloadFraction > 0
          ? 'offloaded'
          : slow || full || slowStart
            ? 'tight'
            : 'comfortable';

      return {
        contextTokens,
        concurrency,
        state,
        // Capacity named first when several apply: running out of memory is the harder wall,
        // and the one a user cannot trade away by accepting a slower answer. Then decode, which
        // costs you every token, before first-token latency, which costs you once per turn.
        ...(state === 'tight'
          ? {
              tightBecause: full
                ? ('capacity' as const)
                : slow
                  ? ('speed' as const)
                  : ('latency' as const),
            }
          : {}),
        tokensPerSec,
        ttftSeconds,
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
